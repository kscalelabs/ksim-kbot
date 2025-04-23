# mypy: disable-error-code="override"
"""Defines simple task for training a walking policy for the default humanoid using an RNN actor."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, TypeVar

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import ksim
import xax
from jaxtyping import Array, PRNGKeyArray
from xax.nn.export import export

from .pseudo_ik import (
    NUM_INPUTS,
    NUM_JOINTS,
    KbotPseudoIKTask,
    KbotPseudoIKTaskConfig,
)

RNN_NUM_INPUTS = NUM_INPUTS - NUM_JOINTS


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AuxOutputs:
    log_probs: Array
    values: Array
    actor_carry: Array
    critic_carry: Array


class KbotRNNActor(eqx.Module):
    """RNN-based actor for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear
    min_std: float = eqx.static_field()
    max_std: float = eqx.static_field()
    var_scale: float = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        depth: int,
    ) -> None:
        num_outputs = NUM_JOINTS

        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=RNN_NUM_INPUTS,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for _ in range(depth)
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs * 2,
            key=key,
        )

        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale

    def forward(
        self,
        joint_pos_n: Array,
        joint_vel_n: Array,
        xyz_target_3: Array,
        quat_target_4: Array,
        carry: Array,
    ) -> tuple[distrax.Distribution, Array]:
        obs_n = jnp.concatenate(
            [
                joint_pos_n,  # NUM_JOINTS
                joint_vel_n,  # NUM_JOINTS
                xyz_target_3,  # 3
                quat_target_4,  # 4
            ],
            axis=-1,
        )
        return self.call_flat_obs(obs_n, carry)

    def call_flat_obs(self, obs_n: Array, carry: Array) -> tuple[distrax.Distribution, Array]:
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

        # Converts the output to a distribution.
        mean_n = out_n[..., :NUM_JOINTS]
        std_n = out_n[..., NUM_JOINTS:]

        # Softplus and clip to ensure positive standard deviations.
        std_n = jnp.clip((jax.nn.softplus(std_n) + self.min_std) * self.var_scale, max=self.max_std)

        dist_n = distrax.Normal(mean_n, std_n)
        return dist_n, jnp.stack(out_carries, axis=0)


class KbotRNNCritic(eqx.Module):
    """RNN-based critic for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        hidden_size: int,
        depth: int,
    ) -> None:
        num_inputs = RNN_NUM_INPUTS + NUM_JOINTS + 3
        num_outputs = 1

        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for _ in range(depth)
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs,
            key=key,
        )

    def forward(
        self,
        joint_pos_n: Array,
        joint_vel_n: Array,
        actuator_force_n: Array,
        xyz_target_3: Array,
        quat_target_4: Array,
        end_effector_pos_3: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        obs_n = jnp.concatenate(
            [
                joint_pos_n,  # NUM_JOINTS
                joint_vel_n,  # NUM_JOINTS
                actuator_force_n,  # NUM_JOINTS
                xyz_target_3,  # 3
                quat_target_4,  # 4
                end_effector_pos_3,  # 3
            ],
            axis=-1,
        )

        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

        return out_n, jnp.stack(out_carries, axis=0)


class KbotRNNModel(eqx.Module):
    actor: KbotRNNActor
    critic: KbotRNNCritic

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        hidden_size: int,
        depth: int,
    ) -> None:
        self.actor = KbotRNNActor(
            key,
            min_std=0.01,
            max_std=1.0,
            var_scale=0.5,
            hidden_size=hidden_size,
            depth=depth,
        )
        self.critic = KbotRNNCritic(
            key,
            hidden_size=hidden_size,
            depth=depth,
        )


@dataclass
class KbotPseudoIKRNNTaskConfig(KbotPseudoIKTaskConfig):
    pass


Config = TypeVar("Config", bound=KbotPseudoIKRNNTaskConfig)


class KbotPseudoIKRNNTask(KbotPseudoIKTask[Config], Generic[Config]):
    def get_model(self, key: PRNGKeyArray) -> KbotRNNModel:
        return KbotRNNModel(
            key,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
        )

    def _run_actor(
        self,
        model: KbotRNNActor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[distrax.Distribution, Array]:
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        xyz_target_3 = commands["cartesian_body_target_command_KC_C_104R_PitchHardstopDriven"]
        quat_target_4 = commands["global_body_quaternion_command_KB_C_501X_Right_Bayonet_Adapter_Hard_Stop"]

        return model.forward(
            joint_pos_n=joint_pos_n,
            joint_vel_n=joint_vel_n,
            xyz_target_3=xyz_target_3,
            quat_target_4=quat_target_4,
            carry=carry,
        )

    def _run_critic(
        self,
        model: KbotRNNCritic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[Array, Array]:
        joint_pos_n = observations["joint_position_observation"]  # 26
        joint_vel_n = observations["joint_velocity_observation"]  # 27
        actuator_force_n = observations["actuator_force_observation"]  # 27
        xyz_target_3 = commands["cartesian_body_target_command_KC_C_104R_PitchHardstopDriven"]  # 3
        quat_target_4 = commands["global_body_quaternion_command_KB_C_501X_Right_Bayonet_Adapter_Hard_Stop"]  # 4
        end_effector_pos_3 = observations["cartesian_body_position_observation_KC_C_104R_PitchHardstopDriven"]  # 3

        return model.forward(
            joint_pos_n=joint_pos_n,
            joint_vel_n=joint_vel_n,
            actuator_force_n=actuator_force_n,
            xyz_target_3=xyz_target_3,
            quat_target_4=quat_target_4,
            end_effector_pos_3=end_effector_pos_3,
            carry=carry,
        )

    def get_ppo_variables(
        self,
        model: KbotRNNModel,
        trajectories: ksim.Trajectory,
        carry: tuple[Array, Array],
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, tuple[Array, Array]]:
        actor_carry, critic_carry = carry

        # Vectorize over the time dimensions.
        action_dist_tj, actor_carry = self._run_actor(model.actor, trajectories.obs, trajectories.command, actor_carry)
        log_probs_tj = action_dist_tj.log_prob(trajectories.action)

        # Gets the value by calling the critic.
        values_t1, critic_carry = self._run_critic(model.critic, trajectories.obs, trajectories.command, critic_carry)

        ppo_variables = ksim.PPOVariables(
            log_probs=log_probs_tj,
            values=values_t1.squeeze(-1),
        )

        return ppo_variables, (actor_carry, critic_carry)

    def get_initial_carry(self, rng: PRNGKeyArray) -> tuple[Array, Array]:
        return (
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
        )

    def sample_action(
        self,
        model: KbotRNNModel,
        carry: tuple[Array, Array],
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
    ) -> ksim.Action:
        actor_carry_in, critic_carry_in = carry

        # Runs the actor model to get the action distribution.
        action_dist_j, actor_carry = self._run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
            carry=actor_carry_in,
        )

        action_j = action_dist_j.sample(seed=rng)

        return ksim.Action(
            action=action_j,
            carry=(actor_carry, critic_carry_in),
            aux_outputs=None,
        )

    def make_export_model(self, model: KbotRNNModel, stochastic: bool = False, batched: bool = False) -> Callable:
        """Makes a callable inference function that directly takes a flattened input vector and returns an action.

        Returns:
            A tuple containing the inference function and the size of the input vector.
        """

        def deterministic_model_fn(obs: Array, carry: Array) -> tuple[Array, Array]:
            dist, carry = model.actor.call_flat_obs(obs, carry)
            return dist.mode(), carry

        def stochastic_model_fn(obs: Array, carry: Array) -> tuple[Array, Array]:
            dist, carry = model.actor.call_flat_obs(obs, carry)
            return dist.sample(seed=jax.random.PRNGKey(0)), carry

        if stochastic:
            model_fn = stochastic_model_fn
        else:
            model_fn = deterministic_model_fn

        if batched:

            def batched_model_fn(obs: Array, carry: Array) -> tuple[Array, Array]:
                return jax.vmap(model_fn)(obs, carry)

            return batched_model_fn

        return model_fn

    def on_after_checkpoint_save(self, ckpt_path: Path, state: xax.State) -> xax.State:
        if not self.config.export_for_inference:
            return state

        model: KbotRNNModel = self.load_ckpt(ckpt_path, part="model")[0]

        model_fn = self.make_export_model(model, stochastic=False, batched=True)

        input_shapes = [
            (RNN_NUM_INPUTS,),
            (
                self.config.depth,
                self.config.hidden_size,
            ),
        ]

        tf_path = (
            ckpt_path.parent / "tf_model"
            if self.config.only_save_most_recent
            else ckpt_path.parent / f"tf_model_{state.num_steps}"
        )

        export(
            model_fn,
            input_shapes,
            tf_path,
        )

        return state


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m ksim_kbot.misc_tasks.pseudo_ik_rnn
    # To visualize the environment, use the following command:
    #   python -m ksim_kbot.misc_tasks.pseudo_ik_rnn run_environment=True
    KbotPseudoIKRNNTask.launch(
        KbotPseudoIKRNNTaskConfig(
            # Training parameters.
            num_envs=3000,
            batch_size=300,
            num_passes=10,
            epochs_per_log_step=1,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            entropy_coef=0.05,
            learning_rate=3e-4,
            rollout_length_seconds=10.0,
            render_length_seconds=10.0,
            save_every_n_steps=25,
            export_for_inference=True,
            # Apparently rendering markers can sometimes cause segfaults.
            # Disable this if you are running into segfaults.
            render_markers=True,
            render_camera_name="iso_camera",
            use_mit_actuators=True,
        ),
    )
