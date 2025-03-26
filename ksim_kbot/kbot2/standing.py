"""Defines simple task for training a walking policy for K-Bot."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import attrs
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import ksim
import mujoco
import optax
import xax
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray
from kscale.web.gen.api import JointMetadataOutput
from mujoco import mjx

OBS_SIZE = 20 * 2 + 3 + 3  # = 46 position + velocity + imu_acc + imu_gyro
CMD_SIZE = 2
NUM_OUTPUTS = 20 * 2  # position + velocity

SINGLE_STEP_HISTORY_SIZE = NUM_OUTPUTS + OBS_SIZE + CMD_SIZE

HISTORY_LENGTH = 0

NUM_INPUTS = (OBS_SIZE + CMD_SIZE) + SINGLE_STEP_HISTORY_SIZE * HISTORY_LENGTH

MAX_TORQUE = {
    "00": 17.0,
    "02": 17.0,
    "03": 40.0,
    "04": 60.0,
}


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AuxOutputs:
    log_probs: Array
    values: Array


@attrs.define(frozen=True)
class HistoryObservation(ksim.Observation):
    def observe(self, state: ksim.RolloutVariables, rng: PRNGKeyArray) -> Array:
        if not isinstance(state.carry, Array):
            raise ValueError("Carry is not a history array")
        return state.carry


@attrs.define(frozen=True, kw_only=True)
class JointDeviationPenalty(ksim.Reward):
    """Penalty for joint deviations."""

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        diff = trajectory.qpos[:, 7:] - jnp.zeros_like(trajectory.qpos[:, 7:])
        return jnp.sum(jnp.square(diff), axis=-1)


@attrs.define(frozen=True, kw_only=True)
class DHControlPenalty(ksim.Reward):
    """Legacy default humanoid control cost that penalizes squared action magnitude."""

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        return jnp.sum(jnp.square(trajectory.action), axis=-1)


@attrs.define(frozen=True, kw_only=True)
class DHHealthyReward(ksim.Reward):
    """Legacy default humanoid healthy reward that gives binary reward based on height."""

    healthy_z_lower: float = attrs.field(default=0.5)
    healthy_z_upper: float = attrs.field(default=1.5)

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        height = trajectory.qpos[:, 2]
        is_healthy = jnp.where(height < self.healthy_z_lower, 0.0, 1.0)
        is_healthy = jnp.where(height > self.healthy_z_upper, 0.0, is_healthy)
        return is_healthy


class KbotActor(eqx.Module):
    """Actor for the walking task."""

    mlp: eqx.nn.MLP
    min_std: float = eqx.static_field()
    max_std: float = eqx.static_field()
    var_scale: float = eqx.static_field()
    mean_scale: float = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        min_std: float,
        max_std: float,
        var_scale: float,
        mean_scale: float,
    ) -> None:
        self.mlp = eqx.nn.MLP(
            in_size=NUM_INPUTS,
            out_size=NUM_OUTPUTS * 2,
            width_size=64,
            depth=5,
            key=key,
            activation=jax.nn.relu,
        )
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale
        self.mean_scale = mean_scale

    def __call__(
        self,
        joint_pos_n: Array,
        joint_vel_n: Array,
        imu_acc_n: Array,
        imu_gyro_n: Array,
        lin_vel_cmd_n: Array,
        history_n: Array,
    ) -> distrax.Normal:
        x_n = jnp.concatenate(
            [
                joint_pos_n,
                joint_vel_n,
                imu_acc_n,
                imu_gyro_n,
                lin_vel_cmd_n,
                # history_n,
            ],
            axis=-1,
        )  # (NUM_INPUTS)

        return self.call_flat_obs(x_n)

    def call_flat_obs(
        self,
        flat_obs_n: Array,
    ) -> distrax.Normal:
        prediction_n = self.mlp(flat_obs_n)
        mean_n = prediction_n[..., :NUM_OUTPUTS]
        std_n = prediction_n[..., NUM_OUTPUTS:]

        # Scale the mean.
        mean_n = jnp.tanh(mean_n) * self.mean_scale

        # Softplus and clip to ensure positive standard deviations.
        std_n = jnp.clip((jax.nn.softplus(std_n) + self.min_std) * self.var_scale, max=self.max_std)

        return distrax.Normal(mean_n, std_n)


class KbotCritic(eqx.Module):
    """Critic for the standing task."""

    mlp: eqx.nn.MLP

    def __init__(self, key: PRNGKeyArray) -> None:
        self.mlp = eqx.nn.MLP(
            in_size=NUM_INPUTS,
            out_size=1,  # Always output a single critic value.
            width_size=64,
            depth=5,
            key=key,
            activation=jax.nn.relu,
        )

    def __call__(
        self,
        joint_pos_n: Array,
        joint_vel_n: Array,
        imu_acc_n: Array,
        imu_gyro_n: Array,
        lin_vel_cmd_n: Array,
        history_n: Array,
    ) -> Array:
        x_n = jnp.concatenate(
            [
                joint_pos_n,
                joint_vel_n,
                imu_acc_n,
                imu_gyro_n,
                lin_vel_cmd_n,
                # history_n,
            ],
            axis=-1,
        )  # (NUM_INPUTS)
        return self.mlp(x_n)


class KbotModel(eqx.Module):
    actor: KbotActor
    critic: KbotCritic

    def __init__(self, key: PRNGKeyArray) -> None:
        self.actor = KbotActor(
            key,
            min_std=0.01,
            max_std=1.0,
            var_scale=1.0,
            mean_scale=1.0,
        )
        self.critic = KbotCritic(key)


@dataclass
class KbotStandingTaskConfig(ksim.PPOConfig):
    """Config for the KBot walking task."""

    robot_urdf_path: str = xax.field(
        value="ksim_kbot/kscale-assets/kbot-v2-lw-feet/",
        help="The path to the assets directory for the robot.",
    )

    action_scale: float = xax.field(
        value=1.0,
        help="The scale to apply to the actions.",
    )

    # Optimizer parameters.
    learning_rate: float = xax.field(
        value=1e-4,
        help="Learning rate for PPO.",
    )
    max_grad_norm: float = xax.field(
        value=0.5,
        help="Maximum gradient norm for clipping.",
    )
    adam_weight_decay: float = xax.field(
        value=0.0,
        help="Weight decay for the Adam optimizer.",
    )

    # Mujoco parameters.
    use_mit_actuators: bool = xax.field(
        value=False,
        help="Whether to use the MIT actuator model, where the actions are position + velocity commands",
    )

    # Rendering parameters.
    render_track_body_id: int | None = xax.field(
        value=None,
        help="The body id to track with the render camera.",
    )

    # Checkpointing parameters.
    export_for_inference: bool = xax.field(
        value=False,
        help="Whether to export the model for inference.",
    )


class KbotStandingTask(ksim.PPOTask[KbotStandingTaskConfig]):
    def get_optimizer(self) -> optax.GradientTransformation:
        """Builds the optimizer.

        This provides a reasonable default optimizer for training PPO models,
        but can be overridden by subclasses who want to do something different.
        """
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            (
                optax.adam(self.config.learning_rate)
                if self.config.adam_weight_decay == 0.0
                else optax.adamw(self.config.learning_rate, weight_decay=self.config.adam_weight_decay)
            ),
        )

        return optimizer

    def get_mujoco_model(self) -> mujoco.MjModel:
        mjcf_path = (Path(self.config.robot_urdf_path) / "scene.mjcf").resolve().as_posix()
        mj_model = mujoco.MjModel.from_xml_path(mjcf_path)

        mj_model.opt.timestep = jnp.array(self.config.dt)
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6
        mj_model.opt.disableflags = mjx.DisableBit.EULERDAMP
        mj_model.opt.solver = mjx.SolverType.CG

        return mj_model

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> dict[str, JointMetadataOutput]:
        metadata = asyncio.run(ksim.get_mujoco_model_metadata(self.config.robot_urdf_path, cache=False))
        if metadata.joint_name_to_metadata is None:
            raise ValueError("Joint metadata is not available")
        return metadata.joint_name_to_metadata

    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: dict[str, JointMetadataOutput] | None = None,
    ) -> ksim.Actuators:
        if self.config.use_mit_actuators:
            if metadata is None:
                raise ValueError("Metadata is required for MIT actuators")
            return ksim.MITPositionVelocityActuators(
                physics_model,
                metadata,
                pos_action_noise=0.1,
                vel_action_noise=0.1,
                pos_action_noise_type="gaussian",
                vel_action_noise_type="gaussian",
                ctrl_clip=[
                    # right arm
                    MAX_TORQUE["03"],
                    MAX_TORQUE["03"],
                    MAX_TORQUE["02"],
                    MAX_TORQUE["02"],
                    MAX_TORQUE["00"],
                    # left arm
                    MAX_TORQUE["03"],
                    MAX_TORQUE["03"],
                    MAX_TORQUE["02"],
                    MAX_TORQUE["02"],
                    MAX_TORQUE["00"],
                    # right leg
                    MAX_TORQUE["04"],
                    MAX_TORQUE["03"],
                    MAX_TORQUE["03"],
                    MAX_TORQUE["04"],
                    MAX_TORQUE["02"],
                    # left leg
                    MAX_TORQUE["04"],
                    MAX_TORQUE["03"],
                    MAX_TORQUE["03"],
                    MAX_TORQUE["04"],
                    MAX_TORQUE["02"],
                ],
            )
        else:
            return ksim.TorqueActuators()

    def get_randomization(self, physics_model: ksim.PhysicsModel) -> list[ksim.Randomization]:
        return [
            ksim.WeightRandomization(scale=0.05),
            ksim.StaticFrictionRandomization(scale_lower=0.1, scale_upper=1.5),
        ]

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomBaseVelocityXYReset(scale=0.01),
            ksim.RandomJointPositionReset(scale=0.02),
            ksim.RandomJointVelocityReset(scale=0.02),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        return [
            ksim.PushEvent(
                probability=0.0005,
                interval_range=(1, 3),
                linear_force_scale=0.5,
            ),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        return [
            ksim.JointPositionObservation(noise=0.02),
            ksim.JointVelocityObservation(noise=0.2),
            ksim.ActuatorForceObservation(),
            ksim.SensorObservation.create(physics_model, "imu_acc", noise=0.8),
            ksim.SensorObservation.create(physics_model, "imu_gyro", noise=0.1),
            HistoryObservation(),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return [
            ksim.LinearVelocityCommand(x_range=(-0.0, 0.0), y_range=(-0.0, 0.0), switch_prob=0.0, zero_prob=0.0),
        ]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            JointDeviationPenalty(scale=-0.5),
            DHControlPenalty(scale=-0.05),
            DHHealthyReward(scale=0.5),
            ksim.BaseHeightReward(scale=1.0, height_target=0.7),
            ksim.ActionSmoothnessPenalty(scale=-0.01),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            ksim.RollTooGreatTermination(max_roll=2.04),
            ksim.PitchTooGreatTermination(max_pitch=2.04),
        ]

    def get_model(self, key: PRNGKeyArray) -> KbotModel:
        return KbotModel(key)

    def get_initial_carry(self, rng: PRNGKeyArray) -> Array:
        return jnp.zeros(HISTORY_LENGTH * SINGLE_STEP_HISTORY_SIZE)

    def _run_actor(
        self,
        model: KbotModel,
        observations: FrozenDict[str, Array],
        commands: FrozenDict[str, Array],
    ) -> distrax.Normal:
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        imu_acc_n = observations["imu_acc_obs"]
        imu_gyro_n = observations["imu_gyro_obs"]
        lin_vel_cmd_n = commands["linear_velocity_command"]
        history_n = observations["history_observation"]
        return model.actor(joint_pos_n, joint_vel_n, imu_acc_n, imu_gyro_n, lin_vel_cmd_n, history_n)

    def _run_critic(
        self,
        model: KbotModel,
        observations: FrozenDict[str, Array],
        commands: FrozenDict[str, Array],
    ) -> Array:
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        imu_acc_n = observations["imu_acc_obs"]
        imu_gyro_n = observations["imu_gyro_obs"]
        lin_vel_cmd_n = commands["linear_velocity_command"]
        history_n = observations["history_observation"]
        return model.critic(joint_pos_n, joint_vel_n, imu_acc_n, imu_gyro_n, lin_vel_cmd_n, history_n)

    def get_on_policy_log_probs(
        self,
        model: KbotModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> Array:
        if trajectories.aux_outputs is None:
            raise ValueError("No aux outputs found in trajectories")
        return trajectories.aux_outputs.log_probs

    def get_on_policy_values(
        self,
        model: KbotModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> Array:
        if trajectories.aux_outputs is None:
            raise ValueError("No aux outputs found in trajectories")
        return trajectories.aux_outputs.values

    def get_log_probs(
        self,
        model: KbotModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> tuple[Array, Array]:
        # Vectorize over both batch and time dimensions.
        par_fn = jax.vmap(self._run_actor, in_axes=(None, 0, 0))
        action_dist_btn = par_fn(model, trajectories.obs, trajectories.command)

        # Compute the log probabilities of the trajectory's actions according
        # to the current policy, along with the entropy of the distribution.
        action_btn = trajectories.action
        log_probs_btn = action_dist_btn.log_prob(action_btn)
        entropy_btn = action_dist_btn.entropy()

        return log_probs_btn, entropy_btn

    def get_values(
        self,
        model: KbotModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> Array:
        # Vectorize over both batch and time dimensions.
        par_fn = jax.vmap(self._run_critic, in_axes=(None, 0, 0))
        values_bt1 = par_fn(model, trajectories.obs, trajectories.command)

        # Remove the last dimension.
        return values_bt1.squeeze(-1)

    def sample_action(
        self,
        model: KbotModel,
        carry: Array,
        physics_model: ksim.PhysicsModel,
        observations: FrozenDict[str, Array],
        commands: FrozenDict[str, Array],
        rng: PRNGKeyArray,
    ) -> tuple[Array, Array, AuxOutputs]:
        action_dist_n = self._run_actor(model, observations, commands)
        action_n = action_dist_n.sample(seed=rng)
        action_log_prob_n = action_dist_n.log_prob(action_n)

        critic_n = self._run_critic(model, observations, commands)
        value_n = critic_n.squeeze(-1)

        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        imu_acc_n = observations["imu_acc_obs"]
        imu_gyro_n = observations["imu_gyro_obs"]
        lin_vel_cmd_n = commands["linear_velocity_command"]

        history_n = jnp.concatenate(
            [
                joint_pos_n,
                joint_vel_n,
                imu_acc_n,
                imu_gyro_n,
                lin_vel_cmd_n,
                action_n,
            ],
            axis=-1,
        )

        if HISTORY_LENGTH > 0:
            # Roll the history by shifting the existing history and adding the new data
            carry_reshaped = carry.reshape(HISTORY_LENGTH, SINGLE_STEP_HISTORY_SIZE)
            shifted_history = jnp.roll(carry_reshaped, shift=-1, axis=0)
            new_history = shifted_history.at[HISTORY_LENGTH - 1].set(history_n)
            history_n = new_history.reshape(-1)
        else:
            history_n = jnp.zeros(0)

        return action_n, history_n, AuxOutputs(log_probs=action_log_prob_n, values=value_n)

    def make_export_model(self, model: KbotModel, stochastic: bool = False, batched: bool = False) -> Callable:
        """Makes a callable inference function that directly takes a flattened input vector and returns an action.

        Returns:
            A tuple containing the inference function and the size of the input vector.
        """

        def deterministic_model_fn(obs: Array) -> Array:
            return model.actor.call_flat_obs(obs).mode()

        def stochastic_model_fn(obs: Array) -> Array:
            distribution = model.actor.call_flat_obs(obs)
            return distribution.sample(seed=jax.random.PRNGKey(0))

        if stochastic:
            model_fn = stochastic_model_fn
        else:
            model_fn = deterministic_model_fn

        if batched:

            def batched_model_fn(obs: Array) -> Array:
                return jax.vmap(model_fn)(obs)

            return batched_model_fn

        return model_fn

    def on_after_checkpoint_save(self, ckpt_path: Path, state: xax.State) -> xax.State:
        state = super().on_after_checkpoint_save(ckpt_path, state)

        model: KbotModel = self.load_checkpoint(ckpt_path, part="model")

        model_fn = self.make_export_model(model, stochastic=False, batched=True)

        input_shapes = [(NUM_INPUTS,)]

        xax.export(
            model_fn,
            input_shapes,  # type: ignore [arg-type]
            ckpt_path.parent / "tf_model",
        )

        return state


if __name__ == "__main__":
    # python -m ksim_kbot.kbot2.standing run_environment=True
    KbotStandingTask.launch(
        KbotStandingTaskConfig(
            num_envs=2048,
            num_batches=64,
            num_passes=10,
            # Simulation parameters.
            dt=0.002,
            ctrl_dt=0.02,
            max_action_latency=0.002,
            min_action_latency=0.0,
            valid_every_n_steps=25,
            valid_first_n_steps=0,
            rollout_length_seconds=10.0,
            # PPO parameters
            gamma=0.97,
            lam=0.95,
            entropy_coef=0.001,
            learning_rate=1e-4,
            clip_param=0.3,
            max_grad_norm=1.0,
            use_mit_actuators=True,
            export_for_inference=True,
            save_every_n_steps=25,
        ),
    )
