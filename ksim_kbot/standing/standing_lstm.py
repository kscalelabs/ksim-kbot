# mypy: disable-error-code="override"
"""Defines simple task for training a walking policy for K-Bot."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import ksim
import xax
from jaxtyping import Array, PRNGKeyArray

from ksim_kbot.standing.standing import AuxOutputs, KbotStandingTask, KbotStandingTaskConfig

OBS_SIZE = 20 * 2 + 3 + 3  # = 46 position + velocity + imu_acc + imu_gyro
CMD_SIZE = 2
NUM_INPUTS = OBS_SIZE + CMD_SIZE
NUM_OUTPUTS = 20 * 2  # position + velocity

HIDDEN_SIZE = 128  # LSTM hidden state size
DEPTH = 2  # Number of LSTM layers


class MultiLayerLSTM(eqx.Module):
    layers: tuple[eqx.nn.LSTMCell, ...]
    depth: int = eqx.field(static=True)
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    def __init__(self, key: PRNGKeyArray, *, input_size: int, hidden_size: int, depth: int) -> None:
        if depth < 1:
            raise ValueError("Depth must be at least 1")

        first_layer = eqx.nn.LSTMCell(input_size=input_size, hidden_size=hidden_size, use_bias=True, key=key)

        other_layers = tuple(
            eqx.nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size, use_bias=True, key=key)
            for _ in range(depth - 1)
        )

        self.layers = (first_layer, *other_layers)
        self.depth = depth
        self.input_size = input_size
        self.hidden_size = hidden_size

    def __call__(
        self,
        x_n: Array,
        hidden_states: Array,  # (depth, 2, hidden_size)
    ) -> tuple[Array, Array, Array]:  # (output_h, output_c, new_hidden_states)
        h_states = hidden_states[:, 0]  # All h states
        c_states = hidden_states[:, 1]  # All c states

        new_h_states = []
        new_c_states = []

        h, c = self.layers[0](x_n, (h_states[0], c_states[0]))
        new_h_states.append(h)
        new_c_states.append(c)

        if self.depth > 1:
            for layer, h_state, c_state in zip(self.layers[1:], h_states[1:], c_states[1:]):
                h, c = layer(h, (h_state, c_state))
                new_h_states.append(h)
                new_c_states.append(c)

        stacked_h = jnp.stack(new_h_states, axis=0)  # (depth, hidden_size)
        stacked_c = jnp.stack(new_c_states, axis=0)  # (depth, hidden_size)

        return h, c, jnp.stack([stacked_h, stacked_c], axis=1)  # h_last, c_last, (depth, 2, hidden_size)


class KbotActor(eqx.Module):
    """Actor for the walking task."""

    multi_layer_lstm: MultiLayerLSTM
    projector: eqx.nn.MLP
    min_std: float = eqx.static_field()
    max_std: float = eqx.static_field()
    var_scale: float = eqx.static_field()
    mean_scale: float = eqx.static_field()
    hidden_size: int = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        min_std: float,
        max_std: float,
        var_scale: float,
        mean_scale: float,
        hidden_size: int,
    ) -> None:
        self.multi_layer_lstm = MultiLayerLSTM(
            key,
            input_size=NUM_INPUTS,
            hidden_size=hidden_size,
            depth=DEPTH,
        )

        self.projector = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=NUM_OUTPUTS * 2,
            width_size=64,
            depth=2,
            key=key,
            activation=jax.nn.relu,
        )

        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale
        self.mean_scale = mean_scale
        self.hidden_size = hidden_size

    def __call__(
        self,
        joint_pos_n: Array,
        joint_vel_n: Array,
        imu_acc_n: Array,
        imu_gyro_n: Array,
        lin_vel_cmd_n: Array,
        hidden_states: Array,
    ) -> tuple[distrax.Normal, Array]:
        x_n = jnp.concatenate(
            [
                joint_pos_n,
                joint_vel_n,
                imu_acc_n,
                imu_gyro_n,
                lin_vel_cmd_n,
            ],
            axis=-1,
        )  # (NUM_INPUTS)

        # Process through LSTM cell
        last_h, _, new_hidden_states = self.multi_layer_lstm(x_n, hidden_states)
        out_n = self.projector(last_h)

        mean_n = out_n[..., :NUM_OUTPUTS]
        std_n = out_n[..., NUM_OUTPUTS:]

        # Scale the mean.
        mean_n = jnp.tanh(mean_n) * self.mean_scale

        # Softplus and clip to ensure positive standard deviations.
        std_n = jnp.clip((jax.nn.softplus(std_n) + self.min_std) * self.var_scale, max=self.max_std)

        return distrax.Normal(mean_n, std_n), new_hidden_states

    def call_flat_obs(
        self,
        flat_obs_n: Array,
        hidden_states: Array,
    ) -> tuple[distrax.Normal, Array]:
        # Process through LSTM cell
        last_h, _, new_hidden_states = self.multi_layer_lstm(flat_obs_n, hidden_states)
        out_n = self.projector(last_h)

        mean_n = out_n[..., :NUM_OUTPUTS]
        std_n = out_n[..., NUM_OUTPUTS:]

        # Scale the mean.
        mean_n = jnp.tanh(mean_n) * self.mean_scale

        # Softplus and clip to ensure positive standard deviations.
        std_n = jnp.clip((jax.nn.softplus(std_n) + self.min_std) * self.var_scale, max=self.max_std)

        return distrax.Normal(mean_n, std_n), new_hidden_states


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
    ) -> Array:
        x_n = jnp.concatenate(
            [
                joint_pos_n,
                joint_vel_n,
                imu_acc_n,
                imu_gyro_n,
                lin_vel_cmd_n,
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
            hidden_size=HIDDEN_SIZE,
        )
        self.critic = KbotCritic(key)


@dataclass
class KbotStandingLSTMTaskConfig(KbotStandingTaskConfig):
    pass


Config = TypeVar("Config", bound=KbotStandingLSTMTaskConfig)


class KbotStandingLSTMTask(KbotStandingTask[Config], Generic[Config]):
    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        return [
            ksim.JointPositionObservation(noise=0.02),
            ksim.JointVelocityObservation(noise=0.2),
            ksim.SensorObservation.create(physics_model, "imu_acc", noise=0.8),
            ksim.SensorObservation.create(physics_model, "imu_gyro", noise=0.1),
            ksim.ActuatorForceObservation(),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            ksim.RollTooGreatTermination(max_roll=2.04),
            ksim.PitchTooGreatTermination(max_pitch=2.04),
        ]

    def get_model(self, key: PRNGKeyArray) -> KbotModel:
        return KbotModel(key)

    def get_initial_carry(self, rng: PRNGKeyArray) -> Array:
        # Initialize the hidden state for LSTM
        return jnp.zeros((DEPTH, 2, HIDDEN_SIZE))

    def _run_actor(
        self,
        model: KbotModel,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[distrax.Normal, Array]:
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        imu_acc_n = observations["imu_acc_obs"]
        imu_gyro_n = observations["imu_gyro_obs"]
        lin_vel_cmd_n = commands["linear_velocity_command"]
        return model.actor(joint_pos_n, joint_vel_n, imu_acc_n, imu_gyro_n, lin_vel_cmd_n, carry)

    def _run_critic(
        self,
        model: KbotModel,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
    ) -> Array:
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        imu_acc_n = observations["imu_acc_obs"]
        imu_gyro_n = observations["imu_gyro_obs"]
        lin_vel_cmd_n = commands["linear_velocity_command"]
        return model.critic(joint_pos_n, joint_vel_n, imu_acc_n, imu_gyro_n, lin_vel_cmd_n)

    def get_log_probs(
        self,
        model: KbotModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> tuple[Array, Array]:
        def scan_fn(
            carry: Array,
            inputs: ksim.Trajectory,
        ) -> tuple[Array, tuple[Array, Array]]:
            action_dist_n, carry = self._run_actor(model, inputs.obs, inputs.command, carry)
            log_probs_n = action_dist_n.log_prob(inputs.action)
            entropy_n = action_dist_n.entropy()
            return carry, (log_probs_n, entropy_n)

        initial_hidden_states = self.get_initial_carry(rng)
        _, (log_probs_tn, entropy_tn) = jax.lax.scan(scan_fn, initial_hidden_states, trajectories)

        return log_probs_tn, entropy_tn

    def sample_action(
        self,
        model: KbotModel,
        carry: Array,
        physics_model: ksim.PhysicsModel,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
    ) -> tuple[Array, Array, AuxOutputs]:
        action_dist_n, next_carry = self._run_actor(model, observations, commands, carry)
        action_n = action_dist_n.sample(seed=rng)
        action_log_prob_n = action_dist_n.log_prob(action_n)

        critic_n = self._run_critic(model, observations, commands)
        value_n = critic_n.squeeze(-1)

        return action_n, next_carry, AuxOutputs(log_probs=action_log_prob_n, values=value_n)


if __name__ == "__main__":
    # python -m ksim_kbot.standing.standing_lstm run_environment=True
    # To resume training:
    # python -m ksim_kbot.standing.standing_lstm load_from_ckpt_path=*.run_*.ckpt.*.bin
    KbotStandingLSTMTask.launch(
        KbotStandingLSTMTaskConfig(
            num_envs=2048,
            batch_size=256,
            num_passes=10,
            # Simulation parameters.
            dt=0.002,
            ctrl_dt=0.02,
            max_action_latency=0.005,
            min_action_latency=0.0,
            valid_every_n_steps=25,
            valid_every_n_seconds=300,
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
