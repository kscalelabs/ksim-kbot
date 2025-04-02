# mypy: disable-error-code="override"
"""Defines simple task for training a getup policy for K-Bot."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import attrs
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import ksim
import mujoco
import xax
from jaxtyping import Array, PRNGKeyArray
from mujoco import mjx

from ksim_kbot.standing.standing import KbotStandingTask, KbotStandingTaskConfig

OBS_SIZE = 20 * 2 + 3 + 3 + 40  # = 46 position + velocity + imu_acc + imu_gyro + last_action
CMD_SIZE = 2
NUM_OUTPUTS = 20 * 2  # position + velocity

SINGLE_STEP_HISTORY_SIZE = NUM_OUTPUTS + OBS_SIZE + CMD_SIZE

HISTORY_LENGTH = 0

NUM_INPUTS = (OBS_SIZE + CMD_SIZE) + SINGLE_STEP_HISTORY_SIZE * HISTORY_LENGTH

MAX_TORQUE = {
    "00": 15.0,
    "02": 20.0,
    "03": 60.0,
    "04": 90.0,
}


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AuxOutputs:
    log_probs: Array
    values: Array


@attrs.define(frozen=True, kw_only=True)
class EnergyTermination(ksim.Termination):
    energy_termination_threshold: float = attrs.field(default=100.0)

    def __call__(self, data: ksim.PhysicsData) -> jax.Array:
        energy = jnp.sum(jnp.abs(data.actuator_force * data.qvel[6:]))
        energy_termination = energy > jnp.inf
        return energy_termination


@attrs.define(frozen=True, kw_only=True)
class ResetDefaultJointPosition(ksim.Reset):
    """Resets the joint positions of the robot to random values."""

    default_targets: tuple[float, ...] = attrs.field(
        default=(
            # xyz
            0.0,
            0.0,
            0.1,
            # quat
            0.7071,
            0.7071,
            0.0,
            0.0,
            # qpos
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    )

    def __call__(self, data: ksim.PhysicsData, rng: PRNGKeyArray) -> ksim.PhysicsData:
        qpos = data.qpos
        match type(data):
            case mujoco.MjData:
                qpos[:] = self.default_targets
            case mjx.Data:
                qpos = qpos.at[:].set(self.default_targets)
        return ksim.utils.mujoco.update_data_field(data, "qpos", qpos)


@attrs.define(frozen=True)
class HeightOrientationObservation(ksim.Observation):
    noise: float = attrs.field(default=0.0)

    def observe(self, rollout_state: ksim.RolloutVariables, rng: PRNGKeyArray) -> Array:
        qpos = rollout_state.physics_state.data.qpos[2:7]  # (N, 5)
        return qpos

    def add_noise(self, observation: Array, rng: PRNGKeyArray) -> Array:
        return observation + jax.random.normal(rng, observation.shape) * self.noise


@attrs.define(frozen=True)
class HistoryObservation(ksim.Observation):
    def observe(self, state: ksim.RolloutVariables, rng: PRNGKeyArray) -> Array:
        if not isinstance(state.carry, Array):
            raise ValueError("Carry is not a history array")
        return state.carry


@attrs.define(frozen=True)
class JointPositionObservation(ksim.Observation):
    noise: float = attrs.field(default=0.0)
    default_targets: tuple[float, ...] = attrs.field(
        default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )

    def observe(self, rollout_state: ksim.RolloutVariables, rng: PRNGKeyArray) -> Array:
        qpos = rollout_state.physics_state.data.qpos[7:]  # (N,)
        diff = qpos - jnp.array(self.default_targets)
        return diff

    def add_noise(self, observation: Array, rng: PRNGKeyArray) -> Array:
        return observation + jax.random.normal(rng, observation.shape) * self.noise


@attrs.define(frozen=True)
class LastActionObservation(ksim.Observation):
    noise: float = attrs.field(default=0.0)

    def observe(self, rollout_state: ksim.RolloutVariables, rng: PRNGKeyArray) -> Array:
        return rollout_state.physics_state.most_recent_action

    def add_noise(self, observation: Array, rng: PRNGKeyArray) -> Array:
        return observation + jax.random.normal(rng, observation.shape) * self.noise


@attrs.define(frozen=True, kw_only=True)
class JointDeviationPenalty(ksim.Reward):
    """Penalty for joint deviations with a height threshold."""

    joint_targets: tuple[float, ...] = attrs.field(
        default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )
    height_threshold: float = attrs.field(default=0.5)

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        height = trajectory.qpos[..., 2]
        diff = trajectory.qpos[..., 7:] - jnp.array(self.joint_targets)
        diff = jnp.sum(jnp.square(diff), axis=-1)
        # Gate the reward subject to the height.
        gate = height > self.height_threshold
        return diff * gate


@attrs.define(frozen=True, kw_only=True)
class StandStillPenalty(ksim.Reward):
    """Penalty for not standing still."""

    height_threshold: float = attrs.field(default=0.5)

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        height = trajectory.qpos[..., 2]
        cost = jnp.sum(jnp.square(trajectory.qvel[..., :2]), axis=-1)
        # Gate the reward subject to the height.
        gate = height > self.height_threshold
        return cost * gate


@attrs.define(frozen=True, kw_only=True)
class UpwardPositionReward(ksim.Reward):
    """Reward for the upward position."""

    target_pos: float = attrs.field(default=0.71)
    sensitivity: float = attrs.field(default=10)

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        pos_after = trajectory.qpos[..., 2]
        diff = jnp.abs(pos_after - self.target_pos)
        return jnp.exp(-self.sensitivity * diff)


@attrs.define(frozen=True, kw_only=True)
class UpwardVelocityReward(ksim.Reward):
    """Incentives upward velocity."""

    velocity_clip: float = attrs.field(default=10.0)

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        z_delta = jnp.clip(trajectory.qvel[..., 2], 0, self.velocity_clip)

        return z_delta


@attrs.define(frozen=True, kw_only=True)
class StationaryPenalty(ksim.Reward):
    """Incentives staying in place laterally."""

    norm: xax.NormType = attrs.field(default="l2")

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        return xax.get_norm(trajectory.qvel[..., :2], self.norm).sum(axis=-1)


class KbotGetupActor(eqx.Module):
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
        last_action_n: Array,
        history_n: Array,
    ) -> distrax.Normal:
        x_n = jnp.concatenate(
            [
                joint_pos_n,
                joint_vel_n,
                imu_acc_n,
                imu_gyro_n,
                lin_vel_cmd_n,
                last_action_n,
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


class KbotGetupCritic(eqx.Module):
    """Critic for the getup task."""

    mlp: eqx.nn.MLP

    def __init__(self, key: PRNGKeyArray) -> None:
        self.mlp = eqx.nn.MLP(
            in_size=NUM_INPUTS + 5,
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
        imu_acc_3: Array,
        imu_gyro_3: Array,
        lin_vel_cmd_2: Array,
        last_action_n: Array,
        height_orientation_5: Array,
        history_n: Array,
    ) -> Array:
        x_n = jnp.concatenate(
            [
                joint_pos_n,
                joint_vel_n,
                imu_acc_3,
                imu_gyro_3,
                lin_vel_cmd_2,
                last_action_n,
                height_orientation_5,
                history_n,
            ],
            axis=-1,
        )  # (NUM_INPUTS + 5)
        return self.mlp(x_n)


class KbotModel(eqx.Module):
    actor: KbotGetupActor
    critic: KbotGetupCritic

    def __init__(self, key: PRNGKeyArray) -> None:
        self.actor = KbotGetupActor(
            key,
            min_std=0.01,
            max_std=1.0,
            var_scale=1.0,
            mean_scale=1.0,
        )
        self.critic = KbotGetupCritic(key)


@dataclass
class KbotGetupTaskConfig(KbotStandingTaskConfig):
    robot_urdf_path: str = xax.field(
        value="ksim_kbot/kscale-assets/kbot-v2-lw-full/",
        help="The path to the assets directory for the robot.",
    )


Config = TypeVar("Config", bound=KbotGetupTaskConfig)


class KbotGetupTask(KbotStandingTask[Config], Generic[Config]):
    def get_model(self, key: PRNGKeyArray) -> KbotModel:
        return KbotModel(key)

    def get_initial_carry(self, rng: PRNGKeyArray) -> Array:
        return jnp.zeros(HISTORY_LENGTH * SINGLE_STEP_HISTORY_SIZE)

    def _run_critic(
        self,
        model: KbotModel,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
    ) -> Array:
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        imu_acc_3 = observations["imu_acc_obs"]
        imu_gyro_3 = observations["imu_gyro_obs"]
        lin_vel_cmd_2 = commands["linear_velocity_step_command"]
        last_action_n = observations["last_action_observation"]
        height_orientation_5 = observations["height_orientation_observation"]
        history_n = observations["history_observation"]
        return model.critic(
            joint_pos_n,
            joint_vel_n,
            imu_acc_3,
            imu_gyro_3,
            lin_vel_cmd_2,
            last_action_n,
            height_orientation_5,
            history_n,
        )

    def get_randomization(self, physics_model: ksim.PhysicsModel) -> list[ksim.Randomization]:
        return []

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomBaseVelocityXYReset(scale=0.01),
            ksim.RandomJointPositionReset(scale=0.02),
            ksim.RandomJointVelocityReset(scale=0.02),
            ResetDefaultJointPosition(),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        return []

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        return [
            HeightOrientationObservation(),
            JointPositionObservation(noise=0.0),
            ksim.JointVelocityObservation(noise=0.0),
            ksim.ActuatorForceObservation(),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_acc", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_gyro", noise=0.0),
            LastActionObservation(noise=0.0),
            HistoryObservation(),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return [
            ksim.LinearVelocityStepCommand(
                x_range=(-0.0, 0.0),
                y_range=(-0.0, 0.0),
                x_fwd_prob=0.0,
                y_fwd_prob=0.0,
            ),
        ]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            UpwardPositionReward(scale=1.0),
            UpwardVelocityReward(scale=0.05),
            # For now we do not penalize for wild moves.
            # StandStillPenalty(scale=-0.2),
            # JointDeviationPenalty(scale=-0.01),
            # StationaryPenalty(scale=-0.05),
            # ksim.ActuatorForcePenalty(scale=-0.01),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            EnergyTermination(),
        ]


if __name__ == "__main__":
    # To run training, use the following command:
    # python -m ksim_kbot.misc_tasks.getup
    # To visualize the environment, use the following command:
    # python -m ksim_kbot.misc_tasks.getup \
    # run_environment=True \
    # run_environment_num_seconds=1 \
    # run_environment_save_path=videos/test.mp4
    KbotGetupTask.launch(
        KbotGetupTaskConfig(
            num_envs=2048,
            batch_size=256,
            num_passes=10,
            # Simulation parameters.
            dt=0.002,
            ctrl_dt=0.02,
            max_action_latency=0.005,
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
        ),
    )
