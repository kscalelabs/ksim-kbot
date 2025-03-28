"""Common utilities for K-Bot 2."""

import attrs
import jax
import jax.numpy as jnp
import ksim
import mujoco
import xax
from jaxtyping import Array, PRNGKeyArray
from mujoco import mjx


@attrs.define(frozen=True)
class HistoryObservation(ksim.Observation):
    def observe(self, state: ksim.RolloutVariables, rng: PRNGKeyArray) -> Array:
        if not isinstance(state.carry, Array):
            raise ValueError("Carry is not a history array")
        return state.carry


@attrs.define(frozen=True, kw_only=True)
class JointDeviationPenalty(ksim.Reward):
    """Penalty for joint deviations."""

    joint_targets: tuple[float, ...] = attrs.field(
        default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        diff = trajectory.qpos[:, 7:] - jnp.array(self.joint_targets)
        return jnp.sum(jnp.square(diff), axis=-1)


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


@attrs.define(frozen=True, kw_only=True)
class ResetDefaultJointPosition(ksim.Reset):
    """Resets the joint positions of the robot to random values."""

    default_targets: tuple[float, ...] = attrs.field()

    def __call__(self, data: ksim.PhysicsData, rng: PRNGKeyArray) -> ksim.PhysicsData:
        qpos = data.qpos
        match type(data):
            case mujoco.MjData:
                qpos[:] = self.default_targets
            case mjx.Data:
                qpos = qpos.at[:].set(self.default_targets)
        return ksim.utils.mujoco.update_data_field(data, "qpos", qpos)


@attrs.define(frozen=True, kw_only=True)
class FeetSlipPenalty(ksim.Reward):
    """Penalty for feet slipping."""

    norm: xax.NormType = attrs.field(default="l2")
    observation_name: str = attrs.field(default="feet_contact_observation")
    command_name: str = attrs.field(default="linear_velocity_step_command")
    com_vel_obs_name: str = attrs.field(default="center_of_mass_velocity_observation")
    command_vel_scale: float = attrs.field(default=0.02)

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        if self.observation_name not in trajectory.obs:
            raise ValueError(f"Observation {self.observation_name} not found; add it as an observation in your task.")
        contact = trajectory.obs[self.observation_name]
        com_vel = trajectory.obs[self.com_vel_obs_name][..., :2]
        return (xax.get_norm(com_vel, self.norm) * contact).sum(axis=-1)


@attrs.define(frozen=True)
class LastActionObservation(ksim.Observation):
    noise: float = attrs.field(default=0.0)

    def observe(self, rollout_state: ksim.RolloutVariables, rng: PRNGKeyArray) -> Array:
        return rollout_state.physics_state.most_recent_action

    def add_noise(self, observation: Array, rng: PRNGKeyArray) -> Array:
        return observation + jax.random.normal(rng, observation.shape) * self.noise


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
