"""Common utilities for K-Bot 2.

If some utilities will become more general, we can move them to ksim or xax.
"""

from typing import Self

import attrs
import jax.numpy as jnp
import ksim
import mujoco
import xax
from jaxtyping import Array, PRNGKeyArray
from ksim.utils.mujoco import get_qpos_data_idxs_by_name, get_site_data_idx_from_name
from mujoco import mjx


@attrs.define(frozen=True)
class HistoryObservation(ksim.Observation):
    def observe(self, state: ksim.RolloutVariables, rng: PRNGKeyArray) -> Array:
        if not isinstance(state.carry, Array):
            raise ValueError("Carry is not a history array")
        return state.carry


@attrs.define(frozen=True)
class JointPositionObservation(ksim.Observation):
    default_targets: tuple[float, ...] = attrs.field()
    noise: float = attrs.field(default=0.0)

    def observe(self, rollout_state: ksim.RolloutVariables, rng: PRNGKeyArray) -> Array:
        qpos = rollout_state.physics_state.data.qpos[7:]  # (N,)
        diff = qpos - jnp.array(self.default_targets)
        return diff


@attrs.define(frozen=True)
class ProjectedGravityObservation(ksim.Observation):
    noise: float = attrs.field(default=0.0)

    def observe(self, state: ksim.RolloutVariables, rng: PRNGKeyArray) -> Array:
        gvec = xax.get_projected_gravity_vector_from_quat(state.physics_state.data.qpos[3:7])
        return gvec


@attrs.define(frozen=True)
class LastActionObservation(ksim.Observation):
    noise: float = attrs.field(default=0.0)

    def observe(self, rollout_state: ksim.RolloutVariables, rng: PRNGKeyArray) -> Array:
        return rollout_state.physics_state.most_recent_action


@attrs.define(frozen=True)
class FeetPositionObservation(ksim.Observation):
    foot_left: int = attrs.field()
    foot_right: int = attrs.field()

    @classmethod
    def create(
        cls,
        *,
        physics_model: ksim.PhysicsModel,
        foot_left_site_name: str,
        foot_right_site_name: str,
    ) -> Self:
        foot_left_idx = get_site_data_idx_from_name(physics_model, foot_left_site_name)
        foot_right_idx = get_site_data_idx_from_name(physics_model, foot_right_site_name)
        return cls(
            foot_left=foot_left_idx,
            foot_right=foot_right_idx,
        )

    def observe(self, rollout_state: ksim.RolloutVariables, rng: PRNGKeyArray) -> Array:
        foot_left_pos = rollout_state.physics_state.data.site_xpos[self.foot_left]
        foot_right_pos = rollout_state.physics_state.data.site_xpos[self.foot_right]
        return jnp.concatenate([foot_left_pos, foot_right_pos], axis=-1)


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
class JointDeviationPenalty(ksim.Reward):
    """Penalty for joint deviations."""

    norm: xax.NormType = attrs.field(default="l2")
    joint_targets: tuple[float, ...] = attrs.field()

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        diff = trajectory.qpos[..., 7:] - jnp.array(self.joint_targets)
        return xax.get_norm(diff, self.norm).sum(axis=-1)


@attrs.define(frozen=True, kw_only=True)
class DHForwardReward(ksim.Reward):
    """Incentives forward movement."""

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        # Take just the x velocity component
        x_delta = -jnp.clip(trajectory.qvel[..., 1], -1.0, 1.0)
        return x_delta


@attrs.define(frozen=True, kw_only=True)
class DHControlPenalty(ksim.Reward):
    """Legacy default humanoid control cost that penalizes squared action magnitude."""

    norm: xax.NormType = attrs.field(default="l2")

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        return xax.get_norm(trajectory.action, self.norm).sum(axis=-1)


@attrs.define(frozen=True, kw_only=True)
class DHHealthyReward(ksim.Reward):
    """Legacy default humanoid healthy reward that gives binary reward based on height."""

    healthy_z_lower: float = attrs.field(default=0.5)
    healthy_z_upper: float = attrs.field(default=1.5)

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        height = trajectory.qpos[..., 2]
        is_healthy = jnp.where(height < self.healthy_z_lower, 0.0, 1.0)
        is_healthy = jnp.where(height > self.healthy_z_upper, 0.0, is_healthy)
        return is_healthy


@attrs.define(frozen=True, kw_only=True)
class FeetSlipPenalty(ksim.Reward):
    """Penalty for feet slipping."""

    scale: float = -1.0
    com_vel_obs_name: str = attrs.field(default="center_of_mass_velocity_observation")
    feet_contact_obs_name: str = attrs.field(default="feet_contact_observation")

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        if self.feet_contact_obs_name not in trajectory.obs:
            raise ValueError(
                f"Observation {self.feet_contact_obs_name} not found; add it as an observation in your task."
            )
        contact = trajectory.obs[self.feet_contact_obs_name]
        body_vel = trajectory.obs[self.com_vel_obs_name][..., :2]
        return jnp.sum(jnp.linalg.norm(body_vel, axis=-1, keepdims=True) * contact, axis=-1)


@attrs.define(frozen=True, kw_only=True)
class FeetHeightPenalty(ksim.Reward):
    """Cost penalizing feet height."""

    scale: float = -1.0
    max_foot_height: float = 0.1

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        swing_peak = trajectory.reward_carry["swing_peak"]  # type: ignore[attr-defined]
        first_contact = trajectory.reward_carry["first_contact"]  # type: ignore[attr-defined]
        error = swing_peak / self.max_foot_height - 1.0
        return jnp.sum(jnp.square(error) * first_contact, axis=-1)


@attrs.define(frozen=True, kw_only=True)
class FeetAirTimeReward(ksim.Reward):
    """Reward for feet air time."""

    scale: float = 1.0
    threshold_min: float = 0.1
    threshold_max: float = 0.4

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        first_contact = trajectory.reward_carry["first_contact"]  # type: ignore[attr-defined]
        air_time = trajectory.reward_carry["feet_air_time"]  # type: ignore[attr-defined]
        air_time = (air_time - self.threshold_min) * first_contact
        air_time = jnp.clip(air_time, max=self.threshold_max - self.threshold_min)
        return jnp.sum(air_time, axis=-1)


@attrs.define(frozen=True, kw_only=True)
class FeetPhaseReward(ksim.Reward):
    """Reward for tracking the desired foot height."""

    scale: float = 1.0
    feet_pos_obs_name: str = attrs.field(default="feet_position_observation")
    max_foot_height: float = 0.12

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        if self.feet_pos_obs_name not in trajectory.obs:
            raise ValueError(f"Observation {self.feet_pos_obs_name} not found; add it as an observation in your task.")
        foot_pos = trajectory.obs[self.feet_pos_obs_name]
        phase = trajectory.reward_carry["phase"]  # type: ignore[attr-defined]

        foot_z = jnp.array([foot_pos[..., 2], foot_pos[..., 5]]).T
        rz = self.gait_phase(phase, swing_height=self.max_foot_height)

        error = jnp.sum(jnp.square(foot_z - rz), axis=-1)
        reward = jnp.exp(-error / 0.01)

        return reward

    def gait_phase(
        self,
        phi: Array | float,
        swing_height: Array | float = 0.08,
    ) -> Array:
        """Interpolation logic for the gait phase.

        Original implementation:
        https://arxiv.org/pdf/2201.00206
        https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/gait.py#L33
        """

        def _cubic_bezier_interpolation(
            y_start: Array | float, y_end: Array | float, x: Array | float
        ) -> Array | float:
            """Cubic Bezier interpolation for the gait phase."""
            y_diff = y_end - y_start
            bezier = x**3 + 3 * (x**2 * (1 - x))
            return y_start + y_diff * bezier

        x = (phi + jnp.pi) / (2 * jnp.pi)
        stance = _cubic_bezier_interpolation(0, swing_height, 2 * x)
        swing = _cubic_bezier_interpolation(swing_height, 0, 2 * x - 1)
        return jnp.where(x <= 0.5, stance, swing)


@attrs.define(frozen=True, kw_only=True)
class OrientationPenalty(ksim.Reward):
    """Penalty for the orientation of the robot."""

    norm: xax.NormType = attrs.field(default="l2")
    obs_name: str = attrs.field(default="upvector_torso_obs")

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        return xax.get_norm(trajectory.obs[self.obs_name][..., :2], self.norm).sum(axis=-1)


@attrs.define(frozen=True, kw_only=True)
class LinearVelocityTrackingReward(ksim.Reward):
    """Reward for tracking the linear velocity."""

    error_scale: float = attrs.field(default=0.25)
    linvel_obs_name: str = attrs.field(default="local_linvel_torso_obs")
    command_name: str = attrs.field(default="linear_velocity_command")
    norm: xax.NormType = attrs.field(default="l2")

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        if self.linvel_obs_name not in trajectory.obs:
            raise ValueError(f"Observation {self.linvel_obs_name} not found; add it as an observation in your task.")
        lin_vel_error = xax.get_norm(
            trajectory.command[self.command_name][..., :2] - trajectory.obs[self.linvel_obs_name][..., :2], self.norm
        ).sum(axis=-1)
        return jnp.exp(-lin_vel_error / self.error_scale)


@attrs.define(frozen=True, kw_only=True)
class AngularVelocityTrackingReward(ksim.Reward):
    """Reward for tracking the angular velocity."""

    error_scale: float = attrs.field(default=0.25)
    angvel_obs_name: str = attrs.field(default="gyro_torso_obs")
    command_name: str = attrs.field(default="angular_velocity_command")
    norm: xax.NormType = attrs.field(default="l2")

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        if self.angvel_obs_name not in trajectory.obs:
            raise ValueError(f"Observation {self.angvel_obs_name} not found; add it as an observation in your task.")
        ang_vel_error = trajectory.command[self.command_name][..., 2] - trajectory.obs[self.angvel_obs_name][..., 2]
        return jnp.exp(-ang_vel_error / self.error_scale)


@attrs.define(frozen=True, kw_only=True)
class AngularVelocityXYPenalty(ksim.Reward):
    """Penalty for the angular velocity."""

    tracking_sigma: float = attrs.field(default=0.25)
    angvel_obs_name: str = attrs.field(default="global_angvel_torso_obs")
    norm: xax.NormType = attrs.field(default="l2")

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        if self.angvel_obs_name not in trajectory.obs:
            raise ValueError(f"Observation {self.angvel_obs_name} not found; add it as an observation in your task.")
        ang_vel = trajectory.obs[self.angvel_obs_name][..., :2]
        return xax.get_norm(ang_vel, self.norm).sum(axis=-1)


@attrs.define(frozen=True, kw_only=True)
class HipDeviationPenalty(ksim.Reward):
    """Penalty for hip joint deviations."""

    norm: xax.NormType = attrs.field(default="l2")
    hip_indices: tuple[int, ...] = attrs.field()
    joint_targets: tuple[float, ...] = attrs.field()

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        diff = (
            trajectory.qpos[..., jnp.array(self.hip_indices)]
            - jnp.array(self.joint_targets)[jnp.array(self.hip_indices)]
        )
        return xax.get_norm(diff, self.norm).sum(axis=-1)

    @classmethod
    def create(
        cls,
        physics_model: ksim.PhysicsModel,
        hip_names: tuple[str, ...],
        joint_targets: tuple[float, ...],
        scale: float = -1.0,
    ) -> Self:
        """Create a sensor observation from a physics model."""
        mappings = get_qpos_data_idxs_by_name(physics_model)
        hip_indices = tuple([int(mappings[name][0]) - 7 for name in hip_names])
        return cls(
            hip_indices=hip_indices,
            joint_targets=joint_targets,
            scale=scale,
        )


@attrs.define(frozen=True, kw_only=True)
class KneeDeviationPenalty(ksim.Reward):
    """Penalty for knee joint deviations."""

    norm: xax.NormType = attrs.field(default="l2")
    knee_indices: tuple[int, ...] = attrs.field()
    joint_targets: tuple[float, ...] = attrs.field()

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        diff = (
            trajectory.qpos[..., jnp.array(self.knee_indices)]
            - jnp.array(self.joint_targets)[jnp.array(self.knee_indices)]
        )
        return xax.get_norm(diff, self.norm).sum(axis=-1)

    @classmethod
    def create(
        cls,
        physics_model: ksim.PhysicsModel,
        knee_names: tuple[str, ...],
        joint_targets: tuple[float, ...],
        scale: float = -1.0,
    ) -> Self:
        """Create a sensor observation from a physics model."""
        mappings = get_qpos_data_idxs_by_name(physics_model)
        knee_indices = tuple([int(mappings[name][0]) - 7 for name in knee_names])
        return cls(
            knee_indices=knee_indices,
            joint_targets=joint_targets,
            scale=scale,
        )
