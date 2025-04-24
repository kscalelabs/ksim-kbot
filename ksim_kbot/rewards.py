"""Common rewards for K-Bot 2.

If some logic will become more general, we can move it to ksim or xax.
"""

from typing import Literal, Self

import attrs
import jax.numpy as jnp
import ksim
import xax
from jax.scipy.spatial.transform import Rotation
from jaxtyping import Array
from ksim.utils.mujoco import get_qpos_data_idxs_by_name


@attrs.define(frozen=True, kw_only=True)
class JointDeviationPenalty(ksim.Reward):
    """Penalty for joint deviations."""

    norm: xax.NormType = attrs.field(default="l2")
    joint_targets: tuple[float, ...] = attrs.field()
    joint_weights: tuple[float, ...] = attrs.field(default=None)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        diff = trajectory.qpos[..., 7:] - jnp.array(self.joint_targets)
        cost = jnp.square(diff) * jnp.array(self.joint_weights)
        reward_value = jnp.sum(cost, axis=-1)
        return reward_value

    @classmethod
    def create(
        cls,
        physics_model: ksim.PhysicsModel,
        scale: float = -1.0,
        *,
        joint_targets: tuple[float, ...],
        joint_weights: tuple[float, ...] | None = None,
    ) -> Self:
        if joint_weights is None:
            joint_weights = tuple([1.0] * len(joint_targets))

        return cls(
            scale=scale,
            joint_targets=joint_targets,
            joint_weights=joint_weights,
        )


@attrs.define(frozen=True, kw_only=True)
class FeetSlipPenalty(ksim.Reward):
    """Penalty for feet slipping."""

    scale: float = -1.0
    com_vel_obs_name: str = attrs.field(default="center_of_mass_velocity_observation")
    feet_contact_obs_name: str = attrs.field(default="feet_contact_observation")

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        if self.feet_contact_obs_name not in trajectory.obs:
            raise ValueError(
                f"Observation {self.feet_contact_obs_name} not found; add it as an observation in your task."
            )
        contact = trajectory.obs[self.feet_contact_obs_name]
        body_vel = trajectory.obs[self.com_vel_obs_name][..., :2]
        normed_body_vel = jnp.linalg.norm(body_vel, axis=-1, keepdims=True)
        reward_value = jnp.sum(normed_body_vel * contact, axis=-1)
        return reward_value


@attrs.define(frozen=True, kw_only=True)
class SensorOrientationPenalty(ksim.Reward):
    """Penalty for the orientation of the robot."""

    norm: xax.NormType = attrs.field(default="l2")
    obs_name: str = attrs.field(default="sensor_observation_upvector_origin")

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        reward_value = xax.get_norm(trajectory.obs[self.obs_name][..., :2], self.norm).sum(axis=-1)
        return reward_value


@attrs.define(frozen=True, kw_only=True)
class OrientationPenalty(ksim.Reward):
    """Penalizes deviation from upright orientation using the upvector approach.

    Rotates a unit up vector [0,0,1] by the current quaternion
    and penalizes any x,y components, which should be zero if perfectly upright.
    """

    scale: float = attrs.field()

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        quat = trajectory.qpos[..., 3:7]
        up = jnp.array([0.0, 0.0, 1.0])
        rot_up = Rotation(quat).apply(up)
        orientation_penalty = jnp.sum(jnp.square(rot_up[..., :2]), axis=-1)
        return orientation_penalty


@attrs.define(frozen=True, kw_only=True)
class LinearVelocityTrackingReward(ksim.Reward):
    """Reward for tracking the linear velocity."""

    error_scale: float = attrs.field(default=0.25)
    linvel_obs_name: str = attrs.field(default="sensor_observation_local_linvel_origin")
    command_name: str = attrs.field(default="linear_velocity_command")
    norm: xax.NormType = attrs.field(default="l2")
    stand_still_threshold: float = attrs.field(default=0.0)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        if self.linvel_obs_name not in trajectory.obs:
            raise ValueError(f"Observation {self.linvel_obs_name} not found; add it as an observation in your task.")

        command = trajectory.command[self.command_name]
        lin_vel_error = xax.get_norm(command - trajectory.obs[self.linvel_obs_name][..., :2], self.norm).sum(axis=-1)
        reward_value = jnp.exp(-lin_vel_error / self.error_scale)

        command_norm = jnp.linalg.norm(command, axis=-1)
        reward_value *= command_norm > self.stand_still_threshold

        return reward_value


@attrs.define(frozen=True, kw_only=True)
class AngularVelocityTrackingReward(ksim.Reward):
    """Reward for tracking the angular velocity."""

    error_scale: float = attrs.field(default=0.25)
    angvel_obs_name: str = attrs.field(default="sensor_observation_gyro_origin")
    command_name: str = attrs.field(default="angular_velocity_command")
    norm: xax.NormType = attrs.field(default="l2")
    stand_still_threshold: float = attrs.field(default=0.0)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        if self.angvel_obs_name not in trajectory.obs:
            raise ValueError(f"Observation {self.angvel_obs_name} not found; add it as an observation in your task.")

        command = trajectory.command[self.command_name]
        ang_vel_error = jnp.square(command.flatten() - trajectory.obs[self.angvel_obs_name][..., 2])
        reward_value = jnp.exp(-ang_vel_error / self.error_scale)

        command_norm = jnp.linalg.norm(command, axis=-1)
        reward_value *= command_norm > self.stand_still_threshold

        return reward_value


@attrs.define(frozen=True, kw_only=True)
class AngularVelocityXYPenalty(ksim.Reward):
    """Penalty for the angular velocity."""

    norm: xax.NormType = attrs.field(default="l2")
    angvel_obs_name: str = attrs.field(default="sensor_observation_global_angvel_origin")
    command_name: str = attrs.field(default="angular_velocity_command")
    stand_still_threshold: float = attrs.field(default=0.0)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        if self.angvel_obs_name not in trajectory.obs:
            raise ValueError(f"Observation {self.angvel_obs_name} not found; add it as an observation in your task.")
        ang_vel = trajectory.obs[self.angvel_obs_name][..., :2]
        command = trajectory.command[self.command_name]
        command_norm = jnp.linalg.norm(command, axis=-1)
        reward_value = xax.get_norm(ang_vel, self.norm).sum(axis=-1)
        reward_value *= command_norm > self.stand_still_threshold
        return reward_value


@attrs.define(frozen=True, kw_only=True)
class HipDeviationPenalty(ksim.Reward):
    """Penalty for hip joint deviations."""

    norm: xax.NormType = attrs.field(default="l2")
    hip_indices: tuple[int, ...] = attrs.field()
    joint_targets: tuple[float, ...] = attrs.field()

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        diff = (
            trajectory.qpos[..., jnp.array(self.hip_indices) + 7]
            - jnp.array(self.joint_targets)[jnp.array(self.hip_indices)]
        )
        reward_value = xax.get_norm(diff, self.norm).sum(axis=-1)
        return reward_value

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

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        diff = (
            trajectory.qpos[..., jnp.array(self.knee_indices) + 7]
            - jnp.array(self.joint_targets)[jnp.array(self.knee_indices)]
        )
        reward_value = xax.get_norm(diff, self.norm).sum(axis=-1)
        return reward_value

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


@attrs.define(frozen=True, kw_only=True)
class TerminationPenalty(ksim.Reward):
    """Penalty for termination."""

    scale: float = attrs.field(default=-1.0)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        reward_value = trajectory.done
        return reward_value


@attrs.define(frozen=True, kw_only=True)
class XYPositionPenalty(ksim.Reward):
    """Penalty for deviation from a target (x, y) position."""

    target_x: float = attrs.field()
    target_y: float = attrs.field()
    norm: xax.NormType = attrs.field(default="l2")

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        current_pos = trajectory.qpos[..., :2]
        target_pos = jnp.array([self.target_x, self.target_y])
        diff = current_pos - target_pos
        reward_value = xax.get_norm(diff, self.norm).sum(axis=-1)
        return reward_value


@attrs.define(frozen=True, kw_only=True)
class FarFromOriginTerminationReward(ksim.Reward):
    """Reward for being far from the origin."""

    max_dist: float = attrs.field()

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        reward_value = jnp.linalg.norm(trajectory.qpos[..., :2], axis=-1) > self.max_dist
        return reward_value


@attrs.define(frozen=True, kw_only=True)
class KsimLinearVelocityTrackingReward(ksim.Reward):
    """Penalty for deviating from the linear velocity command."""

    index: int = attrs.field()
    command_name: str = attrs.field()
    norm: xax.NormType = attrs.field(default="l1")
    temp: float = attrs.field(default=1.0)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        dim = self.index
        lin_vel_cmd = trajectory.command[self.command_name].squeeze(-1)
        lin_vel = trajectory.qvel[..., dim]
        norm = xax.get_norm(lin_vel - lin_vel_cmd, self.norm)
        reward_value = 1.0 / (norm / self.temp + 1.0)
        return reward_value

    def get_name(self) -> str:
        return f"{self.index}_{super().get_name()}"


@attrs.define(frozen=True, kw_only=True)
class JointPositionLimitPenalty(ksim.Reward):
    """Penalty for joint position limits."""

    lower_limits: xax.HashableArray = attrs.field()
    upper_limits: xax.HashableArray = attrs.field()

    @classmethod
    def create(
        cls,
        physics_model: ksim.PhysicsModel,
        *,
        soft_limit_factor: float = 0.95,
        scale: float = -1.0,
    ) -> Self:
        # Note: First joint is freejoint.
        lowers, uppers = physics_model.jnt_range[1:].T
        center = (lowers + uppers) / 2
        range = uppers - lowers
        soft_lowers = center - 0.5 * range * soft_limit_factor
        soft_uppers = center + 0.5 * range * soft_limit_factor

        return cls(
            scale=scale,
            lower_limits=xax.hashable_array(soft_lowers),
            upper_limits=xax.hashable_array(soft_uppers),
        )

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        penalty = -jnp.clip(trajectory.qpos[..., 7:] - self.lower_limits.array, None, 0.0)
        penalty += jnp.clip(trajectory.qpos[..., 7:] - self.upper_limits.array, 0.0, None)
        return jnp.sum(penalty, axis=-1)


@attrs.define(frozen=True, kw_only=True)
class ContactForcePenalty(ksim.Reward):
    """Penalty for too high contact force."""

    max_contact_force: float = attrs.field(default=350.0)
    sensor_names: tuple[str, ...] = attrs.field()

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        for sensor_name in self.sensor_names:
            if sensor_name not in trajectory.obs:
                raise ValueError(f"{sensor_name} not found in trajectory.obs")

        forces_t3b = jnp.stack([trajectory.obs[name] for name in self.sensor_names], axis=-1)
        cost = jnp.clip(jnp.abs(forces_t3b[..., 2, :]) - self.max_contact_force, min=0.0)
        cost = jnp.sum(cost, axis=-1)
        return cost


@attrs.define(frozen=True, kw_only=True)
class StandStillReward(ksim.Reward):
    """Reward for standing still."""

    scale: float = 1.0
    sensitivity: float = 0.01
    norm: xax.NormType = attrs.field(default="l1")
    linear_velocity_cmd_name: str = attrs.field(default="linear_velocity_command")
    angular_velocity_cmd_name: str = attrs.field(default="angular_velocity_command")
    joint_targets: tuple[float, ...] = attrs.field()
    stand_still_threshold: float = attrs.field(default=0.0)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        vel_cmd = trajectory.command[self.linear_velocity_cmd_name]
        ang_vel_cmd = trajectory.command[self.angular_velocity_cmd_name]
        cmd_norm = jnp.linalg.norm(jnp.concatenate([vel_cmd, ang_vel_cmd], axis=-1), axis=-1)

        error = jnp.sum(
            jnp.square(trajectory.qpos[..., 7:] - jnp.array(self.joint_targets)),
            axis=-1,
        )
        reward = jnp.exp(-error / self.sensitivity)
        reward *= cmd_norm < self.stand_still_threshold
        return reward


@attrs.define(frozen=True, kw_only=True)
class FeetPhaseReward(ksim.Reward):
    """Reward for tracking the desired foot height."""

    scale: float = 1.0
    feet_pos_obs_name: str = attrs.field(default="feet_position_observation")
    linear_velocity_cmd_name: str = attrs.field(default="linear_velocity_command")
    angular_velocity_cmd_name: str = attrs.field(default="angular_velocity_command")
    gait_freq_cmd_name: str = attrs.field(default="gait_frequency_command")
    max_foot_height: float = attrs.field(default=0.12)
    ctrl_dt: float = attrs.field(default=0.02)
    sensitivity: float = attrs.field(default=0.01)
    foot_default_height: float = attrs.field(default=0.0)
    stand_still_threshold: float = attrs.field(default=0.0)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        if self.feet_pos_obs_name not in trajectory.obs:
            raise ValueError(f"Observation {self.feet_pos_obs_name} not found; add it as an observation in your task.")
        if self.gait_freq_cmd_name not in trajectory.command:
            raise ValueError(f"Command {self.gait_freq_cmd_name} not found; add it as a command in your task.")

        # generate phase values
        gait_freq_n = trajectory.command[self.gait_freq_cmd_name]

        phase_dt = 2 * jnp.pi * gait_freq_n * self.ctrl_dt
        steps = jnp.int32(trajectory.timestep / self.ctrl_dt)
        steps = jnp.repeat(steps[:, None], 2, axis=1)

        start_phase = jnp.broadcast_to(jnp.array([0.0, jnp.pi]), (steps.shape[0], 2))
        phase = start_phase + steps * phase_dt
        phase = jnp.fmod(phase + jnp.pi, 2 * jnp.pi) - jnp.pi

        # batch reward over the time dimension
        foot_pos = trajectory.obs[self.feet_pos_obs_name]

        foot_z = jnp.array([foot_pos[..., 2], foot_pos[..., 5]]).T
        ideal_z = self.gait_phase(phase, swing_height=jnp.array(self.max_foot_height))
        error = jnp.sum(jnp.square(foot_z - ideal_z), axis=-1)
        reward = jnp.exp(-error / self.sensitivity)

        # no movement for small velocity command
        vel_cmd = trajectory.command[self.linear_velocity_cmd_name]
        ang_vel_cmd = trajectory.command[self.angular_velocity_cmd_name]
        command_norm = jnp.linalg.norm(jnp.concatenate([vel_cmd, ang_vel_cmd], axis=-1), axis=-1)
        reward *= command_norm > self.stand_still_threshold

        return reward

    def gait_phase(
        self,
        phi: Array | float,
        swing_height: Array = jnp.array(0.08),
    ) -> Array:
        """Interpolation logic for the gait phase.

        Original implementation:
        https://arxiv.org/pdf/2201.00206
        https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/gait.py#L33
        """
        x = (phi + jnp.pi) / (2 * jnp.pi)
        x = jnp.clip(x, 0, 1)
        stance = xax.cubic_bezier_interpolation(jnp.array(0), swing_height, 2 * x)
        swing = xax.cubic_bezier_interpolation(swing_height, jnp.array(0), 2 * x - 1)
        return jnp.where(x <= 0.5, stance, swing)


@attrs.define(frozen=True)
class TargetLinearVelocityReward(ksim.Reward):
    """Reward for forward motion."""

    index: Literal["x", "y", "z"] = attrs.field(default="x")
    target_vel: float = attrs.field(default=0.0)
    norm: xax.NormType = attrs.field(default="l1")
    monotonic_fn: Literal["exp", "inv"] = attrs.field(default="inv")
    temp: float = attrs.field(default=1.0)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        vel = trajectory.qvel[..., ksim.cartesian_index_to_dim(self.index)]
        error = xax.get_norm(vel - self.target_vel, self.norm)
        return ksim.norm_to_reward(error, temp=self.temp, monotonic_fn=self.monotonic_fn)

    def get_name(self) -> str:
        return f"{self.index}_{super().get_name()}"


@attrs.define(frozen=True, kw_only=True)
class TargetHeightReward(ksim.Reward):
    """Reward for reaching a target height."""

    target_height: float = attrs.field(default=1.0)
    norm: xax.NormType = attrs.field(default="l1")
    temp: float = attrs.field(default=1.0)
    monotonic_fn: Literal["exp", "inv"] = attrs.field(default="inv")

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        qpos = trajectory.qpos
        error = qpos[..., 2] - self.target_height
        reward_value = ksim.norm_to_reward(
            xax.get_norm(error, self.norm), temp=self.temp, monotonic_fn=self.monotonic_fn
        )
        return reward_value
