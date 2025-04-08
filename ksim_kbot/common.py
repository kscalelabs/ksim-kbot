"""Common utilities for K-Bot 2.

If some utilities will become more general, we can move them to ksim or xax.
"""

from typing import Self

import attrs
import jax
import jax.numpy as jnp
import ksim
import mujoco
import xax
from jaxtyping import Array, PRNGKeyArray
from kscale.web.gen.api import JointMetadataOutput
from ksim.utils.mujoco import get_sensor_data_idxs_by_name, get_site_data_idx_from_name
from mujoco import mjx


class TargetPositionMITActuators(ksim.MITPositionVelocityActuators):
    """MIT-mode actuator controller operating on position."""

    def __init__(
        self,
        physics_model: ksim.PhysicsModel,
        joint_name_to_metadata: dict[str, JointMetadataOutput],
        default_targets: tuple[float, ...] = (),
        *,
        pos_action_noise: float = 0.0,
        pos_action_noise_type: ksim.actuators.NoiseType = "none",
        vel_action_noise: float = 0.0,
        vel_action_noise_type: ksim.actuators.NoiseType = "none",
        torque_noise: float = 0.0,
        torque_noise_type: ksim.actuators.NoiseType = "none",
        ctrl_clip: list[float] | None = None,
        action_scale: float = 1.0,
        freejoint_first: bool = True,
    ) -> None:
        super().__init__(
            physics_model=physics_model,
            joint_name_to_metadata=joint_name_to_metadata,
            pos_action_noise=pos_action_noise,
            pos_action_noise_type=pos_action_noise_type,
            vel_action_noise=vel_action_noise,
            vel_action_noise_type=vel_action_noise_type,
            torque_noise=torque_noise,
            torque_noise_type=torque_noise_type,
            ctrl_clip=ctrl_clip,
            freejoint_first=freejoint_first,
        )
        self.action_scale = action_scale
        self.default_targets = jnp.array(default_targets)

    def get_ctrl(self, action: Array, physics_data: ksim.PhysicsData, rng: PRNGKeyArray) -> Array:
        """Get the control signal from the (position and velocity) action vector."""
        pos_rng, vel_rng, tor_rng = jax.random.split(rng, 3)

        if self.freejoint_first:
            current_pos = physics_data.qpos[7:]  # First 7 are always root pos.
            current_vel = physics_data.qvel[6:]  # First 6 are always root vel.
        else:
            current_pos = physics_data.qpos[:]
            current_vel = physics_data.qvel[:]

        # Adds position and velocity noise.
        target_position = action[: len(current_pos)] * self.action_scale + self.default_targets
        target_velocity = action[len(current_pos) :] * self.action_scale
        target_position = self.add_noise(self.action_noise, self.action_noise_type, target_position, pos_rng)
        target_velocity = self.add_noise(self.vel_action_noise, self.vel_action_noise_type, target_velocity, vel_rng)

        pos_delta = target_position - current_pos
        vel_delta = target_velocity - current_vel
        ctrl = self.kps * pos_delta + self.kds * vel_delta
        return jnp.clip(
            self.add_noise(self.torque_noise, self.torque_noise_type, ctrl, tor_rng),
            -self.ctrl_clip,
            self.ctrl_clip,
        )


class ScaledTorqueActuators(ksim.Actuators):
    """Direct torque control."""

    def __init__(
        self,
        default_targets: Array,
        action_scale: float = 0.5,
        noise: float = 0.0,
        noise_type: ksim.actuators.NoiseType = "none",
    ) -> None:
        super().__init__()

        self._action_scale = action_scale
        self.noise = noise
        self.noise_type = noise_type
        self.default_targets = default_targets

    def get_ctrl(self, action: Array, physics_data: ksim.PhysicsData, rng: PRNGKeyArray) -> Array:
        """Use the scaled action as the torque."""
        action = self.default_targets + action * self._action_scale
        return self.add_noise(self.noise, self.noise_type, action, rng)


@attrs.define(frozen=True)
class JointPositionObservation(ksim.Observation):
    default_targets: tuple[float, ...] = attrs.field()
    noise: float = attrs.field(default=0.0)

    def observe(self, state: ksim.ObservationState, rng: PRNGKeyArray) -> Array:
        qpos = state.physics_state.data.qpos[7:]  # (N,)
        diff = qpos - jnp.array(self.default_targets)
        return diff


@attrs.define(frozen=True)
class ProjectedGravityObservation(ksim.Observation):
    noise: float = attrs.field(default=0.0)

    def observe(self, state: ksim.ObservationState, rng: PRNGKeyArray) -> Array:
        gvec = xax.get_projected_gravity_vector_from_quat(state.physics_state.data.qpos[3:7])
        return gvec


@attrs.define(frozen=True)
class LastActionObservation(ksim.Observation):
    noise: float = attrs.field(default=0.0)

    def observe(self, state: ksim.ObservationState, rng: PRNGKeyArray) -> Array:
        return state.physics_state.most_recent_action


@attrs.define(frozen=True)
class TrueHeightObservation(ksim.Observation):
    """Observation of the true height of the body."""

    def observe(self, state: ksim.ObservationState, rng: PRNGKeyArray) -> Array:
        return jnp.atleast_1d(state.physics_state.data.qpos[2])


@attrs.define(frozen=True, kw_only=True)
class TimestepPhaseObservation(ksim.Observation):
    """Observation of the phase of the timestep."""

    def observe(self, state: ksim.ObservationState, rng: PRNGKeyArray) -> Array:
        return jnp.array([jnp.cos(state.physics_state.data.time), jnp.sin(state.physics_state.data.time)])


@attrs.define(frozen=True)
class FeetPositionObservation(ksim.Observation):
    foot_left: int = attrs.field()
    foot_right: int = attrs.field()
    floor_threshold: float = attrs.field(default=0.0)

    @classmethod
    def create(
        cls,
        *,
        physics_model: ksim.PhysicsModel,
        foot_left_site_name: str,
        foot_right_site_name: str,
        floor_threshold: float = 0.0,
    ) -> Self:
        foot_left_idx = get_site_data_idx_from_name(physics_model, foot_left_site_name)
        foot_right_idx = get_site_data_idx_from_name(physics_model, foot_right_site_name)
        return cls(
            foot_left=foot_left_idx,
            foot_right=foot_right_idx,
            floor_threshold=floor_threshold,
        )

    def observe(self, state: ksim.ObservationState, rng: PRNGKeyArray) -> Array:
        foot_left_pos = state.physics_state.data.site_xpos[self.foot_left] + jnp.array([0.0, 0.0, self.floor_threshold])
        foot_right_pos = state.physics_state.data.site_xpos[self.foot_right] + jnp.array(
            [0.0, 0.0, self.floor_threshold]
        )
        return jnp.concatenate([foot_left_pos, foot_right_pos], axis=-1)


@attrs.define(frozen=True, kw_only=True)
class FeetContactObservation(ksim.FeetContactObservation):
    """Observation of the feet contact."""

    def observe(self, state: ksim.ObservationState, rng: PRNGKeyArray) -> Array:
        feet_contact_12 = super().observe(state, rng)
        return feet_contact_12.flatten()


@attrs.define(frozen=True, kw_only=True)
class GVecTermination(ksim.Termination):
    """Terminates the episode if the robot is facing down."""

    sensor_idx_range: tuple[int, int | None] = attrs.field()
    min_z: float = attrs.field(default=0.0)

    def __call__(self, state: ksim.PhysicsData, curriculum_level: Array) -> Array:
        start, end = self.sensor_idx_range
        return jnp.where(state.sensordata[start:end][-1] < self.min_z, -1, 0)

    @classmethod
    def create(cls, physics_model: ksim.PhysicsModel, sensor_name: str) -> Self:
        sensor_idx_range = get_sensor_data_idxs_by_name(physics_model)[sensor_name]
        return cls(sensor_idx_range=sensor_idx_range)


@attrs.define(frozen=True, kw_only=True)
class ResetDefaultJointPosition(ksim.Reset):
    """Resets the joint positions of the robot to random values."""

    default_targets: tuple[float, ...] = attrs.field()

    def __call__(self, data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> ksim.PhysicsData:
        qpos = data.qpos
        match type(data):
            case mujoco.MjData:
                qpos[:] = self.default_targets
            case mjx.Data:
                qpos = qpos.at[:].set(self.default_targets)
        return ksim.utils.mujoco.update_data_field(data, "qpos", qpos)
