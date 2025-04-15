"""Common utilities for K-Bot 2.

If some utilities will become more general, we can move them to ksim or xax.
"""

from typing import Collection, Self

import attrs
import jax
import jax.numpy as jnp
import ksim
import mujoco
import xax
from jaxtyping import Array, PRNGKeyArray
from kscale.web.gen.api import JointMetadataOutput
from ksim.utils.mujoco import (
    get_sensor_data_idxs_by_name,
    get_site_data_idx_from_name,
    slice_update,
    update_data_field,
)
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
class TimestepPhaseObservation(ksim.TimestepObservation):
    """Observation of the phase of the timestep."""

    ctrl_dt: float = attrs.field(default=0.02)

    def observe(self, state: ksim.ObservationState, rng: PRNGKeyArray) -> Array:
        gait_freq = state.commands["gait_frequency_command"]
        timestep = super().observe(state, rng)
        steps = timestep / self.ctrl_dt
        phase_dt = 2 * jnp.pi * gait_freq * self.ctrl_dt
        start_phase = jnp.array([0, jnp.pi])  # trotting gait
        phase = start_phase + steps * phase_dt
        phase = jnp.fmod(phase + jnp.pi, 2 * jnp.pi) - jnp.pi

        return jnp.array([jnp.cos(phase), jnp.sin(phase)]).flatten()


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


@attrs.define(frozen=True, kw_only=True)
class FarFromOriginTermination(ksim.Termination):
    """Terminates the episode if the robot is too far from the origin.

    This is treated as a positive termination.
    """

    max_dist: float = attrs.field()

    def __call__(self, state: ksim.PhysicsData, curriculum_level: Array) -> Array:
        return jnp.where(jnp.linalg.norm(state.qpos[..., :2], axis=-1) > self.max_dist, -1, 0)


@attrs.define(frozen=True)
class GaitFrequencyCommand(ksim.Command):
    """Command to set the gait frequency of the robot."""

    gait_freq_lower: float = attrs.field(default=1.2)
    gait_freq_upper: float = attrs.field(default=1.5)

    def initial_command(
        self,
        physics_data: ksim.PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        """Returns (1,) array with gait frequency."""
        return jax.random.uniform(rng, (1,), minval=self.gait_freq_lower, maxval=self.gait_freq_upper)

    def __call__(
        self,
        prev_command: Array,
        physics_data: ksim.PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        return prev_command


@attrs.define(frozen=True)
class LinearVelocityCommand(ksim.Command):
    """Command to move the robot in a straight line.

    By convention, X is forward and Y is left. The switching probability is the
    probability of resampling the command at each step. The zero probability is
    the probability of the command being zero - this can be used to turn off
    any command.
    """

    x_range: tuple[float, float] = attrs.field()
    y_range: tuple[float, float] = attrs.field()
    x_zero_prob: float = attrs.field(default=0.0)
    y_zero_prob: float = attrs.field(default=0.0)
    switch_prob: float = attrs.field(default=0.0)
    vis_height: float = attrs.field(default=1.0)
    vis_scale: float = attrs.field(default=0.05)

    def initial_command(self, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        rng_x, rng_y, rng_zero_x, rng_zero_y = jax.random.split(rng, 4)
        (xmin, xmax), (ymin, ymax) = self.x_range, self.y_range
        x = jax.random.uniform(rng_x, (1,), minval=xmin, maxval=xmax)
        y = jax.random.uniform(rng_y, (1,), minval=ymin, maxval=ymax)
        x_zero_mask = jax.random.bernoulli(rng_zero_x, self.x_zero_prob)
        y_zero_mask = jax.random.bernoulli(rng_zero_y, self.y_zero_prob)
        return jnp.concatenate(
            [
                jnp.where(x_zero_mask, 0.0, x),
                jnp.where(y_zero_mask, 0.0, y),
            ]
        )

    def __call__(
        self, prev_command: Array, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray
    ) -> Array:
        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_commands = self.initial_command(physics_data, curriculum_level, rng_b)
        return jnp.where(switch_mask, new_commands, prev_command)

    def get_markers(self) -> Collection[ksim.vis.Marker]:
        return []


@attrs.define(frozen=True)
class AngularVelocityCommand(ksim.Command):
    """Command to turn the robot."""

    scale: float = attrs.field()
    zero_prob: float = attrs.field(default=0.0)
    switch_prob: float = attrs.field(default=0.0)

    def initial_command(self, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        """Returns (1,) array with angular velocity."""
        rng_a, rng_b = jax.random.split(rng)
        zero_mask = jax.random.bernoulli(rng_a, self.zero_prob)
        cmd = jax.random.uniform(rng_b, (1,), minval=-self.scale, maxval=self.scale)
        return jnp.where(zero_mask, jnp.zeros_like(cmd), cmd)

    def __call__(
        self, prev_command: Array, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray
    ) -> Array:
        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_commands = self.initial_command(physics_data, curriculum_level, rng_b)
        return jnp.where(switch_mask, new_commands, prev_command)


@attrs.define(frozen=True, kw_only=True)
class XYPushEvent(ksim.Event):
    """Randomly push the robot after some interval."""

    interval_range: tuple[float, float] = attrs.field()
    force_range: tuple[float, float] = attrs.field()
    curriculum_scale: float = attrs.field(default=1.0)

    def __call__(
        self,
        model: ksim.PhysicsModel,
        data: ksim.PhysicsData,
        event_state: Array,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PhysicsData, Array]:
        # Decrement by physics timestep.
        dt = jnp.float32(model.opt.timestep)
        time_remaining = event_state - dt

        # Update the data if the time remaining is less than 0.
        updated_data, time_remaining = jax.lax.cond(
            time_remaining <= 0.0,
            lambda: self._apply_random_force(data, curriculum_level, rng),
            lambda: (data, time_remaining),
        )

        return updated_data, time_remaining

    def _apply_random_force(
        self, data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray
    ) -> tuple[ksim.PhysicsData, Array]:
        push_theta = jax.random.uniform(rng, maxval=2 * jnp.pi)
        push_magnitude = (
            jax.random.uniform(
                rng,
                minval=self.force_range[0],
                maxval=self.force_range[1],
            )
            * curriculum_level
            * self.curriculum_scale
        )
        push = jnp.array([jnp.cos(push_theta), jnp.sin(push_theta)])
        random_forces = push * push_magnitude + data.qvel[:2]
        new_qvel = slice_update(data, "qvel", slice(0, 2), random_forces)
        updated_data = update_data_field(data, "qvel", new_qvel)

        # Chooses a new remaining interval.
        minval, maxval = self.interval_range
        time_remaining = jax.random.uniform(rng, (), minval=minval, maxval=maxval)

        return updated_data, time_remaining

    def get_initial_event_state(self, rng: PRNGKeyArray) -> Array:
        minval, maxval = self.interval_range
        return jax.random.uniform(rng, (), minval=minval, maxval=maxval)


@attrs.define(frozen=True, kw_only=True)
class TorquePushEvent(ksim.Event):
    """Randomly push the robot with torque (angular velocity) after some interval."""

    interval_range: tuple[float, float] = attrs.field()
    ang_vel_range: tuple[float, float] = attrs.field()  # Min/max push angular velocity per axis
    curriculum_scale: float = attrs.field(default=1.0)

    def __call__(
        self,
        model: ksim.PhysicsModel,
        data: ksim.PhysicsData,
        event_state: Array,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PhysicsData, Array]:
        dt = jnp.float32(model.opt.timestep)
        time_remaining = event_state - dt

        # Update the data if the time remaining is less than 0.
        updated_data, time_remaining = jax.lax.cond(
            time_remaining <= 0.0,
            lambda: self._apply_random_angular_velocity_push(data, curriculum_level, rng),
            lambda: (data, time_remaining),
        )

        return updated_data, time_remaining

    def _apply_random_angular_velocity_push(
        self, data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray
    ) -> tuple[ksim.PhysicsData, Array]:
        """Applies a random angular velocity push to the root body."""
        rng_push, rng_interval = jax.random.split(rng)

        # Sample angular velocity push components
        min_ang_vel, max_ang_vel = self.ang_vel_range
        push_ang_vel = jax.random.uniform(
            rng_push,
            shape=(3,),  # Angular velocity is 3D (wx, wy, wz)
            minval=min_ang_vel,
            maxval=max_ang_vel,
        )
        scaled_push_ang_vel = push_ang_vel * curriculum_level * self.curriculum_scale

        # Apply the push to angular velocity (qvel indices 3:6 for free joint)
        ang_vel_indices = slice(3, 6)

        # Add the push to the current angular velocity
        current_ang_vel = data.qvel[ang_vel_indices]
        new_ang_vel_val = current_ang_vel + scaled_push_ang_vel
        new_qvel = slice_update(data, "qvel", ang_vel_indices, new_ang_vel_val)
        updated_data = update_data_field(data, "qvel", new_qvel)

        minval_interval, maxval_interval = self.interval_range
        time_remaining = jax.random.uniform(rng_interval, (), minval=minval_interval, maxval=maxval_interval)

        return updated_data, time_remaining

    def get_initial_event_state(self, rng: PRNGKeyArray) -> Array:
        minval, maxval = self.interval_range
        return jax.random.uniform(rng, (), minval=minval, maxval=maxval)
