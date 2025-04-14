# mypy: disable-error-code="override"
"""Walking default humanoid task with reference gait tracking."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, TypeVar

import attrs
import bvhio
import glm
import jax
import jax.numpy as jnp
import ksim
import mujoco
import numpy as np
import xax
from bvhio.lib.hierarchy import Joint as BvhioJoint
from jaxtyping import Array, PRNGKeyArray
from ksim.types import PhysicsModel, PhysicsState
from ksim import ObservationState
from ksim.utils.reference_motion import (
    ReferenceMapping,
    get_local_xpos,
    get_reference_cartesian_poses,
    get_reference_joint_id,
    get_reference_qpos,
    local_to_absolute,
    visualize_reference_motion,
)
from scipy.spatial.transform import Rotation as R

import ksim_kbot.common as common
import ksim_kbot.rewards as kbot_rewards

# Import necessary components from walking_rnn
from ksim_kbot.walking.walking_rnn import (
    NUM_JOINTS,
    NUM_INPUTS,  # Import base actor input size
    NUM_CRITIC_INPUTS, # Import base critic input size
    RnnActor,
    RnnCritic,
    RnnModel,
    WalkingRnnTask,
    WalkingRnnTaskConfig,
)
import distrax # Import distrax

HUMANOID_REFERENCE_MAPPINGS = (
    ReferenceMapping("CC_Base_L_ThighTwist01", "RS03_5"),  # hip
    ReferenceMapping("CC_Base_R_ThighTwist01", "RS03_4"),  # hip
    ReferenceMapping("CC_Base_L_CalfTwist01", "KC_D_401L_L_Shin_Drive"),  # knee
    ReferenceMapping("CC_Base_R_CalfTwist01", "KC_D_401R_R_Shin_Drive"),  # knee
    ReferenceMapping("CC_Base_L_Foot", "KB_D_501L_L_LEG_FOOT"),  # foot
    ReferenceMapping("CC_Base_R_Foot", "KB_D_501R_R_LEG_FOOT"),  # foot
    ReferenceMapping("CC_Base_L_UpperarmTwist01", "RS03_6"),  # shoulder
    ReferenceMapping("CC_Base_R_UpperarmTwist01", "RS03_3"),  # shoulder
    ReferenceMapping("CC_Base_L_ForearmTwist01", "KC_C_401L_L_UpForearmDrive"),  # elbow
    ReferenceMapping("CC_Base_R_ForearmTwist01", "KC_C_401R_R_UpForearmDrive"),  # elbow
    ReferenceMapping("CC_Base_L_Hand", "KB_C_501X_Left_Bayonet_Adapter_Hard_Stop"),  # hand
    ReferenceMapping("CC_Base_R_Hand", "KB_C_501X_Right_Bayonet_Adapter_Hard_Stop"),  # hand
)


@dataclass
class WalkingRnnRefMotionTaskConfig(WalkingRnnTaskConfig):
    bvh_path: str = xax.field(
        value=str(Path(__file__).parent.parent / "reference_motions" / "walk_normal_kbot.bvh"),
        help="The path to the BVH file.",
    )
    rotate_bvh_euler: tuple[float, float, float] = xax.field(
        value=(0, np.pi / 2, 0),
        help="Optional rotation to ensure the BVH tree matches the Mujoco model.",
    )
    bvh_scaling_factor: float = xax.field(
        value=1/100.0,
        help="Scaling factor to ensure the BVH tree matches the Mujoco model.",
    )
    bvh_offset: tuple[float, float, float] = xax.field(
        value=(0.02, 0.09, -0.29),
        help="Offset to ensure the BVH tree matches the Mujoco model.",
    )
    mj_base_name: str = xax.field(
        value="floating_base_link",
        help="The Mujoco body name of the base of the humanoid",
    )
    reference_base_name: str = xax.field(
        value="CC_Base_Pelvis",
        help="The BVH joint name of the base of the humanoid",
    )
    visualize_reference_motion: bool = xax.field(
        value=False,
        help="Whether to visualize the reference motion.",
    )
    orientation_penalty: float = xax.field(
        value=-1.0,
        help="The scale to apply to the orientation penalty.",
    )
    naive_reward_scale: float = xax.field(
        value=1.0,
        help="The scale to apply to the naive reward.",
    )


Config = TypeVar("Config", bound=WalkingRnnRefMotionTaskConfig)

@attrs.define(frozen=True)
class NaiveForwardReward(ksim.Reward):
    """Reward for forward motion."""

    vel_clip_max: float = attrs.field(default=1.0)

    def __call__(self, trajectory: ksim.Trajectory, _: None) -> tuple[Array, None]:
        vel = trajectory.qvel[..., 0]
        clipped_vel = jnp.clip(vel, a_max=self.vel_clip_max)
        return clipped_vel, None

@attrs.define(frozen=True)
class TargetLinearVelocityReward(ksim.Reward):
    """Reward for forward motion."""

    index: int = attrs.field(default=0)
    target_vel: float = attrs.field(default=0.0)
    norm: xax.NormType = attrs.field(default="l1")
    monotonic_fn: str = attrs.field(default="inv")
    temp: float = attrs.field(default=1.0)

    def __call__(self, trajectory: ksim.Trajectory, _: None) -> tuple[Array, None]:
        vel = trajectory.qvel[..., self.index]
        error = xax.get_norm(vel - self.target_vel, self.norm)
        return ksim.norm_to_reward(error, temp=self.temp, monotonic_fn=self.monotonic_fn), None


@attrs.define(frozen=True, kw_only=True)
class QposReferenceMotionReward(ksim.Reward):
    reference_qpos: xax.HashableArray
    ctrl_dt: float
    norm: xax.NormType = attrs.field(default="l1")
    sensitivity: float = attrs.field(default=5.0)
    joint_weights: tuple[float, ...] = attrs.field(default=tuple([1.0] * NUM_JOINTS))
    speed: float = attrs.field(default=1.0)

    @property
    def num_frames(self) -> int:
        return self.reference_qpos.array.shape[0]

    def __call__(self, trajectory: ksim.Trajectory, _: None) -> tuple[Array, None]:
        qpos = trajectory.qpos
        step_number = jnp.int32(jnp.round(self.speed * trajectory.timestep / self.ctrl_dt)) % self.num_frames
        reference_qpos = jnp.take(self.reference_qpos.array, step_number, axis=0)
        error = xax.get_norm(reference_qpos[..., 7:] - qpos[..., 7:], self.norm)
        error = error * jnp.array(self.joint_weights)
        mean_error = error.mean(axis=-1)
        reward = jnp.exp(-mean_error * self.sensitivity)
        return reward, None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MotionAuxOutputs:
    tracked_pos: xax.FrozenDict[int, Array]

def create_tracked_marker_update_fn(
    body_id: int, mj_base_id: int, tracked_pos_fn: Callable[[ksim.Trajectory], xax.FrozenDict[int, Array]]
) -> Callable[[ksim.Marker, ksim.Trajectory], None]:
    """Factory function to create a marker update for the tracked positions."""

    def _actual_update_fn(marker: ksim.Marker, transition: ksim.Trajectory) -> None:
        tracked_pos = tracked_pos_fn(transition)
        abs_pos = local_to_absolute(transition.xpos, tracked_pos[body_id], mj_base_id)
        marker.pos = tuple(abs_pos)

    return _actual_update_fn


def create_target_marker_update_fn(
    body_id: int, mj_base_id: int, target_pos_fn: Callable[[ksim.Trajectory], xax.FrozenDict[int, Array]]
) -> Callable[[ksim.Marker, ksim.Trajectory], None]:
    """Factory function to create a marker update for the target positions."""

    def _target_update_fn(marker: ksim.Marker, transition: ksim.Trajectory) -> None:
        target_pos = target_pos_fn(transition)
        abs_pos = local_to_absolute(transition.xpos, target_pos[body_id], mj_base_id)
        marker.pos = tuple(abs_pos)

    return _target_update_fn

@attrs.define(frozen=True, kw_only=True)
class CartesianReferenceMotionReward(ksim.Reward):
    reference_motion: xax.FrozenDict[int, xax.HashableArray]
    mj_base_id: int
    ctrl_dt: float
    norm: xax.NormType = attrs.field(default="l1")
    sensitivity: float = attrs.field(default=5.0)

    @property
    def num_frames(self) -> int:
        return list(self.reference_motion.values())[0].array.shape[0]

    def get_tracked_pos(self, trajectory: ksim.Trajectory) -> xax.FrozenDict[int, Array]:
        assert isinstance(trajectory.aux_outputs, MotionAuxOutputs)
        return trajectory.aux_outputs.tracked_pos

    def get_target_pos(self, trajectory: ksim.Trajectory) -> xax.FrozenDict[int, Array]:
        reference_motion: xax.FrozenDict[int, Array] = jax.tree.map(lambda x: x.array, self.reference_motion)
        step_number = jnp.int32(jnp.round(trajectory.timestep / self.ctrl_dt)) % self.num_frames
        return jax.tree.map(lambda x: jnp.take(x, step_number, axis=0), reference_motion)

    def __call__(self, trajectory: ksim.Trajectory, _: None) -> tuple[Array, None]:
        target_pos = self.get_target_pos(trajectory)
        tracked_pos = self.get_tracked_pos(trajectory)
        error = jax.tree.map(lambda target, tracked: xax.get_norm(target - tracked, self.norm), target_pos, tracked_pos)
        mean_error_over_bodies = jax.tree.reduce(jnp.add, error) / len(error)
        mean_error = mean_error_over_bodies.mean(axis=-1)
        reward = jnp.exp(-mean_error * self.sensitivity)
        return reward, None

    def get_markers(self) -> list[ksim.Marker]:
        markers = []

        # Add markers for reference positions (in blue)
        for body_id in self.reference_motion.keys():

            markers.append(
                ksim.Marker.sphere(
                    pos=(0.0, 0.0, 0.0),
                    radius=0.05,
                    rgba=(0.0, 0.0, 1.0, 0.5),  # blue = actual
                    update_fn=create_tracked_marker_update_fn(body_id, self.mj_base_id, self.get_tracked_pos),
                )
            )

            markers.append(
                ksim.Marker.sphere(
                    pos=(0.0, 0.0, 0.0),
                    radius=0.05,
                    rgba=(1.0, 0.0, 0.0, 0.5),  # red = target
                    update_fn=create_target_marker_update_fn(body_id, self.mj_base_id, self.get_target_pos),
                )
            )

        return markers


@attrs.define(frozen=True, kw_only=True)
class ReferenceQposObservation(ksim.Observation):
    """Observation for the reference joint positions."""

    reference_qpos: xax.HashableArray
    ctrl_dt: float
    speed: float = attrs.field(default=1.0)

    @property
    def num_frames(self) -> int:
        return self.reference_qpos.array.shape[0]

    def observe(self, state: ObservationState, rng: PRNGKeyArray) -> Array:
        physics_state = state.physics_state
        step_number = jnp.int32(jnp.round(self.speed * physics_state.data.time / self.ctrl_dt)) % self.num_frames
        reference_qpos_at_step = jnp.take(self.reference_qpos.array, step_number, axis=0)
        return reference_qpos_at_step[..., 7:]


@attrs.define(frozen=True, kw_only=True)
class ReferenceLocalXposObservation(ksim.Observation):
    """Observation for the reference local cartesian positions of tracked bodies."""

    reference_motion: xax.FrozenDict[int, xax.HashableArray]
    tracked_body_ids: tuple[int, ...]
    ctrl_dt: float

    @property
    def num_frames(self) -> int:
        # Assuming all bodies have the same number of frames
        return list(self.reference_motion.values())[0].array.shape[0]

    def observe(self, state: ObservationState, rng: PRNGKeyArray) -> Array:
        physics_state = state.physics_state
        reference_motion: xax.FrozenDict[int, Array] = jax.tree.map(lambda x: x.array, self.reference_motion)
        step_number = jnp.int32(jnp.round(physics_state.data.time / self.ctrl_dt)) % self.num_frames
        target_pos_dict = jax.tree.map(lambda x: jnp.take(x, step_number, axis=0), reference_motion)
        target_pos_list = [target_pos_dict[body_id] for body_id in self.tracked_body_ids]
        return jnp.concatenate(target_pos_list, axis=-1)


@attrs.define(frozen=True, kw_only=True)
class TrackedLocalXposObservation(ksim.Observation):
    """Observation for the current local cartesian positions of tracked bodies."""

    tracked_body_ids: tuple[int, ...]
    mj_base_id: int

    def observe(self, state: ObservationState, rng: PRNGKeyArray) -> Array:
        physics_state = state.physics_state
        tracked_positions_list: list[Array] = []
        for body_id in self.tracked_body_ids:
            body_pos = get_local_xpos(physics_state.data.xpos, body_id, self.mj_base_id)
            tracked_positions_list.append(jnp.array(body_pos))
        return jnp.concatenate(tracked_positions_list, axis=-1)


# Actor inputs are the same as the base class for now
NUM_ACTOR_INPUTS_REF = NUM_INPUTS

# Critic inputs are the base class inputs plus the new reference observations
NUM_CRITIC_INPUTS_REF = (
    NUM_CRITIC_INPUTS
    + NUM_JOINTS  # reference_qpos
    + (len(HUMANOID_REFERENCE_MAPPINGS) * 3) # reference_local_xpos
    + (len(HUMANOID_REFERENCE_MAPPINGS) * 3) # tracked_local_xpos
)


class WalkingRnnRefMotionTask(WalkingRnnTask[Config], Generic[Config]):
    config: Config
    reference_qpos: xax.HashableArray
    reference_motion: xax.FrozenDict[int, xax.HashableArray]
    tracked_body_ids: tuple[int, ...]
    mj_base_id: int
    qpos_reference_speed: float

    def get_physics_randomizers(self, physics_model: ksim.PhysicsModel) -> list[ksim.PhysicsRandomizer]:
        return [
            ksim.StaticFrictionRandomizer(),
            ksim.ArmatureRandomizer(),
            ksim.MassMultiplicationRandomizer.from_body_name(physics_model, "Torso_Side_Right"),
            ksim.JointDampingRandomizer(),
            ksim.JointZeroPositionRandomizer(),
        ]

    def get_model(self, key: PRNGKeyArray) -> RnnModel:
        """Overrides the base method to use reference motion input sizes."""
        num_tracked_bodies = len(self.tracked_body_ids)
        num_critic_inputs_actual = (
            NUM_CRITIC_INPUTS
            + NUM_JOINTS  # reference_qpos
            + (num_tracked_bodies * 3) # reference_local_xpos
            + (num_tracked_bodies * 3) # tracked_local_xpos
        )
        assert num_critic_inputs_actual == NUM_CRITIC_INPUTS_REF, \
            f"Calculated critic inputs ({num_critic_inputs_actual}) != constant ({NUM_CRITIC_INPUTS_REF})"

        return RnnModel(
            key,
            num_inputs=NUM_ACTOR_INPUTS_REF, # Use actor size derived from base
            num_critic_inputs=num_critic_inputs_actual, # Use dynamically calculated critic size
            min_std=0.01,
            max_std=1.0,
            mean_scale=self.config.action_scale,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
        )

    def run_actor(
        self,
        model: RnnActor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[distrax.Distribution, Array]:
        """Overrides the base method to define actor inputs for this task."""
        timestep_1 = observations["timestep_observation"]
        joint_pos_j = observations["joint_position_observation"]
        joint_vel_j = observations["joint_velocity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        joystick_cmd_1 = commands["joystick_command"]
        joystick_cmd_ohe_6 = jax.nn.one_hot(joystick_cmd_1, num_classes=6).squeeze(-2)

        # Concatenate observations for the actor (same as base class for now)
        obs_n = jnp.concatenate(
            [
                jnp.cos(timestep_1),  # 1
                jnp.sin(timestep_1),  # 1
                joint_pos_j,  # NUM_JOINTS
                joint_vel_j / 10.0,  # NUM_JOINTS
                imu_acc_3 / 50.0,  # 3
                imu_gyro_3 / 3.0,  # 3
                joystick_cmd_ohe_6,  # 6
            ],
            axis=-1,
        )
        assert obs_n.shape[-1] == NUM_ACTOR_INPUTS_REF, \
            f"Actor input shape ({obs_n.shape[-1]}) != constant ({NUM_ACTOR_INPUTS_REF})"

        return model.forward(obs_n, carry)

    def run_critic(
        self,
        model: RnnCritic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[Array, Array]:
        """Overrides the base method to include reference observations for the critic."""
        timestep_1 = observations["timestep_observation"]
        dh_joint_pos_j = observations["joint_position_observation"]
        dh_joint_vel_j = observations["joint_velocity_observation"]
        com_inertia_n = observations["center_of_mass_inertia_observation"]
        com_vel_n = observations["center_of_mass_velocity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        act_frc_obs_n = observations["actuator_force_observation"]
        base_pos_3 = observations["base_position_observation"]
        base_quat_4 = observations["base_orientation_observation"]
        lin_vel_obs_3 = observations["base_linear_velocity_observation"]
        ang_vel_obs_3 = observations["base_angular_velocity_observation"]
        joystick_cmd_1 = commands["joystick_command"]
        joystick_cmd_ohe_6 = jax.nn.one_hot(joystick_cmd_1, num_classes=6).squeeze(-2)
        # New reference observations
        ref_qpos_j = observations["reference_qpos_observation"]
        ref_local_xpos_n = observations["reference_local_xpos_observation"]
        tracked_local_xpos_n = observations["tracked_local_xpos_observation"]


        # Concatenate observations for the critic, including new ones
        obs_n = jnp.concatenate(
            [
                jnp.cos(timestep_1),  # 1
                jnp.sin(timestep_1),  # 1
                dh_joint_pos_j,  # NUM_JOINTS
                dh_joint_vel_j / 10.0,  # NUM_JOINTS
                com_inertia_n,  # 160
                com_vel_n,  # 96
                imu_acc_3 / 50.0,  # 3
                imu_gyro_3 / 3.0,  # 3
                act_frc_obs_n / 100.0,  # NUM_JOINTS
                base_pos_3,  # 3
                base_quat_4,  # 4
                lin_vel_obs_3,  # 3
                ang_vel_obs_3,  # 3
                joystick_cmd_ohe_6,  # 6
                # Add reference observations
                ref_qpos_j, # NUM_JOINTS
                ref_local_xpos_n, # num_tracked_bodies * 3
                tracked_local_xpos_n, # num_tracked_bodies * 3
            ],
            axis=-1,
        )

        assert obs_n.shape[-1] == NUM_CRITIC_INPUTS_REF, \
             f"Critic input shape ({obs_n.shape[-1]}) != constant ({NUM_CRITIC_INPUTS_REF})"


        return model.forward(obs_n, carry)


    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        rewards: list[ksim.Reward] = [
            ksim.StayAliveReward(
                success_reward=1.0,
                scale=8.0,
            ),
            # kbot_rewards.SensorOrientationPenalty(scale=self.config.orientation_penalty),
            # CartesianReferenceMotionReward(
            #     reference_motion=self.reference_motion,
            #     mj_base_id=self.mj_base_id,
            #     ctrl_dt=self.config.ctrl_dt,
            #     scale=1.0,
            # ),
            QposReferenceMotionReward(
                reference_qpos=self.reference_qpos,
                ctrl_dt=self.config.ctrl_dt,
                scale=15.0,
                speed=self.qpos_reference_speed,
                joint_weights=(
                    # right arm
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    # left arm
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    # right leg
                    2.0, # hip pitch
                    1.0, # hip roll
                    1.0, # hip yaw
                    2.0, # knee
                    1.0, # ankle
                    # left leg
                    2.0, # hip pitch
                    1.0, # hip roll
                    1.0, # hip yaw
                    2.0, # knee
                    1.0, # ankle
                ),
            ),
            kbot_rewards.FeetSlipPenalty(scale=-2.0),
            kbot_rewards.FeetAirTimeReward(
                scale=8.0,
            ),
            TargetLinearVelocityReward(
                index=0, # x-axis
                target_vel=0.5,
                scale=1.5,
            ),
            TargetLinearVelocityReward(
                index=1, # y-axis
                target_vel=0.0,
                scale=1.5,
            ),
            # NaiveForwardReward(scale=1.5, vel_clip_max=0.5),
            ksim.LinearVelocityPenalty(index="z", scale=-2.0),
            kbot_rewards.TargetHeightReward(target_height=1.0, scale=1.0),
        ]

        return rewards

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomJointPositionReset(),
            ksim.RandomJointVelocityReset(),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        return [
            ksim.JointPositionObservation(),
            ksim.JointVelocityObservation(),
            ksim.ActuatorForceObservation(),
            ksim.CenterOfMassInertiaObservation(),
            ksim.CenterOfMassVelocityObservation(),
            ksim.BasePositionObservation(),
            ksim.BaseOrientationObservation(),
            ksim.BaseLinearVelocityObservation(),
            ksim.BaseAngularVelocityObservation(),
            ksim.BaseLinearAccelerationObservation(),
            ksim.BaseAngularAccelerationObservation(),
            ksim.ActuatorAccelerationObservation(),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_acc"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_gyro"),
            common.FeetContactObservation.create(
                physics_model=physics_model,
                foot_left_geom_names="KB_D_501L_L_LEG_FOOT_collision_box",
                foot_right_geom_names="KB_D_501R_R_LEG_FOOT_collision_box",
                floor_geom_names="floor",
            ),
            ksim.TimestepObservation(),
            ReferenceQposObservation(
                reference_qpos=self.reference_qpos,
                ctrl_dt=self.config.ctrl_dt,
                speed=self.qpos_reference_speed,
            ),
            ReferenceLocalXposObservation(
                reference_motion=self.reference_motion,
                tracked_body_ids=self.tracked_body_ids,
                ctrl_dt=self.config.ctrl_dt,
            ),
            TrackedLocalXposObservation(
                tracked_body_ids=self.tracked_body_ids,
                mj_base_id=self.mj_base_id,
            ),
        ]

    def get_ppo_variables(
        self,
        model: RnnModel,
        trajectory: ksim.Trajectory,
        model_carry: tuple[Array, Array],
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, tuple[Array, Array]]:
        def scan_fn(
            actor_critic_carry: tuple[Array, Array], transition: ksim.Trajectory
        ) -> tuple[tuple[Array, Array], ksim.PPOVariables]:
            actor_carry, critic_carry = actor_critic_carry
            actor_dist, next_actor_carry = self.run_actor(
                model=model.actor,
                observations=transition.obs,
                commands=transition.command,
                carry=actor_carry,
            )
            log_probs = actor_dist.log_prob(transition.action)
            assert isinstance(log_probs, Array)
            value, next_critic_carry = self.run_critic(
                model=model.critic,
                observations=transition.obs,
                commands=transition.command,
                carry=critic_carry,
            )

            transition_ppo_variables = ksim.PPOVariables(
                log_probs=log_probs,
                values=value.squeeze(-1),
            )

            initial_carry = self.get_initial_model_carry(rng)
            next_carry = jax.tree.map(
                lambda x, y: jnp.where(transition.done, x, y), initial_carry, (next_actor_carry, next_critic_carry)
            )

            return next_carry, transition_ppo_variables

        next_model_carry, ppo_variables = jax.lax.scan(scan_fn, model_carry, trajectory)

        return ppo_variables, next_model_carry

    def sample_action(
        self,
        model: RnnModel,
        model_carry: tuple[Array, Array],
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
        argmax: bool = False,
    ) -> ksim.Action:
        actor_carry_in, critic_carry_in = model_carry

        # Runs the actor model to get the action distribution.
        action_dist_j, actor_carry = self.run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
            carry=actor_carry_in,
        )

        if argmax:
            action_j = action_dist_j.mode()
        else:
            action_j = action_dist_j.sample(seed=rng)

        # Getting the local cartesian positions for all tracked bodies.
        tracked_positions: dict[int, Array] = {}
        for body_id in self.tracked_body_ids:
            body_pos = get_local_xpos(physics_state.data.xpos, body_id, self.mj_base_id)
            tracked_positions[body_id] = jnp.array(body_pos)

        return ksim.Action(
            action=action_j,
            carry=(actor_carry, critic_carry_in),
            aux_outputs=MotionAuxOutputs(
                tracked_pos=xax.FrozenDict(tracked_positions),
            ),
        )

    def run(self) -> None:
        mj_model: PhysicsModel = self.get_mujoco_model()
        root: BvhioJoint = bvhio.readAsHierarchy(self.config.bvh_path)
        reference_base_id = get_reference_joint_id(root, self.config.reference_base_name)
        self.mj_base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, self.config.mj_base_name)
        self.qpos_reference_speed = 1.8

        def rotation_callback(root: BvhioJoint) -> None:
            euler_rotation = np.array(self.config.rotate_bvh_euler)
            quat = R.from_euler("xyz", euler_rotation).as_quat(scalar_first=True)
            root.applyRotation(glm.quat(*quat), bake=True)

        np_reference_qpos = get_reference_qpos(
            model=mj_model,
            mj_base_id=self.mj_base_id,
            bvh_root=root,
            bvh_to_mujoco_names=HUMANOID_REFERENCE_MAPPINGS,
            bvh_base_id=reference_base_id,
            bvh_offset=np.array(self.config.bvh_offset),
            bvh_root_callback=rotation_callback,
            bvh_scaling_factor=self.config.bvh_scaling_factor,
            neutral_qpos=None,
            neutral_similarity_weight=0.1,
            temporal_consistency_weight=0.1,
            n_restarts=3,
            error_acceptance_threshold=1e-4,
            ftol=1e-8,
            xtol=1e-8,
            max_nfev=2000,
            verbose=False,
        )
        self.reference_qpos = xax.hashable_array(jnp.array(np_reference_qpos))

        np_reference_motion = get_reference_cartesian_poses(
            mappings=HUMANOID_REFERENCE_MAPPINGS,
            model=mj_model,
            root=root,
            reference_base_id=reference_base_id,
            root_callback=rotation_callback,
            scaling_factor=self.config.bvh_scaling_factor,
            offset=np.array(self.config.bvh_offset),
        )
        self.reference_motion: xax.FrozenDict[int, xax.HashableArray] = jax.tree.map(
            lambda x: xax.hashable_array(jnp.array(x)), np_reference_motion
        )
        self.tracked_body_ids = tuple(self.reference_motion.keys())

        if self.config.visualize_reference_motion:
            visualize_reference_motion(
                mj_model,
                reference_qpos=np_reference_qpos,
                cartesian_motion=np_reference_motion,
                mj_base_id=self.mj_base_id,
            )
        else:
            super().run()


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m ksim_kbot.walking.walking_reference_motion num_envs=2 batch_size=2
    # To visualize the environment, use the following command:
    #   python -m ksim_kbot.walking.walking_reference_motion run_environment=True
    # To visualize the reference gait, use the following command:
    #   mjpython -m ksim_kbot.walking.walking_reference_motion visualize_reference_motion=True
    # On MacOS or other devices with less memory, you can change the number
    # of environments and batch size to reduce memory usage. Here's an example
    # from the command line:
    #   python -m ksim_kbot.walking.walking_reference_motion num_envs=1 batch_size=1
    WalkingRnnRefMotionTask.launch(
        WalkingRnnRefMotionTaskConfig(
            # Training parameters.
            num_envs=2048,
            batch_size=256,
            num_passes=10,
            epochs_per_log_step=1,
            rollout_length_seconds=15.0,
            render_length_seconds=15.0,
            increase_threshold=5.0,
            decrease_threshold=3.0,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            min_action_latency=0.0,
            # PPO parameters.
            gamma=0.97,
            lam=0.95,
            entropy_coef=0.005,
            learning_rate=1e-3,
            clip_param=0.3,
            max_grad_norm=0.5,
            export_for_inference=True,
            only_save_most_recent=False,
            action_scale = 0.75,
            # visualize_reference_motion=True,
        ),
    )
