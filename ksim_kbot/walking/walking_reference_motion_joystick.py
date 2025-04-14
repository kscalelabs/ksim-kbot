# mypy: disable-error-code="override"
"""Walking default humanoid task with reference gait tracking."""

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar, Callable

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
from ksim.types import PhysicsModel
import ksim_kbot.common as common
from ksim.utils.reference_motion import (
    ReferenceMapping,
    get_reference_qpos,
    get_reference_joint_id,
    local_to_absolute,
    visualize_reference_motion,
    get_reference_cartesian_poses,
)
from scipy.spatial.transform import Rotation as R

import ksim_kbot.rewards as kbot_rewards
from ksim_kbot.walking.walking_rnn import RnnModel, WalkingRnnTask, WalkingRnnTaskConfig, NUM_JOINTS, RnnActor, RnnCritic, NUM_INPUTS, NUM_CRITIC_INPUTS
import distrax

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

    def __call__(self, trajectory: ksim.Trajectory, _: None) -> tuple[Array, None]:
        return trajectory.qvel[..., 0], None


# approx 72 frames for 1 cycle
# with ctrl_dt = 0.02, have ~ 1.44 seconds per cycle
# isaacgym setup had 0.4 cycle time
@attrs.define(frozen=True, kw_only=True)
class QposReferenceMotionReward(ksim.Reward):
    forward_reference_qpos: xax.HashableArray
    backward_reference_qpos: xax.HashableArray
    ctrl_dt: float
    norm: xax.NormType = attrs.field(default="l1")
    sensitivity: float = attrs.field(default=5.0)
    joint_weights: tuple[float, ...] = attrs.field(default=tuple([1.0] * NUM_JOINTS))
    command_name: str = attrs.field(default="joystick_command")
    speed: float = attrs.field(default=1.0)

    @property
    def num_frames(self) -> int:
        return self.forward_reference_qpos.array.shape[0]

    def __call__(self, trajectory: ksim.Trajectory, _: None) -> tuple[Array, None]:
        qpos = trajectory.qpos
        command = trajectory.command[self.command_name]
        
        # Calculate the step number within the reference motion cycle
        step_number = jnp.int32(jnp.round(self.speed * trajectory.timestep / self.ctrl_dt)) % self.num_frames
        
        # Determine the target reference qpos based on the command
        # Command: 0=stand, 1=forward, 2=backward, 3=turn left, 4=turn right
        # is_standing = command[..., 0] == 0
        # is_walking_backward = command[..., 0] == 2

        is_standing = jnp.linalg.norm(command, axis=-1) == 0
        is_walking_backward = command[..., 0] < 0

        target_qpos_fwd = jnp.take(self.forward_reference_qpos.array, step_number, axis=0)
        target_qpos_bwd = jnp.take(self.backward_reference_qpos.array, step_number, axis=0)

        # Select the appropriate target qpos frame based on the command
        is_walking_backward_b = jnp.expand_dims(is_walking_backward, axis=-1)
        target_reference_qpos = jnp.where(
            is_walking_backward_b,
            target_qpos_bwd,
            target_qpos_fwd
        )

        # Compute the reference motion reward error using the selected target
        error = xax.get_norm(target_reference_qpos[..., 7:] - qpos[..., 7:], self.norm)
        error = error * jnp.array(self.joint_weights)
        mean_error = error.mean(axis=-1)
        motion_reward = jnp.exp(-mean_error * self.sensitivity)

        # Apply reward: 1.0 for standing, motion_reward otherwise
        reward = jnp.where(is_standing, 1.0, motion_reward)

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

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        target_pos = self.get_target_pos(trajectory)
        tracked_pos = self.get_tracked_pos(trajectory)
        error = jax.tree.map(lambda target, tracked: xax.get_norm(target - tracked, self.norm), target_pos, tracked_pos)
        mean_error_over_bodies = jax.tree.reduce(jnp.add, error) / len(error)
        mean_error = mean_error_over_bodies.mean(axis=-1)
        reward = jnp.exp(-mean_error * self.sensitivity)
        return reward

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

class WalkingRnnRefMotionJoystickTask(WalkingRnnTask[Config], Generic[Config]):
    config: Config
    reference_qpos: xax.HashableArray
    mj_base_id: int

    def get_model(self, key: PRNGKeyArray) -> RnnModel:
        return RnnModel(
            key,
            num_inputs=NUM_INPUTS - 4, # lin vel instead of joystick
            num_critic_inputs=NUM_CRITIC_INPUTS - 4, # lin vel instead of joystick
            min_std=0.01,
            max_std=1.0,
            mean_scale=self.config.action_scale,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
        )

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        backward_ref_qpos_array = self.reference_qpos.array[::-1]
        backward_reference_qpos_hashable = xax.hashable_array(backward_ref_qpos_array)

        rewards: list[ksim.Reward] = [
            ksim.StayAliveReward(
                success_reward=1.0,
                scale=2.0,
            ),
            kbot_rewards.OrientationPenalty(scale=self.config.orientation_penalty),
            QposReferenceMotionReward(
                forward_reference_qpos=self.reference_qpos,
                backward_reference_qpos=backward_reference_qpos_hashable,
                ctrl_dt=self.config.ctrl_dt,
                scale=6.0,
                speed=3.6,
                command_name="linear_velocity_command",
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
            # kbot_rewards.FeetSlipPenalty(scale=-0.25),
            # kbot_rewards.FeetAirTimeReward(
            #     scale=5.0,
            #     # threshold_min=0.0,
            #     # threshold_max=0.4,
            # ),
            # ksim.JoystickReward(scale=2.0, linear_velocity_clip_max=1.0, angular_velocity_clip_max=1.0),
            kbot_rewards.LinearVelocityTrackingReward(
                scale=2.0,
            ),
            ksim.LinearVelocityPenalty(index="z", scale=-1.0),
            kbot_rewards.TargetHeightReward(target_height=1.0, scale=1.0),
        ]

        return rewards

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomJointPositionReset(),
            ksim.RandomJointVelocityReset(),
        ]

    def run_actor(
        self,
        model: RnnActor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[distrax.Distribution, Array]:
        timestep_1 = observations["timestep_observation"]
        joint_pos_j = observations["joint_position_observation"]
        joint_vel_j = observations["joint_velocity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        # joystick_cmd_1 = commands["joystick_command"]
        # joystick_cmd_ohe_6 = jax.nn.one_hot(joystick_cmd_1, num_classes=6).squeeze(-2)
        linear_vel_cmd_2 = commands["linear_velocity_command"]

        obs_n = jnp.concatenate(
            [
                jnp.cos(timestep_1),  # 1
                jnp.sin(timestep_1),  # 1
                joint_pos_j,  # NUM_JOINTS
                joint_vel_j / 10.0,  # NUM_JOINTS
                imu_acc_3 / 50.0,  # 3
                imu_gyro_3 / 3.0,  # 3
                linear_vel_cmd_2,  # 2
            ],
            axis=-1,
        )

        return model.forward(obs_n, carry)

    def run_critic(
        self,
        model: RnnCritic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[Array, Array]:
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
        # joystick_cmd_1 = commands["joystick_command"]
        # joystick_cmd_ohe_6 = jax.nn.one_hot(joystick_cmd_1, num_classes=6).squeeze(-2)
        linear_vel_cmd_2 = commands["linear_velocity_command"]

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
                linear_vel_cmd_2,  # 2
            ],
            axis=-1,
        )

        return model.forward(obs_n, carry)

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return [
            # ksim.JoystickCommand(
            #     switch_prob=self.config.ctrl_dt / 15,
            #     # ranges=((0, 1),) if self.config.joystick_only_forward else ((0, 4),),
            #     ranges=((0,2),)
            # ),
            common.LinearVelocityCommand(
                x_range=(-1, 1),
                y_range=(-1, 1),
                x_zero_prob=0.3,
                y_zero_prob=0.3,
                switch_prob=self.config.ctrl_dt / 15,
            )
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        return [
            ksim.JointPositionObservation(noise=0.01),
            ksim.JointVelocityObservation(noise=0.1),
            ksim.ActuatorForceObservation(noise=0.0),
            ksim.CenterOfMassInertiaObservation(noise=0.0),
            ksim.CenterOfMassVelocityObservation(noise=0.0),
            ksim.BasePositionObservation(noise=0.0),
            ksim.BaseOrientationObservation(noise=0.0),
            ksim.BaseLinearVelocityObservation(noise=0.0),
            ksim.BaseAngularVelocityObservation(noise=0.0),
            ksim.BaseLinearAccelerationObservation(),
            ksim.BaseAngularAccelerationObservation(),
            ksim.ActuatorAccelerationObservation(),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_acc", noise=0.4),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_gyro", noise=0.5),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="local_linvel_origin", noise=0.0),
            common.FeetContactObservation.create(
                physics_model=physics_model,
                foot_left_geom_names="KB_D_501L_L_LEG_FOOT_collision_box",
                foot_right_geom_names="KB_D_501R_R_LEG_FOOT_collision_box",
                floor_geom_names="floor",
            ),
            ksim.TimestepObservation(),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="upvector_origin", noise=0.0),
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
    ) -> ksim.Action:
        actor_carry_in, critic_carry_in = model_carry

        # Runs the actor model to get the action distribution.
        action_dist_j, actor_carry = self.run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
            carry=actor_carry_in,
        )

        action_j = action_dist_j.sample(seed=rng)

        return ksim.Action(
            action=action_j,
            carry=(actor_carry, critic_carry_in),
            aux_outputs=None,  # No auxiliary outputs needed for qpos matching
        )

    def run(self) -> None:
        mj_model: PhysicsModel = self.get_mujoco_model()
        root: BvhioJoint = bvhio.readAsHierarchy(self.config.bvh_path)
        reference_base_id = get_reference_joint_id(root, self.config.reference_base_name)
        self.mj_base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, self.config.mj_base_name)

        def rotation_callback(root: BvhioJoint) -> None:
            euler_rotation = np.array(self.config.rotate_bvh_euler)
            quat = R.from_euler("xyz", euler_rotation).as_quat(scalar_first=True)
            root.applyRotation(glm.quat(*quat), bake=True)

        cartesian_motion = get_reference_cartesian_poses(
            mappings=HUMANOID_REFERENCE_MAPPINGS,
            model=mj_model,
            root=root,
            reference_base_id=reference_base_id,
            root_callback=rotation_callback,
            scaling_factor=self.config.bvh_scaling_factor,
            offset=np.array(self.config.bvh_offset),
        )

        np_reference_qpos = get_reference_qpos(
            model=mj_model,
            mj_base_id=self.mj_base_id,
            bvh_root=root,
            bvh_to_mujoco_names=HUMANOID_REFERENCE_MAPPINGS,
            bvh_base_id=reference_base_id,
            bvh_offset=np.array(self.config.bvh_offset),
            bvh_root_callback=rotation_callback,
            bvh_scaling_factor=self.config.bvh_scaling_factor,
            neutral_qpos=None,  # Or provide a neutral pose if desired
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

        # Visualize reference motion (optional)
        if self.config.visualize_reference_motion:
            visualize_reference_motion(
                mj_model,
                reference_qpos=np_reference_qpos,
                cartesian_motion=cartesian_motion,
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
    WalkingRnnRefMotionJoystickTask.launch(
        WalkingRnnRefMotionTaskConfig(
            # Training parameters.
            num_envs=2048,
            batch_size=256,
            num_passes=10,
            epochs_per_log_step=1,
            rollout_length_seconds=15.0,
            render_length_seconds=15.0,
            increase_threshold=10.0,
            decrease_threshold=5.0,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            min_action_latency=0.0,
            # PPO parameters.
            gamma=0.97,
            lam=0.95,
            entropy_coef=0.005,
            learning_rate=4e-3,
            clip_param=0.3,
            max_grad_norm=0.5,
            export_for_inference=True,
            only_save_most_recent=False,
            action_scale = 0.5,
            # visualize_reference_motion=True,
        ),
    )
