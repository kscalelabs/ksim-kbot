# mypy: disable-error-code="override"
"""Walking default humanoid task with reference gait tracking."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Optional, Self, TypeVar

import attrs
import bvhio
import distrax
import equinox as eqx
import glm
import jax
import jax.numpy as jnp
import ksim
import mujoco
import numpy as np
import xax
from bvhio.lib.hierarchy import Joint as BvhioJoint
from jaxtyping import Array, PRNGKeyArray
from kscale.web.gen.api import JointMetadataOutput
from ksim.types import PhysicsModel
from ksim.utils.priors import (
    MotionReferenceData,
    MotionReferenceMapping,
    generate_reference_motion,
    get_local_xpos,
    get_reference_joint_id,
    local_to_absolute,
    visualize_reference_motion,
)
from scipy.spatial.transform import Rotation as R

import ksim_kbot.rewards as kbot_rewards
from ksim_kbot import common
from ksim_kbot.walking.walking_joystick import (
    MAX_TORQUE,
    NUM_CRITIC_INPUTS,
    NUM_INPUTS,
    NUM_OUTPUTS,
    KbotWalkingTask,
    KbotWalkingTaskConfig,
)

JOINT_TARGETS = (
    # right arm
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    # left arm
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    # right leg
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    # left leg
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)


HUMANOID_REFERENCE_MAPPINGS = (
    MotionReferenceMapping("CC_Base_L_ThighTwist01", "RS03_5"),  # hip
    MotionReferenceMapping("CC_Base_R_ThighTwist01", "RS03_4"),  # hip
    MotionReferenceMapping("CC_Base_L_CalfTwist01", "KC_D_401L_L_Shin_Drive"),  # knee
    MotionReferenceMapping("CC_Base_R_CalfTwist01", "KC_D_401R_R_Shin_Drive"),  # knee
    MotionReferenceMapping("CC_Base_L_Foot", "KB_D_501L_L_LEG_FOOT"),  # foot
    MotionReferenceMapping("CC_Base_R_Foot", "KB_D_501R_R_LEG_FOOT"),  # foot
    MotionReferenceMapping("CC_Base_L_UpperarmTwist01", "RS03_6"),  # shoulder
    MotionReferenceMapping("CC_Base_R_UpperarmTwist01", "RS03_3"),  # shoulder
    MotionReferenceMapping("CC_Base_L_ForearmTwist01", "KC_C_401L_L_UpForearmDrive"),  # elbow
    MotionReferenceMapping("CC_Base_R_ForearmTwist01", "KC_C_401R_R_UpForearmDrive"),  # elbow
    MotionReferenceMapping("CC_Base_L_Hand", "KB_C_501X_Left_Bayonet_Adapter_Hard_Stop"),  # hand
    MotionReferenceMapping("CC_Base_R_Hand", "KB_C_501X_Right_Bayonet_Adapter_Hard_Stop"),  # hand
)

NUM_JOINTS = 20


# Actor inputs are the same as the base class for now
NUM_ACTOR_INPUTS_REF = (
    NUM_INPUTS + NUM_JOINTS + (len(HUMANOID_REFERENCE_MAPPINGS) * 3)  # reference_qpos  # reference_local_xpos
)

# Critic inputs are the base class inputs plus the new reference observations
NUM_CRITIC_INPUTS_REF = (
    NUM_CRITIC_INPUTS
    + NUM_JOINTS  # reference_qpos
    + (len(HUMANOID_REFERENCE_MAPPINGS) * 3)  # reference_local_xpos
    + (len(HUMANOID_REFERENCE_MAPPINGS) * 3)  # tracked_local_xpos
)


@dataclass
class WalkingRefMotionTaskConfig(KbotWalkingTaskConfig):
    bvh_path: str = xax.field(
        value=str(Path(__file__).parent.parent / "reference_motions" / "cheer_hands_waving_dance_kbot.bvh"),
        help="The path to the BVH file.",
    )
    rotate_bvh_euler: tuple[float, float, float] = xax.field(
        value=(0, np.pi / 2, 0),
        help="Optional rotation to ensure the BVH tree matches the Mujoco model.",
    )
    bvh_scaling_factor: float = xax.field(
        value=1 / 100.0,
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
    qpos_reference_speed: float = xax.field(
        value=1.8,
        help="The speed to apply to the reference motion.",
    )


Config = TypeVar("Config", bound=WalkingRefMotionTaskConfig)


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
    reference_motion_data: MotionReferenceData
    mj_base_id: int
    norm: xax.NormType = attrs.field(default="l1")
    sensitivity: float = attrs.field(default=5.0)

    def get_tracked_pos(self, trajectory: ksim.Trajectory) -> xax.FrozenDict[int, Array]:
        assert trajectory.aux_outputs is not None
        return trajectory.aux_outputs["tracked_pos"]

    def get_target_pos(self, trajectory: ksim.Trajectory) -> xax.FrozenDict[int, Array]:
        return self.reference_motion_data.get_cartesian_pose_at_time(trajectory.timestep)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        target_pos = self.get_target_pos(trajectory)
        tracked_pos = self.get_tracked_pos(trajectory)
        target_pos_filtered: xax.FrozenDict[int, Array] = xax.FrozenDict(
            {k: v for k, v in target_pos.items() if k in tracked_pos}
        )
        error = jax.tree.map(
            lambda target, tracked: xax.get_norm(target - tracked, self.norm), target_pos_filtered, tracked_pos
        )
        mean_error_over_bodies = jax.tree.reduce(jnp.add, error) / len(error)
        mean_error = mean_error_over_bodies.mean(axis=-1)
        reward = jnp.exp(-mean_error * self.sensitivity)
        return reward

    def get_markers(self) -> list[ksim.Marker]:
        markers = []

        # Add markers for reference positions (in blue)
        for body_id in self.reference_motion_data.cartesian_poses.keys():
            markers.append(
                ksim.Marker.sphere(
                    pos=(0.0, 0.0, 0.0),
                    radius=0.03,
                    rgba=(0.0, 0.0, 1.0, 0.5),  # blue = actual
                    update_fn=create_tracked_marker_update_fn(body_id, self.mj_base_id, self.get_tracked_pos),
                )
            )

            markers.append(
                ksim.Marker.sphere(
                    pos=(0.0, 0.0, 0.0),
                    radius=0.03,
                    rgba=(1.0, 0.0, 0.0, 0.5),  # red = target
                    update_fn=create_target_marker_update_fn(body_id, self.mj_base_id, self.get_target_pos),
                )
            )

        return markers


@attrs.define(frozen=True, kw_only=True)
class QposReferenceMotionReward(ksim.Reward):
    reference_motion_data: MotionReferenceData
    joint_weights: tuple[float, ...] = attrs.field(default=tuple([1.0] * NUM_JOINTS))
    joint_indices: tuple[int, ...] = attrs.field(default=None)
    speed: float = attrs.field(default=1.0)
    sensitivity: float = attrs.field(default=5.0)
    norm: xax.NormType = attrs.field(default="l1")

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        qpos = trajectory.qpos
        effective_time = trajectory.timestep * self.speed
        reference_qpos = self.reference_motion_data.get_qpos_at_time(effective_time)
        if self.joint_indices is not None:
            error = xax.get_norm(
                reference_qpos[..., jnp.array(self.joint_indices)] - qpos[..., jnp.array(self.joint_indices)],
                self.norm,
            )
        else:
            error = xax.get_norm(
                reference_qpos[..., 7:] - qpos[..., 7:],
                self.norm,
            )
        error = error * jnp.array(self.joint_weights)
        mean_error = error.mean(axis=-1)
        reward = jnp.exp(-mean_error * self.sensitivity)
        return reward

    @classmethod
    def create(
        cls,
        physics_model: ksim.PhysicsModel,
        reference_motion_data: MotionReferenceData,
        scale: float,
        joint_names: Optional[tuple[str, ...]] = None,
        joint_weights: Optional[tuple[float, ...]] = None,
        speed: float = 1.0,
    ) -> Self:
        """Create a sensor observation from a physics model.

        NOTE - fix this logic.
        """
        if joint_names is not None:
            joint_names = tuple(physics_model.joints.keys())
            mappings = ksim.get_qpos_data_idxs_by_name(physics_model)
            joint_indices = tuple([int(mappings[name][0]) for name in joint_names])
            actual_weights = joint_weights if joint_weights is not None else tuple([1.0] * len(joint_indices))
            return cls(
                reference_motion_data=reference_motion_data,
                joint_indices=joint_indices,
                joint_weights=actual_weights,
                speed=speed,
                scale=scale,
            )
        else:
            return cls(
                reference_motion_data=reference_motion_data,
                speed=speed,
                scale=scale,
            )


class KbotActor(eqx.Module):
    """Actor for the standing task."""

    mlp: eqx.nn.MLP
    min_std: float = eqx.static_field()
    max_std: float = eqx.static_field()
    var_scale: float = eqx.static_field()
    mean_scale: float = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        mean_scale: float,
    ) -> None:
        self.mlp = eqx.nn.MLP(
            in_size=num_inputs,
            out_size=NUM_OUTPUTS * 2,
            width_size=256,
            depth=5,
            key=key,
            activation=jax.nn.relu,
        )
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale
        self.mean_scale = mean_scale

    def forward(
        self,
        timestep_phase_4: Array,
        joint_pos_n: Array,
        joint_vel_n: Array,
        imu_acc_3: Array,
        imu_gyro_3: Array,
        lin_vel_cmd_2: Array,
        ang_vel_cmd: Array,
        gait_freq_cmd: Array,
        last_action_n: Array,
        ref_qpos_j: Array,
        ref_local_xpos_n: Array,
    ) -> distrax.Normal:
        x_n = jnp.concatenate(
            [
                timestep_phase_4,
                joint_pos_n,
                joint_vel_n,
                imu_acc_3,
                imu_gyro_3,
                lin_vel_cmd_2,
                ang_vel_cmd,
                gait_freq_cmd,
                last_action_n,
                ref_qpos_j,
                ref_local_xpos_n,
            ],
            axis=-1,
        )
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

    def __init__(self, key: PRNGKeyArray, *, num_inputs: int) -> None:
        self.mlp = eqx.nn.MLP(
            in_size=num_inputs,
            out_size=1,  # Always output a single critic value.
            width_size=256,
            depth=5,
            key=key,
            activation=jax.nn.relu,
        )

    def forward(
        self,
        timestep_phase_4: Array,
        joint_pos_n: Array,
        joint_vel_n: Array,
        imu_acc_3: Array,
        imu_gyro_3: Array,
        projected_gravity_3: Array,
        lin_vel_cmd_2: Array,
        ang_vel_cmd: Array,
        gait_freq_cmd: Array,
        last_action_n: Array,
        feet_contact_2: Array,
        feet_position_6: Array,
        base_position_3: Array,
        base_orientation_4: Array,
        base_linear_velocity_3: Array,
        base_angular_velocity_3: Array,
        actuator_force_n: Array,
        true_height_1: Array,
        # motion reference observations
        ref_qpos_j: Array,
        ref_local_xpos_n: Array,
        tracked_local_xpos_n: Array,
    ) -> Array:
        x_n = jnp.concatenate(
            [
                timestep_phase_4,
                joint_pos_n,
                joint_vel_n,
                imu_acc_3,
                imu_gyro_3,
                projected_gravity_3,
                lin_vel_cmd_2,
                ang_vel_cmd,
                gait_freq_cmd,
                last_action_n,
                feet_contact_2,
                feet_position_6,
                base_position_3,
                base_orientation_4,
                base_linear_velocity_3,
                base_angular_velocity_3,
                actuator_force_n,
                true_height_1,
                # motion reference observations
                ref_qpos_j,
                ref_local_xpos_n,
                tracked_local_xpos_n,
            ],
            axis=-1,
        )
        return self.mlp(x_n)


class KbotModel(eqx.Module):
    actor: KbotActor
    critic: KbotCritic
    num_inputs: int = eqx.static_field()
    num_critic_inputs: int = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        num_critic_inputs: int,
    ) -> None:
        self.num_inputs = num_inputs
        self.num_critic_inputs = num_critic_inputs
        self.actor = KbotActor(
            key,
            num_inputs=num_inputs,
            min_std=0.01,
            max_std=1.0,
            var_scale=1.0,
            mean_scale=1.0,
        )
        self.critic = KbotCritic(
            key,
            num_inputs=num_critic_inputs,
        )


class WalkingRefMotionTask(KbotWalkingTask[Config], Generic[Config]):
    config: Config
    reference_motion_data: MotionReferenceData
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

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        super_observations = super().get_observations(physics_model)
        observations: list[ksim.Observation] = [
            common.ReferenceQposObservation(
                reference_motion_data=self.reference_motion_data,
                speed=self.config.qpos_reference_speed,
            ),
            common.ReferenceLocalXposObservation(
                reference_motion_data=self.reference_motion_data,
                tracked_body_ids=self.tracked_body_ids,
            ),
            common.TrackedLocalXposObservation(
                tracked_body_ids=self.tracked_body_ids,
                mj_base_id=self.mj_base_id,
            ),
        ]

        return super_observations + observations

    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: dict[str, JointMetadataOutput] | None = None,
    ) -> ksim.Actuators:
        assert metadata is not None, "Metadata is required"
        return common.TargetPositionMITActuators(
            physics_model,
            metadata,
            default_targets=JOINT_TARGETS,
            pos_action_noise=0.05,
            vel_action_noise=0.05,
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
            action_scale=self.config.action_scale,
        )

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomJointPositionReset(),
            ksim.RandomJointVelocityReset(),
        ]
        # return [ksim.InitialMotionStateReset(
        #         reference_motion=self.reference_motion_data,
        #         freejoint=True,
        #     ),
        # ]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        rewards: list[ksim.Reward] = [
            ksim.StayAliveReward(
                success_reward=1.0,
                scale=8.0,
            ),
            # kbot_rewards.SensorOrientationPenalty(scale=self.config.orientation_penalty),
            CartesianReferenceMotionReward(
                reference_motion_data=self.reference_motion_data,
                mj_base_id=self.mj_base_id,
                scale=1.0,
            ),
            QposReferenceMotionReward.create(
                physics_model=physics_model,
                # joint_names=("right_arm", "left_arm", "right_leg", "left_leg"),
                reference_motion_data=self.reference_motion_data,
                scale=15.0,
                speed=self.config.qpos_reference_speed,
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
                    2.0,  # hip pitch
                    1.0,  # hip roll
                    1.0,  # hip yaw
                    2.0,  # knee
                    1.0,  # ankle
                    # left leg
                    2.0,  # hip pitch
                    1.0,  # hip roll
                    1.0,  # hip yaw
                    2.0,  # knee
                    1.0,  # ankle
                ),
            ),
            kbot_rewards.FeetSlipPenalty(scale=-2.0),
            kbot_rewards.TargetLinearVelocityReward(
                index="x",
                target_vel=0.5,
                scale=3.0,
            ),
            kbot_rewards.TargetLinearVelocityReward(
                index="y",
                target_vel=0.0,
                scale=1.5,
            ),
            # ksim.LinearVelocityPenalty(index="z", scale=-2.0),
            # kbot_rewards.TargetHeightReward(target_height=1.0, scale=1.0),
        ]

        return rewards

    def sample_action(
        self,
        model: KbotModel,
        model_carry: tuple[Array, Array],
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
        argmax: bool = False,
    ) -> ksim.Action:
        action_dist_j = self.run_actor(
            model.actor,
            observations,
            commands,
        )
        action_j = action_dist_j.mode() if argmax else action_dist_j.sample(seed=rng)
        # Getting the local cartesian positions for all tracked bodies.
        tracked_positions: dict[int, Array] = {}
        for body_id in self.tracked_body_ids:
            body_pos = get_local_xpos(physics_state.data.xpos, body_id, self.mj_base_id)
            tracked_positions[body_id] = jnp.array(body_pos)

        return ksim.Action(
            action=action_j,
            carry=None,
            aux_outputs=xax.FrozenDict({"tracked_pos": xax.FrozenDict(tracked_positions)}),
        )

    def run(self) -> None:
        mj_model: PhysicsModel = self.get_mujoco_model()
        root: BvhioJoint = bvhio.readAsHierarchy(self.config.bvh_path)
        reference_base_id = get_reference_joint_id(root, self.config.reference_base_name)
        self.mj_base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, self.config.mj_base_name)
        self.qpos_reference_speed = self.config.qpos_reference_speed

        def rotation_callback(root: BvhioJoint) -> None:
            euler_rotation = np.array(self.config.rotate_bvh_euler)
            quat = R.from_euler("xyz", euler_rotation).as_quat(scalar_first=True)
            root.applyRotation(glm.quat(*quat), bake=True)

        self.reference_motion_data = generate_reference_motion(
            model=mj_model,
            mj_base_id=self.mj_base_id,
            bvh_root=root,
            bvh_to_mujoco_names=HUMANOID_REFERENCE_MAPPINGS,
            bvh_base_id=reference_base_id,
            ctrl_dt=self.config.ctrl_dt,
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
        self.tracked_body_ids = tuple(self.reference_motion_data.cartesian_poses.keys())

        if self.config.visualize_reference_motion:
            np_reference_qpos = np.asarray(self.reference_motion_data.qpos.array)
            np_cartesian_motion = jax.tree.map(
                lambda x: np.asarray(x.array),
                self.reference_motion_data.cartesian_poses,
            )
            visualize_reference_motion(
                mj_model,
                reference_qpos=np_reference_qpos,
                cartesian_motion=np_cartesian_motion,
                mj_base_id=self.mj_base_id,
            )
        else:
            super().run()

    def get_model(self, key: PRNGKeyArray) -> KbotModel:
        return KbotModel(
            key,
            num_inputs=NUM_ACTOR_INPUTS_REF,
            num_critic_inputs=NUM_CRITIC_INPUTS_REF,
        )

    def run_actor(
        self, model: KbotActor, observations: xax.FrozenDict[str, Array], commands: xax.FrozenDict[str, Array]
    ) -> distrax.Normal:
        timestep_phase_4 = observations["timestep_phase_observation"]
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        lin_vel_cmd_2 = commands["linear_velocity_command"]
        ang_vel_cmd = commands["angular_velocity_command"]
        gait_freq_cmd = commands["gait_frequency_command"]
        last_action_n = observations["last_action_observation"]

        # motion reference observations
        ref_qpos_j = observations["reference_qpos_observation"]
        ref_local_xpos_n = observations["reference_local_xpos_observation"]

        return model.forward(
            timestep_phase_4=timestep_phase_4,
            joint_pos_n=joint_pos_n,
            joint_vel_n=joint_vel_n,
            imu_acc_3=imu_acc_3,
            imu_gyro_3=imu_gyro_3,
            lin_vel_cmd_2=lin_vel_cmd_2,
            ang_vel_cmd=ang_vel_cmd,
            gait_freq_cmd=gait_freq_cmd,
            last_action_n=last_action_n,
            # motion reference observations
            ref_qpos_j=ref_qpos_j,
            ref_local_xpos_n=ref_local_xpos_n,
        )

    def run_critic(
        self, model: KbotCritic, observations: xax.FrozenDict[str, Array], commands: xax.FrozenDict[str, Array]
    ) -> Array:
        timestep_phase_4 = observations["timestep_phase_observation"]
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        projected_gravity_3 = observations["projected_gravity_observation"]
        lin_vel_cmd_2 = commands["linear_velocity_command"]
        ang_vel_cmd = commands["angular_velocity_command"]
        gait_freq_cmd = commands["gait_frequency_command"]
        last_action_n = observations["last_action_observation"]
        # critic observations
        feet_contact_2 = observations["feet_contact_observation"]
        feet_position_6 = observations["feet_position_observation"]
        base_position_3 = observations["base_position_observation"]
        base_orientation_4 = observations["base_orientation_observation"]
        base_linear_velocity_3 = observations["base_linear_velocity_observation"]
        base_angular_velocity_3 = observations["base_angular_velocity_observation"]
        actuator_force_n = observations["actuator_force_observation"]
        true_height_1 = observations["true_height_observation"]

        # motion reference observations
        ref_qpos_j = observations["reference_qpos_observation"]
        ref_local_xpos_n = observations["reference_local_xpos_observation"]
        tracked_local_xpos_n = observations["tracked_local_xpos_observation"]

        return model.forward(
            timestep_phase_4=timestep_phase_4,
            joint_pos_n=joint_pos_n,
            joint_vel_n=joint_vel_n,
            imu_acc_3=imu_acc_3,
            imu_gyro_3=imu_gyro_3,
            lin_vel_cmd_2=lin_vel_cmd_2,
            ang_vel_cmd=ang_vel_cmd,
            gait_freq_cmd=gait_freq_cmd,
            last_action_n=last_action_n,
            # critic observations
            feet_contact_2=feet_contact_2,
            feet_position_6=feet_position_6,
            projected_gravity_3=projected_gravity_3,
            base_position_3=base_position_3,
            base_orientation_4=base_orientation_4,
            base_linear_velocity_3=base_linear_velocity_3,
            base_angular_velocity_3=base_angular_velocity_3,
            actuator_force_n=actuator_force_n,
            true_height_1=true_height_1,
            # motion reference observations
            ref_qpos_j=ref_qpos_j,
            ref_local_xpos_n=ref_local_xpos_n,
            tracked_local_xpos_n=tracked_local_xpos_n,
        )


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m ksim_kbot.walking.walking_reference_motion disable_multiprocessing=True
    # To visualize the environment, use the following command:
    #   python -m ksim_kbot.walking.walking_reference_motion run_model_viewer=True
    # To visualize the reference gait, use the following command:
    #   python -m ksim_kbot.walking.walking_reference_motion visualize_reference_motion=True
    # On MacOS or other devices with less memory, you can change the number
    # of environments and batch size to reduce memory usage. Here's an example
    # from the command line:
    #   python -m ksim_kbot.walking.walking_reference_motion num_envs=1 batch_size=1
    WalkingRefMotionTask.launch(
        WalkingRefMotionTaskConfig(
            # Training parameters.
            num_envs=8192,
            batch_size=256,
            num_passes=10,
            epochs_per_log_step=1,
            rollout_length_seconds=1.25,
            render_length_seconds=5.0,
            # Simulation parameters.
            valid_every_n_seconds=700,
            iterations=8,
            ls_iterations=8,
            dt=0.002,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            # PPO parameters.
            gamma=0.97,
            lam=0.95,
            entropy_coef=0.005,
            learning_rate=1e-3,
            clip_param=0.3,
            max_grad_norm=0.5,
            export_for_inference=True,
            only_save_most_recent=False,
            action_scale=1.0,
            # Gait matching parameters.
            bvh_path=str(Path(__file__).parent.parent / "reference_motions" / "walk_normal_kbot.bvh"),
            rotate_bvh_euler=(0, np.pi / 2, 0),
            bvh_offset=(0.02, 0.09, -0.29),
            bvh_scaling_factor=1 / 100.0,
            mj_base_name="Torso_Side_Right",
            reference_base_name="CC_Base_Pelvis",
            visualize_reference_motion=False,
        ),
    )
