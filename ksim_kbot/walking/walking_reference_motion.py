"""Walking default humanoid task with reference gait tracking."""

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

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
from ksim.utils.reference_motion import (
    ReferenceMapping,
    generate_reference_gait,
    get_local_xpos,
    get_reference_joint_id,
    visualize_reference_gait,
)
from scipy.spatial.transform import Rotation as R

from ksim_kbot.walking.walking import NaiveForwardReward
from ksim_kbot.walking.walking_rnn import RnnModel, WalkingRnnTask, WalkingRnnTaskConfig

HISTORY_LENGTH = 0
SINGLE_STEP_HISTORY_SIZE = 0


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GaitMatchingAuxOutputs:
    tracked_pos: xax.FrozenDict[int, Array]


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
        value=(0, 0, 0),
        help="Optional rotation to ensure the BVH tree matches the Mujoco model.",
    )
    bvh_scaling_factor: float = xax.field(
        value=1.0,
        help="Scaling factor to ensure the BVH tree matches the Mujoco model.",
    )
    offset_reference_gait: tuple[float, float, float] = xax.field(
        value=(0.1, 0.09, -0.29),
        help="Offset to apply to the reference gait.",
    )
    mj_base_name: str = xax.field(
        value="pelvis",
        help="The Mujoco body name of the base of the humanoid",
    )
    reference_base_name: str = xax.field(
        value="CC_Base_Pelvis",
        help="The BVH joint name of the base of the humanoid",
    )
    visualize_reference_gait: bool = xax.field(
        value=False,
        help="Whether to visualize the reference gait.",
    )


Config = TypeVar("Config", bound=WalkingRnnRefMotionTaskConfig)


@attrs.define(frozen=True, kw_only=True)
class MatchReferenceMotionReward(ksim.Reward):
    reference_gait: xax.FrozenDict[int, xax.HashableArray]
    ctrl_dt: float
    norm: xax.NormType = attrs.field(default="l1")
    sensitivity: float = attrs.field(default=5.0)

    @property
    def num_frames(self) -> int:
        return list(self.reference_gait.values())[0].array.shape[0]

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        assert isinstance(trajectory.aux_outputs, GaitMatchingAuxOutputs)
        reference_gait: xax.FrozenDict[int, Array] = jax.tree.map(lambda x: x.array, self.reference_gait)
        step_number = jnp.int32(jnp.round(trajectory.timestep / self.ctrl_dt)) % self.num_frames
        target_pos = jax.tree.map(lambda x: jnp.take(x, step_number, axis=0), reference_gait)
        tracked_pos = trajectory.aux_outputs.tracked_pos
        error = jax.tree.map(lambda target, tracked: xax.get_norm(target - tracked, self.norm), target_pos, tracked_pos)
        mean_error_over_bodies = jax.tree.reduce(jnp.add, error) / len(error)
        mean_error = mean_error_over_bodies.mean(axis=-1)
        reward = jnp.exp(-mean_error * self.sensitivity)
        return reward


class WalkingRnnRefMotionTask(WalkingRnnTask[Config], Generic[Config]):
    config: Config

    def get_initial_model_carry(self, rng: PRNGKeyArray) -> tuple[Array, Array]:
        return super().get_initial_model_carry(rng)

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        rewards: list[ksim.Reward] = [
            ksim.StayAliveReward(
                success_reward=1.0,
                scale=1.0,
            ),
            MatchReferenceMotionReward(
                reference_gait=self.reference_gait,
                ctrl_dt=self.config.ctrl_dt,
                scale=0.1,
            ),
        ]
        if self.config.use_naive_reward:
            rewards += [
                NaiveForwardReward(clip_max=self.config.naive_clip_max, scale=1.0),
            ]
        else:
            rewards += [
                ksim.LinearVelocityTrackingReward(index="x", command_name="linear_velocity_command_x", scale=1.0),
                ksim.LinearVelocityTrackingReward(index="y", command_name="linear_velocity_command_y", scale=0.1),
                ksim.AngularVelocityTrackingReward(index="z", command_name="angular_velocity_command_z", scale=0.01),
            ]
        return rewards

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
        action_n = super().sample_action(model, model_carry, physics_model, physics_state, observations, commands, rng)

        # Getting the local cartesian positions for all tracked bodies.
        tracked_positions: dict[int, Array] = {}
        for body_id in self.tracked_body_ids:
            body_pos = get_local_xpos(physics_state.data.xpos, body_id, self.mj_base_id)
            tracked_positions[body_id] = jnp.array(body_pos)

        return ksim.Action(
            action=action_n.action,
            carry=model_carry,
            aux_outputs=GaitMatchingAuxOutputs(
                tracked_pos=xax.FrozenDict(tracked_positions),
            ),
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

        np_reference_gait = generate_reference_gait(
            mappings=HUMANOID_REFERENCE_MAPPINGS,
            model=mj_model,
            root=root,
            reference_base_id=reference_base_id,
            root_callback=rotation_callback,
            scaling_factor=self.config.bvh_scaling_factor,
            offset=np.array(self.config.offset_reference_gait),
        )
        self.reference_gait: xax.FrozenDict[int, xax.HashableArray] = jax.tree.map(
            lambda x: xax.hashable_array(jnp.array(x)), np_reference_gait
        )
        self.tracked_body_ids = tuple(self.reference_gait.keys())

        if self.config.visualize_reference_gait:
            visualize_reference_gait(
                mj_model,
                base_id=self.mj_base_id,
                reference_gait=np_reference_gait,
            )
        else:
            super().run()


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m ksim_kbot.walking.walking_reference_motion
    # To visualize the environment, use the following command:
    #   python -m ksim_kbot.walking.walking_reference_motion run_environment=True
    # To visualize the reference gait, use the following command:
    #   mujoco python -m ksim_kbot.walking.walking_reference_motion num_envs=1 batch_size=1
    # On MacOS or other devices with less memory, you can change the number
    # of environments and batch size to reduce memory usage. Here's an example
    # from the command line:
    #   python -m ksim_kbot.walking.walking_reference_motion num_envs=1 batch_size=1
    WalkingRnnRefMotionTask.launch(
        WalkingRnnRefMotionTaskConfig(
            # Training parameters.
            num_envs=2048,
            batch_size=256,
            num_passes=4,
            epochs_per_log_step=1,
            rollout_length_seconds=10.0,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            min_action_latency=0.0,
            rotate_bvh_euler=(0, np.pi / 2, 0),
            bvh_scaling_factor=1 / 100,
            offset_reference_gait=(0.02, 0.09, -0.29),
            mj_base_name="floating_base_link",
            reference_base_name="CC_Base_Pelvis",
        ),
    )
