# mypy: disable-error-code="override"
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
    get_reference_qpos,
    get_reference_joint_id,
    visualize_reference_motion,
    get_reference_cartesian_poses,
)
from scipy.spatial.transform import Rotation as R

import ksim_kbot.rewards as kbot_rewards
from ksim_kbot.walking.walking_rnn import RnnModel, WalkingRnnTask, WalkingRnnTaskConfig


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



@attrs.define(frozen=True, kw_only=True)
class QposReferenceMotionReward(ksim.Reward):
    reference_qpos: xax.HashableArray
    ctrl_dt: float
    norm: xax.NormType = attrs.field(default="l1")
    sensitivity: float = attrs.field(default=5.0)

    @property
    def num_frames(self) -> int:
        return self.reference_qpos.array.shape[0]

    def __call__(self, trajectory: ksim.Trajectory, _: None) -> tuple[Array, None]:
        qpos = trajectory.qpos
        step_number = jnp.int32(jnp.round(trajectory.timestep / self.ctrl_dt)) % self.num_frames
        reference_qpos = jnp.take(self.reference_qpos.array, step_number, axis=0)
        error = xax.get_norm(reference_qpos[..., 7:] - qpos[..., 7:], self.norm)
        mean_error = error.mean(axis=-1)
        reward = jnp.exp(-mean_error * self.sensitivity)
        return reward, None


class WalkingRnnRefMotionTask(WalkingRnnTask[Config], Generic[Config]):
    config: Config
    reference_qpos: xax.HashableArray
    mj_base_id: int

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        rewards: list[ksim.Reward] = [
            ksim.StayAliveReward(
                success_reward=1.0,
                scale=3.0,
            ),
            kbot_rewards.OrientationPenalty(scale=self.config.orientation_penalty),
            QposReferenceMotionReward(
                reference_qpos=self.reference_qpos,
                ctrl_dt=self.config.ctrl_dt,
                scale=10.0,
            ),
            NaiveForwardReward(scale=0.5),
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
            ksim.FeetContactObservation.create(
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
    WalkingRnnRefMotionTask.launch(
        WalkingRnnRefMotionTaskConfig(
            # Training parameters.
            num_envs=2048,
            batch_size=256,
            num_passes=10,
            epochs_per_log_step=1,
            rollout_length_seconds=15.0,
            render_length_seconds=15.0,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            min_action_latency=0.0,
            # PPO parameters.
            gamma=0.97,
            lam=0.95,
            entropy_coef=0.005,
            learning_rate=1e-4,
            clip_param=0.3,
            max_grad_norm=0.5,
            export_for_inference=True,
            only_save_most_recent=False,
            # visualize_reference_motion=True,
        ),
    )
