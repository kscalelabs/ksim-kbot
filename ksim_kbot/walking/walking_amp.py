# mypy: disable-error-code="override"
"""Walking default humanoid task with reference motion tracking."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, TypeVar

import bvhio
import distrax
import equinox as eqx
import glm
import jax
import jax.numpy as jnp
import ksim
import mujoco
import numpy as np
import optax
import xax
from bvhio.lib.hierarchy import Joint as BvhioJoint
from jaxtyping import Array, PRNGKeyArray
from kscale.web.gen.api import JointMetadataOutput
from ksim.utils.priors import (
    MotionReferenceData,
    generate_reference_motion,
    get_local_xpos,
    get_reference_joint_id,
    visualize_reference_motion,
)
from mujoco_scenes.mjcf import load_mjmodel
from scipy.spatial.transform import Rotation as R
from xax.xax.nn import export

import ksim_kbot.rewards as kbot_rewards
from ksim_kbot import common
from ksim_kbot.standing.standing import MAX_TORQUE
from ksim_kbot.walking.walking_reference_motion_rnn import (
    HUMANOID_REFERENCE_MAPPINGS,
    NUM_ACTOR_INPUTS_REF,
    NUM_CRITIC_INPUTS,
    WalkingRnnRefMotionTaskConfig,
)
from ksim_kbot.walking.walking_rnn import RnnActor, RnnCritic, RnnModel

NUM_JOINTS = 20
NUM_ACTOR_INPUTS_REF -= 6
NUM_CRITIC_INPUTS += (
    -6
    + NUM_JOINTS  # reference_qpos
    + (len(HUMANOID_REFERENCE_MAPPINGS) * 3)  # reference_local_xpos
    + (len(HUMANOID_REFERENCE_MAPPINGS) * 3)  # tracked_local_xpos
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

logger = logging.getLogger(__name__)


class Discriminator(eqx.Module):
    """AMP discriminator."""

    layers: list[Callable[[Array], Array]]

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        hidden_size: int,
        depth: int,
        num_frames: int,
    ) -> None:
        num_inputs = NUM_JOINTS
        num_outputs = 1

        layers: list[Callable[[Array], Array]] = []
        for _ in range(depth):
            key, subkey = jax.random.split(key)
            layers += [
                eqx.nn.Conv1d(
                    in_channels=num_inputs,
                    out_channels=hidden_size,
                    kernel_size=num_frames,
                    # We left-pad the input so that the discriminator output
                    # at time T can be reasonably interpretted as the logit
                    # for the motion from time T - N to T. In other words, we
                    # this means that all the reward for time T will be based
                    # on the motion from the previous N frames.
                    padding=[(num_frames - 1, 0)],
                    key=subkey,
                ),
                jax.nn.relu,
            ]
            num_inputs = hidden_size

        layers += [
            eqx.nn.Conv1d(
                in_channels=num_inputs,
                out_channels=num_outputs,
                kernel_size=1,
                key=key,
            )
        ]

        self.layers = layers

    def forward(self, x: Array) -> Array:
        x_nt = x.transpose(1, 0)
        for layer in self.layers:
            x_nt = layer(x_nt)
        return x_nt.squeeze(0)


@dataclass
class WalkingAmpTaskConfig(WalkingRnnRefMotionTaskConfig, ksim.AMPConfig):
    # Disciminator parameters.
    discriminator_hidden_size: int = xax.field(
        value=512,
        help="The hidden size for the discriminator.",
    )
    discriminator_depth: int = xax.field(
        value=3,
        help="The depth for the discriminator.",
    )

    num_frames: int = xax.field(
        value=10,
        help="The number of frames to use for the discriminator.",
    )

    max_discriminator_grad_norm: float = xax.field(
        value=2.0,
        help="Maximum gradient norm for clipping.",
    )
    discriminator_learning_rate: float = xax.field(
        value=1e-3,
        help="Learning rate for the discriminator.",
    )

    amp_scale: float = xax.field(value=1.0)


Config = TypeVar("Config", bound=WalkingAmpTaskConfig)


class WalkingAmpTask(ksim.AMPTask[Config], Generic[Config]):
    config: Config
    reference_motion_data: MotionReferenceData
    tracked_body_ids: tuple[int, ...]
    mj_base_id: int
    qpos_reference_speed: float

    def get_policy_model(self, key: PRNGKeyArray) -> RnnModel:
        return RnnModel(
            key,
            num_inputs=NUM_ACTOR_INPUTS_REF,
            num_joints=NUM_JOINTS * 2,  # (pos + vel and std + mean)
            num_critic_inputs=NUM_CRITIC_INPUTS,
            num_mixtures=self.config.num_mixtures,
            min_std=0.01,
            max_std=1.0,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
        )

    def get_policy_optimizer(self) -> optax.GradientTransformation:
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            (
                optax.adam(self.config.learning_rate)
                if self.config.adam_weight_decay == 0.0
                else optax.adamw(self.config.learning_rate, weight_decay=self.config.adam_weight_decay)
            ),
        )

        return optimizer

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

        obs_n = jnp.concatenate(
            [
                jnp.cos(timestep_1),  # 1
                jnp.sin(timestep_1),  # 1
                joint_pos_j,  # NUM_JOINTS
                joint_vel_j / 10.0,  # NUM_JOINTS
                imu_acc_3 / 50.0,  # 3
                imu_gyro_3 / 3.0,  # 3
            ],
            axis=-1,
        )

        msg = f"Actor input shape ({obs_n.shape[-1]}) != constant ({NUM_ACTOR_INPUTS_REF})"
        assert obs_n.shape[-1] == NUM_ACTOR_INPUTS_REF, msg

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
        # reference observations
        ref_qpos_j = observations["reference_qpos_observation"]
        ref_local_xpos_n = observations["reference_local_xpos_observation"]
        tracked_local_xpos_n = observations["tracked_local_xpos_observation"]

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
                # reference observations
                ref_qpos_j,  # NUM_JOINTS
                ref_local_xpos_n,  # num_tracked_bodies * 3
                tracked_local_xpos_n,  # num_tracked_bodies * 3
            ],
            axis=-1,
        )

        return model.forward(obs_n, carry)

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        # rewards: list[ksim.Reward] = super().get_rewards(physics_model)
        rewards = [
            ksim.StayAliveReward(scale=5.0),
            kbot_rewards.LinearVelocityTrackingReward(
                scale=1.0,
                stand_still_threshold=0.0,
            ),
            ksim.AngularVelocityPenalty(index="x", scale=-0.01),
            ksim.AngularVelocityPenalty(index="y", scale=-0.01),
        ]
        rewards.append(ksim.AMPReward(scale=self.config.amp_scale))

        return rewards

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomJointPositionReset(),
            ksim.RandomJointVelocityReset(),
            # ksim.InitialMotionStateReset(),
        ]

    def get_discriminator_model(self, key: PRNGKeyArray) -> Discriminator:
        return Discriminator(
            key,
            hidden_size=self.config.discriminator_hidden_size,
            depth=self.config.discriminator_depth,
            num_frames=self.config.num_frames,
        )

    def get_discriminator_optimizer(self) -> optax.GradientTransformation:
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.max_discriminator_grad_norm),
            optax.adam(self.config.discriminator_learning_rate),
        )
        return optimizer

    def call_discriminator(self, model: Discriminator, motion: Array) -> Array:
        return model.forward(motion).squeeze()

    def create_reference_motion(self, mj_model: mujoco.MjModel) -> MotionReferenceData:
        root: BvhioJoint = bvhio.readAsHierarchy(self.config.bvh_path)
        reference_base_id = ksim.get_reference_joint_id(root, self.config.reference_base_name)
        mj_base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, self.config.mj_base_name)

        def rotation_callback(root: BvhioJoint) -> None:
            euler_rotation = np.array(self.config.rotate_bvh_euler)
            quat = R.from_euler("xyz", euler_rotation).as_quat(scalar_first=True)
            root.applyRotation(glm.quat(*quat), bake=True)

        reference_motion = ksim.generate_reference_motion(
            model=mj_model,
            mj_base_id=mj_base_id,
            bvh_root=root,
            bvh_to_mujoco_names=HUMANOID_REFERENCE_MAPPINGS,
            bvh_base_id=reference_base_id,
            bvh_offset=np.array(self.config.bvh_offset),
            bvh_root_callback=rotation_callback,
            bvh_scaling_factor=self.config.bvh_scaling_factor,
            ctrl_dt=self.config.ctrl_dt,
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

        return reference_motion

    def get_real_motions(self, mj_model: mujoco.MjModel) -> Array:
        reference_motion = self.create_reference_motion(mj_model)
        # cartesian_poses = jax.tree.map(lambda x: np.asarray(x.array), reference_motion.cartesian_poses)
        return jnp.array(
            reference_motion.qpos.array[None, ..., 7:]
        )  # Remove the root joint absolute coordinates + orientation.

    def trajectory_to_motion(self, trajectory: ksim.Trajectory) -> Array:
        return trajectory.qpos[..., 7:]  # Remove the root joint absolute coordinates + orientation.

    def motion_to_qpos(self, motion: Array) -> Array:
        qpos_root_init = jnp.array([0.0, 0.0, 1.5])
        qpos_root_broadcast = jnp.broadcast_to(qpos_root_init, (*motion.shape[:-1], 3))
        return jnp.concatenate([qpos_root_broadcast, motion], axis=-1)

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
            aux_outputs=xax.FrozenDict({"tracked_pos": xax.FrozenDict(tracked_positions)}),
        )

    def get_physics_randomizers(self, physics_model: ksim.PhysicsModel) -> list[ksim.PhysicsRandomizer]:
        return [
            ksim.StaticFrictionRandomizer(),
            ksim.ArmatureRandomizer(),
            ksim.MassMultiplicationRandomizer.from_body_name(physics_model, "Torso_Side_Right"),
            ksim.JointDampingRandomizer(),
            ksim.JointZeroPositionRandomizer(),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        return [
            ksim.PushEvent(
                x_force=1.0,
                y_force=1.0,
                z_force=0.0,
                x_angular_force=0.1,
                y_angular_force=0.1,
                z_angular_force=0.3,
                interval_range=(0.25, 0.75),
            ),
        ]

    def get_initial_model_carry(self, rng: PRNGKeyArray) -> tuple[Array, Array]:
        return (
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
        )

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> dict[str, JointMetadataOutput]:
        metadata = asyncio.run(ksim.get_mujoco_model_metadata(self.config.robot_urdf_path, cache=False))
        if metadata.joint_name_to_metadata is None:
            raise ValueError("Joint metadata is not available")
        return metadata.joint_name_to_metadata

    def get_mujoco_model(self) -> mujoco.MjModel:
        mjcf_path = (Path(self.config.robot_urdf_path) / "robot.mjcf").resolve().as_posix()
        logger.info("Loading MJCF model from %s", mjcf_path)
        mj_model = load_mjmodel(mjcf_path, scene="smooth")

        return mj_model

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
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="local_linvel_origin", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="global_linvel_origin", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="global_angvel_origin", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="upvector_origin", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="orientation_origin", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="gyro_origin", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="left_foot_force", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="right_foot_force", noise=0.0),
            common.FeetContactObservation.create(
                physics_model=physics_model,
                foot_left_geom_names="KB_D_501L_L_LEG_FOOT_collision_box",
                foot_right_geom_names="KB_D_501R_R_LEG_FOOT_collision_box",
                floor_geom_names="floor",
            ),
            ksim.TimestepObservation(),
            common.ReferenceQposObservation(
                reference_motion_data=self.reference_motion_data,
                speed=self.qpos_reference_speed,
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

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return [
            common.LinearVelocityCommand(
                x_range=(-0.3, 0.7),
                y_range=(-0.2, 0.2),
                x_zero_prob=0.1,
                y_zero_prob=0.2,
                switch_prob=self.config.ctrl_dt / 3,  # once per 3 seconds
            )
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            common.GVecTermination.create(physics_model, sensor_name="upvector_origin"),
            ksim.MinimumHeightTermination(min_height=0.2),
        ]

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        return ksim.ConstantCurriculum(level=1.0)

    def run(self) -> None:
        mj_model: ksim.PhysicsModel = self.get_mujoco_model()
        root: BvhioJoint = bvhio.readAsHierarchy(self.config.bvh_path)
        reference_base_id = get_reference_joint_id(root, self.config.reference_base_name)
        self.mj_base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, self.config.mj_base_name)
        self.qpos_reference_speed = 1.8

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
            np_reference_qpos = np.asarray(self.reference_motion_data.qpos)
            np_cartesian_motion = jax.tree.map(np.asarray, self.reference_motion_data.cartesian_poses)

            visualize_reference_motion(
                mj_model,
                reference_qpos=np_reference_qpos,
                cartesian_motion=np_cartesian_motion,
                mj_base_id=self.mj_base_id,
            )
        else:
            super().run()

    def make_export_model(self, model: RnnActor, stochastic: bool = False, batched: bool = False) -> Callable:
        """Makes a callable inference function that directly takes a flattened input vector and returns an action.

        Returns:
            A tuple containing the inference function and the size of the input vector.
        """

        def deterministic_model_fn(obs: Array, carry: Array) -> tuple[Array, Array]:
            dist, carry = model.actor.call_flat_obs(obs, carry)
            return dist.mode(), carry

        def stochastic_model_fn(obs: Array, carry: Array) -> tuple[Array, Array]:
            dist, carry = model.actor.call_flat_obs(obs, carry)
            return dist.sample(seed=jax.random.PRNGKey(0)), carry

        if stochastic:
            model_fn = stochastic_model_fn
        else:
            model_fn = deterministic_model_fn

        if batched:

            def batched_model_fn(obs: Array, carry: Array) -> tuple[Array, Array]:
                return jax.vmap(model_fn)(obs, carry)

            return batched_model_fn

        return model_fn

    def on_after_checkpoint_save(self, ckpt_path: Path, state: xax.State) -> xax.State:
        if not self.config.export_for_inference:
            return state

        model: RnnModel = self.load_ckpt(ckpt_path, part="model")[0]

        model_fn = self.make_export_model(model, stochastic=False, batched=True)
        input_shapes = [
            (NUM_ACTOR_INPUTS_REF,),
            (
                self.config.depth,
                self.config.hidden_size,
            ),
        ]

        tf_path = (
            ckpt_path.parent / "tf_model"
            if self.config.only_save_most_recent
            else ckpt_path.parent / f"tf_model_{state.num_steps}"
        )

        export(
            model_fn,
            input_shapes,
            tf_path,
        )

        return state


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m ksim_kbot.walking.walking_amp disable_multiprocessing=True num_envs=2 batch_size=2
    # To visualize the environment, use the following command:
    #   python -m ksim_kbot.walking.walking_amp run_model_viewer=True
    # To visualize the reference gait, use the following command:
    #   mjpython -m ksim_kbot.walking.walking_amp visualize_reference_motion=True
    # On MacOS or other devices with less memory, you can change the number
    # of environments and batch size to reduce memory usage. Here's an example
    # from the command line:
    #   python -m ksim_kbot.walking.walking_amp num_envs=1 batch_size=1
    WalkingAmpTask.launch(
        WalkingAmpTaskConfig(
            # Training parameters.
            num_envs=1024,
            batch_size=256,
            num_passes=10,
            epochs_per_log_step=1,
            rollout_length_seconds=10.0,
            render_length_seconds=10.0,
            increase_threshold=5.0,
            decrease_threshold=3.0,
            # Simulation parameters.
            valid_every_n_seconds=240,
            iterations=8,
            ls_iterations=8,
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            # PPO parameters.
            gamma=0.97,
            lam=0.95,
            entropy_coef=0.005,
            learning_rate=1e-3,
            clip_param=0.3,
            max_grad_norm=0.5,
            # Gait matching parameters.
            rotate_bvh_euler=(0, np.pi / 2, 0),
            bvh_offset=(0.02, 0.09, -0.29),
            bvh_scaling_factor=1 / 100,
            mj_base_name="floating_base_link",
            reference_base_name="CC_Base_Pelvis",
            export_for_inference=True,
            only_save_most_recent=False,
            visualize_reference_motion=False,
        ),
    )
