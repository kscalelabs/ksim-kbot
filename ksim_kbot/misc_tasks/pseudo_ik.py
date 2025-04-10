# mypy: ignore-errors
"""Pseudo-Inverse Kinematics task for K-Bot"""

import asyncio
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Self, TypeVar, Collection

import attrs
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import ksim
import mujoco
import optax
import xax
from jaxtyping import Array, PRNGKeyArray
from kscale.web.gen.api import JointMetadataOutput
from ksim.utils.mujoco import remove_mujoco_joints_except, add_new_mujoco_body
from ksim.utils.types import norm_validator
from mujoco import mjx
from xax.nn.export import export

import ksim_kbot.common
from ksim_kbot.standing.standing import MAX_TORQUE
from ksim.vis import Marker

NUM_JOINTS = 5  # disabling all DoFs except for the right arm.
NUM_OUTPUTS = NUM_JOINTS * 2
NUM_INPUTS = NUM_JOINTS + NUM_JOINTS + 3 + 4 + NUM_OUTPUTS



@attrs.define(kw_only=True)
class BodyOrientationMarker(Marker):
    command_name: str = attrs.field()

    def __attrs_post_init__(self) -> None:
        if self.target_name is None or self.target_type != "body":
            raise ValueError("Base body name must be provided. Make sure to create with `get`.")

    def update(self, trajectory: ksim.Trajectory) -> None:
        command = trajectory.command[self.command_name]
        # Check if command is zeros (null quaternion)
        is_null = jnp.all(jnp.isclose(command, 0.0))

        # Only update orientation if command is not null
        if not is_null:
            self.geom = mujoco.mjtGeom.mjGEOM_ARROW
            self.orientation = command
        else:
            self.geom = mujoco.mjtGeom.mjGEOM_SPHERE

    @classmethod
    def get(
        cls,
        command_name: str,
        base_body_name: str,
        size: float,
        magnitude: float,
        rgba: tuple[float, float, float, float],
    ) -> Self:
        return cls(
            command_name=command_name,
            target_name=base_body_name,
            target_type="body",
            geom=mujoco.mjtGeom.mjGEOM_ARROW,
            scale=(size, size, magnitude),
            rgba=rgba,
        )


@attrs.define(frozen=True)
class BodyOrientationCommand(ksim.Command):

    base_body_name: str = attrs.field()
    base_id: int = attrs.field()
    switch_prob: float = attrs.field()
    null_prob: float = attrs.field()  # Probability of sampling null quaternion
    vis_magnitude: float = attrs.field()
    vis_size: float = attrs.field()
    vis_color: tuple[float, float, float, float] = attrs.field()

    def initial_command(
        self,
        physics_data: ksim.PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        rng_a, rng_b = jax.random.split(rng)
        is_null = jax.random.bernoulli(rng_a, self.null_prob)
        quat = jax.random.normal(rng_b, (4,))
        random_quat = quat / jnp.linalg.norm(quat)
        return jnp.where(is_null, jnp.zeros(4), random_quat)

    def __call__(
        self,
        prev_command: Array,
        physics_data: ksim.PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_commands = self.initial_command(physics_data, curriculum_level, rng_b)
        return jnp.where(switch_mask, new_commands, prev_command)

    def get_markers(self) -> Collection[Marker]:
        return [
            BodyOrientationMarker.get(
                self.command_name, self.base_body_name, self.vis_size, self.vis_magnitude, self.vis_color
            )
        ]

    def get_name(self) -> str:
        return f"{self.base_body_name}_{super().get_name()}"

    @classmethod
    def create(
        cls,
        model: ksim.PhysicsModel,
        base_body_name: str,
        switch_prob: float = 0.1,
        null_prob: float = 0.1,
        vis_magnitude: float = 0.5,
        vis_size: float = 0.05,
        vis_color: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.8),
    ) -> Self:
        base_id = ksim.get_body_data_idx_from_name(model, base_body_name)
        return cls(
            base_body_name=base_body_name,
            base_id=base_id,
            switch_prob=switch_prob,
            null_prob=null_prob,
            vis_magnitude=vis_magnitude,
            vis_size=vis_size,
            vis_color=vis_color,
        )

@attrs.define(frozen=True, kw_only=True)
class BodyOrientationTrackingReward(ksim.Reward):
    """Rewards the closeness of the body orientation to the target quaternion."""

    tracked_body_idx: int = attrs.field()
    command_name: str = attrs.field()
    norm: xax.NormType = attrs.field(default="l1", validator=norm_validator)
    eps: float = attrs.field(default=1e-6)

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        body_quat = trajectory.xquat[..., self.tracked_body_idx, :]
        target_quat = trajectory.command[self.command_name]

        # Ensure unit quaternions (safe for numerical drift).
        body_quat = body_quat / jnp.linalg.norm(body_quat, axis=-1, keepdims=True).clip(min=self.eps)
        target_quat = target_quat / jnp.linalg.norm(target_quat, axis=-1, keepdims=True).clip(min=self.eps)

        # Compute absolute dot product to handle antipodal symmetry.
        dot = jnp.sum(body_quat * target_quat, axis=-1)
        dot = jnp.clip(jnp.abs(dot), self.eps, 1.0 - self.eps)

        # Quaternion angular distance (geodesic distance on S^3).
        angle_error = 2.0 * jnp.arccos(dot)

        # Previous angle error.
        prev_error = jnp.concatenate([angle_error[..., :1], angle_error[..., :-1]], axis=-1)
        error_diff = angle_error - prev_error

        # Negative delta error as reward: improving alignment yields positive reward.
        reward = -error_diff

        return reward

    @classmethod
    def create(
        cls,
        model: ksim.PhysicsModel,
        command_name: str,
        tracked_body_name: str,
        norm: xax.NormType = "l1",
        scale: float = 1.0,
    ) -> Self:
        body_idx = ksim.get_body_data_idx_from_name(model, tracked_body_name)
        return cls(
            tracked_body_idx=body_idx,
            norm=norm,
            command_name=command_name,
            scale=scale,
        )

@attrs.define(frozen=True, kw_only=True)
class CartesianBodyTargetVectorReward(ksim.Reward):
    """Rewards the alignment of the body's velocity vector to the direction of the target."""

    tracked_body_idx: int = attrs.field()
    base_body_idx: int = attrs.field()
    command_name: str = attrs.field()
    dt: float = attrs.field()
    normalize_velocity: bool = attrs.field()
    distance_threshold: float = attrs.field()
    epsilon: float = attrs.field(default=1e-6)

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        body_pos_TL = trajectory.xpos[..., self.tracked_body_idx, :] - trajectory.xpos[..., self.base_body_idx, :]

        body_pos_right_shifted_TL = jnp.roll(body_pos_TL, shift=1, axis=0)

        # Zero out the first velocity
        body_pos_right_shifted_TL = body_pos_right_shifted_TL.at[0].set(body_pos_TL[0])

        body_vel_TL = (body_pos_TL - body_pos_right_shifted_TL) / self.dt

        target_vector = trajectory.command[self.command_name][..., :3] - body_pos_TL
        normalized_target_vector = target_vector / (
            jnp.linalg.norm(target_vector, axis=-1, keepdims=True) + self.epsilon
        )

        # Threshold to only apply reward to the body when it is far from the target.
        distance_scalar = jnp.linalg.norm(target_vector, axis=-1)
        far_from_target = distance_scalar > self.distance_threshold

        velocity_scalar = jnp.linalg.norm(body_vel_TL, axis=-1)
        high_velocity = velocity_scalar > 0.1

        if self.normalize_velocity:
            normalized_body_vel = body_vel_TL / (jnp.linalg.norm(body_vel_TL, axis=-1, keepdims=True) + self.epsilon)
            original_products = normalized_body_vel * normalized_target_vector
        else:
            original_products = body_vel_TL * normalized_target_vector

        # This will give maximum reward if near the target (and velocity is normalized)
        return jnp.where(far_from_target & high_velocity, jnp.sum(original_products, axis=-1), 1.1)

    @classmethod
    def create(
        cls,
        model: ksim.PhysicsModel,
        command_name: str,
        tracked_body_name: str,
        base_body_name: str,
        dt: float,
        normalize_velocity: bool = True,
        scale: float = 1.0,
        epsilon: float = 1e-6,
        distance_threshold: float = 0.1,
    ) -> Self:
        body_idx = ksim.get_body_data_idx_from_name(model, tracked_body_name)
        base_idx = ksim.get_body_data_idx_from_name(model, base_body_name)
        return cls(
            tracked_body_idx=body_idx,
            base_body_idx=base_idx,
            scale=scale,
            command_name=command_name,
            dt=dt,
            normalize_velocity=normalize_velocity,
            epsilon=epsilon,
            distance_threshold=distance_threshold,
        )

@attrs.define(frozen=True, kw_only=True)
class BodyPositionObservation(ksim.Observation):
    body_idx: int = attrs.field()
    body_name: str = attrs.field()

    @classmethod
    def create(
        cls,
        *,
        physics_model: ksim.PhysicsModel,
        body_name: str,
        noise: float = 0.0,
    ) -> Self:
        body_idx = ksim.get_body_data_idx_from_name(physics_model, body_name)
        return cls(
            body_idx=body_idx,
            body_name=body_name,
            noise=noise,
        )

    def observe(self, state: ksim.ObservationState, rng: PRNGKeyArray) -> Array:
        return state.physics_state.data.xpos[self.body_idx]

    def get_name(self) -> str:
        return f"{self.body_name}_{super().get_name()}"

@attrs.define(frozen=True, kw_only=True)
class BodyOrientationObservation(ksim.Observation):
    body_idx: int = attrs.field()
    body_name: str = attrs.field()

    @classmethod
    def create(
        cls,
        *,
        physics_model: ksim.PhysicsModel,
        body_name: str,
        noise: float = 0.0,
    ) -> Self:
        body_idx = ksim.get_body_data_idx_from_name(physics_model, body_name)
        return cls(
            body_idx=body_idx,
            body_name=body_name,
            noise=noise,
        )

    def observe(self, state: ksim.ObservationState, rng: PRNGKeyArray) -> Array:
        return state.physics_state.data.xquat[self.body_idx]

    def get_name(self) -> str:
        return f"{self.body_name}_{super().get_name()}"


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AuxOutputs:
    log_probs: Array
    values: Array


class KbotActor(eqx.Module):
    """Actor for the walking task."""

    mlp: eqx.nn.MLP
    min_std: float = eqx.static_field()
    max_std: float = eqx.static_field()
    var_scale: float = eqx.static_field()
    mean_scale: float = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        min_std: float,
        max_std: float,
        var_scale: float,
        mean_scale: float,
        hidden_size: int,
        depth: int,
    ) -> None:
        self.mlp = eqx.nn.MLP(
            in_size=NUM_INPUTS,
            out_size=NUM_OUTPUTS * 2,
            width_size=hidden_size,
            depth=depth,
            key=key,
            activation=jax.nn.elu,
        )
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale
        self.mean_scale = mean_scale

    def __call__(
        self,
        joint_pos_n: Array,
        joint_vel_n: Array,
        xyz_target_3: Array,
        quat_target_4: Array,
        prev_action_n: Array,
    ) -> distrax.Normal:
        obs_n = jnp.concatenate(
            [
                joint_pos_n,  # NUM_JOINTS
                joint_vel_n,  # NUM_JOINTS
                xyz_target_3,  # 3
                quat_target_4,  # 4
                prev_action_n,  # NUM_OUTPUTS
            ],
            axis=-1,
        )

        return self.call_flat_obs(obs_n)

    def call_flat_obs(self, obs_n: Array) -> distrax.Normal:
        prediction_n = self.mlp(obs_n)
        mean_n = prediction_n[..., :NUM_OUTPUTS]
        std_n = prediction_n[..., NUM_OUTPUTS:]

        mean_n = mean_n * self.mean_scale

        # Softplus and clip to ensure positive standard deviations.
        std_n = jnp.clip((jax.nn.softplus(std_n) + self.min_std) * self.var_scale, max=self.max_std)

        return distrax.Normal(mean_n, std_n)


class KbotCritic(eqx.Module):
    """Critic for the walking task."""

    mlp: eqx.nn.MLP

    def __init__(self, key: PRNGKeyArray, *, hidden_size: int, depth: int) -> None:
        num_inputs = NUM_INPUTS + NUM_JOINTS + 3 + 4
        num_outputs = 1

        self.mlp = eqx.nn.MLP(
            in_size=num_inputs,
            out_size=num_outputs,
            width_size=hidden_size,
            depth=depth,
            key=key,
            activation=jax.nn.relu,
        )

    def __call__(
        self,
        joint_pos_n: Array,
        joint_vel_n: Array,
        actuator_force_n: Array,
        xyz_target_3: Array,
        quat_target_4: Array,
        end_effector_pos_3: Array,
        end_effector_quat_4: Array,
        prev_action_n: Array,
    ) -> Array:
        x_n = jnp.concatenate(
            [
                joint_pos_n,  # NUM_JOINTS
                joint_vel_n,  # NUM_JOINTS
                actuator_force_n,  # NUM_JOINTS
                xyz_target_3,  # 3
                quat_target_4,  # 4
                end_effector_pos_3,  # 3
                end_effector_quat_4,  # 4
                prev_action_n,  # NUM_OUTPUTS
            ],
            axis=-1,
        )
        return self.mlp(x_n)


class KbotModel(eqx.Module):
    actor: KbotActor
    critic: KbotCritic

    def __init__(self, key: PRNGKeyArray, *, hidden_size: int, depth: int) -> None:
        self.actor = KbotActor(
            key,
            min_std=0.0, #0.001,
            max_std=1.0,
            var_scale=1.0,
            mean_scale=1.0,
            hidden_size=hidden_size,
            depth=depth,
        )
        self.critic = KbotCritic(key, hidden_size=hidden_size, depth=depth)


@dataclass
class KbotPseudoIKTaskConfig(ksim.PPOConfig):
    """Config for the KBot pseudo-IK task."""

    robot_urdf_path: str = xax.field(
        value="ksim_kbot/kscale-assets/kbot-v2-lw-feet/",
        help="The path to the assets directory for the robot.",
    )
    # Optimizer parameters.
    learning_rate: float = xax.field(
        value=3e-4,
        help="Learning rate for PPO.",
    )
    max_grad_norm: float = xax.field(
        value=2.0,
        help="Maximum gradient norm for clipping.",
    )
    adam_weight_decay: float = xax.field(
        value=0.0,
        help="Weight decay for the Adam optimizer.",
    )

    # Mujoco parameters.
    use_mit_actuators: bool = xax.field(
        value=False,
        help="Whether to use the MIT actuator model, where the actions are position commands",
    )
    # Rendering parameters.
    render_track_body_id: int | None = xax.field(
        value=0,
        help="The body id to track with the render camera.",
    )

    # Checkpointing parameters.
    export_for_inference: bool = xax.field(
        value=False,
        help="Whether to export the model for inference.",
    )

    depth: int = xax.field(
        value=5,
        help="The depth of the models.",
    )
    hidden_size: int = xax.field(
        value=256,
        help="The hidden size of the models.",
    )


Config = TypeVar("Config", bound=KbotPseudoIKTaskConfig)


class KbotPseudoIKTask(ksim.PPOTask[Config], Generic[Config]):
    def get_optimizer(self) -> optax.GradientTransformation:
        """Builds the optimizer.

        This provides a reasonable default optimizer for training PPO models,
        but can be overridden by subclasses who want to do something different.
        """
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            (
                optax.adam(self.config.learning_rate)
                if self.config.adam_weight_decay == 0.0
                else optax.adamw(self.config.learning_rate, weight_decay=self.config.adam_weight_decay)
            ),
        )

        return optimizer

    def get_mujoco_model(self) -> tuple[mujoco.MjModel, dict[str, JointMetadataOutput]]:
        mjcf_path = (Path(self.config.robot_urdf_path) / "robot_scene_collisions_simplified.mjcf").resolve().as_posix()
        mj_model_joint_removed = remove_mujoco_joints_except(
            mjcf_path,
            [
                "dof_right_shoulder_pitch_03",
                "dof_right_shoulder_roll_03",
                "dof_right_shoulder_yaw_02",
                "dof_right_elbow_02",
                "dof_right_wrist_00",
            ],
        )

        # save to a temp file in the same directory
        temp_path = (
            (Path(self.config.robot_urdf_path) / f"robot_scene_joint_removed_{uuid.uuid4()}.mjcf").resolve().as_posix()
        )

        with open(temp_path, "w") as f:
            f.write(mj_model_joint_removed)

        # add body
        mj_model_added_body = add_new_mujoco_body(
            temp_path,
            parent_body_name="KB_C_501X_Right_Bayonet_Adapter_Hard_Stop",
            new_body_name="ik_target",
            pos=(0.0, 0.0, -0.1),
            quat=(1.0, 0.0, 0.0, 0.0),
            add_visual=True,
            visual_geom_color=(0, 1, 0, 1),
            visual_geom_size=(0.03, 0.03, 0.03),
        )

        temp_path_2 = (
            (Path(self.config.robot_urdf_path) / f"robot_scene_added_body_{uuid.uuid4()}.mjcf").resolve().as_posix()
        )

        with open(temp_path_2, "w") as f:
            f.write(mj_model_added_body)

        mj_model = mujoco.MjModel.from_xml_path(temp_path_2)

        # remove the temp file
        os.remove(temp_path)
        os.remove(temp_path_2)
        mj_model.opt.timestep = jnp.array(self.config.dt)
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6
        mj_model.opt.disableflags = mjx.DisableBit.EULERDAMP
        mj_model.opt.solver = mjx.SolverType.CG

        return mj_model

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> dict[str, JointMetadataOutput]:
        metadata = asyncio.run(ksim.get_mujoco_model_metadata(self.config.robot_urdf_path, cache=False))
        if metadata.joint_name_to_metadata is None:
            raise ValueError("Joint metadata is not available")
        return metadata.joint_name_to_metadata

    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: dict[str, JointMetadataOutput] | None = None,
    ) -> ksim.Actuators:
        if self.config.use_mit_actuators:
            if metadata is None:
                raise ValueError("Metadata is required for MIT actuators")
            return ksim.MITPositionVelocityActuators(
                # return ksim.MITPositionActuators(
                physics_model,
                metadata,
                # pos_action_noise=0.1,
                # vel_action_noise=0.1,
                # pos_action_noise_type="gaussian",
                # vel_action_noise_type="gaussian",
                # torque_noise=0.2,
                # torque_noise_type="gaussian",
                ctrl_clip=[
                    # right arm
                    MAX_TORQUE["03"],
                    MAX_TORQUE["03"],
                    MAX_TORQUE["02"],
                    MAX_TORQUE["02"],
                    MAX_TORQUE["00"],
                ],
                freejoint_first=False,
            )
        else:
            return ksim.TorqueActuators()

    def get_physics_randomizers(self, physics_model: ksim.PhysicsModel) -> list[ksim.PhysicsRandomizer]:
        return [
            ksim.StaticFrictionRandomizer(scale_lower=0.5, scale_upper=2.0, freejoint_first=False),
            # ksim.JointZeroPositionRandomization(scale_lower=-0.01, scale_upper=0.01, freejoint_first=False),
            ksim.ArmatureRandomizer(scale_lower=1.0, scale_upper=1.05, freejoint_first=False),
            ksim.MassMultiplicationRandomizer.from_body_name(physics_model, "KC_C_104R_PitchHardstopDriven"),
            ksim.JointDampingRandomizer(scale_lower=0.95, scale_upper=1.05, freejoint_first=False),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        return []

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomJointPositionReset(scale=1.0),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        return [
            ksim.JointPositionObservation(freejoint_first=False, noise=0.01, noise_type="gaussian"),
            ksim.JointVelocityObservation(freejoint_first=False, noise=0.1, noise_type="gaussian"),
            ksim.ActuatorForceObservation(),
            ksim.ActuatorAccelerationObservation(freejoint_first=False),
            ksim.ContactObservation.create(
                contact_group="arms",
                physics_model=physics_model,
                geom_names=[
                    "right_upper_arm_collision",
                    "left_upper_arm_collision",
                    "right_forearm_collision",
                    "left_forearm_collision",
                    "torso_collision",
                    "legs_collision",
                ],
            ),
            BodyPositionObservation.create(
                physics_model=physics_model,
                body_name="ik_target",
            ),
            BodyOrientationObservation.create(
                physics_model=physics_model,
                body_name="ik_target",
            ),
            ksim_kbot.common.LastActionObservation(noise=0.0),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return [
            ksim.PositionCommand.create(
                model=physics_model,
                box_min=(0.0, -0.2, -0.2),
                box_max=(0.3, 0.2, 0.2),
                vis_target_name="floating_base_link",
                vis_radius=0.05,
                vis_color=(1.0, 0.0, 0.0, 0.8),
                unique_name="target",
                min_speed=0.2,
                max_speed=4.0,
                switch_prob=self.config.ctrl_dt * 10,
                jump_prob=self.config.ctrl_dt * 5,
            ),  # type: ignore[call-arg]
            BodyOrientationCommand.create(
                model=physics_model,
                base_body_name="ik_target",
                switch_prob=self.config.ctrl_dt,
                null_prob=self.config.ctrl_dt * 5,
                vis_magnitude=0.5,
                vis_size=0.05,
                vis_color=(0.0, 1.0, 0.0, 0.8),
            ),
        ]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            ksim.PositionTrackingReward.create(
                model=physics_model,
                tracked_body_name="ik_target",
                base_body_name="floating_base_link",
                scale=10.0,
                command_name="target_position_command",
            ),
            BodyOrientationTrackingReward.create(
                model=physics_model,
                command_name="ik_target_body_orientation_command",
                tracked_body_name="ik_target",
                scale=0.1,
            ),
            # CartesianBodyTargetVectorReward.create(
            #     model=physics_model,
            #     command_name="target_position_command",
            #     tracked_body_name="ik_target",
            #     base_body_name="floating_base_link",
            #     scale=3.0,
            #     normalize_velocity=True,
            #     distance_threshold=0.1,
            #     dt=self.config.dt,
            # ),
            ksim.ObservationMeanPenalty(observation_name="contact_observation_arms", scale=-0.1),
            ksim.ActuatorForcePenalty(scale=-0.000001, norm="l1"),
            ksim.ActionSmoothnessPenalty(scale=-0.02, norm="l2"),
            ksim.JointVelocityPenalty(scale=-0.001, freejoint_first=False, norm="l2"),
            ksim.ActuatorJerkPenalty(scale=-0.001, ctrl_dt=self.config.ctrl_dt, norm="l2"),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            ksim.FastAccelerationTermination(),
        ]

    def get_model(self, key: PRNGKeyArray) -> KbotModel:
        return KbotModel(key, hidden_size=self.config.hidden_size, depth=self.config.depth)

    def get_initial_model_carry(self, rng: PRNGKeyArray) -> None:
        return None

    def _run_actor(
        self,
        model: KbotModel,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
    ) -> distrax.Normal:
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"] / 50.0
        xyz_target_3 = commands["target_position_command"][..., :3]
        quat_target_4 = commands["ik_target_body_orientation_command"]
        prev_action_n = observations["last_action_observation"]
        return model.actor(
            joint_pos_n=joint_pos_n,
            joint_vel_n=joint_vel_n,
            xyz_target_3=xyz_target_3,
            quat_target_4=quat_target_4,
            prev_action_n=prev_action_n,
        )

    def _run_critic(
        self,
        model: KbotModel,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
    ) -> Array:
        joint_pos_n = observations["joint_position_observation"]  # 26
        joint_vel_n = observations["joint_velocity_observation"] / 50.0  # 27
        actuator_force_n = observations["actuator_force_observation"]  # 27
        xyz_target_3 = commands["target_position_command"][..., :3]  # 3
        quat_target_4 = commands["ik_target_body_orientation_command"]  # 4
        end_effector_pos_3 = observations["ik_target_body_position_observation"]  # 3
        end_effector_quat_4 = observations["ik_target_body_orientation_observation"]  # 4
        prev_action_n = observations["last_action_observation"]  # 5

        return model.critic(
            joint_pos_n=joint_pos_n,
            joint_vel_n=joint_vel_n,
            actuator_force_n=actuator_force_n,
            xyz_target_3=xyz_target_3,
            quat_target_4=quat_target_4,
            end_effector_pos_3=end_effector_pos_3,
            end_effector_quat_4=end_effector_quat_4,
            prev_action_n=prev_action_n,
        )

    def get_ppo_variables(
        self,
        model: KbotModel,
        trajectories: ksim.Trajectory,
        carry: None,
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, None]:

        # vectorize over the time dimensions
        def get_log_prob(transition: ksim.Trajectory) -> Array:
            action_dist_n = self._run_actor(model, transition.obs, transition.command)
            log_probs_n = action_dist_n.log_prob(transition.action / model.actor.mean_scale)
            return log_probs_n

        log_probs_tn = jax.vmap(get_log_prob)(trajectories)

        values_tn = jax.vmap(self._run_critic, in_axes=(None, 0, 0))(model, trajectories.obs, trajectories.command)

        ppo_variables = ksim.PPOVariables(
            log_probs=log_probs_tn,
            values=values_tn.squeeze(-1),
        )

        return ppo_variables, None

    def sample_action(
        self,
        model: KbotModel,
        model_carry: None,
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
    ) -> ksim.Action:
        action_dist_n = self._run_actor(model, observations, commands)
        action_n = action_dist_n.sample(seed=rng)
        action_log_prob_n = action_dist_n.log_prob(action_n)

        critic_n = self._run_critic(model, observations, commands)
        value_n = critic_n.squeeze(-1)

        return ksim.Action(action=action_n, aux_outputs=AuxOutputs(log_probs=action_log_prob_n, values=value_n))

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        return ksim.RewardLevelCurriculum(
            reward_name="ik_target_position_tracking_reward",
            increase_threshold=0.15,
            decrease_threshold=0.12,
            min_level_steps=10,
            num_levels=10,
        )

    def make_export_model(self, model: KbotModel, stochastic: bool = False, batched: bool = False) -> Callable:
        """Makes a callable inference function that directly takes a flattened input vector and returns an action.

        Returns:
            A tuple containing the inference function and the size of the input vector.
        """

        def deterministic_model_fn(obs: Array) -> Array:
            return model.actor.call_flat_obs(obs).mode()

        def stochastic_model_fn(obs: Array) -> Array:
            dist = model.actor.call_flat_obs(obs)
            return dist.sample(seed=jax.random.PRNGKey(0))

        if stochastic:
            model_fn = stochastic_model_fn
        else:
            model_fn = deterministic_model_fn

        if batched:

            def batched_model_fn(obs: Array) -> Array:
                return jax.vmap(model_fn)(obs)

            return batched_model_fn

        return model_fn

    def on_after_checkpoint_save(self, ckpt_path: Path, state: xax.State) -> xax.State:
        if not self.config.export_for_inference:
            return state

        model: KbotModel = self.load_ckpt_with_template(
            ckpt_path,
            part="model",
            model_template=self.get_model(key=jax.random.PRNGKey(0)),
        )

        model_fn = self.make_export_model(model, stochastic=False, batched=True)

        input_shapes = [(NUM_INPUTS,)]

        tf_path = (
            ckpt_path.parent / "tf_model"
            if self.config.only_save_most_recent
            else ckpt_path.parent / f"tf_model_{state.num_steps}"
        )

        export(
            model_fn,
            input_shapes,  # type: ignore [arg-type]
            tf_path,
        )

        return state


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m ksim_kbot.misc_tasks.psuedo_ik
    # To visualize the environment, use the following command:
    #   python -m ksim_kbot.misc_tasks.psuedo_ik run_environment=True
    # On MacOS or other devices with less memory, you can change the number
    # of environments and batch size to reduce memory usage. Here's an example
    # from the command line:
    #   python -m ksim_kbot.misc_tasks.psuedo_ik num_envs=8 batch_size=4
    KbotPseudoIKTask.launch(
        KbotPseudoIKTaskConfig(
            # Training parameters.
            num_envs=8192,
            batch_size=1024,
            num_passes=10,
            epochs_per_log_step=1,
            # Logging parameters.
            log_full_trajectory_every_n_seconds=60,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.05,
            min_action_latency=0.0,
            entropy_coef=0.005,
            learning_rate=3e-4,
            rollout_length_seconds=10.0,
            render_length_seconds=10.0,
            save_every_n_steps=25,
            export_for_inference=True,
            # Apparently rendering markers can sometimes cause segfaults.
            # Disable this if you are running into segfaults.
            render_markers=True,
            render_camera_name="iso_camera",
            disable_multiprocessing=False,
            use_mit_actuators=True,
        ),
    )
