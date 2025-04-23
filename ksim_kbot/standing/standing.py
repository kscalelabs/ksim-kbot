"""Defines simple task for training a standing policy for K-Bot."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

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
from ksim.curriculum import ConstantCurriculum, Curriculum
from mujoco import mjx
from mujoco_scenes.mjcf import load_mjmodel

from ksim_kbot import common, rewards

logger = logging.getLogger(__name__)

OBS_SIZE = 20 * 2 + 2 + 3 + 3 + 3 + 40  # = position + velocity + imu_acc + imu_gyro + projected_gravity + last_action
CMD_SIZE = 3
NUM_OUTPUTS = 20 * 2  # position + velocity

SINGLE_STEP_HISTORY_SIZE = NUM_OUTPUTS + OBS_SIZE + CMD_SIZE

HISTORY_LENGTH = 0

NUM_INPUTS = (OBS_SIZE + CMD_SIZE) + SINGLE_STEP_HISTORY_SIZE * HISTORY_LENGTH

MAX_TORQUE = {
    "00": 1.0,
    "02": 14.0,
    "03": 40.0,
    "04": 60.0,
}

Config = TypeVar("Config", bound="KbotStandingTaskConfig")


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
        min_std: float,
        max_std: float,
        var_scale: float,
        mean_scale: float,
    ) -> None:
        self.mlp = eqx.nn.MLP(
            in_size=NUM_INPUTS,
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
        timestep_phase_2: Array,
        joint_pos_n: Array,
        joint_vel_n: Array,
        imu_acc_3: Array,
        imu_gyro_3: Array,
        projected_gravity_3: Array,
        lin_vel_cmd_x: Array,
        lin_vel_cmd_y: Array,
        ang_vel_cmd_z: Array,
        last_action_n: Array,
        history_n: Array,
    ) -> distrax.Normal:
        x_n = jnp.concatenate(
            [
                timestep_phase_2,
                joint_pos_n,
                joint_vel_n,
                imu_acc_3,
                imu_gyro_3,
                projected_gravity_3,
                lin_vel_cmd_x,
                lin_vel_cmd_y,
                ang_vel_cmd_z,
                last_action_n,
                history_n,
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

    def __init__(self, key: PRNGKeyArray) -> None:
        self.mlp = eqx.nn.MLP(
            in_size=NUM_INPUTS + 2 + 3 + 4 + 3 + 3 + 20,
            out_size=1,  # Always output a single critic value.
            width_size=256,
            depth=5,
            key=key,
            activation=jax.nn.relu,
        )

    def forward(
        self,
        timestep_phase_2: Array,
        joint_pos_n: Array,
        joint_vel_n: Array,
        imu_acc_3: Array,
        imu_gyro_3: Array,
        projected_gravity_3: Array,
        lin_vel_cmd_x: Array,
        lin_vel_cmd_y: Array,
        ang_vel_cmd_z: Array,
        last_action_n: Array,
        feet_contact_2: Array,
        base_position_3: Array,
        base_orientation_4: Array,
        base_linear_velocity_3: Array,
        base_angular_velocity_3: Array,
        actuator_force_n: Array,
        history_n: Array,
    ) -> Array:
        x_n = jnp.concatenate(
            [
                timestep_phase_2,
                joint_pos_n,
                joint_vel_n,
                imu_acc_3,
                imu_gyro_3,
                projected_gravity_3,
                lin_vel_cmd_x,
                lin_vel_cmd_y,
                ang_vel_cmd_z,
                last_action_n,
                feet_contact_2,
                base_position_3,
                base_orientation_4,
                base_linear_velocity_3,
                base_angular_velocity_3,
                actuator_force_n,
                history_n,
            ],
            axis=-1,
        )
        return self.mlp(x_n)


class KbotModel(eqx.Module):
    actor: KbotActor
    critic: KbotCritic

    def __init__(self, key: PRNGKeyArray) -> None:
        self.actor = KbotActor(
            key,
            min_std=0.01,
            max_std=1.0,
            var_scale=1.0,
            mean_scale=1.0,
        )
        self.critic = KbotCritic(key)


@dataclass
class KbotStandingTaskConfig(ksim.PPOConfig):
    """Config for the KBot standing task."""

    robot_urdf_path: str = xax.field(
        value="ksim_kbot/kscale-assets/kbot-v2-feet/",
        help="The path to the assets directory for the robot.",
    )

    action_scale: float = xax.field(
        value=1.0,
        help="The scale to apply to the actions.",
    )

    # Optimizer parameters.
    learning_rate: float = xax.field(
        value=1e-4,
        help="Learning rate for PPO.",
    )
    max_grad_norm: float = xax.field(
        value=0.5,
        help="Maximum gradient norm for clipping.",
    )
    adam_weight_decay: float = xax.field(
        value=0.0,
        help="Weight decay for the Adam optimizer.",
    )

    # Mujoco parameters.
    use_mit_actuators: bool = xax.field(
        value=False,
        help="Whether to use the MIT actuator model, where the actions are position + velocity commands",
    )

    domain_randomize: bool = xax.field(
        value=True,
        help="Whether to domain randomize the model.",
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

    terrain_type: str = xax.field(value="smooth", help="The type of terrain to use. Options are 'smooth' or 'rough'")


class KbotStandingTask(ksim.PPOTask[KbotStandingTaskConfig], Generic[Config]):
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

    def get_mujoco_model(self) -> mujoco.MjModel:
        mjcf_path = (Path(self.config.robot_urdf_path) / "robot.mjcf").resolve().as_posix()
        logger.info("Loading MJCF model from %s", mjcf_path)

        mj_model = load_mjmodel(mjcf_path, scene=self.config.terrain_type)

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
            return common.TargetPositionMITActuators(
                physics_model,
                metadata,
                default_targets=(
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
                    -0.23,
                    0.0,
                    0.0,
                    -0.441,
                    0.195,
                    # left leg
                    0.23,
                    0.0,
                    0.0,
                    0.441,
                    -0.195,
                ),
                pos_action_noise=0.1,
                vel_action_noise=0.1,
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
        else:
            return ksim.TorqueActuators()

    def get_randomization(self, physics_model: ksim.PhysicsModel) -> list[ksim.PhysicsRandomizer]:
        if self.config.domain_randomize:
            return [
                ksim.StaticFrictionRandomizer(scale_lower=0.5, scale_upper=2.0),
                ksim.JointZeroPositionRandomizer(scale_lower=-0.01, scale_upper=0.01),
                ksim.ArmatureRandomizer(scale_lower=1.0, scale_upper=1.05),
                ksim.MassMultiplicationRandomizer.from_body_name(physics_model, "Torso_Side_Right"),
                ksim.JointDampingRandomizer(scale_lower=0.95, scale_upper=1.05),
            ]
        else:
            return []

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        scale = 0.0 if self.config.domain_randomize else 0.01
        return [
            ksim.RandomBaseVelocityXYReset(scale=scale),
            ksim.RandomJointPositionReset(scale=scale),
            ksim.RandomJointVelocityReset(scale=scale),
            common.ResetDefaultJointPosition(
                default_targets=(
                    0.0,
                    0.0,
                    0.9,
                    # quat
                    1.0,
                    0.0,
                    0.0,
                    0.0,
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
                    -0.23,
                    0.0,
                    0.0,
                    -0.441,
                    0.195,
                    # left leg
                    0.23,
                    0.0,
                    0.0,
                    0.441,
                    -0.195,
                )
            ),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        if self.config.domain_randomize:
            return [
                ksim.PushEvent(
                    x_force=1.0,
                    y_force=1.0,
                    z_force=0.0,
                    x_angular_force=0.0,
                    y_angular_force=0.0,
                    z_angular_force=0.0,
                    interval_range=(4.0, 7.0),
                ),
            ]
        else:
            return []

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> Curriculum:
        return ConstantCurriculum(level=1.0)

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        if self.config.domain_randomize:
            vel_obs_noise = 0.0
            imu_acc_noise = 0.5
            imu_gyro_noise = 0.2
            gvec_noise = 0.0
            base_position_noise = 0.0
            base_orientation_noise = 0.0
            base_linear_velocity_noise = 0.0
            base_angular_velocity_noise = 0.0
            base_angular_velocity_noise = 0.0
        else:
            vel_obs_noise = 0.0
            imu_acc_noise = 0.0
            imu_gyro_noise = 0.0
            gvec_noise = 0.0
            base_position_noise = 0.0
            base_orientation_noise = 0.0
            base_linear_velocity_noise = 0.0
            base_angular_velocity_noise = 0.0
            base_angular_velocity_noise = 0.0
        return [
            common.TimestepPhaseObservation(),
            common.JointPositionObservation(
                default_targets=(
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
                    -0.23,
                    0.0,
                    0.0,
                    -0.441,
                    0.195,
                    # left leg
                    0.23,
                    0.0,
                    0.0,
                    0.441,
                    -0.195,
                ),
                noise=0.01,
            ),
            ksim.JointVelocityObservation(noise=vel_obs_noise),
            ksim.ActuatorForceObservation(),
            ksim.SensorObservation.create(
                physics_model=physics_model,
                sensor_name="imu_acc",
                noise=imu_acc_noise,
            ),
            ksim.SensorObservation.create(
                physics_model=physics_model,
                sensor_name="imu_gyro",
                noise=imu_gyro_noise,
            ),
            common.ProjectedGravityObservation(noise=gvec_noise),
            common.LastActionObservation(noise=0.0),
            # Additional critic observations
            ksim.BasePositionObservation(noise=base_position_noise),
            ksim.BaseOrientationObservation(noise=base_orientation_noise),
            ksim.BaseLinearVelocityObservation(noise=base_linear_velocity_noise),
            ksim.BaseAngularVelocityObservation(noise=base_angular_velocity_noise),
            ksim.CenterOfMassVelocityObservation(),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="local_linvel_origin", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="global_linvel_origin", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="global_angvel_origin", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="upvector_origin", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="orientation_origin", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="gyro_origin", noise=0.0),
            common.FeetContactObservation.create(
                physics_model=physics_model,
                foot_left_geom_names="KB_D_501L_L_LEG_FOOT_collision_box",
                foot_right_geom_names="KB_D_501R_R_LEG_FOOT_collision_box",
                floor_geom_names="floor",
            ),
            common.FeetPositionObservation.create(
                physics_model=physics_model,
                foot_left_site_name="left_foot",
                foot_right_site_name="right_foot",
                floor_threshold=0.00,
            ),
            common.TrueHeightObservation(),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        switch_prob = 0.0
        return [
            ksim.LinearVelocityCommand(index="x", range=(0.0, 0.0), zero_prob=1.0, switch_prob=switch_prob),  # type: ignore[attr-defined]
            ksim.LinearVelocityCommand(index="y", range=(0.0, 0.0), zero_prob=1.0, switch_prob=switch_prob),  # type: ignore[attr-defined]
            ksim.AngularVelocityCommand(index="z", scale=0.0, zero_prob=1.0, switch_prob=switch_prob),  # type: ignore[attr-defined]
        ]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            rewards.XYPositionPenalty(target_x=0.0, target_y=0.0, scale=-0.1),
            ksim.ActionSmoothnessPenalty(scale=-0.001),
            rewards.JointDeviationPenalty(
                scale=-0.02,
                joint_targets=(
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
                    -0.23,
                    0.0,
                    0.0,
                    -0.441,
                    0.195,
                    # left leg
                    0.23,
                    0.0,
                    0.0,
                    0.441,
                    -0.195,
                ),
                joint_weights=(
                    # right arm
                    0.1,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    # left arm
                    0.1,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    # right leg
                    0.01,  # pitch
                    1.0,
                    1.0,
                    0.01,  # knee
                    1.0,
                    # left leg
                    0.01,  # pitch
                    1.0,
                    1.0,
                    0.01,  # knee
                    1.0,
                ),
            ),
            rewards.FeetSlipPenalty(scale=-0.05),
            ksim.StayAliveReward(scale=1.0),
            # common.TerminationPenalty(scale=-5.0),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            common.GVecTermination.create(physics_model, sensor_name="upvector_origin"),
        ]

    def get_model(self, key: PRNGKeyArray) -> KbotModel:
        return KbotModel(key)

    def get_initial_carry(self, rng: PRNGKeyArray) -> tuple[Array, Array]:
        return (
            jnp.zeros(HISTORY_LENGTH * SINGLE_STEP_HISTORY_SIZE),
            jnp.zeros(HISTORY_LENGTH * SINGLE_STEP_HISTORY_SIZE),
        )

    def run_actor(
        self,
        model: KbotActor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> distrax.Normal:
        timestep_phase_2 = observations["timestep_phase_observation"]
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"] / 10.0
        imu_acc_3 = observations["sensor_observation_imu_acc"] / 50.0
        imu_gyro_3 = observations["sensor_observation_imu_gyro"] / 3.0
        projected_gravity_3 = observations["projected_gravity_observation"]
        lin_vel_cmd_x = commands["linear_velocity_command_x"]
        lin_vel_cmd_y = commands["linear_velocity_command_y"]
        ang_vel_cmd_z = commands["angular_velocity_command_z"]
        last_action_n = observations["last_action_observation"]
        history_n = carry

        return model.forward(
            timestep_phase_2=timestep_phase_2,
            joint_pos_n=joint_pos_n,
            joint_vel_n=joint_vel_n,
            imu_acc_3=imu_acc_3,
            imu_gyro_3=imu_gyro_3,
            projected_gravity_3=projected_gravity_3,
            lin_vel_cmd_x=lin_vel_cmd_x,
            lin_vel_cmd_y=lin_vel_cmd_y,
            ang_vel_cmd_z=ang_vel_cmd_z,
            last_action_n=last_action_n,
            history_n=history_n,
        )

    def run_critic(
        self,
        model: KbotCritic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> Array:
        timestep_phase_2 = observations["timestep_phase_observation"]
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"] / 10.0
        imu_acc_3 = observations["sensor_observation_imu_acc"] / 50.0
        imu_gyro_3 = observations["sensor_observation_imu_gyro"] / 3.0
        projected_gravity_3 = observations["projected_gravity_observation"]
        lin_vel_cmd_x = commands["linear_velocity_command_x"]
        lin_vel_cmd_y = commands["linear_velocity_command_y"]
        ang_vel_cmd_z = commands["angular_velocity_command_z"]
        last_action_n = observations["last_action_observation"]
        # critic observations
        feet_contact_2 = observations["feet_contact_observation"]
        base_position_3 = observations["base_position_observation"]
        base_orientation_4 = observations["base_orientation_observation"]
        base_linear_velocity_3 = observations["base_linear_velocity_observation"]
        base_angular_velocity_3 = observations["base_angular_velocity_observation"]
        actuator_force_n = observations["actuator_force_observation"] / 100.0

        history_n = carry

        return model.forward(
            timestep_phase_2=timestep_phase_2,
            joint_pos_n=joint_pos_n,
            joint_vel_n=joint_vel_n,
            imu_acc_3=imu_acc_3,
            imu_gyro_3=imu_gyro_3,
            projected_gravity_3=projected_gravity_3,
            lin_vel_cmd_x=lin_vel_cmd_x,
            lin_vel_cmd_y=lin_vel_cmd_y,
            ang_vel_cmd_z=ang_vel_cmd_z,
            last_action_n=last_action_n,
            feet_contact_2=feet_contact_2,
            actuator_force_n=actuator_force_n,
            base_position_3=base_position_3,
            base_orientation_4=base_orientation_4,
            base_linear_velocity_3=base_linear_velocity_3,
            base_angular_velocity_3=base_angular_velocity_3,
            history_n=history_n,
        )

    def get_ppo_variables(
        self,
        model: KbotModel,
        trajectories: ksim.Trajectory,
        carry: tuple[Array, Array],
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, tuple[Array, Array]]:
        actor_carry, critic_carry = carry
        action_dist_j = self.run_actor(model.actor, trajectories.obs, trajectories.command, actor_carry)
        log_probs_j = action_dist_j.log_prob(trajectories.action)

        values_1 = self.run_critic(model.critic, trajectories.obs, trajectories.command, critic_carry)

        ppo_variables = ksim.PPOVariables(
            log_probs=log_probs_j,
            values=values_1.squeeze(-1),
        )
        # TODO: update the carry properly
        return ppo_variables, (actor_carry, critic_carry)

    def sample_action(
        self,
        model: KbotModel,
        model_carry: Array,
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
        argmax: bool = False,
    ) -> ksim.Action:
        actor_carry, _ = model_carry
        action_dist_n = self.run_actor(model.actor, observations, commands, actor_carry)
        action_j = action_dist_n.sample(seed=rng)

        timestep_phase_2 = observations["timestep_phase_observation"]
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"] / 10.0
        imu_acc_3 = observations["sensor_observation_imu_acc"] / 50.0
        imu_gyro_3 = observations["sensor_observation_imu_gyro"] / 3.0
        projected_gravity_3 = observations["projected_gravity_observation"]
        lin_vel_cmd_x = commands["linear_velocity_command_x"]
        lin_vel_cmd_y = commands["linear_velocity_command_y"]
        ang_vel_cmd_z = commands["angular_velocity_command_z"]
        last_action_n = observations["last_action_observation"]
        current_history_data = jnp.concatenate(
            [
                timestep_phase_2,
                joint_pos_n,
                joint_vel_n,
                imu_acc_3,
                imu_gyro_3,
                projected_gravity_3,
                lin_vel_cmd_x,
                lin_vel_cmd_y,
                ang_vel_cmd_z,
                last_action_n,
                action_j,
            ],
            axis=-1,
        )

        if HISTORY_LENGTH > 0:
            # Roll the history by shifting the existing history and adding the new data
            carry_reshaped = actor_carry.reshape(HISTORY_LENGTH, SINGLE_STEP_HISTORY_SIZE)
            shifted_history = jnp.roll(carry_reshaped, shift=-1, axis=0)
            new_history = shifted_history.at[HISTORY_LENGTH - 1].set(current_history_data)
            history_array = new_history.reshape(-1)
            next_carry = (history_array, history_array)
        else:
            next_carry = (jnp.zeros(0), jnp.zeros(0))

        return ksim.Action(action=action_j, carry=next_carry, aux_outputs=None)


if __name__ == "__main__":
    # To run training, use the following command:
    # python -m ksim_kbot.standing.standing disable_multiprocessing=True
    # To visualize the environment, use the following command:
    # python -m ksim_kbot.standing.standing run_environment=True \
    #  run_environment_num_seconds=1 \
    #  run_environment_save_path=videos/test.mp4
    KbotStandingTask.launch(
        KbotStandingTaskConfig(
            num_envs=8192,
            batch_size=256,
            num_passes=10,
            epochs_per_log_step=1,
            # Simulation parameters.
            dt=0.002,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            rollout_length_seconds=1.25,
            # PPO parameters
            action_scale=0.5,
            gamma=0.97,
            lam=0.95,
            entropy_coef=0.005,
            learning_rate=1e-4,
            clip_param=0.3,
            max_grad_norm=0.5,
            use_mit_actuators=True,
            save_every_n_steps=25,
            export_for_inference=True,
            domain_randomize=True,
        ),
    )
