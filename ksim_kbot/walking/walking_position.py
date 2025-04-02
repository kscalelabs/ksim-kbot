# mypy: disable-error-code="override"
"""Defines simple task for training a standing policy for K-Bot."""

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import ksim
import mujoco
import xax
from jaxtyping import Array, PRNGKeyArray
from kscale.web.gen.api import JointMetadataOutput
from mujoco import mjx

from ksim_kbot import common
from ksim_kbot.walking.walking import KbotWalkingTask, KbotWalkingTaskConfig

OBS_SIZE = 10 + 3 + 3 + 20 + 4  # = 36 position + velocity + imu_acc + imu_gyro + last_action
CMD_SIZE = 2
NUM_OUTPUTS = 10  # position + velocity

SINGLE_STEP_HISTORY_SIZE = NUM_OUTPUTS + OBS_SIZE + CMD_SIZE

HISTORY_LENGTH = 0

NUM_INPUTS = (OBS_SIZE + CMD_SIZE) + SINGLE_STEP_HISTORY_SIZE * HISTORY_LENGTH

MAX_TORQUE = {
    "00": 1.0,
    "02": 30.0,
    "03": 60.0,
    "04": 100.0,
}


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
        """Just use the action as the torque, the simplest actuator model."""
        action = self.default_targets + action * self._action_scale
        return self.add_noise(self.noise, self.noise_type, action, rng)


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

    def __call__(
        self,
        joint_pos_n: Array,
        joint_vel_n: Array,
        imu_acc_3: Array,
        imu_gyro_3: Array,
        lin_vel_cmd_2: Array,
        last_action_n: Array,
        phase_4: Array,
        history_n: Array,
    ) -> distrax.Normal:
        x_n = jnp.concatenate(
            [
                joint_pos_n,
                joint_vel_n,
                imu_acc_3,
                imu_gyro_3,
                lin_vel_cmd_2,
                last_action_n,
                phase_4,
                # history_n,
            ],
            axis=-1,
        )  # (NUM_INPUTS)

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
            in_size=NUM_INPUTS + 3 + 2 + 3 + 4 + 3 + 3 + 10,
            out_size=1,  # Always output a single critic value.
            width_size=256,
            depth=5,
            key=key,
            activation=jax.nn.relu,
        )

    def __call__(
        self,
        joint_pos_n: Array,
        joint_vel_n: Array,
        imu_acc_3: Array,
        imu_gyro_3: Array,
        lin_vel_cmd_2: Array,
        last_action_n: Array,
        projected_gravity_3: Array,
        feet_contact_2: Array,
        actuator_force_n: Array,
        base_position_3: Array,
        base_orientation_4: Array,
        base_linear_velocity_3: Array,
        base_angular_velocity_3: Array,
        phase_4: Array,
        history_n: Array,
    ) -> Array:
        x_n = jnp.concatenate(
            [
                joint_pos_n,
                joint_vel_n,
                imu_acc_3,
                imu_gyro_3,
                lin_vel_cmd_2,
                last_action_n,
                projected_gravity_3,
                feet_contact_2,
                actuator_force_n,
                base_position_3,
                base_orientation_4,
                base_linear_velocity_3,
                base_angular_velocity_3,
                phase_4,
                # history_n,
            ],
            axis=-1,
        )  # (NUM_INPUTS)
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
class KbotWalkingPositionTaskConfig(KbotWalkingTaskConfig):
    """Config for the K-Bot walking task."""

    use_gait_rewards: bool = xax.field(value=False)


Config = TypeVar("Config", bound=KbotWalkingTaskConfig)


class KbotWalkingPositionTask(KbotWalkingTask[Config], Generic[Config]):
    def get_mujoco_model(self) -> mujoco.MjModel:
        mjcf_path = (Path(self.config.robot_urdf_path) / "scene_fixed_position.mjcf").resolve().as_posix()
        mj_model = mujoco.MjModel.from_xml_path(mjcf_path)

        mj_model.opt.timestep = jnp.array(self.config.dt)
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6
        mj_model.opt.disableflags = mjx.DisableBit.EULERDAMP
        mj_model.opt.solver = mjx.SolverType.CG

        return mj_model

    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: dict[str, JointMetadataOutput] | None = None,
    ) -> ksim.Actuators:
        if self.config.domain_randomize:
            noise = 0.1
        else:
            noise = 0.0
        return ScaledTorqueActuators(
            default_targets=jnp.array(
                [
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
                ]
            ),
            action_scale=0.5,
            noise=noise,
            noise_type="gaussian",
        )

    def get_randomization(self, physics_model: ksim.PhysicsModel) -> list[ksim.Randomization]:
        if self.config.domain_randomize:
            return [
                ksim.StaticFrictionRandomization(scale_lower=0.5, scale_upper=2.0),
                ksim.JointZeroPositionRandomization(scale_lower=-0.05, scale_upper=0.05),
                ksim.ArmatureRandomization(scale_lower=1.0, scale_upper=1.05),
                ksim.MassMultiplicationRandomization.from_body_name(physics_model, "Torso_Side_Right"),
                ksim.JointDampingRandomization(scale_lower=0.95, scale_upper=1.05),
                # TODO: Add this back in.
                # ksim.FloorFrictionRandomization.from_body_name(
                #     model=physics_model,
                #     scale_lower=0.2,
                #     scale_upper=0.6,
                #     floor_body_name="floor",
                # ),
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
                    1.01,
                    1.0,
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
                    x_force=0.2,
                    y_force=0.2,
                    z_force=0.0,
                    interval_range=(1.0, 2.0),
                ),
            ]
        else:
            return []

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        if self.config.domain_randomize:
            imu_acc_noise = 0.5
            imu_gyro_noise = 0.2
            gvec_noise = 0.0
            base_position_noise = 0.0
            base_orientation_noise = 0.0
            base_linear_velocity_noise = 0.0
            base_angular_velocity_noise = 0.0
            base_angular_velocity_noise = 0.0
            noise = 0.0
        else:
            imu_acc_noise = 0.0
            imu_gyro_noise = 0.0
            gvec_noise = 0.0
            base_position_noise = 0.0
            base_orientation_noise = 0.0
            base_linear_velocity_noise = 0.0
            base_angular_velocity_noise = 0.0
            noise = 0.0

        return [
            common.JointPositionObservation(
                default_targets=(
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
            ksim.JointVelocityObservation(noise=imu_gyro_noise),
            ksim.ActuatorForceObservation(),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_acc", noise=imu_acc_noise),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_gyro", noise=imu_gyro_noise),
            ksim.BasePositionObservation(noise=base_position_noise),
            ksim.BaseOrientationObservation(noise=base_orientation_noise),
            ksim.BaseLinearVelocityObservation(noise=base_linear_velocity_noise),
            ksim.BaseAngularVelocityObservation(noise=base_angular_velocity_noise),
            ksim.CenterOfMassVelocityObservation(),
            ksim.FeetContactObservation.create(
                physics_model=physics_model,
                foot_left_geom_names="KB_D_501L_L_LEG_FOOT_collision_box",
                foot_right_geom_names="KB_D_501R_R_LEG_FOOT_collision_box",
                floor_geom_names="floor",
            ),
            # Bring back ksim.FeetPositionObservation
            common.FeetPositionObservation.create(
                physics_model=physics_model,
                foot_left_geom_name="KB_D_501L_L_LEG_FOOT_collision_box",
                foot_right_geom_name="KB_D_501R_R_LEG_FOOT_collision_box",
                floor_threshold=0.04,
            ),
            common.PhaseObservation(),
            common.LastActionObservation(),
            common.ProjectedGravityObservation(noise=gvec_noise),
            common.HistoryObservation(),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="local_linvel_torso", noise=noise),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="global_linvel_torso", noise=noise),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="global_angvel_torso", noise=noise),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="upvector_torso", noise=noise),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="orientation_torso", noise=noise),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="gyro_torso", noise=noise),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return [
            ksim.LinearVelocityCommand(
                x_range=(0.2, 0.2),
                y_range=(0.0, 0.0),
                x_zero_prob=0.0,
                y_zero_prob=1.0,
            ),
            ksim.AngularVelocityCommand(
                scale=0.0,
                zero_prob=1.0,
            ),
        ]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        rewards = [
            common.JointDeviationPenalty(
                scale=-0.1,
                joint_targets=(
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
            ),
            common.TerminationPenalty(scale=-1.0),
            common.OrientationPenalty(scale=-1.0),
            common.HipDeviationPenalty.create(
                physics_model=physics_model,
                hip_names=(
                    "dof_right_hip_roll_03",
                    "dof_right_hip_yaw_03",
                    "dof_left_hip_roll_03",
                    "dof_left_hip_yaw_03",
                ),
                joint_targets=(
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
                scale=-0.25,
            ),
            common.KneeDeviationPenalty.create(
                physics_model=physics_model,
                knee_names=("dof_left_knee_04", "dof_right_knee_04"),
                joint_targets=(
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
                scale=-0.1,
            ),
            common.LinearVelocityTrackingReward(scale=1.0),
            common.AngularVelocityTrackingReward(scale=0.75),
            common.AngularVelocityXYPenalty(scale=-0.15),
            # TODO: Add this back in.
            # AvoidLimitsReward(scale=0.1),
            # Either termination or healthy reward.
            # ksim.HealthyReward(scale=0.25),
        ]
        if self.config.use_gait_rewards:
            gait_rewards = [
                common.FeetSlipPenalty(scale=-0.25),
                common.FeetAirTimeReward(scale=2.0),
                common.FeetPhaseReward(scale=1.0),
            ]
            rewards += gait_rewards

        return rewards

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            common.GVecTermination.create(physics_model, sensor_name="upvector_torso"),
        ]

    def get_model(self, key: PRNGKeyArray) -> KbotModel:
        return KbotModel(key)

    def get_initial_carry(self, rng: PRNGKeyArray) -> Array:
        return jnp.zeros(HISTORY_LENGTH * SINGLE_STEP_HISTORY_SIZE)

    def _run_actor(
        self,
        model: KbotModel,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
    ) -> distrax.Normal:
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        lin_vel_cmd_2 = commands["linear_velocity_command"]
        last_action_n = observations["last_action_observation"]
        phase_4 = observations["phase_observation"]
        history_n = observations["history_observation"]
        return model.actor(
            joint_pos_n, joint_vel_n, imu_acc_3, imu_gyro_3, lin_vel_cmd_2, last_action_n, phase_4, history_n
        )

    def _run_critic(
        self,
        model: KbotModel,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
    ) -> Array:
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        lin_vel_cmd_2 = commands["linear_velocity_command"]
        last_action_n = observations["last_action_observation"]
        phase_4 = observations["phase_observation"]
        projected_gravity_3 = observations["projected_gravity_observation"]
        feet_contact_2 = observations["feet_contact_observation"]
        base_position_3 = observations["base_position_observation"]
        base_orientation_4 = observations["base_orientation_observation"]
        base_linear_velocity_3 = observations["base_linear_velocity_observation"]
        base_angular_velocity_3 = observations["base_angular_velocity_observation"]
        actuator_force_n = observations["actuator_force_observation"]
        history_n = observations["history_observation"]
        return model.critic(
            joint_pos_n,
            joint_vel_n,
            imu_acc_3,
            imu_gyro_3,
            lin_vel_cmd_2,
            last_action_n,
            projected_gravity_3,
            feet_contact_2,
            base_position_3,
            base_orientation_4,
            base_linear_velocity_3,
            base_angular_velocity_3,
            phase_4,
            actuator_force_n,
            history_n,
        )


if __name__ == "__main__":
    # To run training, use the following command:
    # python -m ksim_kbot.walking.walking_posirion num_envs=1 batch_size=1 rollout_length_seconds=1.0
    # To visualize the environment, use the following command:
    # python -m ksim_kbot.walking.walking_position \
    #  run_environment=True \
    #  run_environment_num_seconds=1 \
    #  run_environment_save_path=videos/test.mp4
    KbotWalkingPositionTask.launch(
        KbotWalkingPositionTaskConfig(
            num_envs=1024,
            batch_size=512,
            num_passes=10,
            epochs_per_log_step=1,
            # Simulation parameters.
            dt=0.002,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            min_action_latency=0.0,
            valid_every_n_steps=25,
            valid_first_n_steps=0,
            save_every_n_steps=25,
            rollout_length_seconds=5.0,
            # PPO parameters
            gamma=0.97,
            lam=0.95,
            entropy_coef=0.005,
            learning_rate=1e-4,
            clip_param=0.3,
            max_grad_norm=1.0,
            use_mit_actuators=False,
            export_for_inference=True,
            domain_randomize=False,
            use_gait_rewards=True,
        ),
    )
