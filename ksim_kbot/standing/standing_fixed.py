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
from jaxtyping import Array, PRNGKeyArray
from kscale.web.gen.api import JointMetadataOutput
from mujoco import mjx

from ksim_kbot import common
from ksim_kbot.standing.standing import MAX_TORQUE, KbotStandingTask, KbotStandingTaskConfig

OBS_SIZE = 10 * 2 + 3 + 3 + 3 + 20  # = 38 position + velocity + imu_acc + imu_gyro + last_action
CMD_SIZE = 3
NUM_OUTPUTS = 10 * 2  # position + velocity

SINGLE_STEP_HISTORY_SIZE = NUM_OUTPUTS + OBS_SIZE + CMD_SIZE

HISTORY_LENGTH = 0

NUM_INPUTS = (OBS_SIZE + CMD_SIZE) + SINGLE_STEP_HISTORY_SIZE * HISTORY_LENGTH


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

    def forward(
        self,
        joint_pos_n: Array,
        joint_vel_n: Array,
        imu_acc_3: Array,
        imu_gyro_3: Array,
        projected_gravity_3: Array,
        lin_vel_cmd_x: Array,
        lin_vel_cmd_y: Array,
        ang_vel_cmd_z: Array,
        last_action_n: Array,
        # history_n: Array,
    ) -> distrax.Normal:
        x_n = jnp.concatenate(
            [
                joint_pos_n,
                joint_vel_n,
                imu_acc_3,
                imu_gyro_3,
                projected_gravity_3,
                lin_vel_cmd_x,
                lin_vel_cmd_y,
                ang_vel_cmd_z,
                last_action_n,
                # history_n,
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
            in_size=NUM_INPUTS + 3 + 2 + 3 + 4 + 3 + 3 + 10,
            out_size=1,  # Always output a single critic value.
            width_size=256,
            depth=5,
            key=key,
            activation=jax.nn.relu,
        )

    def forward(
        self,
        joint_pos_n: Array,
        joint_vel_n: Array,
        imu_acc_3: Array,
        imu_gyro_3: Array,
        projected_gravity_3: Array,
        lin_vel_cmd_x: Array,
        lin_vel_cmd_y: Array,
        ang_vel_cmd_z: Array,
        last_action_n: Array,
        # NOTE - bring this back in
        # feet_contact_2: Array,
        # actuator_force_n: Array,
        # base_position_3: Array,
        # base_orientation_4: Array,
        # base_linear_velocity_3: Array,
        # base_angular_velocity_3: Array,
        # history_n: Array,
    ) -> Array:
        x_n = jnp.concatenate(
            [
                joint_pos_n,
                joint_vel_n,
                imu_acc_3,
                imu_gyro_3,
                projected_gravity_3,
                lin_vel_cmd_x,
                lin_vel_cmd_y,
                ang_vel_cmd_z,
                last_action_n,
                # feet_contact_2,
                # actuator_force_n,
                # base_position_3,
                # base_orientation_4,
                # base_linear_velocity_3,
                # base_angular_velocity_3,
                # history_n,
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
class KbotStandingFixedTaskConfig(KbotStandingTaskConfig):
    pass


Config = TypeVar("Config", bound=KbotStandingFixedTaskConfig)


class KbotStandingFixedTask(KbotStandingTask[KbotStandingFixedTaskConfig], Generic[Config]):
    def get_mujoco_model(self) -> mujoco.MjModel:
        mjcf_path = (Path(self.config.robot_urdf_path) / "scene_fixed.mjcf").resolve().as_posix()
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
        if self.config.use_mit_actuators:
            if metadata is None:
                raise ValueError("Metadata is required for MIT actuators")
            return ksim.MITPositionVelocityActuators(
                physics_model,
                metadata,
                pos_action_noise=0.1,
                vel_action_noise=0.1,
                pos_action_noise_type="gaussian",
                vel_action_noise_type="gaussian",
                ctrl_clip=[
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
            )
        else:
            return ksim.TorqueActuators()

    def get_randomization(self, physics_model: ksim.PhysicsModel) -> list[ksim.Randomization]:
        if self.config.domain_randomize:
            return [
                ksim.StaticFrictionRandomization(scale_lower=0.5, scale_upper=2.0),
                ksim.JointZeroPositionRandomization(scale_lower=-0.05, scale_upper=0.05),
                ksim.ArmatureRandomization(scale_lower=1.0, scale_upper=1.05),
                ksim.MassMultiplicationRandomization.from_body_name(physics_model, "Torso_Side_Right"),
                ksim.JointDampingRandomization(scale_lower=0.95, scale_upper=1.05),
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

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
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
            ksim.JointVelocityObservation(noise=0.5),
            ksim.ActuatorForceObservation(),
            common.ProjectedGravityObservation(noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_acc", noise=0.5),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_gyro", noise=0.2),
            ksim.BasePositionObservation(noise=0.0),
            ksim.BaseOrientationObservation(noise=0.0),
            ksim.BaseLinearVelocityObservation(noise=0.0),
            ksim.BaseAngularVelocityObservation(noise=0.0),
            ksim.CenterOfMassVelocityObservation(),
            ksim.FeetContactObservation.create(
                physics_model=physics_model,
                foot_left_geom_names="KB_D_501L_L_LEG_FOOT_collision_box",
                foot_right_geom_names="KB_D_501R_R_LEG_FOOT_collision_box",
                floor_geom_names="floor",
            ),
            ksim.FeetPositionObservation.create(
                physics_model=physics_model,
                foot_left_body_name="KB_D_501L_L_LEG_FOOT",
                foot_right_body_name="KB_D_501R_R_LEG_FOOT",
            ),
            common.LastActionObservation(noise=0.0),
        ]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            common.JointDeviationPenalty(
                scale=-0.3,
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
            common.DHControlPenalty(scale=-0.05),
            common.DHHealthyReward(scale=0.5),
            ksim.ActuatorForcePenalty(scale=-0.01),
            ksim.BaseHeightReward(scale=1.0, height_target=0.9),
            ksim.LinearVelocityPenalty(index="z", scale=-0.01),
            ksim.AngularVelocityPenalty(index="x", scale=-0.01),
            ksim.AngularVelocityPenalty(index="y", scale=-0.01),
            common.FeetSlipPenalty(scale=-0.25),
        ]

    def get_model(self, key: PRNGKeyArray) -> KbotModel:
        return KbotModel(key)

    def get_initial_carry(self, rng: PRNGKeyArray) -> Array:
        return jnp.zeros(HISTORY_LENGTH * SINGLE_STEP_HISTORY_SIZE)


if __name__ == "__main__":
    # To run training, use the following command:
    # python -m ksim_kbot.standing.standing_fixed
    # To visualize the environment, use the following command:
    # python -m ksim_kbot.standing.standing_fixed run_environment=True \
    #  run_environment_num_seconds=1 \
    #  run_environment_save_path=videos/test.mp4
    KbotStandingFixedTask.launch(
        KbotStandingFixedTaskConfig(
            num_envs=8192,
            batch_size=512,
            num_passes=10,
            epochs_per_log_step=1,
            # Simulation parameters.
            dt=0.002,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            min_action_latency=0.0,
            rollout_length_seconds=1.25,
            log_full_trajectory_every_n_steps=5,
            # PPO parameters
            gamma=0.97,
            lam=0.95,
            entropy_coef=0.001,
            learning_rate=1e-4,
            clip_param=0.3,
            max_grad_norm=1.0,
            use_mit_actuators=True,
            export_for_inference=True,
            domain_randomize=True,
        ),
    )
