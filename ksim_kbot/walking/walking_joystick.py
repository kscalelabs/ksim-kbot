# mypy: ignore-errors
"""Defines simple task for training a standing policy for K-Bot."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, TypeVar

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import ksim
import xax
from jaxtyping import Array, PRNGKeyArray
from kscale.web.gen.api import JointMetadataOutput
from ksim.curriculum import ConstantCurriculum, Curriculum
from xax.nn.export import export

from ksim_kbot import common, rewards as kbot_rewards
from ksim_kbot.standing.standing import MAX_TORQUE, KbotStandingTask, KbotStandingTaskConfig

OBS_SIZE = 20 * 2 + 4 + 3 + 3 + 40  # = position + velocity + phase + imu_acc + imu_gyro + last_action
CMD_SIZE = 2 + 1 + 1
NUM_INPUTS = OBS_SIZE + CMD_SIZE
NUM_OUTPUTS = 20 * 2  # position + velocity
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
        timestep_phase_4: Array,
        joint_pos_n: Array,
        joint_vel_n: Array,
        imu_acc_3: Array,
        imu_gyro_3: Array,
        lin_vel_cmd_2: Array,
        ang_vel_cmd: Array,
        gait_freq_cmd: Array,
        last_action_n: Array,
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
            in_size=NUM_INPUTS + 2 + 6 + 3 + 3 + 4 + 3 + 3 + 20 + 1,
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
class KbotWalkingTaskConfig(KbotStandingTaskConfig):
    """Config for the K-Bot walking task."""

    gait_freq_lower: float = xax.field(value=1.25)
    gait_freq_upper: float = xax.field(value=1.5)
    # to be removed
    log_full_trajectory_every_n_steps: int = xax.field(value=5)
    log_full_trajectory_on_first_step: bool = xax.field(value=False)
    log_full_trajectory_every_n_seconds: float = xax.field(value=1.0)

    stand_still_threshold: float = xax.field(value=0.05)


Config = TypeVar("Config", bound=KbotWalkingTaskConfig)


class KbotWalkingTask(KbotStandingTask[Config], Generic[Config]):
    config: Config

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

    def get_physics_randomizers(self, physics_model: ksim.PhysicsModel) -> list[ksim.PhysicsRandomizer]:
        if self.config.domain_randomize:
            return [
                ksim.StaticFrictionRandomizer(scale_lower=0.9, scale_upper=1.1),
                ksim.ArmatureRandomizer(),
                # ksim.AllBodiesMassMultiplicationRandomizer(),
                ksim.MassAdditionRandomizer.from_body_name(physics_model, "Torso_Side_Right"),
                ksim.JointDampingRandomizer(),
                ksim.JointZeroPositionRandomizer(scale_lower=-0.03, scale_upper=0.03),
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
                    # quat
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                )
                + JOINT_TARGETS
            ),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        if self.config.domain_randomize:
            return [
                common.XYPushEvent(
                    interval_range=(5.0, 10.0),
                    force_range=(0.1, 0.5),
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
                default_targets=JOINT_TARGETS,
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
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="left_foot_force", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="right_foot_force", noise=0.0),
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
            # NOTE: Add collisions to hands
            # ksim.ContactObservation(
            #     physics_model=physics_model,
            #     geom_names=(
            #         "KB_C_501X_Right_Bayonet_Adapter_Hard_Stop",
            #         "RS03_4",
            #     ),
            #     contact_group="right_hand_leg",
            # ),
            # ksim.ContactObservation(
            #     physics_model=physics_model,
            #     geom_names=(
            #         "KB_C_501X_Left_Bayonet_Adapter_Hard_Stop",
            #         "RS03_5",
            #     ),
            #     contact_group="left_hand_leg",
            # ),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        # NOTE: increase to 360
        return [
            common.LinearVelocityCommand(
                x_range=(-0.3, 0.7),
                y_range=(-0.2, 0.2),
                x_zero_prob=1.0,
                y_zero_prob=1.0,
                switch_prob=0.0,
            ),
            common.AngularVelocityCommand(
                scale=0.1,
                zero_prob=0.9,
                switch_prob=0.0,
            ),
            common.GaitFrequencyCommand(
                gait_freq_lower=self.config.gait_freq_lower,
                gait_freq_upper=self.config.gait_freq_upper,
            ),
        ]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        rewards: list[ksim.Reward] = [
            kbot_rewards.JointDeviationPenalty(
                scale=-0.1,
                joint_targets=JOINT_TARGETS,
                joint_weights=(
                    # right arm
                    1.2,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    # left arm
                    1.2,
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
            kbot_rewards.KneeDeviationPenalty.create(
                physics_model=physics_model,
                knee_names=("dof_left_knee_04", "dof_right_knee_04"),
                joint_targets=JOINT_TARGETS,
                scale=-0.1,
            ),
            kbot_rewards.HipDeviationPenalty.create(
                physics_model=physics_model,
                hip_names=(
                    "dof_right_hip_roll_03",
                    "dof_right_hip_yaw_03",
                    "dof_left_hip_roll_03",
                    "dof_left_hip_yaw_03",
                ),
                joint_targets=JOINT_TARGETS,
                scale=-0.25,
            ),
            kbot_rewards.TerminationPenalty(scale=-1.0),
            kbot_rewards.SensorOrientationPenalty(scale=-2.0),
            # kbot_rewards.OrientationPenalty(scale=-2.0),
            kbot_rewards.LinearVelocityTrackingReward(scale=1.0),
            kbot_rewards.AngularVelocityTrackingReward(scale=0.5),
            kbot_rewards.AngularVelocityXYPenalty(scale=-0.15),
            # Stateful rewards
            kbot_rewards.FeetPhaseReward(
                foot_default_height=0.04,
                max_foot_height=0.12,
                scale=2.1,
                stand_still_threshold=self.config.stand_still_threshold,
            ),
            kbot_rewards.FeetSlipPenalty(scale=-0.25),
            # force penalties
            kbot_rewards.JointPositionLimitPenalty.create(
                physics_model=physics_model,
                soft_limit_factor=0.95,
                scale=-1.0,
            ),
            kbot_rewards.ContactForcePenalty(
                scale=-0.01,
                sensor_names=("sensor_observation_left_foot_force", "sensor_observation_right_foot_force"),
            ),
            # NOTE: Investigate the effect of these penalties
            # ksim.ActuatorForcePenalty(scale=-0.005),
            # ksim.ActionSmoothnessPenalty(scale=-0.005),
            # ksim.AvoidLimitsReward(-0.01)
            kbot_rewards.StandStillPenalty(
                scale=-1.0,
                linear_velocity_cmd_name="linear_velocity_command",
                angular_velocity_cmd_name="angular_velocity_command",
                joint_targets=JOINT_TARGETS,
                stand_still_threshold=self.config.stand_still_threshold,
            ),
        ]

        return rewards

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [common.GVecTermination.create(physics_model, sensor_name="upvector_origin")]

    def get_model(self, key: PRNGKeyArray) -> KbotModel:
        return KbotModel(key)

    def get_initial_carry(self, rng: PRNGKeyArray) -> tuple[Array, Array]:
        return None, None

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
        )

    def get_ppo_variables(
        self,
        model: KbotModel,
        trajectories: ksim.Trajectory,
        carry: None,
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, None]:
        # Vectorize over the time dimensions.

        def get_log_prob(transition: ksim.Trajectory) -> Array:
            action_dist_n = self.run_actor(model.actor, transition.obs, transition.command)
            log_probs_n = action_dist_n.log_prob(transition.action / model.actor.mean_scale)
            return log_probs_n

        log_probs_tn = jax.vmap(get_log_prob)(trajectories)

        values_tn = jax.vmap(self.run_critic, in_axes=(None, 0, 0))(
            model.critic, trajectories.obs, trajectories.command
        )

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
        argmax: bool = False,
    ) -> ksim.Action:
        action_dist_j = self.run_actor(
            model.actor,
            observations,
            commands,
        )
        action_j = action_dist_j.mode() if argmax else action_dist_j.sample(seed=rng)
        return ksim.Action(action=action_j, carry=None, aux_outputs=None)

    def get_initial_model_carry(self, rng: PRNGKeyArray) -> None:
        return None

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
    # python -m ksim_kbot.walking.walking_joystick num_envs=2 batch_size=2
    # To run training, use the following command:
    # python -m ksim_kbot.walking.walking_joystick disable_multiprocessing=True
    # To visualize the environment, use the following command:
    # python -m ksim_kbot.walking.walking_joystick run_environment=True \
    #  run_environment_num_seconds=1 \
    #  run_environment_save_path=videos/test.mp4
    KbotWalkingTask.launch(
        KbotWalkingTaskConfig(
            num_envs=8192,
            batch_size=256,
            num_passes=10,
            epochs_per_log_step=1,
            # Simulation parameters.
            dt=0.002,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            min_action_latency=0.0,
            rollout_length_seconds=1.25,
            # PPO parameters
            action_scale=1.0,
            gamma=0.97,
            lam=0.95,
            entropy_coef=0.005,
            learning_rate=1e-4,
            clip_param=0.3,
            max_grad_norm=0.5,
            valid_every_n_steps=25,
            save_every_n_steps=25,
            export_for_inference=True,
            only_save_most_recent=False,
            # Task parameters
            domain_randomize=True,
            gait_freq_lower=1.25,
            gait_freq_upper=1.5,
            reward_clip_min=0.0,
            reward_clip_max=1000.0,
            stand_still_threshold=0.05,
        ),
    )
