# mypy: ignore-errors
"""Defines simple task for training a walking + pseudo ik policy for K-Bot."""

import os
import uuid
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
from ksim.curriculum import Curriculum
from mujoco import mjx

from ksim_kbot import common
from ksim_kbot.misc_tasks.pseudo_ik import BodyPositionObservation, CartesianBodyTargetVectorReward
from ksim_kbot.walking.walking_joystick import (
    JOINT_TARGETS,
    NUM_CRITIC_INPUTS,
    NUM_INPUTS,
    NUM_OUTPUTS,
    KbotWalkingTask,
    KbotWalkingTaskConfig,
)

EXTRA_INPUTS = 3
EXTRA_CRITIC_INPUTS = 3


class KbotActor(eqx.Module):
    """Actor for the walking + pseudo ik task."""

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
            in_size=NUM_INPUTS + EXTRA_INPUTS,
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
        xyz_target_3: Array,
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
                xyz_target_3,
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
    """Critic for the walking + pseudo ik task."""

    mlp: eqx.nn.MLP

    def __init__(self, key: PRNGKeyArray) -> None:
        self.mlp = eqx.nn.MLP(
            in_size=NUM_CRITIC_INPUTS + EXTRA_INPUTS + EXTRA_CRITIC_INPUTS,
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
        xyz_target_3: Array,
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
        end_effector_pos_3: Array,
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
                xyz_target_3,
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
                end_effector_pos_3,
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
class KbotWalkingPseudoIKTaskConfig(KbotWalkingTaskConfig):
    pass


Config = TypeVar("Config", bound=KbotWalkingTaskConfig)


class KbotWalkingPseudoIKTask(KbotWalkingTask[Config], Generic[Config]):
    config: Config

    def get_mujoco_model(self) -> tuple[mujoco.MjModel, dict[str, JointMetadataOutput]]:
        mjcf_path = (Path(self.config.robot_urdf_path) / "robot_scene.mjcf").resolve().as_posix()

        # add body
        mj_model_added_body = ksim.add_new_mujoco_body(
            mjcf_path,
            parent_body_name="KB_C_501X_Right_Bayonet_Adapter_Hard_Stop",
            new_body_name="ik_target",
            pos=(0.0, 0.0, -0.1),
            quat=(1.0, 0.0, 0.0, 0.0),
            add_visual=True,
            visual_geom_color=(0, 1, 0, 1),
            visual_geom_size=(0.03, 0.03, 0.03),
        )

        temp_path = (
            (Path(self.config.robot_urdf_path) / f"robot_scene_added_body_{uuid.uuid4()}.mjcf").resolve().as_posix()
        )

        with open(temp_path, "w") as f:
            f.write(mj_model_added_body)

        mj_model = mujoco.MjModel.from_xml_path(temp_path)

        # remove the temp file
        os.remove(temp_path)
        mj_model.opt.timestep = jnp.array(self.config.dt)
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6
        mj_model.opt.disableflags = mjx.DisableBit.EULERDAMP
        mj_model.opt.solver = mjx.SolverType.CG

        return mj_model

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
        xyz_target_3 = commands["target_position_command"][..., :3]
        last_action_n = observations["last_action_observation"]
        return model.forward(
            timestep_phase_4=timestep_phase_4,
            joint_pos_n=joint_pos_n,
            joint_vel_n=joint_vel_n,
            imu_acc_3=imu_acc_3,
            imu_gyro_3=imu_gyro_3,
            lin_vel_cmd_2=lin_vel_cmd_2,
            ang_vel_cmd=ang_vel_cmd,
            xyz_target_3=xyz_target_3,
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
        xyz_target_3 = commands["target_position_command"][..., :3]
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
        end_effector_pos_3 = observations["ik_target_body_position_observation"]

        return model.forward(
            timestep_phase_4=timestep_phase_4,
            joint_pos_n=joint_pos_n,
            joint_vel_n=joint_vel_n,
            imu_acc_3=imu_acc_3,
            imu_gyro_3=imu_gyro_3,
            lin_vel_cmd_2=lin_vel_cmd_2,
            ang_vel_cmd=ang_vel_cmd,
            xyz_target_3=xyz_target_3,
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
            end_effector_pos_3=end_effector_pos_3,
        )

    def get_model(self, key: PRNGKeyArray) -> KbotModel:
        return KbotModel(key)

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> Curriculum:
        return ksim.EpisodeLengthCurriculum(
            num_levels=10,
            increase_threshold=20.0,
            decrease_threshold=10.0,
            min_level_steps=5,
            dt=self.config.ctrl_dt,  # not sure what this is for
        )

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        if self.config.domain_randomize:
            vel_obs_noise = 0.0
            imu_acc_noise = 0.5
            imu_gyro_noise = 0.5
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
            BodyPositionObservation.create(
                physics_model=physics_model,
                body_name="ik_target",
            ),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        # NOTE: increase to 360
        return [
            common.LinearVelocityCommand(
                x_range=(-0.3, 0.7),
                y_range=(-0.2, 0.2),
                x_zero_prob=0.1,
                y_zero_prob=0.2,
                switch_prob=self.config.ctrl_dt / 3,
            ),
            common.AngularVelocityCommand(
                scale=0.1,
                zero_prob=0.9,
                switch_prob=self.config.ctrl_dt / 3,
            ),
            common.GaitFrequencyCommand(
                gait_freq_lower=self.config.gait_freq_lower,
                gait_freq_upper=self.config.gait_freq_upper,
            ),
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
        ]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        rewards = super().get_rewards(physics_model)
        rewards.extend(
            [
                ksim.PositionTrackingReward.create(
                    model=physics_model,
                    tracked_body_name="ik_target",
                    base_body_name="floating_base_link",
                    scale=10.0,
                    command_name="target_position_command",
                ),
                CartesianBodyTargetVectorReward.create(
                    model=physics_model,
                    command_name="target_position_command",
                    tracked_body_name="ik_target",
                    base_body_name="floating_base_link",
                    scale=3.0,
                    normalize_velocity=True,
                    distance_threshold=0.1,
                    dt=self.config.dt,
                ),
            ]
        )

        return rewards

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [common.GVecTermination.create(physics_model, sensor_name="upvector_origin", min_z=0.1)]

    def get_initial_carry(self, rng: PRNGKeyArray) -> tuple[Array, Array]:
        return None, None

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


if __name__ == "__main__":
    # python -m ksim_kbot.walking.walking_joystick num_envs=2 batch_size=2
    # To run training, use the following command:
    # python -m ksim_kbot.misc_tasks.walking_pseudo_ik.py disable_multiprocessing=True
    # To visualize the environment, use the following command:
    # python -m ksim_kbot.misc_tasks.walking_pseudo_ik.py run_environment=True \
    #  run_environment_num_seconds=1 \
    #  run_environment_save_path=videos/test.mp4
    KbotWalkingPseudoIKTask.launch(
        KbotWalkingPseudoIKTaskConfig(
            num_envs=8192,
            batch_size=256,
            num_passes=10,
            epochs_per_log_step=1,
            # Simulation parameters.
            dt=0.002,
            ctrl_dt=0.02,
            max_action_latency=0.005,
            min_action_latency=0.0,
            rollout_length_seconds=5.0,
            render_length_seconds=5.0,
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
            stand_still=False,
        ),
    )
