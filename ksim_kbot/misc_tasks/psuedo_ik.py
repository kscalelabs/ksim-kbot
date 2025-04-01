"""Pseudo-Inverse Kinematics task for the default humanoid."""

import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Generic, Literal, TypeVar

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
import optax
import xax
from jaxtyping import Array, PRNGKeyArray
from kscale.web.gen.api import JointMetadataOutput
from mujoco import mjx

import ksim
from ksim.utils.mujoco import remove_joints_except

NoiseType = Literal["none", "uniform", "gaussian"]


NUM_JOINTS = 5  # disabling all DoFs except for the right arm.

NUM_INPUTS = NUM_JOINTS + NUM_JOINTS + 3 + 3
NUM_OUTPUTS = NUM_JOINTS * 2

MAX_TORQUE = {
    "00": 1.0,
    "02": 14.0,
    "03": 40.0,
    "04": 60.0,
}


class ModifiedMITPositionVelocityActuators(ksim.MITPositionActuators):
    """MIT-mode actuator controller operating on both position and velocity."""

    def __init__(
        self,
        physics_model: ksim.PhysicsModel,
        joint_name_to_metadata: dict[str, JointMetadataOutput],
        pos_action_noise: float = 0.0,
        pos_action_noise_type: NoiseType = "none",
        vel_action_noise: float = 0.0,
        vel_action_noise_type: NoiseType = "none",
        torque_noise: float = 0.0,
        torque_noise_type: NoiseType = "none",
        ctrl_clip: list[float] | None = None,
    ) -> None:
        super().__init__(
            physics_model=physics_model,
            joint_name_to_metadata=joint_name_to_metadata,
            action_noise=pos_action_noise,
            action_noise_type=pos_action_noise_type,
            torque_noise=torque_noise,
            torque_noise_type=torque_noise_type,
            ctrl_clip=ctrl_clip,
        )

        self.vel_action_noise = vel_action_noise
        self.vel_action_noise_type = vel_action_noise_type

    def get_ctrl(self, action: Array, physics_data: ksim.PhysicsData, rng: PRNGKeyArray) -> Array:
        """Get the control signal from the (position and velocity) action vector."""
        pos_rng, vel_rng, tor_rng = jax.random.split(rng, 3)
        current_pos = physics_data.qpos[:]
        current_vel = physics_data.qvel[:]

        # Adds position and velocity noise.
        target_position = action[: len(current_pos)]
        target_velocity = action[len(current_pos) :]
        target_position = self.add_noise(self.action_noise, self.action_noise_type, target_position, pos_rng)
        target_velocity = self.add_noise(self.vel_action_noise, self.vel_action_noise_type, target_velocity, vel_rng)

        pos_delta = target_position - current_pos
        vel_delta = target_velocity - current_vel

        ctrl = self.kps * pos_delta + self.kds * vel_delta
        return jnp.clip(
            self.add_noise(self.torque_noise, self.torque_noise_type, ctrl, tor_rng),
            -self.ctrl_clip,
            self.ctrl_clip,
        )

    def get_default_action(self, physics_data: ksim.PhysicsData) -> Array:
        """Get the default action (zeros) with the correct shape."""
        qpos_dim = len(physics_data.qpos)
        return jnp.zeros(qpos_dim * 2)

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
            out_size=NUM_OUTPUTS,
            width_size=64,
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
        xyz_target_3: Array,
    ) -> distrax.Normal:
        obs_n = jnp.concatenate(
            [
                joint_pos_n,  # NUM_JOINTS
                joint_vel_n,  # NUM_JOINTS
                imu_acc_3,  # 3
                xyz_target_3,  # 3
            ],
            axis=-1,
        )

        prediction_n = self.mlp(obs_n)
        mean_n = prediction_n[..., :NUM_JOINTS]
        std_n = prediction_n[..., NUM_JOINTS:]

        # Scale the mean.
        mean_n = jnp.tanh(mean_n) * self.mean_scale

        # Softplus and clip to ensure positive standard deviations.
        std_n = jnp.clip((jax.nn.softplus(std_n) + self.min_std) * self.var_scale, max=self.max_std)

        return distrax.Normal(mean_n, std_n)


class KbotCritic(eqx.Module):
    """Critic for the walking task."""

    mlp: eqx.nn.MLP

    def __init__(self, key: PRNGKeyArray) -> None:
        num_inputs = NUM_INPUTS + NUM_JOINTS
        num_outputs = 1

        self.mlp = eqx.nn.MLP(
            in_size=num_inputs,
            out_size=num_outputs,
            width_size=64,
            depth=5,
            key=key,
            activation=jax.nn.relu,
        )

    def __call__(
        self,
        joint_pos_n: Array,
        joint_vel_n: Array,
        actuator_force_n: Array,
        imu_acc_3: Array,
        xyz_target_3: Array,
    ) -> Array:
        x_n = jnp.concatenate(
            [
                joint_pos_n,  # NUM_JOINTS
                joint_vel_n,  # NUM_JOINTS
                actuator_force_n,  # NUM_JOINTS
                imu_acc_3,  # 3
                xyz_target_3,  # 3
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
    kp: float = xax.field(
        value=1.0,
        help="The Kp for the actuators",
    )
    kd: float = xax.field(
        value=0.1,
        help="The Kd for the actuators",
    )
    armature: float = xax.field(
        value=1e-2,
        help="A value representing the effective inertia of the actuator armature",
    )
    friction: float = xax.field(
        value=1e-6,
        help="The dynamic friction loss for the actuator",
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
        mjcf_path = (Path(self.config.robot_urdf_path) / "robot_scene.mjcf").resolve().as_posix()
        mj_model_joint_removed = remove_joints_except(mjcf_path, ["dof_right_shoulder_pitch_03", "dof_right_shoulder_roll_03", "dof_right_shoulder_yaw_02", "dof_right_elbow_02", "dof_right_wrist_00"])
        # save to a temp file in the same directory
        temp_path = (Path(self.config.robot_urdf_path) / "robot_scene_joint_removed.mjcf").resolve().as_posix()
        with open(temp_path, "w") as f:
            f.write(mj_model_joint_removed)

        mj_model = mujoco.MjModel.from_xml_path(temp_path)

        # remove the temp file
        os.remove(temp_path)

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
            return ModifiedMITPositionVelocityActuators(
                physics_model,
                metadata,
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
                ],
            )
        else:
            return ksim.TorqueActuators()

    def get_randomization(self, physics_model: ksim.PhysicsModel) -> list[ksim.Randomization]:
        return [
            # ksim.StaticFrictionRandomization(),
            # ksim.ArmatureRandomization(),
            # ksim.MassMultiplicationRandomization.from_body_name(physics_model, "upper_arm_right"),
            # ksim.JointDampingRandomization(),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        return []

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomJointPositionReset(),
            ksim.RandomJointVelocityReset(),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        return [
            ksim.JointPositionObservation(freejoint_first=False),
            ksim.JointVelocityObservation(freejoint_first=False),
            ksim.ActuatorForceObservation(),
            # ksim.ActuatorAccelerationObservation(freejoint_first=False),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_acc"),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return [
            ksim.CartesianBodyTargetCommand.create(
                model=physics_model,
                command_name="cartesian_body_target_command_KC_C_104R_PitchHardstopDriven",
                pivot_name="KC_C_104R_PitchHardstopDriven",
                base_name="floating_base_link",
                sample_sphere_radius=0.5,
                positive_x=True,  # only sample in the positive x direction
                positive_y=False,
                positive_z=False,
                switch_prob=self.config.ctrl_dt / 1,  # will last 1 seconds in expectation
                vis_radius=0.05,
                vis_color=(1.0, 0.0, 0.0, 0.8),
            ),
        ]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            ksim.CartesianBodyTargetReward.create(
                model=physics_model,
                tracked_body_name="KB_C_501X_Right_Bayonet_Adapter_Hard_Stop",
                base_body_name="floating_base_link",
                norm="l2",
                scale=0.1,
                sensitivity=1.0,
                command_name="cartesian_body_target_command_KC_C_104R_PitchHardstopDriven",
            ),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            ksim.FastAccelerationTermination(),
            # TODO: add for collisions
        ]

    def get_model(self, key: PRNGKeyArray) -> KbotModel:
        return KbotModel(key)

    def get_initial_carry(self, rng: PRNGKeyArray) -> None:
        return None

    def _run_actor(
        self,
        model: KbotModel,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
    ) -> distrax.Normal:
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"] / 50.0
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        xyz_target_3 = commands["cartesian_body_target_command_KC_C_104R_PitchHardstopDriven"]
        return model.actor(
            joint_pos_n=joint_pos_n,
            joint_vel_n=joint_vel_n,
            imu_acc_3=imu_acc_3,
            xyz_target_3=xyz_target_3,
        )

    def _run_critic(
        self,
        model: KbotModel,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
    ) -> Array:
        joint_pos_n = observations["joint_position_observation"]  # 26
        joint_vel_n = observations["joint_velocity_observation"] / 100.0  # 27
        actuator_force_n = observations["actuator_force_observation"]  # 27
        imu_acc_3 = observations["sensor_observation_imu_acc"]  # 3
        xyz_target_3 = commands["cartesian_body_target_command_KC_C_104R_PitchHardstopDriven"]  # 3
        return model.critic(
            joint_pos_n=joint_pos_n,
            joint_vel_n=joint_vel_n,
            actuator_force_n=actuator_force_n,
            imu_acc_3=imu_acc_3,
            xyz_target_3=xyz_target_3,
        )

    def get_on_policy_log_probs(
        self,
        model: KbotModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> Array:
        if not isinstance(trajectories.aux_outputs, AuxOutputs):
            raise ValueError("No aux outputs found in trajectories")
        return trajectories.aux_outputs.log_probs

    def get_on_policy_values(
        self,
        model: KbotModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> Array:
        if not isinstance(trajectories.aux_outputs, AuxOutputs):
            raise ValueError("No aux outputs found in trajectories")
        return trajectories.aux_outputs.values

    def get_log_probs(
        self,
        model: KbotModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> tuple[Array, Array]:
        # Vectorize over both batch and time dimensions.
        par_fn = jax.vmap(self._run_actor, in_axes=(None, 0, 0))
        action_dist_btn = par_fn(model, trajectories.obs, trajectories.command)

        # Compute the log probabilities of the trajectory's actions according
        # to the current policy, along with the entropy of the distribution.
        action_btn = trajectories.action / model.actor.mean_scale
        log_probs_btn = action_dist_btn.log_prob(action_btn)
        entropy_btn = action_dist_btn.entropy()

        return log_probs_btn, entropy_btn

    def get_values(
        self,
        model: KbotModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> Array:
        # Vectorize over both batch and time dimensions.
        par_fn = jax.vmap(self._run_critic, in_axes=(None, 0, 0))
        values_bt1 = par_fn(model, trajectories.obs, trajectories.command)

        # Remove the last dimension.
        return values_bt1.squeeze(-1)

    def sample_action(
        self,
        model: KbotModel,
        carry: None,
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
    ) -> tuple[Array, None, AuxOutputs]:
        action_dist_n = self._run_actor(model, observations, commands)
        action_n = action_dist_n.sample(seed=rng)
        action_log_prob_n = action_dist_n.log_prob(action_n)

        critic_n = self._run_critic(model, observations, commands)
        value_n = critic_n.squeeze(-1)

        return action_n, None, AuxOutputs(log_probs=action_log_prob_n, values=value_n)


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
            num_envs=2048,
            batch_size=256,
            num_passes=10,
            epochs_per_log_step=1,
            # Logging parameters.
            log_full_trajectory_every_n_seconds=60,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            min_action_latency=0.0,
            rollout_length_seconds=4.0,
        ),
    )
