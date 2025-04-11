# mypy: disable-error-code="override"
"""Defines simple task for training a walking policy for the K-Bot using an RNN actor."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import ksim
import xax
from jaxtyping import Array, PRNGKeyArray
from kscale.web.gen.api import JointMetadataOutput
from ksim.curriculum import ConstantCurriculum, Curriculum

from ksim_kbot import common, rewards as kbot_rewards
from ksim_kbot.standing.standing import MAX_TORQUE
from ksim_kbot.walking.walking_joystick import KbotWalkingTask, KbotWalkingTaskConfig

# Define Input/Output Sizes based on KbotWalkingTask
# Actor Inputs: phase(4) + pos(20) + vel(20) + imu_acc(3) + imu_gyro(3) + lin_cmd(2) + ang_cmd(1) + freq_cmd(1) + last_action(40)
NUM_ACTOR_INPUTS = 4 + 20 + 20 + 3 + 3 + 2 + 1 + 1 + 40
# Critic Inputs: actor_inputs(94) + proj_grav(3) + feet_contact(2) + base_pos(3) + base_orient(4) + base_lin_vel(3) + base_ang_vel(3) + act_force(20)
# Note: Critic inputs from KbotWalkingTask forward method used here.
NUM_CRITIC_INPUTS = NUM_ACTOR_INPUTS + 3 + 2 + 3 + 4 + 3 + 3 + 20
# Action Outputs: pos_target(20) + vel_target(20)
NUM_ACTION_OUTPUTS = 20 + 20


class RnnActor(eqx.Module):
    """RNN-based actor for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear
    num_outputs: int = eqx.static_field()
    min_std: float = eqx.static_field()
    max_std: float = eqx.static_field()
    var_scale: float = eqx.static_field()
    mean_scale: float = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        num_outputs: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        mean_scale: float,
        hidden_size: int,
        depth: int,
    ) -> None:
        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for _ in range(depth)
            ]
        )

        # Project to output
        key, output_proj_key = jax.random.split(key)
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs * 2,
            key=output_proj_key,
        )

        self.num_outputs = num_outputs
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale
        self.mean_scale = mean_scale

    def forward(self, obs_n: Array, carry: Array) -> tuple[distrax.Distribution, Array]:
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            current_carry = carry[i]
            x_n = rnn(x_n, current_carry)
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

        # Converts the output to a distribution.
        mean_n = out_n[..., : self.num_outputs]
        std_n = out_n[..., self.num_outputs :]

        # Scale the mean
        mean_n = jnp.tanh(mean_n) * self.mean_scale

        # Softplus and clip to ensure positive standard deviations.
        std_n = jnp.clip((jax.nn.softplus(std_n) + self.min_std) * self.var_scale, max=self.max_std)

        dist_n = distrax.Normal(mean_n, std_n)
        return dist_n, jnp.stack(out_carries, axis=0)


class RnnCritic(eqx.Module):
    """RNN-based critic for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        hidden_size: int,
        depth: int,
    ) -> None:
        num_outputs = 1

        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for _ in range(depth)
            ]
        )

        # Project to output
        key, output_proj_key = jax.random.split(key)
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs,
            key=output_proj_key,
        )

    def forward(self, obs_n: Array, carry: Array) -> tuple[Array, Array]:
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            current_carry = carry[i]
            x_n = rnn(x_n, current_carry)
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

        return out_n, jnp.stack(out_carries, axis=0)


class RnnModel(eqx.Module):
    actor: RnnActor
    critic: RnnCritic

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        actor_num_inputs: int,
        critic_num_inputs: int,
        num_action_outputs: int,
        min_std: float,
        max_std: float,
        mean_scale: float,
        hidden_size: int,
        depth: int,
    ) -> None:
        actor_key, critic_key = jax.random.split(key)
        self.actor = RnnActor(
            actor_key,
            num_inputs=actor_num_inputs,
            num_outputs=num_action_outputs,
            min_std=min_std,
            max_std=max_std,
            var_scale=0.5,
            mean_scale=mean_scale,
            hidden_size=hidden_size,
            depth=depth,
        )
        self.critic = RnnCritic(
            critic_key,
            num_inputs=critic_num_inputs,
            hidden_size=hidden_size,
            depth=depth,
        )


@dataclass
class WalkingRnnTaskConfig(KbotWalkingTaskConfig):
    """Config for the K-Bot RNN walking task."""
    hidden_size: int = xax.field(value=256, help="Hidden size for RNN layers.")
    depth: int = xax.field(value=2, help="Number of RNN layers.")
    actor_mean_scale: float = xax.field(value=1.0, help="Scaling factor for actor mean output.")


Config = TypeVar("Config", bound=WalkingRnnTaskConfig)


class KbotWalkingRnnTask(KbotWalkingTask[Config], Generic[Config]):
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
            default_targets=(
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                -0.23, 0.0, 0.0, -0.441, 0.195,
                0.23, 0.0, 0.0, 0.441, -0.195,
            ),
            pos_action_noise=0.1,
            vel_action_noise=0.1,
            pos_action_noise_type="gaussian",
            vel_action_noise_type="gaussian",
            ctrl_clip=[
                MAX_TORQUE["03"], MAX_TORQUE["03"], MAX_TORQUE["02"], MAX_TORQUE["02"], MAX_TORQUE["00"],
                MAX_TORQUE["03"], MAX_TORQUE["03"], MAX_TORQUE["02"], MAX_TORQUE["02"], MAX_TORQUE["00"],
                MAX_TORQUE["04"], MAX_TORQUE["03"], MAX_TORQUE["03"], MAX_TORQUE["04"], MAX_TORQUE["02"],
                MAX_TORQUE["04"], MAX_TORQUE["03"], MAX_TORQUE["03"], MAX_TORQUE["04"], MAX_TORQUE["02"],
            ],
            action_scale=self.config.action_scale,
        )

    def get_physics_randomizers(self, physics_model: ksim.PhysicsModel) -> list[ksim.PhysicsRandomizer]:
        if self.config.domain_randomize:
            return [
                ksim.StaticFrictionRandomizer(),
                ksim.ArmatureRandomizer(),
                ksim.MassMultiplicationRandomizer.from_body_name(physics_model, "Torso_Side_Right"),
                ksim.JointDampingRandomizer(),
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
                    0.0, 0.0, 1.01,
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0,
                    -0.23, 0.0, 0.0, -0.441, 0.195,
                    0.23, 0.0, 0.0, 0.441, -0.195,
                )
            ),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        if self.config.domain_randomize:
            return [
                ksim.PushEvent(
                    x_force=0.8, y_force=0.8, z_force=0.0,
                    x_angular_force=0.0, y_angular_force=0.0, z_angular_force=0.0,
                    interval_range=(5.0, 10.0),
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
        else:
            vel_obs_noise = 0.0
            imu_acc_noise = 0.0
            imu_gyro_noise = 0.0
            gvec_noise = 0.0
            base_position_noise = 0.0
            base_orientation_noise = 0.0
            base_linear_velocity_noise = 0.0
            base_angular_velocity_noise = 0.0
        return [
            common.TimestepPhaseObservation(),
            common.JointPositionObservation(
                default_targets=(
                    0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0,
                    -0.23, 0.0, 0.0, -0.441, 0.195,
                    0.23, 0.0, 0.0, 0.441, -0.195,
                ),
                noise=0.01,
            ),
            ksim.JointVelocityObservation(noise=vel_obs_noise),
            ksim.ActuatorForceObservation(),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_acc", noise=imu_acc_noise),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_gyro", noise=imu_gyro_noise),
            common.ProjectedGravityObservation(noise=gvec_noise),
            common.LastActionObservation(noise=0.0),
            ksim.BasePositionObservation(noise=base_position_noise),
            ksim.BaseOrientationObservation(noise=base_orientation_noise),
            ksim.BaseLinearVelocityObservation(noise=base_linear_velocity_noise),
            ksim.BaseAngularVelocityObservation(noise=base_angular_velocity_noise),
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
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="upvector_origin", noise=0.0),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return [
            common.LinearVelocityCommand(
                x_range=(-0.7, 0.7), y_range=(-0.2, 0.2), x_zero_prob=0.1, y_zero_prob=0.2, switch_prob=0.0
            ),
            common.AngularVelocityCommand(scale=0.1, zero_prob=0.9, switch_prob=0.0),
            common.GaitFrequencyCommand(
                gait_freq_lower=self.config.gait_freq_lower,
                gait_freq_upper=self.config.gait_freq_upper,
            ),
        ]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        rewards: list[ksim.Reward] = [
             kbot_rewards.JointDeviationPenalty(
                scale=-0.02,
                joint_targets=(
                     0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0,
                     -0.23, 0.0, 0.0, -0.441, 0.195,
                     0.23, 0.0, 0.0, 0.441, -0.195,
                 ),
                 joint_weights=(
                     1.2, 1.0, 1.0, 1.0, 1.0,
                     1.2, 1.0, 1.0, 1.0, 1.0,
                     0.01, 1.0, 1.0, 0.01, 1.0,
                     0.01, 1.0, 1.0, 0.01, 1.0,
                 ),
             ),
             kbot_rewards.KneeDeviationPenalty.create(
                 physics_model=physics_model,
                 knee_names=("dof_left_knee_04", "dof_right_knee_04"),
                 joint_targets=(
                     0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0,
                     -0.23, 0.0, 0.0, -0.441, 0.195,
                     0.23, 0.0, 0.0, 0.441, -0.195,
                 ),
                 scale=-0.1,
             ),
             kbot_rewards.LinearVelocityTrackingReward(scale=1.0),
             kbot_rewards.AngularVelocityTrackingReward(scale=0.5),
             kbot_rewards.AngularVelocityXYPenalty(scale=-0.15),
             kbot_rewards.TerminationPenalty(scale=-1.0),
             kbot_rewards.FarFromOriginTerminationReward(max_dist=5.0, scale=1.0),
             kbot_rewards.OrientationPenalty(scale=-2.0),
        ]
        if self.config.use_gait_rewards:
            rewards += [
                kbot_rewards.FeetSlipPenalty(scale=-0.25),
                kbot_rewards.FeetAirTimeReward(scale=2.0),
                kbot_rewards.FeetPhaseReward(max_foot_height=0.11, scale=1.0),
            ]
        return rewards

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            common.GVecTermination.create(physics_model, sensor_name="upvector_origin"),
            common.FarFromOriginTermination(max_dist=5.0),
        ]

    def get_model(self, key: PRNGKeyArray) -> RnnModel:
        return RnnModel(
            key,
            actor_num_inputs=NUM_ACTOR_INPUTS,
            critic_num_inputs=NUM_CRITIC_INPUTS,
            num_action_outputs=NUM_ACTION_OUTPUTS,
            min_std=0.01,
            max_std=1.0,
            mean_scale=self.config.actor_mean_scale,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
        )

    def run_actor(
        self,
        model: RnnActor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[distrax.Distribution, Array]:
        timestep_phase_4 = observations["timestep_phase_observation"]
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        lin_vel_cmd_2 = commands["linear_velocity_command"]
        ang_vel_cmd = commands["angular_velocity_command"]
        gait_freq_cmd = commands["gait_frequency_command"]
        last_action_n = observations["last_action_observation"]

        obs_n = jnp.concatenate(
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

        return model.forward(obs_n, carry)

    def run_critic(
        self,
        model: RnnCritic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[Array, Array]:
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
        feet_contact_2 = observations["feet_contact_observation"]
        base_position_3 = observations["base_position_observation"]
        base_orientation_4 = observations["base_orientation_observation"]
        base_linear_velocity_3 = observations["base_linear_velocity_observation"]
        base_angular_velocity_3 = observations["base_angular_velocity_observation"]
        actuator_force_n = observations["actuator_force_observation"]

        obs_n = jnp.concatenate(
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
                base_position_3,
                base_orientation_4,
                base_linear_velocity_3,
                base_angular_velocity_3,
                actuator_force_n,
            ],
            axis=-1,
        )

        return model.forward(obs_n, carry)

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
                lambda x, y: jnp.where(transition.done, x, y),
                initial_carry,
                (next_actor_carry, next_critic_carry)
            )

            return next_carry, transition_ppo_variables

        next_model_carry, ppo_variables = jax.lax.scan(scan_fn, model_carry, trajectory)

        return ppo_variables, next_model_carry

    def get_initial_model_carry(self, rng: PRNGKeyArray) -> tuple[Array, Array]:
        actor_carry = jnp.zeros(shape=(self.config.depth, self.config.hidden_size))
        critic_carry = jnp.zeros(shape=(self.config.depth, self.config.hidden_size))
        return actor_carry, critic_carry

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

        actor_dist, next_actor_carry = self.run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
            carry=actor_carry_in,
        )

        action_j = actor_dist.sample(seed=rng)

        return ksim.Action(
            action=action_j,
            carry=(next_actor_carry, critic_carry_in),
            aux_outputs=None,
        )

    def reset_reward_carry(self, rng: PRNGKeyArray) -> xax.FrozenDict[str, Array]:
        key, _ = jax.random.split(rng)
        gait_freq = jax.random.uniform(
            key, (1,), minval=self.config.gait_freq_lower, maxval=self.config.gait_freq_upper
        )
        phase_dt = 2 * jnp.pi * gait_freq * self.config.ctrl_dt
        phase = jnp.array([0, jnp.pi])
        reward_carry = xax.FrozenDict(
            {
                "first_contact": jnp.zeros(2, dtype=bool),
                "feet_air_time": jnp.zeros(2),
                "last_contact": jnp.zeros(2, dtype=bool),
                "swing_peak": jnp.zeros(2),
                "phase_dt": phase_dt,
                "phase": phase,
            }
        )
        return reward_carry

    def update_reward_carry(
        self,
        reward_carry: xax.FrozenDict[str, Array],
        observations: xax.FrozenDict[str, Array],
        physics_state: ksim.PhysicsState,
        commands: xax.FrozenDict[str, Array],
    ) -> xax.FrozenDict[str, Array]:
        phase_tp1 = reward_carry["phase"] + reward_carry["phase_dt"]
        phase = jnp.fmod(phase_tp1 + jnp.pi, 2 * jnp.pi) - jnp.pi

        contact = observations["feet_contact_observation"]
        contact_bool = contact.astype(bool)

        contact_filt = contact_bool | reward_carry["last_contact"]
        first_contact = (reward_carry["feet_air_time"] > 0.0) * contact_filt
        last_contact = contact_bool

        feet_air_time = reward_carry["feet_air_time"] + self.config.ctrl_dt
        feet_air_time *= ~contact_bool

        position_feet = observations["feet_position_observation"]
        position_feet_z = jnp.array([position_feet[2], position_feet[5]])
        swing_peak = jnp.maximum(reward_carry["swing_peak"], position_feet_z)
        swing_peak = swing_peak * ~contact_bool

        next_reward_carry = reward_carry.copy(
            {
                "first_contact": first_contact,
                "feet_air_time": feet_air_time,
                "last_contact": last_contact,
                "swing_peak": swing_peak,
                "phase": phase,
            }
        )
        return next_reward_carry


if __name__ == "__main__":
    KbotWalkingRnnTask.launch(
        WalkingRnnTaskConfig(
            hidden_size=256,
            depth=2,
            actor_mean_scale=0.5,
            num_envs=2048,
            batch_size=256,
            num_passes=4,
            epochs_per_log_step=1,
            rollout_length_seconds=1.25,
            dt=0.002,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            min_action_latency=0.0,
            action_scale=1.0,
            gamma=0.97,
            lam=0.95,
            entropy_coef=0.005,
            learning_rate=1e-4,
            clip_param=0.3,
            max_grad_norm=0.5,
            log_full_trajectory_every_n_steps=5,
            save_every_n_steps=25,
            export_for_inference=False,
            only_save_most_recent=False,
            use_gait_rewards=True,
            domain_randomize=True,
            light_domain_randomize=False,
            gait_freq_lower=1.25,
            gait_freq_upper=1.5,
            reward_clip_min=0.0,
            reward_clip_max=1000.0,
        ),
    )
