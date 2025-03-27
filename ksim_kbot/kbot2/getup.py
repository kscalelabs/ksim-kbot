"""Defines simple task for training a standing policy for K-Bot."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, TypeVar

import attrs
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import ksim
import mujoco
import optax
import xax
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray
from kscale.web.gen.api import JointMetadataOutput
from mujoco import mjx


OBS_SIZE = 20 * 2 + 3 + 3 + 40 # = 46 position + velocity + imu_acc + imu_gyro + last_action
CMD_SIZE = 2
NUM_OUTPUTS = 20 * 2  # position + velocity

SINGLE_STEP_HISTORY_SIZE = NUM_OUTPUTS + OBS_SIZE + CMD_SIZE

HISTORY_LENGTH = 0

NUM_INPUTS = (OBS_SIZE + CMD_SIZE) + SINGLE_STEP_HISTORY_SIZE * HISTORY_LENGTH

MAX_TORQUE = {
    "00": 15.0,
    "02": 20.0,
    "03": 60.0,
    "04": 90.0,
}

Config = TypeVar("Config", bound="KbotGetupTaskConfig")


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AuxOutputs:
    log_probs: Array
    values: Array


@attrs.define(frozen=True, kw_only=True)
class EnergyTermination(ksim.Termination):
    energy_termination_threshold: float = attrs.field(default=100.0)

    def __call__(self, data: ksim.PhysicsData) -> jax.Array:
        energy = jnp.sum(jnp.abs(data.actuator_force * data.qvel[6:]))
        energy_termination = energy > jnp.inf
        return energy_termination


@attrs.define(frozen=True, kw_only=True)
class ResetDefaultJointPosition(ksim.Reset):
    """Resets the joint positions of the robot to random values."""

    default_targets: tuple[float, ...] = attrs.field(
        default=(
            # xyz
            0.0,
            0.0,
            0.05,
            # quat
            0.7071,
            0.7071,
            0.0,
            0.0,
            # qpos
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        )
    )

    def __call__(self, data: ksim.PhysicsData, rng: PRNGKeyArray) -> ksim.PhysicsData:
        qpos = data.qpos
        match type(data):
            case mujoco.MjData:
                qpos[:] = self.default_targets
            case mjx.Data:
                qpos = qpos.at[:].set(self.default_targets)
        return ksim.utils.mujoco.update_data_field(data, "qpos", qpos)


@attrs.define(frozen=True)
class HistoryObservation(ksim.Observation):
    def observe(self, state: ksim.RolloutVariables, rng: PRNGKeyArray) -> Array:
        if not isinstance(state.carry, Array):
            raise ValueError("Carry is not a history array")
        return state.carry


@attrs.define(frozen=True)
class JointPositionObservation(ksim.Observation):
    noise: float = attrs.field(default=0.0)
    default_targets: tuple[float, ...] = attrs.field(
        default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )

    def observe(self, rollout_state: ksim.RolloutVariables, rng: PRNGKeyArray) -> Array:
        qpos = rollout_state.physics_state.data.qpos[7:]  # (N,)
        diff = qpos - jnp.array(self.default_targets)
        return diff

    def add_noise(self, observation: Array, rng: PRNGKeyArray) -> Array:
        return observation + jax.random.normal(rng, observation.shape) * self.noise


@attrs.define(frozen=True)
class LastActionObservation(ksim.Observation):
    noise: float = attrs.field(default=0.0)

    def observe(self, rollout_state: ksim.RolloutVariables, rng: PRNGKeyArray) -> Array:
        return rollout_state.physics_state.most_recent_action

    def add_noise(self, observation: Array, rng: PRNGKeyArray) -> Array:
        return observation + jax.random.normal(rng, observation.shape) * self.noise


@attrs.define(frozen=True, kw_only=True)
class JointDeviationPenalty(ksim.Reward):
    """Penalty for joint deviations."""

    joint_targets: tuple[float, ...] = attrs.field(
        default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )
    height_threshold: float = attrs.field(default=0.7)

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        height = trajectory.qpos[:, 2]
        diff = trajectory.qpos[:, 7:] - jnp.array(self.joint_targets)
        diff =jnp.sum(jnp.square(diff), axis=-1)
        # y2 = jnp.zeros_like(diff.shape[0])
        x = jax.lax.cond(
            jnp.all(height < self.height_threshold),
            lambda: diff,
            lambda: jnp.zeros_like(diff),
        )
        return x


@attrs.define(frozen=True, kw_only=True)
class StandStillReward(ksim.Reward):
    """Reward for standing still."""
    height_threshold: float = attrs.field(default=0.7)

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        height = trajectory.qpos[:, 2]
        cost = jnp.sum(jnp.square(trajectory.qvel[..., :2]), axis=-1)
        rew = jnp.exp(-0.5 * cost)
        # y = rew
        # y2 = jnp.zeros_like(rew.shape[0])
        x = jax.lax.cond(
            jnp.all(height < self.height_threshold),
            lambda: rew,
            lambda: jnp.zeros_like(rew),
        )
        return x


@attrs.define(frozen=True, kw_only=True)
class UpwardPositionPenalty(ksim.Reward):
    """Reward for the upward position."""

    dt: float = attrs.field(default=0.02)
    target_pos: float = attrs.field(default=0.91)

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        pos_after = trajectory.qpos[:, 2]
        uph_cost = jnp.abs(pos_after - self.target_pos) #/ self.dt
        return uph_cost




@attrs.define(frozen=True, kw_only=True)
class UpwardVelocityReward(ksim.Reward):
    """Incentives upward velocity."""
    velocity_clip: float = attrs.field(default=10.0)

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        z_delta = jnp.clip(trajectory.qvel[..., 2], 0, self.velocity_clip)

        return z_delta
    

@attrs.define(frozen=True, kw_only=True)
class StationaryPenalty(ksim.Reward):
    """Incentives staying in place laterally."""

    norm: xax.NormType = attrs.field(default="l2")

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        return xax.get_norm(trajectory.qvel[..., :2], self.norm).sum(axis=-1)


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
        imu_acc_n: Array,
        imu_gyro_n: Array,
        lin_vel_cmd_n: Array,
        last_action_n: Array,
        history_n: Array,
    ) -> distrax.Normal:
        x_n = jnp.concatenate(
            [
                joint_pos_n,
                joint_vel_n,
                imu_acc_n,
                imu_gyro_n,
                lin_vel_cmd_n,
                last_action_n,
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
            in_size=NUM_INPUTS,
            out_size=1,  # Always output a single critic value.
            width_size=64,
            depth=5,
            key=key,
            activation=jax.nn.relu,
        )

    def __call__(
        self,
        joint_pos_n: Array,
        joint_vel_n: Array,
        imu_acc_n: Array,
        imu_gyro_n: Array,
        lin_vel_cmd_n: Array,
        last_action_n: Array,
        history_n: Array,
    ) -> Array:
        x_n = jnp.concatenate(
            [
                joint_pos_n,
                joint_vel_n,
                imu_acc_n,
                imu_gyro_n,
                lin_vel_cmd_n,
                last_action_n,
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
class KbotGetupTaskConfig(ksim.PPOConfig):
    """Config for the KBot getup task."""

    robot_urdf_path: str = xax.field(
        value="ksim_kbot/kscale-assets/kbot-v2-lw-full/",
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

    # Rendering parameters.
    render_track_body_id: int | None = xax.field(
        value=None,
        help="The body id to track with the render camera.",
    )

    # Checkpointing parameters.
    export_for_inference: bool = xax.field(
        value=False,
        help="Whether to export the model for inference.",
    )



class KbotGetupTask(ksim.PPOTask[KbotGetupTaskConfig], Generic[Config]):
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
        mjcf_path = (Path(self.config.robot_urdf_path) / "scene.mjcf").resolve().as_posix()
        mj_model = mujoco.MjModel.from_xml_path(mjcf_path)

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
            )
        else:
            return ksim.TorqueActuators()

    def get_randomization(self, physics_model: ksim.PhysicsModel) -> list[ksim.Randomization]:
        return [
            ksim.WeightRandomization(scale=0.05),
            ksim.StaticFrictionRandomization(scale_lower=0.5, scale_upper=2.0),
            # ksim.JointZeroPositionRandomization(scale_lower=-0.05, scale_upper=0.05),
            # ksim.FloorFrictionRandomization.from_body_name(
            #     model=physics_model,
            #     scale_lower=0.2,
            #     scale_upper=0.6,
            #     floor_body_name="floor",
            # ),
            ksim.ArmatureRandomization(scale_lower=1.0, scale_upper=1.05),
            ksim.TorsoMassRandomization.from_body_name(
                model=physics_model,
                scale_lower=-1.0,
                scale_upper=1.0,
                torso_body_name="Torso_Side_Right",
            ),
            ksim.JointDampingRandomization(scale_lower=0.95, scale_upper=1.05),
        ]

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomBaseVelocityXYReset(scale=0.01),
            ksim.RandomJointPositionReset(scale=0.02),
            ksim.RandomJointVelocityReset(scale=0.02),
            ResetDefaultJointPosition(),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        return [
            ksim.PushEvent(
                probability=0.0005,
                interval_range=(1, 3),
                linear_force_scale=0.2,
            ),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        return [
            JointPositionObservation(noise=0.02),
            ksim.JointVelocityObservation(noise=0.5),
            ksim.ActuatorForceObservation(),
            ksim.SensorObservation.create(physics_model, "imu_acc", noise=0.5),
            ksim.SensorObservation.create(physics_model, "imu_gyro", noise=0.2),
            LastActionObservation(noise=0.0),
            HistoryObservation(),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return [
            ksim.LinearVelocityCommand(x_range=(-0.0, 0.0), y_range=(-0.0, 0.0), switch_prob=0.0, zero_prob=0.0),
        ]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            UpwardPositionPenalty(scale=-0.5),
            UpwardVelocityReward(scale=0.05),
            StationaryPenalty(scale=-0.05),
            ksim.ActuatorForcePenalty(scale=-0.01),
            StandStillReward(scale=0.2),
            JointDeviationPenalty(scale=-0.01),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            EnergyTermination(),
        ]

    def get_model(self, key: PRNGKeyArray) -> KbotModel:
        return KbotModel(key)

    def get_initial_carry(self, rng: PRNGKeyArray) -> Array:
        return jnp.zeros(HISTORY_LENGTH * SINGLE_STEP_HISTORY_SIZE)

    def _run_actor(
        self,
        model: KbotModel,
        observations: FrozenDict[str, Array],
        commands: FrozenDict[str, Array],
    ) -> distrax.Normal:
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        imu_acc_n = observations["imu_acc_obs"]
        imu_gyro_n = observations["imu_gyro_obs"]
        lin_vel_cmd_n = commands["linear_velocity_command"]
        last_action_n = observations["last_action_observation"]
        history_n = observations["history_observation"]
        return model.actor(joint_pos_n, joint_vel_n, imu_acc_n, imu_gyro_n, lin_vel_cmd_n, last_action_n, history_n)

    def _run_critic(
        self,
        model: KbotModel,
        observations: FrozenDict[str, Array],
        commands: FrozenDict[str, Array],
    ) -> Array:
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        imu_acc_n = observations["imu_acc_obs"]
        imu_gyro_n = observations["imu_gyro_obs"]
        lin_vel_cmd_n = commands["linear_velocity_command"]
        last_action_n = observations["last_action_observation"]
        history_n = observations["history_observation"]
        return model.critic(joint_pos_n, joint_vel_n, imu_acc_n, imu_gyro_n, lin_vel_cmd_n, last_action_n, history_n)

    def get_on_policy_log_probs(
        self,
        model: KbotModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> Array:
        if trajectories.aux_outputs is None:
            raise ValueError("No aux outputs found in trajectories")
        return trajectories.aux_outputs.log_probs

    def get_on_policy_values(
        self,
        model: KbotModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> Array:
        if trajectories.aux_outputs is None:
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
        action_btn = trajectories.action
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
        carry: Array,
        physics_model: ksim.PhysicsModel,
        observations: FrozenDict[str, Array],
        commands: FrozenDict[str, Array],
        rng: PRNGKeyArray,
    ) -> tuple[Array, Array, AuxOutputs]:
        action_dist_n = self._run_actor(model, observations, commands)
        action_n = action_dist_n.sample(seed=rng)
        action_log_prob_n = action_dist_n.log_prob(action_n)

        critic_n = self._run_critic(model, observations, commands)
        value_n = critic_n.squeeze(-1)

        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        imu_acc_n = observations["imu_acc_obs"]
        imu_gyro_n = observations["imu_gyro_obs"]
        lin_vel_cmd_n = commands["linear_velocity_command"]
        last_action_n = observations["last_action_observation"]
        history_n = jnp.concatenate(
            [
                joint_pos_n,
                joint_vel_n,
                imu_acc_n,
                imu_gyro_n,
                lin_vel_cmd_n,
                last_action_n,
                action_n,
            ],
            axis=-1,
        )

        if HISTORY_LENGTH > 0:
            # Roll the history by shifting the existing history and adding the new data
            carry_reshaped = carry.reshape(HISTORY_LENGTH, SINGLE_STEP_HISTORY_SIZE)
            shifted_history = jnp.roll(carry_reshaped, shift=-1, axis=0)
            new_history = shifted_history.at[HISTORY_LENGTH - 1].set(history_n)
            history_n = new_history.reshape(-1)
        else:
            history_n = jnp.zeros(0)

        return action_n, history_n, AuxOutputs(log_probs=action_log_prob_n, values=value_n)

    def make_export_model(self, model: KbotModel, stochastic: bool = False, batched: bool = False) -> Callable:
        """Makes a callable inference function that directly takes a flattened input vector and returns an action.

        Returns:
            A tuple containing the inference function and the size of the input vector.
        """

        def deterministic_model_fn(obs: Array) -> Array:
            return model.actor.call_flat_obs(obs).mode()

        def stochastic_model_fn(obs: Array) -> Array:
            distribution = model.actor.call_flat_obs(obs)
            return distribution.sample(seed=jax.random.PRNGKey(0))

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
        state = super().on_after_checkpoint_save(ckpt_path, state)

        model: KbotModel = self.load_checkpoint(ckpt_path, part="model")

        model_fn = self.make_export_model(model, stochastic=False, batched=True)

        input_shapes = [(NUM_INPUTS,)]

        xax.export(
            model_fn,
            input_shapes,  # type: ignore [arg-type]
            ckpt_path.parent / "tf_model",
        )

        return state


if __name__ == "__main__":
    # python -m ksim_kbot.kbot2.getup run_environment=True
    KbotGetupTask.launch(
        KbotGetupTaskConfig(
            num_envs=2,
            num_batches=1,
            num_passes=10,
            # Simulation parameters.
            dt=0.002,
            ctrl_dt=0.02,
            max_action_latency=0.005,
            min_action_latency=0.0,
            valid_every_n_steps=25,
            valid_first_n_steps=0,
            rollout_length_seconds=10.0,
            # PPO parameters
            gamma=0.97,
            lam=0.95,
            entropy_coef=0.001,
            learning_rate=1e-4,
            clip_param=0.3,
            max_grad_norm=1.0,
            use_mit_actuators=True,
            export_for_inference=True,
            save_every_n_steps=25,
        ),
    )
