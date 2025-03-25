# mypy: disable-error-code="override"
"""Defines simple task for training a walking policy for K-Bot."""

from dataclasses import dataclass

import attrs
import jax
import jax.numpy as jnp
import ksim
import xax
from jaxtyping import Array

from .standing import KbotStandingTask, KbotStandingTaskConfig


@attrs.define(frozen=True, kw_only=True)
class UpwardReward(ksim.Reward):
    """Incentives forward movement."""

    velocity_clip: float = attrs.field(default=10.0)

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        # Just try to maximize the velocity in the Z direction.
        z_delta = jnp.clip(trajectory.qvel[..., 2], 0, self.velocity_clip)
        return z_delta


@attrs.define(frozen=True, kw_only=True)
class StationaryPenalty(ksim.Reward):
    """Incentives staying in place laterally."""

    norm: xax.NormType = attrs.field(default="l2")

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        return xax.get_norm(trajectory.qvel[..., :2], self.norm).sum(axis=-1)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AuxOutputs:
    log_probs: Array
    values: Array


@dataclass
class KbotJumpingTaskConfig(KbotStandingTaskConfig):
    pass


class KbotJumpingTask(KbotStandingTask[KbotJumpingTaskConfig]):
    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            UpwardReward(scale=0.5),
            StationaryPenalty(scale=-0.1),
            ksim.ActuatorForcePenalty(scale=-0.01),
            ksim.LinearVelocityZPenalty(scale=-0.01),
            ksim.AngularVelocityXYPenalty(scale=-0.01),
        ]


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m ksim_kbot.kbot2.jumping
    # To visualize the environment, use the following command:
    #   python -m ksim_kbot.kbot2.jumping run_environment=True
    KbotJumpingTask.launch(
        KbotJumpingTaskConfig(
            num_envs=2048,
            num_batches=64,
            num_passes=8,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.005,
            min_action_latency=0.0,
            valid_every_n_steps=25,
            valid_every_n_seconds=300,
            log_single_traj_every_n_valid_steps=5,
            valid_first_n_steps=0,
            rollout_length_seconds=10.0,
            eval_rollout_length_seconds=5.0,
            # PPO parameters
            gamma=0.97,
            lam=0.95,
            entropy_coef=0.001,
            learning_rate=3e-4,
            clip_param=0.3,
            max_grad_norm=1.0,
            use_mit_actuators=True,
        ),
    )
