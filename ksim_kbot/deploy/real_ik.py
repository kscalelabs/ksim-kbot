"""Example script to deploy a SavedModel on K-Bot with IK control."""

import argparse
import asyncio
import logging
import select
import sys
import termios
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Generator

import numpy as np
import pykos
import tensorflow as tf

logger = logging.getLogger(__name__)

DT = 0.02  # Policy time step (50Hz)
ACTION_SCALE = 1.0


class Mode(Enum):
    REAL = "real"
    SIM = "sim"
    BOTH = "both"


@dataclass
class Actuator:
    actuator_id: int
    nn_id: int
    kp: float
    kd: float
    max_torque: float
    joint_name: str


ACTUATOR_LIST: list[Actuator] = [
    # Right arm (nn_id 0-4)
    Actuator(actuator_id=21, nn_id=0, kp=40.0, kd=4.0, max_torque=40.0, joint_name="dof_right_shoulder_pitch_03"),
    Actuator(actuator_id=22, nn_id=1, kp=40.0, kd=4.0, max_torque=40.0, joint_name="dof_right_shoulder_roll_03"),
    Actuator(actuator_id=23, nn_id=2, kp=30.0, kd=1.0, max_torque=14.0, joint_name="dof_right_shoulder_yaw_02"),
    Actuator(actuator_id=24, nn_id=3, kp=30.0, kd=1.0, max_torque=14.0, joint_name="dof_right_elbow_02"),
    Actuator(actuator_id=25, nn_id=4, kp=20.0, kd=0.45473329537059787, max_torque=1.0, joint_name="dof_right_wrist_00"),
    # Left arm (nn_id 5-9)
    Actuator(actuator_id=11, nn_id=5, kp=40.0, kd=4.0, max_torque=40.0, joint_name="dof_left_shoulder_pitch_03"),
    Actuator(actuator_id=12, nn_id=6, kp=40.0, kd=4.0, max_torque=40.0, joint_name="dof_left_shoulder_roll_03"),
    Actuator(actuator_id=13, nn_id=7, kp=30.0, kd=1.0, max_torque=14.0, joint_name="dof_left_shoulder_yaw_02"),
    Actuator(actuator_id=14, nn_id=8, kp=30.0, kd=1.0, max_torque=14.0, joint_name="dof_left_elbow_02"),
    Actuator(actuator_id=15, nn_id=9, kp=20.0, kd=0.45473329537059787, max_torque=1.0, joint_name="dof_left_wrist_00"),
    # Right leg (nn_id 10-14)
    Actuator(actuator_id=41, nn_id=10, kp=85.0, kd=5.0, max_torque=60.0, joint_name="dof_right_hip_pitch_04"),
    Actuator(actuator_id=42, nn_id=11, kp=40.0, kd=4.0, max_torque=40.0, joint_name="dof_right_hip_roll_03"),
    Actuator(actuator_id=43, nn_id=12, kp=40.0, kd=4.0, max_torque=40.0, joint_name="dof_right_hip_yaw_03"),
    Actuator(actuator_id=44, nn_id=13, kp=85.0, kd=5.0, max_torque=60.0, joint_name="dof_right_knee_04"),
    Actuator(actuator_id=45, nn_id=14, kp=30.0, kd=1.0, max_torque=14.0, joint_name="dof_right_ankle_02"),
    # Left leg (nn_id 15-19)
    Actuator(actuator_id=31, nn_id=15, kp=85.0, kd=5.0, max_torque=60.0, joint_name="dof_left_hip_pitch_04"),
    Actuator(actuator_id=32, nn_id=16, kp=40.0, kd=4.0, max_torque=40.0, joint_name="dof_left_hip_roll_03"),
    Actuator(actuator_id=33, nn_id=17, kp=40.0, kd=4.0, max_torque=40.0, joint_name="dof_left_hip_yaw_03"),
    Actuator(actuator_id=34, nn_id=18, kp=85.0, kd=5.0, max_torque=60.0, joint_name="dof_left_knee_04"),
    Actuator(actuator_id=35, nn_id=19, kp=30.0, kd=1.0, max_torque=14.0, joint_name="dof_left_ankle_02"),
]

# Only use right arm actuators for IK control
ACTIVE_ACTUATOR_LIST = [
    Actuator(actuator_id=21, nn_id=0, kp=40.0, kd=4.0, max_torque=40.0, joint_name="dof_right_shoulder_pitch_03"),
    Actuator(actuator_id=22, nn_id=1, kp=40.0, kd=4.0, max_torque=40.0, joint_name="dof_right_shoulder_roll_03"),
    Actuator(actuator_id=23, nn_id=2, kp=30.0, kd=1.0, max_torque=14.0, joint_name="dof_right_shoulder_yaw_02"),
    Actuator(actuator_id=24, nn_id=3, kp=30.0, kd=1.0, max_torque=14.0, joint_name="dof_right_elbow_02"),
    Actuator(actuator_id=25, nn_id=4, kp=20.0, kd=0.45473329537059787, max_torque=1.0, joint_name="dof_right_wrist_00"),
]


class TargetState:
    def __init__(self) -> None:
        self.xyz_target = np.array([0.3, -0.1, 0.0])
        self.quat_target = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
        self.step_size = 0.01

    async def update_from_key(self, key: str) -> None:
        if key == "a":
            self.xyz_target[0] -= self.step_size
        elif key == "d":
            self.xyz_target[0] += self.step_size
        elif key == "w":
            self.xyz_target[1] += self.step_size
        elif key == "s":
            self.xyz_target[1] -= self.step_size
        elif key == "q":
            self.xyz_target[2] -= self.step_size
        elif key == "e":
            self.xyz_target[2] += self.step_size
        logger.debug("Target position updated to: %s", self.xyz_target)

    def get_target(self) -> tuple[np.ndarray, np.ndarray]:
        return self.xyz_target.copy(), self.quat_target.copy()


async def keyboard_input(target_state: TargetState) -> None:
    @contextmanager
    def cbreak() -> Generator[None, None, None]:
        """Context manager for terminal cbreak mode - allows char-by-char input without echo."""
        try:
            # Save original terminal settings
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)

            # Configure terminal for cbreak mode (not full raw mode)
            new_settings = termios.tcgetattr(fd)
            new_settings[3] = new_settings[3] & ~termios.ECHO  # Disable echo
            new_settings[3] = new_settings[3] & ~termios.ICANON  # Disable canonical mode
            new_settings[6][termios.VMIN] = 0  # No blocking
            new_settings[6][termios.VTIME] = 0  # No timeout
            termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)

            # Make sure we restore terminal settings no matter what
            yield
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    # Print instructions
    logger.info("\nKeyboard control active:")
    logger.info("  w/s: forward/backward")
    logger.info("  a/d: left/right")
    logger.info("  q/e: down/up")
    logger.info("  Press Ctrl+C to exit\n")

    with cbreak():
        try:
            while True:
                # Check if input is available
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key == "\x03":  # Ctrl+C
                        raise KeyboardInterrupt
                    await target_state.update_from_key(key)

                # Yield to other tasks
                await asyncio.sleep(0.01)
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Keyboard input stopping...")


async def get_observation(
    kos_instances: dict[str, pykos.KOS], prev_action: np.ndarray, target_state: TargetState
) -> np.ndarray:
    kos = kos_instances["real"]
    xyz_target, quat_target = target_state.get_target()
    if kos_instances["sim"] is not None:
        actuator_states, _ = await asyncio.gather(
            kos.actuator.get_actuators_state([ac.actuator_id for ac in ACTIVE_ACTUATOR_LIST]),
            kos_instances["sim"].sim.update_marker(
                name="target",
                offset=target_state.xyz_target.tolist(),
            ),
        )
    else:
        actuator_states = await kos.actuator.get_actuators_state([ac.actuator_id for ac in ACTIVE_ACTUATOR_LIST])

    state_dict_pos = {state.actuator_id: state.position for state in actuator_states.states}
    pos_obs = np.deg2rad(
        np.array([state_dict_pos[ac.actuator_id] for ac in sorted(ACTIVE_ACTUATOR_LIST, key=lambda x: x.nn_id)])
    )
    state_dict_vel = {state.actuator_id: state.velocity for state in actuator_states.states}
    vel_obs = (
        np.deg2rad(
            np.array([state_dict_vel[ac.actuator_id] for ac in sorted(ACTIVE_ACTUATOR_LIST, key=lambda x: x.nn_id)])
        )
        / 50.0
    )

    # Log target position for debug purposes
    logger.debug("xyz_target: %s", xyz_target)

    observation = np.concatenate([pos_obs, vel_obs, xyz_target, quat_target, prev_action], axis=-1)
    return observation


async def send_actions(kos_instances: dict[str, pykos.KOS], position: np.ndarray, velocity: np.ndarray) -> None:
    position = np.rad2deg(position)
    velocity = np.rad2deg(velocity)
    actuator_commands: list[pykos.services.actuator.ActuatorCommand] = [
        {
            "actuator_id": ac.actuator_id,
            "position": position[ac.nn_id] * ACTION_SCALE,
            "velocity": velocity[ac.nn_id] * ACTION_SCALE,
        }
        for ac in ACTIVE_ACTUATOR_LIST
    ]
    # Send commands to all KOS instances
    await asyncio.gather(*(kos.actuator.command_actuators(actuator_commands) for kos in kos_instances.values()))


async def configure_actuators(kos_instances: dict[str, pykos.KOS]) -> None:
    for kos in kos_instances.values():
        for ac in ACTIVE_ACTUATOR_LIST:
            await kos.actuator.configure_actuator(
                actuator_id=ac.actuator_id,
                kp=ac.kp,
                kd=ac.kd,
                torque_enabled=True,
                max_torque=ac.max_torque,
            )


async def reset(kos_dict: dict[str, pykos.KOS]) -> None:
    zero_commands: list[pykos.services.actuator.ActuatorCommand] = [
        {
            "actuator_id": ac.actuator_id,
            "position": 0.0,
            "velocity": 0.0,
        }
        for ac in ACTIVE_ACTUATOR_LIST
    ]

    logger.info("zero_commands: %s", zero_commands)
    # Send reset commands to all KOS instances
    for kos in kos_dict.values():
        await kos.actuator.command_actuators(zero_commands)


async def disable(kos_dict: dict[str, pykos.KOS]) -> None:
    for kos in kos_dict.values():
        for ac in ACTIVE_ACTUATOR_LIST:
            await kos.actuator.configure_actuator(
                actuator_id=ac.actuator_id,
                torque_enabled=False,
            )


async def main(model_path: str, episode_length: int, mode: Mode) -> None:
    model = tf.saved_model.load(model_path)
    target_state = TargetState()
    # Initialize KOS instances based on mode as a dictionary instead of a list
    kos_dict = {}
    if mode in [Mode.REAL, Mode.BOTH]:
        kos_dict["real"] = pykos.KOS(ip="100.99.151.89")  # Fixed IP for real robot
    if mode in [Mode.SIM, Mode.BOTH]:
        kos_dict["sim"] = pykos.KOS(ip="100.100.42.71")  # Simulator always on localhost
        await kos_dict["sim"].sim.add_marker(
            name="target",
            marker_type="sphere",
            target_name="floating_base_link",
            target_type="body",
            offset=target_state.xyz_target.tolist(),
            scale=[0.05, 0.05, 0.05],
            color={"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0},
            label=True,
        )

    if not kos_dict:
        raise ValueError("No KOS instances configured for the selected mode")

    await disable(kos_dict)
    time.sleep(1)
    logger.info("Configuring actuators...")
    await configure_actuators(kos_dict)
    await asyncio.sleep(1)
    logger.info("Resetting...")
    await reset(kos_dict)

    # Start keyboard input task
    keyboard_task = asyncio.create_task(keyboard_input(target_state))
    keyboard_task.add_done_callback(lambda _: logger.info("Keyboard task done"))

    prev_action = np.zeros(len(ACTIVE_ACTUATOR_LIST) * 2)
    observation = (await get_observation(kos_dict, prev_action, target_state)).reshape(1, -1)

    # warm up model
    model.infer(observation)

    for i in range(5, -1, -1):
        logger.info("Starting in %d seconds...", i)
        await asyncio.sleep(1)

    target_time = time.time() + DT
    observation = await get_observation(kos_dict, prev_action, target_state)

    end_time = time.time() + episode_length

    try:
        while time.time() < end_time:
            observation = observation.reshape(1, -1)
            action = np.array(model.infer(observation)).reshape(-1)
            position = action[: len(ACTIVE_ACTUATOR_LIST)]
            velocity = action[len(ACTIVE_ACTUATOR_LIST) :]
            observation, _ = await asyncio.gather(
                get_observation(kos_dict, prev_action, target_state),
                send_actions(kos_dict, position, velocity),
            )
            prev_action = action

            if time.time() < target_time:
                await asyncio.sleep(max(0, target_time - time.time()))
            else:
                logger.info("Loop overran by %s seconds", time.time() - target_time)

            target_time += DT

    except asyncio.CancelledError:
        logger.info("Exiting...")
        keyboard_task.cancel()  # Make sure to cancel the keyboard task
        await disable(kos_dict)
        logger.info("Actuators disabled")
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received...")
        keyboard_task.cancel()  # Make sure to cancel the keyboard task
        await disable(kos_dict)
        logger.info("Actuators disabled")
        raise
    finally:
        await kos_dict["sim"].sim.remove_marker(name="target")

    logger.info("Episode finished!")
    # Clean up keyboard task
    keyboard_task.cancel()
    # Disable actuators
    await disable(kos_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--episode_length", type=int, default=60)  # seconds
    parser.add_argument("--mode", type=str, choices=[m.value for m in Mode], default=Mode.REAL.value)
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    asyncio.run(main(args.model_path, args.episode_length, Mode(args.mode)))
