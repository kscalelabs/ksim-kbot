"""Example script to deploy a SavedModel in KOS-Sim."""

import argparse
import asyncio
import logging
import signal
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from typing import Callable

# import keyboard
import numpy as np
import pykos
import tensorflow as tf
from askin import KeyboardController

logger = logging.getLogger(__name__)

DT = 0.04  # Policy time step (50Hz)


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

ACTIVE_ACTUATOR_LIST = [
    # Right arm (nn_id 0-4)
    Actuator(actuator_id=21, nn_id=0, kp=40.0, kd=4.0, max_torque=40.0, joint_name="dof_right_shoulder_pitch_03"),
    Actuator(actuator_id=22, nn_id=1, kp=40.0, kd=4.0, max_torque=40.0, joint_name="dof_right_shoulder_roll_03"),
    Actuator(actuator_id=23, nn_id=2, kp=30.0, kd=1.0, max_torque=14.0, joint_name="dof_right_shoulder_yaw_02"),
    Actuator(actuator_id=24, nn_id=3, kp=30.0, kd=1.0, max_torque=14.0, joint_name="dof_right_elbow_02"),
    Actuator(actuator_id=25, nn_id=4, kp=20.0, kd=0.45473329537059787, max_torque=1.0, joint_name="dof_right_wrist_00"),
]


class TargetState:
    def __init__(self) -> None:
        self.xyz_target = np.array([0.3, -0.1, 0.0])
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

    def get_target(self) -> np.ndarray:
        return self.xyz_target.copy()


async def get_observation(kos: pykos.KOS, prev_action: np.ndarray, target_state: TargetState) -> np.ndarray:
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

    xyz_target = target_state.get_target()
    # quat_target = np.array([0.0, 0.0, 0.0, 0.0])

    logger.debug("xyz_target %s", xyz_target)

    observation = np.concatenate([pos_obs, vel_obs, xyz_target, prev_action], axis=-1)
    return observation


async def send_actions(kos: pykos.KOS, position: np.ndarray, velocity: np.ndarray) -> None:
    position = np.rad2deg(position)
    velocity = np.rad2deg(velocity)
    actuator_commands: list[pykos.services.actuator.ActuatorCommand] = [
        {
            "actuator_id": ac.actuator_id,
            "position": position[ac.nn_id],
            "velocity": velocity[ac.nn_id],
        }
        for ac in ACTIVE_ACTUATOR_LIST
    ]
    logger.debug(actuator_commands)

    await kos.actuator.command_actuators(actuator_commands)


async def configure_actuators(kos: pykos.KOS) -> None:
    for ac in ACTUATOR_LIST:
        await kos.actuator.configure_actuator(
            actuator_id=ac.actuator_id,
            kp=ac.kp,
            kd=ac.kd,
            torque_enabled=True,
            max_torque=ac.max_torque,
        )


async def reset(kos: pykos.KOS) -> None:
    # Define standing joint positions based on standing.py

    await kos.sim.reset(
        pos={"x": 0.0, "y": 0.0, "z": 1.01},  # Using the z-value from your existing reset function
        quat={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        joints=[{"name": actuator.joint_name, "pos": 0.0} for actuator in ACTUATOR_LIST],
    )


def spawn_kos_sim(no_render: bool) -> tuple[subprocess.Popen, Callable]:
    """Spawn the KOS-Sim KBot2 process and return the process object."""
    logger.info("Starting KOS-Sim kbot2-feet...")
    args = ["kos-sim", "kbot2-feet"]
    if no_render:
        args.append("--no-render")
    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    logger.info("Waiting for KOS-Sim to start...")
    time.sleep(5)

    def cleanup(sig: int | None = None, frame: types.FrameType | None = None) -> None:
        logger.info("Terminating KOS-Sim...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        if sig:
            sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)

    return process, cleanup


async def main(model_path: str, ip: str, no_render: bool, episode_length: int) -> None:
    model = tf.saved_model.load(model_path)
    sim_process = None
    cleanup_fn = None

    try:
        # Try to connect to existing KOS-Sim
        logger.info("Attempting to connect to existing KOS-Sim...")
        kos = pykos.KOS(ip=ip)
        await kos.sim.get_parameters()
        logger.info("Connected to existing KOS-Sim instance.")
    except Exception as e:
        logger.info("Could not connect to existing KOS-Sim: %s", e)
        logger.info("Starting a new KOS-Sim instance locally...")
        sim_process, cleanup_fn = spawn_kos_sim(no_render)
        kos = pykos.KOS()
        attempts = 0
        while attempts < 5:
            try:
                await kos.sim.get_parameters()
                logger.info("Connected to new KOS-Sim instance.")
                break
            except Exception as connect_error:
                attempts += 1
                logger.info("Failed to connect to KOS-Sim: %s", connect_error)
                time.sleep(2)

        if attempts == 5:
            raise RuntimeError("Failed to connect to KOS-Sim")

    await configure_actuators(kos)
    await reset(kos)

    target_state = TargetState()
    # Instantiate the KeyboardController
    keyboard_controller = KeyboardController(target_state.update_from_key, timeout=0.001)

    # Start keyboard input task using the controller
    await keyboard_controller.start()

    # prev_action = np.zeros(len(ACTUATOR_LIST) * 2)
    prev_action = np.zeros(len(ACTIVE_ACTUATOR_LIST) * 2)
    observation = (await get_observation(kos, prev_action, target_state)).reshape(1, -1)

    if no_render:
        await kos.process_manager.start_kclip("deployment")

    # Add target state marker to kos-sim
    await kos.sim.add_marker(
        name="target",
        marker_type="sphere",
        target_name="floating_base_link",
        target_type="body",
        offset=target_state.xyz_target.tolist(),
        scale=[0.05, 0.05, 0.05],
        color={"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0},
        label=True,
    )

    # warm up model
    model.infer(observation)

    target_time = time.time() + DT
    observation = (await get_observation(kos, prev_action, target_state)).reshape(1, -1)

    end_time = time.time() + episode_length

    try:
        while time.time() < end_time:
            observation = observation.reshape(1, -1)
            # move it all to the infer call
            action = np.array(model.infer(observation)).reshape(-1)

            position = action[: len(ACTIVE_ACTUATOR_LIST)]
            velocity = action[len(ACTIVE_ACTUATOR_LIST) :]
            observation, _, _ = await asyncio.gather(
                get_observation(kos, prev_action, target_state),
                send_actions(kos, position, velocity),
                kos.sim.update_marker(
                    name="target",
                    offset=target_state.xyz_target.tolist(),
                ),
            )
            prev_action = action

            if time.time() < target_time:
                await asyncio.sleep(max(0, target_time - time.time()))
            else:
                logger.info("Loop overran by %s seconds", time.time() - target_time)

            target_time += DT

    except asyncio.CancelledError:
        # Stop the keyboard controller
        await keyboard_controller.stop()
        logger.info("Exiting due to CancelledError...")
        if no_render:
            try:
                save_path = await kos.process_manager.stop_kclip("deployment")
                logger.info("KClip saved to %s", save_path)
            except Exception as e:
                logger.warning("Could not stop kclip: %s", e)

        if cleanup_fn:
            cleanup_fn()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received...")
        # Stop the keyboard controller
        await keyboard_controller.stop()
        if no_render:
            try:
                save_path = await kos.process_manager.stop_kclip("deployment")
                logger.info("KClip saved to %s", save_path)
            except Exception as e:
                logger.warning(f"Could not stop kclip: {e}")
        if cleanup_fn:
             cleanup_fn()
        # Reraise to ensure clean exit
        raise
    finally:
        # Ensure keyboard controller is stopped on any exit path
        await keyboard_controller.stop()
        try:
             await kos.sim.remove_marker(name="target")
             logger.info("Target marker removed.")
        except Exception as e:
            logger.warning("Could not remove target marker: %s", e)
        # Ensure cleanup function is called if it exists
        if cleanup_fn:
            logger.debug("Calling cleanup function in finally block.")
            cleanup_fn()

    logger.info("Episode finished!")

    # Stop kclip if still running (e.g., normal loop finish)
    if no_render:
        try:
            save_path = await kos.process_manager.stop_kclip("deployment")
            logger.info("KClip saved to %s after episode finish.", save_path)
        except Exception as e:
            logger.debug("Could not stop kclip after episode finish (may already be stopped): %s", e)


# (optionally) start the KOS-Sim server before running this script
# `kos-sim kbot2-feet`

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--episode_length", type=int, default=5)  # seconds
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--log-file", type=str, help="Path to write log output")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    if args.log_file:
        logging.basicConfig(
            level=log_level,
            format=log_format,
            filename=args.log_file,
            filemode="w",
        )
    else:
        logging.basicConfig(level=log_level, format=log_format)

    asyncio.run(main(args.model_path, args.ip, args.no_render, args.episode_length))
