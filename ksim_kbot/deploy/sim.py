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

import numpy as np
import pykos
import tensorflow as tf

logger = logging.getLogger(__name__)

DT = 0.02  # Policy time step (50Hz)

DEFAULT_POSITIONS = np.array(
    [
        # right arm (nn_id 0-4)
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        # left arm (nn_id 5-9)
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        # right leg (nn_id 10-14)
        -0.23,
        0.0,
        0.0,
        -0.441,
        0.195,
        # left leg (nn_id 15-19)
        0.23,
        0.0,
        0.0,
        0.441,
        -0.195,
    ]
)


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


async def get_observation(kos: pykos.KOS, prev_action: np.ndarray) -> np.ndarray:
    (actuator_states, imu) = await asyncio.gather(
        kos.actuator.get_actuators_state([ac.actuator_id for ac in ACTUATOR_LIST]),
        kos.imu.get_imu_values(),
    )
    state_dict_pos = {state.actuator_id: state.position for state in actuator_states.states}
    pos_obs = np.deg2rad(
        np.array([state_dict_pos[ac.actuator_id] for ac in sorted(ACTUATOR_LIST, key=lambda x: x.nn_id)])
    )
    pos_diff = pos_obs - DEFAULT_POSITIONS
    state_dict_vel = {state.actuator_id: state.velocity for state in actuator_states.states}
    vel_obs = np.deg2rad(
        np.array([state_dict_vel[ac.actuator_id] for ac in sorted(ACTUATOR_LIST, key=lambda x: x.nn_id)])
    )
    imu_obs = np.array([imu.accel_x, imu.accel_y, imu.accel_z, imu.gyro_x, imu.gyro_y, imu.gyro_z])
    cmd = np.array([0.0, 0.0])
    observation = np.concatenate([pos_diff, vel_obs, imu_obs, cmd, prev_action], axis=-1)
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
        for ac in ACTUATOR_LIST
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
        joints=[{"name": actuator.joint_name, "pos": pos} for actuator, pos in zip(ACTUATOR_LIST, DEFAULT_POSITIONS)],
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
    prev_action = np.zeros(len(ACTUATOR_LIST) * 2)
    observation = (await get_observation(kos, prev_action)).reshape(1, -1)

    if no_render:
        await kos.process_manager.start_kclip("deployment")

    # warm up model
    model.infer(observation)

    target_time = time.time() + DT
    observation = (await get_observation(kos, prev_action)).reshape(1, -1)

    end_time = time.time() + episode_length

    try:
        while time.time() < end_time:
            observation = observation.reshape(1, -1)
            # move it all to the infer call
            action = np.array(model.infer(observation)).reshape(-1)

            position = action[: len(ACTUATOR_LIST)]
            velocity = action[len(ACTUATOR_LIST) :]
            observation, _ = await asyncio.gather(
                get_observation(kos, prev_action),
                send_actions(kos, position, velocity),
            )
            prev_action = action

            if time.time() < target_time:
                await asyncio.sleep(max(0, target_time - time.time()))
            else:
                logger.info("Loop overran by %s seconds", time.time() - target_time)

            target_time += DT

    except asyncio.CancelledError:
        logger.info("Exiting...")
        if no_render:
            save_path = await kos.process_manager.stop_kclip("deployment")
            logger.info("KClip saved to %s", save_path)

        if cleanup_fn:
            cleanup_fn()

        raise KeyboardInterrupt

    logger.info("Episode finished!")

    if no_render:
        await kos.process_manager.stop_kclip("deployment")

    if cleanup_fn:
        cleanup_fn()


# (optionally) start the KOS-Sim server before running this script
# `kos-sim kbot-v2-feet`

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
