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
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)
DT = 0.02  # time step (50Hz)

DEFAULT_POSITIONS = np.array(
    [
        0,
        0,
        0,
        0,
        0,  # right arm
        0,
        0,
        0,
        0,
        0,  # left arm
        -0.23,
        0,
        0,
        -0.441,
        0.195,  # right leg
        0.23,
        0,
        0,
        0.441,
        -0.195,  # left leg
    ]
)

OBS_SIZE = 20 + 20 + 3 + 3 + 3 + 40 + 4  # pos_diff (20) + vel_obs (20) + imu (6) - adjust if needed
CMD_SIZE = 2
HIST_LEN = 5


@dataclass
class Actuator:
    actuator_id: int
    nn_id: int
    kp: float
    kd: float
    max_torque: float
    joint_name: str


ACTUATOR_LIST: list[Actuator] = [
    # right arm
    Actuator(21, 0, 40.0, 4.0, 60.0, "dof_right_shoulder_pitch_03"),
    Actuator(22, 1, 40.0, 4.0, 60.0, "dof_right_shoulder_roll_03"),
    Actuator(23, 2, 30.0, 1.0, 17.0, "dof_right_shoulder_yaw_02"),
    Actuator(24, 3, 30.0, 1.0, 17.0, "dof_right_elbow_02"),
    Actuator(25, 4, 20.0, 0.45473329537059787, 1.0, "dof_right_wrist_00"),
    # left arm
    Actuator(11, 5, 40.0, 4.0, 60.0, "dof_left_shoulder_pitch_03"),
    Actuator(12, 6, 40.0, 4.0, 60.0, "dof_left_shoulder_roll_03"),
    Actuator(13, 7, 30.0, 1.0, 17.0, "dof_left_shoulder_yaw_02"),
    Actuator(14, 8, 30.0, 1.0, 17.0, "dof_left_elbow_02"),
    Actuator(15, 9, 20.0, 0.45473329537059787, 1.0, "dof_left_wrist_00"),
    # right leg
    Actuator(41, 10, 85.0, 5.0, 80.0, "dof_right_hip_pitch_04"),
    Actuator(42, 11, 40.0, 4.0, 60.0, "dof_right_hip_roll_03"),
    Actuator(43, 12, 40.0, 4.0, 60.0, "dof_right_hip_yaw_03"),
    Actuator(44, 13, 85.0, 5.0, 80.0, "dof_right_knee_04"),
    Actuator(45, 14, 30.0, 1.0, 17.0, "dof_right_ankle_02"),
    # left leg
    Actuator(31, 15, 85.0, 5.0, 80.0, "dof_left_hip_pitch_04"),
    Actuator(32, 16, 40.0, 4.0, 60.0, "dof_left_hip_roll_03"),
    Actuator(33, 17, 40.0, 4.0, 60.0, "dof_left_hip_yaw_03"),
    Actuator(34, 18, 85.0, 5.0, 80.0, "dof_left_knee_04"),
    Actuator(35, 19, 30.0, 1.0, 17.0, "dof_left_ankle_02"),
]


async def get_observation(
    kos: pykos.KOS, prev_action: np.ndarray, cmd: np.ndarray, phase: np.ndarray, history: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = [ac.actuator_id for ac in ACTUATOR_LIST]
    act_states, imu, raw_quat = await asyncio.gather(
        kos.actuator.get_actuators_state(ids), kos.imu.get_imu_values(), kos.imu.get_quaternion()
    )
    pos_dict = {s.actuator_id: s.position for s in act_states.states}
    pos_obs = np.deg2rad([pos_dict[ac.actuator_id] for ac in sorted(ACTUATOR_LIST, key=lambda x: x.nn_id)])
    pos_diff = pos_obs - DEFAULT_POSITIONS

    vel_dict = {s.actuator_id: s.velocity for s in act_states.states}
    vel_obs = np.deg2rad([vel_dict[ac.actuator_id] for ac in sorted(ACTUATOR_LIST, key=lambda x: x.nn_id)])

    imu_obs = np.array([imu.accel_x, imu.accel_y, imu.accel_z, imu.gyro_x, imu.gyro_y, imu.gyro_z])

    r = R.from_quat([raw_quat.x, raw_quat.y, raw_quat.z, raw_quat.w])
    gvec = r.apply(np.array([0, 0, -1]), inverse=True)
    # During training gravity vector is taken from the first torso frame
    gvec = np.array([gvec[1], -gvec[2], -gvec[0]])

    phase += 2 * np.pi * 1.2550827 * DT
    phase = np.fmod(phase + np.pi, 2 * np.pi) - np.pi
    phase_vec = np.array([np.cos(phase), np.sin(phase)]).flatten()

    obs = np.concatenate([pos_diff, vel_obs, imu_obs, gvec, cmd, prev_action, phase_vec])
    full_obs = np.concatenate([obs, history])
    return obs, full_obs, phase


def update_history(history: np.ndarray, obs: np.ndarray, obs_size: int, cmd_size: int, hist_len: int) -> np.ndarray:
    step_size = obs_size + cmd_size
    history = history.reshape(hist_len, step_size)
    history = np.roll(history, shift=-1, axis=0)
    history[-1] = obs
    return history.flatten()


async def send_actions(kos: pykos.KOS, position: np.ndarray, velocity: np.ndarray) -> None:
    position = np.rad2deg(position)
    velocity = np.rad2deg(velocity)
    commands = [
        {"actuator_id": ac.actuator_id, "position": 0.0 if ac.actuator_id in [11, 12, 13, 14, 15, 21, 22, 23, 24, 25] else position[ac.nn_id], "velocity": velocity[ac.nn_id]}
        for ac in ACTUATOR_LIST
    ]
    await kos.actuator.command_actuators(commands)  # type: ignore[arg-type]


async def configure_actuators(kos: pykos.KOS) -> None:
    for ac in ACTUATOR_LIST:
        if ac.actuator_id in [11, 12, 13, 14, 15, 21, 22, 23, 24, 25]:
            await kos.actuator.configure_actuator(
                actuator_id=ac.actuator_id,
                kp=ac.kp,
                kd=ac.kd,
                torque_enabled=True,
                max_torque=ac.max_torque,
            )


async def reset(kos: pykos.KOS) -> None:
    await kos.sim.reset(
        pos={"x": 0.0, "y": 0.0, "z": 1.01},
        quat={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        joints=[{"name": ac.joint_name, "pos": pos} for ac, pos in zip(ACTUATOR_LIST, DEFAULT_POSITIONS)],
    )


def spawn_kos_sim(no_render: bool) -> tuple[subprocess.Popen, Callable]:
    args = ["kos-sim", "kbot2-feet"] + (["--no-render"] if no_render else [])
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(5)

    def cleanup(sig: int | None = None, frame: types.FrameType | None = None) -> None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        if sig:
            sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    return proc, cleanup


async def main(model_path: str, ip: str, no_render: bool, episode_length: int) -> None:
    model = tf.saved_model.load(model_path)
    try:
        kos = pykos.KOS(ip=ip)
        await kos.sim.get_parameters()
    except Exception:
        _, cleanup = spawn_kos_sim(no_render)
        kos = pykos.KOS()
        for _ in range(5):
            try:
                await kos.sim.get_parameters()
                break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("Failed to connect to KOS-Sim")

    await configure_actuators(kos)
    await reset(kos)

    # command_state = CommandState()
    # keyboard_controller = KeyboardController(command_state.update_from_key)
    # await keyboard_controller.start()

    history = np.zeros(HIST_LEN * (OBS_SIZE + CMD_SIZE))
    cmd = np.array([0.3, 0.0])
    # cmd = command_state.get_command()
    phase = np.array([0, np.pi])
    prev_action = np.zeros(len(ACTUATOR_LIST) * 2)
    obs, full_obs, phase = await get_observation(kos, prev_action, cmd, phase, history)
    if no_render:
        await kos.process_manager.start_kclip("deployment")

    # warm-up
    model.infer(full_obs.reshape(1, -1))

    target_time = time.time() + DT
    end_time = time.time() + episode_length
    while time.time() < end_time:
        action = np.array(model.infer(full_obs.reshape(1, -1))).reshape(-1)
        history = update_history(history, obs, OBS_SIZE, CMD_SIZE, HIST_LEN)
        pos = action[: len(ACTUATOR_LIST)] + DEFAULT_POSITIONS
        vel = action[len(ACTUATOR_LIST) :]
        obs, full_obs, phase = (
            await asyncio.gather(
                get_observation(kos, prev_action, cmd, phase, history),
                send_actions(kos, pos, vel),
            )
        )[0]
        prev_action = action
        if time.time() < target_time:
            await asyncio.sleep(max(0, target_time - time.time()))
        else:
            logger.info("Loop overran by %s seconds", time.time() - target_time)

        target_time += DT

    if no_render:
        await kos.process_manager.stop_kclip("deployment")
    if "cleanup" in locals():
        cleanup()


# Run with:
# python -m ksim_kbot.deploy.sim_history --model_path ksim_kbot/deploy/assets/mlp_example_history
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--episode_length", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    asyncio.run(main(args.model_path, args.ip, args.no_render, args.episode_length))
