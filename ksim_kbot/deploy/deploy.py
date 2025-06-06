"""Class to deploy a SavedModel on K-Bot."""

import asyncio
import os
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pykos
import tensorflow as tf
from loguru import logger


@dataclass
class Actuator:
    actuator_id: int
    nn_id: int
    kp: float
    kd: float
    max_torque: float
    joint_name: str


class Deploy(ABC):
    """Abstract base class for deploying a policy on K-Bot."""

    # Class-level constants
    DT = 0.02  # Policy time step (50Hz)
    GRAVITY = 9.81  # m/s
    ACTION_SCALE = 1.0

    actuator_list: list[Actuator] = [
        # Right arm (nn_id 0-4)
        Actuator(actuator_id=21, nn_id=0, kp=40.0, kd=4.0, max_torque=40.0, joint_name="dof_right_shoulder_pitch_03"),
        Actuator(actuator_id=22, nn_id=1, kp=40.0, kd=4.0, max_torque=40.0, joint_name="dof_right_shoulder_roll_03"),
        Actuator(actuator_id=23, nn_id=2, kp=30.0, kd=1.0, max_torque=14.0, joint_name="dof_right_shoulder_yaw_02"),
        Actuator(actuator_id=24, nn_id=3, kp=30.0, kd=1.0, max_torque=14.0, joint_name="dof_right_elbow_02"),
        Actuator(
            actuator_id=25, nn_id=4, kp=20.0, kd=0.45473329537059787, max_torque=1.0, joint_name="dof_right_wrist_00"
        ),
        # Left arm (nn_id 5-9)
        Actuator(actuator_id=11, nn_id=5, kp=40.0, kd=4.0, max_torque=40.0, joint_name="dof_left_shoulder_pitch_03"),
        Actuator(actuator_id=12, nn_id=6, kp=40.0, kd=4.0, max_torque=40.0, joint_name="dof_left_shoulder_roll_03"),
        Actuator(actuator_id=13, nn_id=7, kp=30.0, kd=1.0, max_torque=14.0, joint_name="dof_left_shoulder_yaw_02"),
        Actuator(actuator_id=14, nn_id=8, kp=30.0, kd=1.0, max_torque=14.0, joint_name="dof_left_elbow_02"),
        Actuator(
            actuator_id=15, nn_id=9, kp=20.0, kd=0.45473329537059787, max_torque=1.0, joint_name="dof_left_wrist_00"
        ),
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

    def __init__(self, model_path: str, mode: str, ip: str) -> None:
        self.model_path = model_path
        self.mode = mode
        self.ip = ip
        self.model = tf.saved_model.load(model_path)
        self.kos = pykos.KOS(ip=self.ip)

        self.default_positions_deg = np.zeros(len(self.actuator_list))
        self.default_positions_rad = np.zeros(len(self.actuator_list))

        self.prev_action = np.zeros(len(self.actuator_list) * 2)

        self.rollout_dict: Dict[str, List[Any]] = {"command": []}

    async def send_actions(self, position: np.ndarray, velocity: np.ndarray) -> None:
        """Send actions to the robot's actuators.

        Args:
            position: Position commands in radians
            velocity: Velocity commands in radians/s
        """
        position = np.rad2deg(position)
        velocity = np.rad2deg(velocity)
        actuator_commands: list[pykos.services.actuator.ActuatorCommand] = [
            {
                "actuator_id": ac.actuator_id,
                "position": (position[ac.nn_id]),
                "velocity": velocity[ac.nn_id],
            }
            for ac in self.actuator_list
        ]

        if self.mode == "real-deploy":
            await self.kos.actuator.command_actuators(actuator_commands)
            self.rollout_dict["command"].append(actuator_commands)
        elif self.mode == "real-check":
            logger.info(f"Sending actuator commands: {actuator_commands}")
            self.rollout_dict["command"].append(actuator_commands)
        elif self.mode == "sim":
            # For all other modes, log and send commands
            await self.kos.actuator.command_actuators(actuator_commands)
            self.rollout_dict["command"].append(actuator_commands)

    async def reset(self) -> None:
        """Reset all actuators to their default positions."""
        reset_commands: list[pykos.services.actuator.ActuatorCommand] = [
            {
                "actuator_id": ac.actuator_id,
                "position": 0.0,
                "velocity": 0.0,
            }
            for ac in self.actuator_list
        ]
        if self.mode in {"real-check", "real-deploy"}:
            await self.kos.actuator.command_actuators(reset_commands)

        elif self.mode == "sim":
            await self.kos.sim.reset(
                pos={"x": 0.0, "y": 0.0, "z": 1.01},
                quat={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            )
            assert self.default_positions_deg is not None, "Default positions are not initialized"
            await self.kos.actuator.command_actuators(reset_commands)

        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    async def disable(self) -> None:
        """Disable all actuators."""
        for ac in self.actuator_list:
            await self.kos.actuator.configure_actuator(
                actuator_id=ac.actuator_id,
                kp=ac.kp,
                kd=ac.kd,
                torque_enabled=False,
                max_torque=ac.max_torque,
            )

    async def enable(self) -> None:
        """Enable all actuators."""
        for ac in self.actuator_list:
            await self.kos.actuator.configure_actuator(
                actuator_id=ac.actuator_id,
                kp=ac.kp,
                kd=ac.kd,
                torque_enabled=True,
                max_torque=ac.max_torque,
            )

    def save_rollout(self) -> None:
        """Save the rollout to a file."""
        if self.rollout_dict is not None:
            file_dir = os.path.dirname(os.path.abspath(__file__))
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            date = time.strftime("%Y%m%d")
            date_dir = f"{file_dir}/deployment_logs/{date}"

            # Check if date directory exists, create if it doesn't
            if not os.path.exists(date_dir):
                os.makedirs(date_dir)

            with open(f"{date_dir}/{self.mode}_{timestamp}.pkl", "wb") as f:
                pickle.dump(self.rollout_dict, f)

    @abstractmethod
    async def get_observation(self) -> np.ndarray:
        """Get observation from the robot."""
        pass

    async def warmup(self) -> None:
        """Warmup the robot."""
        observation = await self.get_observation()
        self.model.infer(observation)

    async def preflight(self) -> None:
        """Preflight checks for the robot."""
        await self.enable()
        await asyncio.sleep(1)
        logger.info("Resetting...")
        await self.reset()

        zero_pos_target_list = []
        for ac in self.actuator_list:
            zero_pos_target_list.append(
                {
                    "actuator_id": ac.actuator_id,
                    "position": 0.0,
                }
            )

        actuator_commands: list[pykos.services.actuator.ActuatorCommand] = [
            {
                "actuator_id": 12,
                "position": -12.0 / 2.0,
                "velocity": 0.0,
            },
            {
                "actuator_id": 22,
                "position": 12.0 / 2.0,
                "velocity": 0.0,
            },
            {
                "actuator_id": 14,
                "position": -30.0 / 2.0,
                "velocity": 0.0,
            },
            {
                "actuator_id": 24,
                "position": 30.0 / 2.0,
                "velocity": 0.0,
            },
        ]

        await self.kos.actuator.command_actuators(actuator_commands)

        reset_commands: list[pykos.services.actuator.ActuatorCommand] = [
            {
                "actuator_id": ac.actuator_id,
                "position": pos,
                "velocity": 0.0,
            }
            for ac, pos in zip(self.actuator_list, self.default_positions_deg)
        ]

        await self.kos.actuator.command_actuators(reset_commands)

        logger.warning(f"Deploying with Action Scale: {self.ACTION_SCALE}")
        if self.mode == "real-deploy":
            input("Press Enter to continue...")

        await self.warmup()

        if self.mode == "real-deploy":
            for i in range(5, -1, -1):
                logger.info(f"Starting in {i} seconds...")
                await asyncio.sleep(1)

        if self.mode == "sim":
            await self.kos.sim.reset(pos={"x": 0.0, "y": 0.0, "z": 1.01}, quat={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0})

    async def postflight(self) -> None:
        """Postflight actions for the robot."""
        await self.disable()
        logger.info("Actuators disabled")
        self.save_rollout()
        logger.info("Episode finished!")

    async def run(self, episode_length: int) -> None:
        """Run the policy on the robot.

        Args:
            episode_length: Length of the episode in seconds
        """
        await self.preflight()

        observation = await self.get_observation()
        target_time = time.time() + self.DT
        end_time = time.time() + episode_length

        try:
            while time.time() < end_time:
                action = np.array(self.model.infer(observation)).reshape(-1)

                #! Only scale action on observation but not onto default positions
                position = action[: len(self.actuator_list)] * self.ACTION_SCALE + self.default_positions_rad
                velocity = action[len(self.actuator_list) :] * self.ACTION_SCALE

                observation, _ = await asyncio.gather(
                    self.get_observation(),
                    self.send_actions(position, velocity),
                )
                self.prev_action = action.copy()

                if time.time() < target_time:
                    logger.debug(f"Sleeping for {max(0, target_time - time.time())} seconds")
                    await asyncio.sleep(max(0, target_time - time.time()))
                    self.rollout_dict["loop_overrun_time"].append(0.0)
                else:
                    logger.info(f"Loop overran by {time.time() - target_time} seconds")
                    self.rollout_dict["loop_overrun_time"].append(time.time() - target_time)

                target_time += self.DT

        except asyncio.CancelledError:
            logger.info("Exiting...")
            await self.postflight()
            raise KeyboardInterrupt

        await self.postflight()
