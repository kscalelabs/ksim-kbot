"""Class to deploy a SavedModel on K-Bot."""

import argparse
import asyncio
import os
import time
from loguru import logger
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
import pykos
import tensorflow as tf

@dataclass
class Actuator:
    actuator_id: int
    nn_id: int
    kp: float
    kd: float
    max_torque: float
    joint_name: str


#*********************#
#* Base Deploy Class  #
#*********************#



class Deploy(ABC):
    """Abstract base class for deploying a SavedModel on K-Bot.
    """
    
    # Class-level constants
    DT = 0.02  # Policy time step (50Hz)
    GAIT_DT = 1.25
    GRAVITY = 9.81  # m/s
    ACTION_SCALE = 1.0


    actuator_list: list[Actuator] = [
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

    def __init__(self, model_path: str, mode: str, ip: str = "localhost"):
        self.model_path = model_path
        self.mode = mode
        self.ip = ip
        self.model = tf.saved_model.load(model_path)
        self.kos = pykos.KOS(ip=self.ip)

        self.default_positions_deg = np.zeros(len(self.actuator_list))
        self.default_positions_rad = np.zeros(len(self.actuator_list))
        
        self.prev_action = np.zeros(len(self.actuator_list) * 2)
        
    

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
                "position": (
                   position[ac.nn_id]
                ),
                "velocity": velocity[ac.nn_id],
            }
            for ac in self.actuator_list
        ]

        if self.mode == "real-deploy":
            await self.kos.actuator.command_actuators(actuator_commands)
        elif self.mode == "real-check":
            logger.info("Sending actuator commands: %s", actuator_commands)

    async def configure_actuators(self) -> None:
        """Configure all actuators with their respective parameters."""
        for ac in self.actuator_list:
            await self.kos.actuator.configure_actuator(
                actuator_id=ac.actuator_id,
                kp=ac.kp,
                kd=ac.kd,
                torque_enabled=True,
                max_torque=ac.max_torque,
            )

    async def reset(self) -> None:
        """Reset all actuators to their default positions."""

        if self.mode == "real":
            reset_commands: list[pykos.services.actuator.ActuatorCommand] = [
                {
                    "actuator_id": ac.actuator_id,
                    "position": pos,
                    "velocity": 0.0,
                }
                for ac, pos in zip(self.actuator_list, self.default_positions_deg)
            ]

            await self.kos.actuator.command_actuators(reset_commands)
        elif self.mode == "sim":
            await self.kos.sim.reset(
                pos={"x": 0.0, "y": 0.0, "z": 1.01},
                quat={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                joints=[{"name": ac.joint_name, "pos": pos} for ac, pos in zip(self.actuator_list, self.default_positions_deg)],
            )

    async def disable(self) -> None:
        """Disable all actuators."""
        for ac in self.actuator_list:
            await self.kos.actuator.configure_actuator(
                actuator_id=ac.actuator_id,
                torque_enabled=False,
            )
    
    @abstractmethod
    async def get_observation(self) -> np.ndarray:
        """
        Get observation from the robot.
        """
        pass
    
    async def run(self, episode_length: int) -> None:
        """Run the policy on the robot.
        
        Args:
            episode_length: Length of the episode in seconds
        """
        self.model = tf.saved_model.load(self.model_path)
        self.kos = pykos.KOS(ip=self.ip)
        
        await self.disable()
        time.sleep(1)
        logger.info("Configuring actuators...")
        await self.configure_actuators()
        await asyncio.sleep(1)
        logger.info("Resetting...")

        observation = await self.get_observation()
        # warm up model
        self.model.infer(observation)

        await self.reset()

        self.reset_phase()
        
        logger.warning(f"Deploying with Action Scale: {self.ACTION_SCALE}")
        if self.mode == "real-deploy":
            input("Press Enter to continue...")

        for i in range(5, -1, -1):
            logger.info(f"Starting in {i} seconds...")
            await asyncio.sleep(1)

        await self.reset()

        target_time = time.time() + self.DT
        observation = await self.get_observation()

        end_time = time.time() + episode_length

        try:
            while time.time() < end_time:
                
                action = np.array(self.model.infer(observation)).reshape(-1) 

                #! Only Scale Action on observation but not onto Default Positions
                position = action[: len(self.actuator_list)] * self.ACTION_SCALE + self.default_positions_rad
                velocity = action[len(self.actuator_list) :] * self.ACTION_SCALE
                
                observation, _ = await asyncio.gather(
                    self.get_observation(),
                    self.send_actions(position, velocity),
                )
                self.prev_action = action.copy()

                if time.time() < target_time:
                    await asyncio.sleep(max(0, target_time - time.time()))
                else:
                    logger.info(f"Loop overran by {time.time() - target_time} seconds")

                target_time += self.DT

        except asyncio.CancelledError:
            logger.info("Exiting...")
            await self.disable()
            logger.info("Actuators disabled")
            raise KeyboardInterrupt

        logger.info("Episode finished!")
        await self.disable()


#*********************#
#* Fixed Arm Deploy   #
#*********************#


class FixedArmDeploy(Deploy):
    """Deploy class for fixed-arm policies."""

    def __init__(self, model_path: str, mode: str, ip: str = "localhost"):
        super().__init__(model_path, mode, ip)

    async def send_actions(self, position: np.ndarray, velocity: np.ndarray) -> None:
        """Send actions to the robot's actuators with additional functionality.
        
        Args:
            position: Position commands in radians
            velocity: Velocity commands in radians/s
        """
        position = np.rad2deg(position)
        velocity = np.rad2deg(velocity)
        actuator_commands: list[pykos.services.actuator.ActuatorCommand] = [
            {
                "actuator_id": ac.actuator_id,
                "position": (
                   0.0 if ac.actuator_id in [11, 12, 13, 14, 15, 21, 22, 23, 24, 25] else position[ac.nn_id]
                ),
                "velocity": velocity[ac.nn_id],
            }
            for ac in self.actuator_list
        ]

        if self.mode == "real-deploy":
            await self.kos.actuator.command_actuators(actuator_commands)
        elif self.mode == "real-check":
            logger.info(f"Sending actuator commands: {actuator_commands}")
        else:
            # For all other modes, log and send commands
            await self.kos.actuator.command_actuators(actuator_commands)

        
if __name__ == "__main__":
    logger.error("Not a standalone script")
    exit(1)
