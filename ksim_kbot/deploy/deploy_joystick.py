"""Module for deploying joystick-controlled policies on K-Bot."""

import argparse
import asyncio
import os
import sys
import time

import numpy as np
from loguru import logger  # to be removed

from ksim_kbot.deploy.deploy import FixedArmDeploy


class JoystickDeploy(FixedArmDeploy):
    """Deploy class for joystick-controlled policies."""

    def __init__(self, enable_joystick: bool, model_path: str, mode: str, ip: str) -> None:
        super().__init__(model_path, mode, ip)
        self.enable_joystick = enable_joystick
        self.gait = np.asarray([1.25])

        self.default_positions_rad: np.ndarray = np.array(
            [
                # right arm
                0,
                0,
                0,
                1.57,
                0,
                # left arm
                0,
                0,
                0,
                -1.57,
                0,
                # right leg
                -0.237,
                0,
                0,
                -0.51,
                0.2356,
                # left leg
                0.237,
                0,
                0,
                0.51,
                -0.2356,
            ]
        )

        self.default_positions_deg: np.ndarray = np.rad2deg(self.default_positions_rad)
        self.phase = np.array([0, np.pi])

        self.rollout_dict = {
            "model_name": "/".join(model_path.split("/")[-2:]),
            "timestamp": [],
            "loop_overrun_time": [],
            "command": [],
            "pos_diff": [],
            "vel_obs": [],
            "imu_accel": [],
            "imu_gyro": [],
            "controller_cmd": [],
            "prev_action": [],
            "phase": [],
        }

    def get_command(self) -> np.ndarray:
        """Get command from the joystick."""
        if self.enable_joystick:
            return np.array([0.0, 0.0, 0.0])
        else:
            return np.array([0.0, 0.0, 0.0])

    async def get_observation(self) -> np.ndarray:
        """Get observation from the robot for joystick-controlled policies.

        Returns:
            Observation vector and updated phase
        """
        # * IMU Observation
        (actuator_states, imu) = await asyncio.gather(
            self.kos.actuator.get_actuators_state([ac.actuator_id for ac in self.actuator_list]),
            self.kos.imu.get_imu_values(),
        )
        imu_accel = np.array([imu.accel_x, imu.accel_y, imu.accel_z])
        imu_gyro = np.array([imu.gyro_x, imu.gyro_y, imu.gyro_z])

        # * Pos Diff. Difference of current position from default position
        state_dict_pos = {state.actuator_id: state.position for state in actuator_states.states}
        pos_obs = [state_dict_pos[ac.actuator_id] for ac in sorted(self.actuator_list, key=lambda x: x.nn_id)]
        pos_obs = np.deg2rad(np.array(pos_obs))
        pos_diff = pos_obs - self.default_positions_rad  #! K-Sim is in radians

        # * Vel Obs. Velocity at each joint
        state_dict_vel = {state.actuator_id: state.velocity for state in actuator_states.states}
        vel_obs = np.deg2rad(
            np.array([state_dict_vel[ac.actuator_id] for ac in sorted(self.actuator_list, key=lambda x: x.nn_id)])
        )

        # * Phase, tracking a sinusoidal
        self.phase += 2 * np.pi * self.gait * self.DT
        self.phase = np.fmod(self.phase + np.pi, 2 * np.pi) - np.pi
        phase_vec = np.array([np.cos(self.phase), np.sin(self.phase)]).flatten()

        cmd = self.get_command()

        self.rollout_dict["timestamp"].append(time.time())
        self.rollout_dict["pos_diff"].append(pos_diff)
        self.rollout_dict["vel_obs"].append(vel_obs)
        self.rollout_dict["imu_accel"].append(imu_accel)
        self.rollout_dict["imu_gyro"].append(imu_gyro)
        self.rollout_dict["controller_cmd"].append(cmd)
        self.rollout_dict["prev_action"].append(self.prev_action)
        self.rollout_dict["phase"].append(phase_vec)

        observation = np.concatenate(
            [phase_vec, pos_diff, vel_obs, imu_accel, imu_gyro, cmd, self.gait, self.prev_action]
        ).reshape(1, -1)

        return observation


def main() -> None:
    """Parse arguments and run the deploy script."""
    parser = argparse.ArgumentParser(description="Deploy a SavedModel on K-Bot")
    parser.add_argument("--model_path", type=str, required=True, help="File in assets folder eg. mlp_example")
    parser.add_argument(
        "--mode", type=str, required=True, choices=["sim", "real-deploy", "real-check"], help="Mode of deployment"
    )
    parser.add_argument("--enable_joystick", action="store_true", help="Enable joystick")
    parser.add_argument("--scale_action", type=float, default=0.1, help="Action Scale, default 0.1")
    parser.add_argument("--ip", type=str, default="localhost", help="IP address of KOS")
    parser.add_argument("--episode_length", type=int, default=30, help="Length of episode in seconds")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    file_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(file_dir, "assets", args.model_path)

    # Configure logging
    log_level = "DEBUG" if args.debug else "INFO"

    # Set global log level
    logger.remove()
    logger.add(sys.stderr, level=log_level)  # This will keep the default colorized format

    deploy = JoystickDeploy(args.enable_joystick, model_path, args.mode, args.ip)
    deploy.ACTION_SCALE = args.scale_action

    try:
        asyncio.run(deploy.run(args.episode_length))
    except Exception as e:
        logger.error("Error: %s", e)
        asyncio.run(deploy.disable())
        raise e


"""
python -m ksim_kbot.deploy.deploy_joystick \
--model_path noisy_joystick_example/tf_model_1576 \
--mode sim \
--scale_action 1.0 \
--debug
"""

if __name__ == "__main__":
    main()
