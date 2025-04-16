"""Module for deploying joystick-controlled policies on K-Bot."""

import argparse
import asyncio
import os
import sys

import numpy as np
from loguru import logger  # to be removed
from scipy.spatial.transform import Rotation as R

from ksim_kbot.deploy.deploy import FixedArmDeploy


class JoystickDeploy(FixedArmDeploy):
    """Deploy class for joystick-controlled policies."""

    def __init__(self, enable_joystick: bool, model_path: str, mode: str, ip: str, imu_type: str) -> None:
        super().__init__(model_path, mode, ip)
        self.enable_joystick = enable_joystick
        self.gait = np.asarray([1.25])
        self.imu_type = imu_type

        self.default_positions_rad: np.ndarray = np.array(
            [
                0,
                np.deg2rad(-12),
                0,
                np.deg2rad(30),
                0,  # right arm
                0,
                np.deg2rad(12),
                0,
                np.deg2rad(-30),
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

        self.default_positions_deg: np.ndarray = np.rad2deg(self.default_positions_rad)
        self.phase = np.array([0, np.pi])

        self.rollout_dict = {
            "command": [],
            "pos_diff": [],
            "vel_obs": [],
            "controller_cmd": [],
            "prev_action": [],
            "phase": [],
        }
        if self.imu_type == "raw":
            self.rollout_dict["imu_accel"] = []
            self.rollout_dict["imu_gyro"] = []
        else:
            self.rollout_dict["proj_grav"] = []

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
        (actuator_states, imu, raw_quat) = await asyncio.gather(
            self.kos.actuator.get_actuators_state([ac.actuator_id for ac in self.actuator_list]),
            self.kos.imu.get_imu_values(),
            self.kos.imu.get_quaternion(),
        )


        imu_obs = []
        if self.imu_type == "raw":
            imu_accel = np.array([imu.accel_x, imu.accel_y, imu.accel_z])
            imu_gyro = np.array([imu.gyro_x, imu.gyro_y, imu.gyro_z])
            imu_obs = [imu_accel, imu_gyro]
        else:
            r = R.from_quat([raw_quat.x, raw_quat.y, raw_quat.z, raw_quat.w])
            proj_grav_world = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True)
            proj_grav = proj_grav_world
            imu_obs = [proj_grav]

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

        if self.mode in ["sim", "real-check", "real-deploy"]:
            self.rollout_dict["pos_diff"].append(pos_diff)
            self.rollout_dict["vel_obs"].append(vel_obs)
            if self.imu_type == "raw":
                self.rollout_dict["imu_accel"].append(imu_accel)
                self.rollout_dict["imu_gyro"].append(imu_gyro)
            else:
                self.rollout_dict["proj_grav"].append(proj_grav)
            self.rollout_dict["controller_cmd"].append(cmd)
            self.rollout_dict["prev_action"].append(self.prev_action)
            self.rollout_dict["phase"].append(phase_vec)

        observation = np.concatenate(
            [phase_vec, pos_diff, vel_obs] + imu_obs + [cmd, self.gait, self.prev_action]
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
    parser.add_argument("--scale_action", type=float, default=1.0, help="Action Scale, default 0.1")
    parser.add_argument("--ip", type=str, default="localhost", help="IP address of KOS")
    parser.add_argument("--episode_length", type=int, default=5, help="Length of episode in seconds")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--imu_type", type=str, choices=["raw", "grav"], default="grav", help="Type of IMU data to use in observation ('raw' or 'grav')")

    args = parser.parse_args()
    
    if args.imu_type not in ["raw", "grav"]:
        raise ValueError(f"Unknown imu_type: {args.imu_type}")

    file_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(file_dir, "assets", args.model_path)

    # Configure logging
    log_level = "DEBUG" if args.debug else "INFO"

    # Set global log level
    logger.remove()
    logger.add(sys.stderr, level=log_level)  # This will keep the default colorized format
    logger.add(f"{file_dir}/deployment_checks/last_deployment.log", level=log_level)

    deploy = JoystickDeploy(args.enable_joystick, model_path, args.mode, args.ip, args.imu_type)
    deploy.ACTION_SCALE = args.scale_action

    try:
        asyncio.run(deploy.run(args.episode_length))
    except Exception as e:
        logger.error("Error: %s", e)
        asyncio.run(deploy.disable())
        raise e


if __name__ == "__main__":
    # python -m ksim_kbot.deploy.deploy_joystick \
    # --model_path ksim_kbot/deploy/assets/noisy_joystick_example/tf_model_1576 \
    # --mode sim \
    # --scale_action 1.0 \
    # --debug
    main()
