"""Module for deploying joystick-controlled policies on K-Bot."""

import argparse
import asyncio
import os
import sys
import time

import numpy as np
from askin import KeyboardController
from loguru import logger  # to be removed
from scipy.spatial.transform import Rotation

from ksim_kbot.deploy.deploy import FixedArmDeploy


class JoystickCommand:
    def __init__(self) -> None:
        self.command = np.array([0.0, 0.0, 0.0])  # x, y, yaw
        self.step_size = 0.01

    async def update(self, key: str) -> None:
        if key == "a":
            self.command[1] += self.step_size
        elif key == "d":
            self.command[1] += -1 * self.step_size
        elif key == "w":
            self.command[0] += self.step_size
        elif key == "s":
            self.command[0] += -1 * self.step_size


class JoystickRNNDeploy(FixedArmDeploy):
    """Deploy class for joystick-controlled policies."""

    def __init__(
        self, enable_joystick: bool, model_path: str, mode: str, ip: str, carry_shape: tuple[int, int]
    ) -> None:
        super().__init__(model_path, mode, ip)
        self.enable_joystick = enable_joystick

        if self.enable_joystick:
            self.joystick_command = JoystickCommand()
            self.controller = KeyboardController(key_handler=self.joystick_command.update, timeout=0.001)

        self.gait = np.asarray([1.25])

        self.carry = np.zeros(carry_shape)[None, :]

        self.default_positions_rad: np.ndarray = np.array(
            [
                0,
                0,
                0,
                1.4,
                0,  # right arm
                0,
                0,
                0,
                -1.4,
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
            "projected_gravity": [],
            "imu_gyro": [],
            "controller_cmd": [],
            "prev_action": [],
            "phase": [],
        }

    def get_command(self) -> np.ndarray:
        """Get command from the joystick."""
        if self.enable_joystick:
            return self.joystick_command.command
        else:
            return np.array([0.0, 0.0, 0.0])

    async def get_observation(self) -> np.ndarray:
        """Get observation from the robot for joystick-controlled policies.

        Returns:
            Observation vector and updated phase
        """
        # * IMU Observation
        (actuator_states, imu, quat) = await asyncio.gather(
            self.kos.actuator.get_actuators_state([ac.actuator_id for ac in self.actuator_list]),
            self.kos.imu.get_imu_values(),
            self.kos.imu.get_quaternion(),
        )

        imu_gyro = np.array([imu.gyro_x, imu.gyro_y, imu.gyro_z])

        r = Rotation.from_quat(np.array([quat.w, quat.x, quat.y, quat.z]), scalar_first=True)
        proj_grav_world = r.apply(np.array([0.0, 0.0, 1.0]), inverse=True)
        projected_gravity = proj_grav_world

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
            self.rollout_dict["projected_gravity"].append(projected_gravity)
            self.rollout_dict["imu_gyro"].append(imu_gyro)
            self.rollout_dict["controller_cmd"].append(cmd)
            self.rollout_dict["prev_action"].append(self.prev_action)
            self.rollout_dict["phase"].append(phase_vec)

        observation = np.concatenate(
            [phase_vec, pos_diff, vel_obs, projected_gravity, imu_gyro, cmd, self.gait]
        ).reshape(1, -1)

        return observation

    async def warmup(self) -> None:
        """Warmup the robot."""
        observation = await self.get_observation()
        self.model.infer(observation, self.carry)

    async def preflight(self) -> None:
        """Preflight the robot."""
        await super().preflight()
        if self.enable_joystick:
            await self.controller.start()

    async def postflight(self) -> None:
        """Postflight the robot."""
        await super().postflight()
        if self.enable_joystick:
            await self.controller.stop()

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
                action, next_carry = self.model.infer(observation, self.carry)
                action = np.array(action).reshape(-1)
                self.carry = np.array(next_carry)

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
                else:
                    logger.info(f"Loop overran by {time.time() - target_time} seconds")

                target_time += self.DT

        except asyncio.CancelledError:
            logger.info("Exiting...")
            await self.postflight()
            raise KeyboardInterrupt

        await self.postflight()


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
    parser.add_argument("--episode_length", type=int, default=5, help="Length of episode in seconds")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    file_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(file_dir, "assets", args.model_path)

    # Configure logging
    log_level = "DEBUG" if args.debug else "INFO"

    # Set global log level
    logger.remove()
    logger.add(sys.stderr, level=log_level)  # This will keep the default colorized format
    logger.add(f"{file_dir}/deployment_checks/last_deployment.log", level=log_level)

    deploy = JoystickRNNDeploy(args.enable_joystick, model_path, args.mode, args.ip, carry_shape=(5, 256))
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
