"""Module for deploying joystick-controlled policies on K-Bot."""

import argparse
import asyncio
import os
import sys
import time

import numpy as np
import xax
from askin import KeyboardController
from loguru import logger  # to be removed

from ksim_kbot.deploy.deploy import Deploy


class UserInput:
    def __init__(self, joystick_step_size: float, action_scale: float, action_scale_step_size: float) -> None:
        self.joystick_command = np.array([0.0, 0.0, 0.0])  # x, y, yaw
        self.joystick_step_size = joystick_step_size

        self.initial_action_scale = action_scale
        self.action_scale = action_scale
        self.action_scale_step_size = action_scale_step_size

        logger.info(f"User Input Initialized: Joystick Step Size: {self.joystick_step_size}, Action Scale: {self.action_scale}, Action Scale Step Size: {self.action_scale_step_size}")
        
    async def update(self, key: str) -> None:
        if key == "a":
            self.joystick_command[1] += self.joystick_step_size
            logger.info(f"Joystick: {self.joystick_command}")
        elif key == "d":
            self.joystick_command[1] += -1 * self.joystick_step_size
            logger.info(f"Joystick: {self.joystick_command}")
        elif key == "w":
            self.joystick_command[0] += self.joystick_step_size
            logger.info(f"Joystick: {self.joystick_command}")
        elif key == "s":
            self.joystick_command[0] += -1 * self.joystick_step_size
            logger.info(f"Joystick: {self.joystick_command}")
        elif key == "=": # + Key without shift!
            self.action_scale += self.action_scale_step_size
            if self.action_scale > 1.0:
                self.action_scale = 1.0
            logger.info(f"Action Scale Increased: {self.action_scale}")
        elif key == "-":
            self.action_scale += -self.action_scale_step_size
            if self.action_scale < 0.0:
                self.action_scale = 0.0
            logger.info(f"Action Scale Decreased: {self.action_scale}")
        elif key == "0":
            self.action_scale = self.initial_action_scale
            logger.info(f"Action Scale Reset: {self.action_scale}")
class JoystickRNNDeploy(Deploy):
    """Deploy class for joystick-controlled policies."""

    def __init__(
        self, enable_joystick: bool, model_path: str, mode: str, ip: str, carry_shape: tuple[int, int], action_scale: float, action_scale_step_size: float
    ) -> None:
        
        super().__init__(model_path, mode, ip)
        self.enable_joystick = enable_joystick

        self.joystick_step_size = 0.01
        if not self.enable_joystick:
            self.joystick_step_size = 0.0

        # Action Scale Limits
        if action_scale > 1.0:
            action_scale = 1.0
            logger.warning("Action Scale Exceeded 1.0, Clipping to 1.0")
        if action_scale < 0.0:
            action_scale = 0.0
            logger.warning("Action Scale Less than 0.0, Clipping to 0.0")

        logger.info(f"Action Scale: {action_scale}")

        self.user_input = UserInput(joystick_step_size=self.joystick_step_size, action_scale=action_scale, action_scale_step_size=action_scale_step_size)
        self.controller = KeyboardController(key_handler=self.user_input.update, timeout=0.001)

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
            "model_name": "/".join(model_path.split("/")[-2:]),
            "timestamp": [],
            "loop_overrun_time": [],
            "command": [],
            "pos_diff": [],
            "vel_obs": [],
            "projected_gravity": [],
            "quat": [],
            "imu_gyro": [],
            "controller_cmd": [],
            "prev_action": [],
            "phase": [],
            "imu_accel": [],
            "euler_angles": [],
        }

    def get_command(self) -> np.ndarray:
        """Get command from the joystick."""
        if self.enable_joystick:
            return self.user_input.joystick_command * self.user_input.action_scale
        else:
            return np.array([0.0, 0.0, 0.0])

    async def get_observation(self) -> np.ndarray:
        """Get observation from the robot for joystick-controlled policies.

        Returns:
            Observation vector and updated phase
        """
        # * IMU Observation
        (actuator_states, imu, euler_angles, quat) = await asyncio.gather(
            self.kos.actuator.get_actuators_state([ac.actuator_id for ac in self.actuator_list]),
            self.kos.imu.get_imu_values(),
            self.kos.imu.get_euler_angles(),
            self.kos.imu.get_quaternion(),
        )

        imu_gyro = np.array([imu.gyro_x, imu.gyro_y, imu.gyro_z])
        projected_gravity = xax.get_projected_gravity_vector_from_quat(np.array([quat.w, quat.x, quat.y, quat.z]))

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

        imu_accel = np.array([imu.accel_x, imu.accel_y, imu.accel_z])
        euler_angles = np.array([euler_angles.roll, euler_angles.pitch, euler_angles.yaw])

        self.rollout_dict["timestamp"].append(time.time())
        self.rollout_dict["pos_diff"].append(pos_diff)
        self.rollout_dict["vel_obs"].append(vel_obs)
        self.rollout_dict["projected_gravity"].append(projected_gravity)
        self.rollout_dict["quat"].append(np.array([quat.w, quat.x, quat.y, quat.z]))
        self.rollout_dict["imu_gyro"].append(imu_gyro)
        self.rollout_dict["controller_cmd"].append(cmd)
        self.rollout_dict["prev_action"].append(self.prev_action)
        self.rollout_dict["phase"].append(phase_vec)
        self.rollout_dict["imu_accel"].append(imu_accel)
        self.rollout_dict["euler_angles"].append(euler_angles)

        observation = np.concatenate([phase_vec, pos_diff, vel_obs, projected_gravity, cmd, self.gait]).reshape(1, -1)

        return observation

    async def warmup(self) -> None:
        """Warmup the robot."""
        observation = await self.get_observation()
        self.model.infer(observation, self.carry)

    async def preflight(self) -> None:
        """Preflight the robot."""
        await super().preflight()
        
        await self.controller.start()

    async def postflight(self) -> None:
        """Postflight the robot."""
        await super().postflight()
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
                position = action[: len(self.actuator_list)] * self.user_input.action_scale + self.default_positions_rad
                velocity = action[len(self.actuator_list) :] * self.user_input.action_scale

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


def main() -> None:
    """Parse arguments and run the deploy script."""
    parser = argparse.ArgumentParser(description="Deploy a SavedModel on K-Bot")
    parser.add_argument("--model_path", type=str, required=True, help="File in assets folder eg. mlp_example")
    parser.add_argument(
        "--mode", type=str, required=True, choices=["sim", "real-deploy", "real-check"], help="Mode of deployment"
    )
    parser.add_argument("--enable_joystick", action="store_true", help="Enable joystick")
    parser.add_argument("--scale_action", type=float, default=0.1, help="Action Scale, default 0.1")
    parser.add_argument("--scale_action_step_size", type=float, default=0.05, help="Action Scale Step Size, default 0.05")
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

    deploy = JoystickRNNDeploy(args.enable_joystick, model_path, args.mode, args.ip, carry_shape=(5, 256), action_scale=args.scale_action, action_scale_step_size=args.scale_action_step_size)

    try:
        asyncio.run(deploy.run(args.episode_length))
    except Exception as e:
        logger.error("Error: %s", e)
        asyncio.run(deploy.disable())
        raise e


"""
python -m ksim_kbot.deploy.deploy_joystick_rnn \
--model_path noisy_joystick_example/tf_model_1407 \
--mode sim \
--scale_action 1.0 \
--debug
"""

if __name__ == "__main__":
    main()
