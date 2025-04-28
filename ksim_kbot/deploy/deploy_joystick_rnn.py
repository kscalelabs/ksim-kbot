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

# Add TensorFlow memory configuration
import tensorflow as tf
# Limit memory growth to avoid excessive allocation
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
            logger.info(f"Memory growth enabled for {device}")
        except Exception as e:
            logger.warning(f"Error setting memory growth: {e}")

from ksim_kbot.deploy.deploy import Deploy


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


class JoystickRNNDeploy(Deploy):
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
                0.0,  # dof_right_shoulder_pitch_03
                0.0,  # dof_right_shoulder_roll_03
                0.0,  # dof_right_shoulder_yaw_02
                0.0,  # dof_right_elbow_02
                0.0,  # dof_right_wrist_00
                0.0,  # dof_left_shoulder_pitch_03
                0.0,  # dof_left_shoulder_roll_03
                0.0,  # dof_left_shoulder_yaw_02
                0.0,  # dof_left_elbow_02
                0.0,  # dof_left_wrist_00
                0.0,  # dof_right_hip_pitch_04
                0.0,  # dof_right_hip_roll_03
                0.0,  # dof_right_hip_yaw_03
                0.0,  # dof_right_knee_04
                0.0,  # dof_right_ankle_02
                0.0,  # dof_left_hip_pitch_04
                0.0,  # dof_left_hip_roll_03
                0.0,  # dof_left_hip_yaw_03
                0.0,  # dof_left_knee_04
                0.0,  # dof_left_ankle_02
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
            return self.joystick_command.command
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
        projected_gravity = xax.rotate_vector_by_quat(np.array([0.0, 0.0, -9.81]), np.array([quat.w, quat.x, quat.y, quat.z]), inverse=True)
        # projected_gravity = xax.get_projected_gravity_vector_from_quat(np.array([quat.w, quat.x, quat.y, quat.z]))

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

        # observation = np.concatenate([phase_vec, pos_diff, vel_obs, projected_gravity, cmd, self.gait]).reshape(1, -1)
        observation = np.concatenate([pos_obs, vel_obs, projected_gravity]).reshape(1, -1)
        
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
                # Get action from model
                action, next_carry = self.model.infer(observation, self.carry)
                action = np.array(action).reshape(-1)
                self.carry = np.array(next_carry)
                
                logger.debug(f"Action shape: {action.shape}, expected positions: {len(self.actuator_list)}, with action scale: {self.ACTION_SCALE}")
                
                # Handle position-only output from model (no velocities)
                num_actuators = len(self.actuator_list)
                
                # Ensure action has at least the right number of positions
                if action.size == 0:
                    logger.error("Model returned empty action array")
                    position = self.default_positions_rad.copy()
                elif action.size < num_actuators:
                    logger.warning(f"Action size {action.size} is smaller than needed {num_actuators}")
                    position = self.default_positions_rad.copy()
                    position[:action.size] = action[:action.size] * self.ACTION_SCALE + self.default_positions_rad[:action.size]
                else:
                    # Use only the positions from the model (truncate if needed)
                    position = action[:num_actuators] * self.ACTION_SCALE + self.default_positions_rad
                
                # Always use zero velocities since model doesn't output them
                velocity = np.zeros(num_actuators)
                
                # Log what we're doing
                if action.size != num_actuators:
                    logger.info(f"Using {action.size} position values from model, zero velocities for all joints")
                else:
                    logger.debug("Using all position values from model, zero velocities for all joints")

                observation, _ = await asyncio.gather(
                    self.get_observation(),
                    self.send_actions(position, velocity),
                )
                self.prev_action = np.concatenate([action[:num_actuators], velocity]) if action.size > 0 else np.zeros(num_actuators * 2)

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
    parser.add_argument("--model_path", type=str, required=True, help="Path to model, either in assets folder or absolute path")
    parser.add_argument(
        "--mode", type=str, required=True, choices=["sim", "real-deploy", "real-check"], help="Mode of deployment"
    )
    parser.add_argument("--enable_joystick", action="store_true", help="Enable joystick")
    parser.add_argument("--scale_action", type=float, default=1.0, help="Action Scale, default 0.1")
    parser.add_argument("--ip", type=str, default="localhost", help="IP address of KOS")
    parser.add_argument("--episode_length", type=int, default=30, help="Length of episode in seconds")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--cpu_only", action="store_true", help="Force CPU inference only")

    args = parser.parse_args()

    # Force CPU if requested
    if args.cpu_only:
        logger.info("Forcing CPU inference only")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Configure TensorFlow to use less memory
    tf.config.optimizer.set_jit(False)  # Disable XLA compilation
    
    # Configure path
    file_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.isabs(args.model_path):
        model_path = args.model_path
        logger.info(f"Using absolute model path: {model_path}")
    else:
        model_path = os.path.join(file_dir, "assets", args.model_path)
        logger.info(f"Using relative model path: {model_path}")

    # Configure logging
    log_level = "DEBUG" if args.debug else "INFO"

    # Set global log level
    logger.remove()
    logger.add(sys.stderr, level=log_level)  # This will keep the default colorized format

    deploy = JoystickRNNDeploy(args.enable_joystick, model_path, args.mode, args.ip, carry_shape=(5, 128))
    deploy.ACTION_SCALE = args.scale_action

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
