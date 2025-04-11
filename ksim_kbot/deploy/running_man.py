#!/usr/bin/env python3
"""Script to play back trajectories from traj_dict.pkl on K-Bot."""

import argparse
import asyncio
import logging
import time
import pickle
import numpy as np
import pykos
from typing import Dict

logger = logging.getLogger(__name__)

# Constants from real.py
GRAVITY = 9.81  # m/s
ACTION_SCALE = 1.0

# Map joint names to actuator IDs (copied from real.py)
JOINT_TO_ACTUATOR = {
    "dof_right_shoulder_pitch_03": 21,
    "dof_right_shoulder_roll_03": 22,
    "dof_right_shoulder_yaw_02": 23,
    "dof_right_elbow_02": 24,
    "dof_right_wrist_00": 25,
    "dof_left_shoulder_pitch_03": 11,
    "dof_left_shoulder_roll_03": 12,
    "dof_left_shoulder_yaw_02": 13,
    "dof_left_elbow_02": 14,
    "dof_left_wrist_00": 15,
    "dof_right_hip_pitch_04": 41,
    "dof_right_hip_roll_03": 42,
    "dof_right_hip_yaw_03": 43,
    "dof_right_knee_04": 44,
    "dof_right_ankle_02": 45,
    "dof_left_hip_pitch_04": 31,
    "dof_left_hip_roll_03": 32,
    "dof_left_hip_yaw_03": 33,
    "dof_left_knee_04": 34,
    "dof_left_ankle_02": 35,
}

# PD gains and torque limits from real.py
ACTUATOR_CONFIGS = {
    # Right arm
    21: {"kp": 40.0, "kd": 4.0, "max_torque": 40.0},
    22: {"kp": 40.0, "kd": 4.0, "max_torque": 40.0},
    23: {"kp": 30.0, "kd": 1.0, "max_torque": 14.0},
    24: {"kp": 30.0, "kd": 1.0, "max_torque": 14.0},
    25: {"kp": 20.0, "kd": 0.45, "max_torque": 1.0},
    # Left arm
    11: {"kp": 40.0, "kd": 4.0, "max_torque": 40.0},
    12: {"kp": 40.0, "kd": 4.0, "max_torque": 40.0},
    13: {"kp": 30.0, "kd": 1.0, "max_torque": 14.0},
    14: {"kp": 30.0, "kd": 1.0, "max_torque": 14.0},
    15: {"kp": 20.0, "kd": 0.45, "max_torque": 1.0},
    # Right leg
    41: {"kp": 85.0, "kd": 5.0, "max_torque": 60.0},
    42: {"kp": 40.0, "kd": 4.0, "max_torque": 40.0},
    43: {"kp": 40.0, "kd": 4.0, "max_torque": 40.0},
    44: {"kp": 85.0, "kd": 5.0, "max_torque": 60.0},
    45: {"kp": 30.0, "kd": 1.0, "max_torque": 14.0},
    # Left leg
    31: {"kp": 85.0, "kd": 5.0, "max_torque": 60.0},
    32: {"kp": 40.0, "kd": 4.0, "max_torque": 40.0},
    33: {"kp": 40.0, "kd": 4.0, "max_torque": 40.0},
    34: {"kp": 85.0, "kd": 5.0, "max_torque": 60.0},
    35: {"kp": 30.0, "kd": 1.0, "max_torque": 14.0},
}


async def configure_actuators(kos: pykos.KOS) -> None:
    """Configure all actuators with their PD gains and torque limits."""
    for actuator_id, config in ACTUATOR_CONFIGS.items():
        await kos.actuator.configure_actuator(
            actuator_id=actuator_id,
            kp=config["kp"],
            kd=config["kd"],
            torque_enabled=True,
            max_torque=config["max_torque"],
        )


async def disable_actuators(kos: pykos.KOS) -> None:
    """Disable all actuators."""
    for actuator_id in ACTUATOR_CONFIGS:
        await kos.actuator.configure_actuator(
            actuator_id=actuator_id,
            torque_enabled=False,
        )


async def send_trajectory_frame(kos: pykos.KOS, positions: Dict[str, float]) -> None:
    """Send a single frame of trajectory positions to the robot."""
    actuator_commands = []
    for joint_name, position in positions.items():
        if joint_name != "floating_base" and joint_name in JOINT_TO_ACTUATOR:
            actuator_id = JOINT_TO_ACTUATOR[joint_name]
            actuator_commands.append({
                "actuator_id": actuator_id,
                "position": np.rad2deg(position) * ACTION_SCALE,
                "velocity": 0.0,  # Using only position control for simplicity
            })
    
    await kos.actuator.command_actuators(actuator_commands)


async def play_trajectory(kos: pykos.KOS, traj_dict: Dict, frequency: float, num_loops: int = 1) -> None:
    """Play the trajectory at the specified frequency."""
    dt = 1.0 / frequency
    
    # Get number of frames from first joint trajectory
    first_joint = next(iter(traj_dict.values()))
    num_frames = first_joint.shape[0]
    
    logger.info(f"Trajectory has {num_frames} frames, playing at {frequency}Hz")
    
    for loop in range(num_loops):
        logger.info(f"Starting loop {loop+1}/{num_loops}")
        
        for frame in range(num_frames):
            target_time = time.time() + dt
            
            # Extract positions for this frame
            frame_positions = {}
            for joint_name, trajectory in traj_dict.items():
                if joint_name != "floating_base":  # Skip floating base (not an actuator)
                    frame_positions[joint_name] = float(trajectory[frame][0])
            
            # Send positions to robot
            await send_trajectory_frame(kos, frame_positions)
            
            # Sleep to maintain frequency
            sleep_time = max(0, target_time - time.time())
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                logger.warning(f"Frame {frame} overran by {time.time() - target_time:.4f}s")


async def main(traj_file: str, ip: str, frequency: float, num_loops: int) -> None:
    """Main function to load and play trajectory."""
    # Load trajectory dictionary
    try:
        with open(traj_file, "rb") as f:
            traj_dict = pickle.load(f)
        logger.info(f"Loaded trajectory with {len(traj_dict)} joints")
    except Exception as e:
        logger.error(f"Failed to load trajectory: {e}")
        return
    
    # Connect to robot
    kos = pykos.KOS(ip=ip)
    
    # Disable actuators first
    await disable_actuators(kos)
    await asyncio.sleep(1)
    
    # Configure actuators
    logger.info("Configuring actuators...")
    await configure_actuators(kos)
    await asyncio.sleep(1)
    
    # Countdown
    for i in range(5, -1, -1):
        logger.info(f"Starting in {i} seconds...")
        await asyncio.sleep(1)
    
    try:
        # Play trajectory
        await play_trajectory(kos, traj_dict, frequency, num_loops)
    except asyncio.CancelledError:
        logger.info("Playback cancelled")
    except Exception as e:
        logger.error(f"Error during playback: {e}")
    finally:
        # Disable actuators when done
        logger.info("Disabling actuators...")
        await disable_actuators(kos)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play trajectory on K-Bot")
    parser.add_argument("--traj_file", type=str, default="./assets/traj_dict.pkl",
                        help="Path to trajectory pickle file")
    parser.add_argument("--ip", type=str, default="localhost",
                        help="IP address of robot")
    parser.add_argument("--frequency", type=float, default=60.0,
                        help="Playback frequency in Hz")
    parser.add_argument("--loops", type=int, default=1,
                        help="Number of times to loop through trajectory")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    
    # Run main function
    asyncio.run(main(args.traj_file, args.ip, args.frequency, args.loops)) 