"""Module for checking and visualizing deployment data from K-Bot."""

import argparse
import glob
import logging
import os
import os.path as osp
import pickle
from dataclasses import dataclass
from datetime import datetime

import colorlogging
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)
colorlogging.configure()


STATS = {
    "deployment_length": -1.0,
    "avg_loop_overrun_time": -1.0,
    "max_loop_overrun_time": -1.0,
    "max_commanded_velocity": -1.0,
    "avg_imu_freq": -1.0,
}


@dataclass
class Actuator:
    actuator_id: int
    nn_id: int
    kp: float
    kd: float
    max_torque: float
    joint_name: str


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


def load_latest_deployment(log_type: str = "real-check") -> tuple[dict, str] | None:
    """Load the latest deployment pickle file of the specified type.

    Args:
        log_type: Type of log to load ('sim', 'real-check', or 'real-deploy').

    Returns:
        Tuple of (data, filename) or None if no files found.
    """
    # Get the directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Find all pickle files matching the pattern
    pkl_files = glob.glob(os.path.join(current_dir, f"{log_type}_*.pkl"))

    if not pkl_files:
        print(f"No {log_type} pickle files found.")
        return None

    # Extract timestamps and sort files
    file_timestamps = []
    for file in pkl_files:
        # Extract the timestamp part from the filename
        filename = os.path.basename(file)
        timestamp_str = filename.split("_")[1].split(".")[0]  # Extract YYYYMMDD-HHMMSS

        # Convert to datetime object for comparison
        try:
            if "-" in timestamp_str:
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d-%H%M%S")
            else:
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d")
            file_timestamps.append((file, timestamp))
        except ValueError as e:
            logger.warning(f"Could not parse timestamp from {filename}: {e}")
            continue

    if not file_timestamps:
        logger.warning(f"No valid {log_type} files found.")
        return None

    # Sort by timestamp (newest first)
    def sort_by_timestamp(item: tuple[str, datetime]) -> datetime:
        return item[1]

    file_timestamps.sort(key=sort_by_timestamp, reverse=True)

    # Get the latest file
    latest_file = file_timestamps[0][0]
    latest_filename = os.path.basename(latest_file)
    print(f"Loading latest {log_type} file: {latest_filename}")

    # Load the pickle file
    with open(latest_file, "rb") as f:
        data = pickle.load(f)

    return data, latest_filename


def find_deployment_file(date_str: str, log_type: str) -> tuple[dict, str, str] | None:
    """Find the latest deployment file of a given type and date.

    Checks for a folder with the date name and looks for files inside.
    Only searches within date-named folders, not directly in the deployment_logs directory.
    """
    # Get the directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Check if there's a folder with the date name
    date_folder = os.path.join(current_dir, date_str)
    if not os.path.isdir(date_folder):
        print(f"Date folder not found: {date_str}")
        return None

    print(f"Found date folder: {date_str}")

    # Construct pattern based on log type
    if log_type == "sim":
        pattern = f"sim_{date_str}*.pkl"
    elif log_type == "real-check":
        pattern = f"real-check_{date_str}*.pkl"
    elif log_type == "real-deploy":
        pattern = f"real-deploy_{date_str}*.pkl"
    else:
        print(f"Invalid log type: {log_type}")
        return None

    # Look for files in the date folder
    pkl_files = glob.glob(os.path.join(date_folder, pattern))

    if not pkl_files:
        print(f"No {log_type} files found in folder {date_str}.")
        return None

    # Extract timestamps and sort files
    file_timestamps = []
    for file in pkl_files:
        # Extract the timestamp part from the filename
        filename = os.path.basename(file)
        try:
            timestamp_str = filename.split("_")[1].split(".")[0]  # Extract YYYYMMDD-HHMMSS or just YYYYMMDD

            # Check if the timestamp contains time
            if "-" in timestamp_str:
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d-%H%M%S")
            else:
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d")

            file_timestamps.append((file, timestamp))
        except (IndexError, ValueError) as e:
            print(f"Warning: Could not parse timestamp from {filename}: {e}")
            continue

    if not file_timestamps:
        print(f"No valid {log_type} files found in folder {date_str}.")
        return None

    # Sort by timestamp (newest first)
    file_timestamps.sort(key=lambda item: item[1], reverse=True)

    # Get the latest file
    latest_file = file_timestamps[0][0]
    latest_filename = os.path.basename(latest_file)
    print(f"Loading deployment file: {latest_filename}")

    # Load the pickle file
    try:
        with open(latest_file, "rb") as f:
            data = pickle.load(f)
        return data, latest_filename, date_folder
    except Exception as e:
        print(f"Error loading {latest_filename}: {e}")
        return None


def plot_command_data(data: dict, output_dir: str) -> None:
    """Plot command data with subplots for each actuator."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a mapping from nn_id to joint_name
    nn_id_to_joint = {a.nn_id: a.joint_name for a in actuator_list}

    # For position and velocity, create 4 figures with 5 subplots each
    fig_pos_1, axs_pos_1 = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    fig_pos_2, axs_pos_2 = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    fig_pos_3, axs_pos_3 = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    fig_pos_4, axs_pos_4 = plt.subplots(5, 1, figsize=(10, 15), sharex=True)

    fig_vel_1, axs_vel_1 = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    fig_vel_2, axs_vel_2 = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    fig_vel_3, axs_vel_3 = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    fig_vel_4, axs_vel_4 = plt.subplots(5, 1, figsize=(10, 15), sharex=True)

    pos_figs = [(fig_pos_1, axs_pos_1), (fig_pos_2, axs_pos_2), (fig_pos_3, axs_pos_3), (fig_pos_4, axs_pos_4)]
    vel_figs = [(fig_vel_1, axs_vel_1), (fig_vel_2, axs_vel_2), (fig_vel_3, axs_vel_3), (fig_vel_4, axs_vel_4)]

    # Extract commands for each actuator over time
    steps = np.arange(len(data["command"]))
    actuator_positions: dict[int, list[float]] = {i: [] for i in range(20)}
    actuator_velocities: dict[int, list[float]] = {i: [] for i in range(20)}

    # Track maximum velocity for stats
    max_velocity = 0.0

    for step_data in data["command"]:
        # Each step_data is a list of actuator commands
        for actuator_data in step_data:
            actuator_id = actuator_data["actuator_id"]
            # Find the nn_id for this actuator_id
            nn_id = next((a.nn_id for a in actuator_list if a.actuator_id == actuator_id), None)
            if nn_id is not None:
                actuator_positions[nn_id].append(actuator_data["position"])
                velocity = float(actuator_data["velocity"])
                actuator_velocities[nn_id].append(velocity)
                # Update max velocity stat
                max_velocity = max(max_velocity, abs(velocity))

    # Update max commanded velocity stat
    STATS["max_commanded_velocity"] = max_velocity

    # Plot each actuator's data
    for nn_id in range(20):
        fig_idx = nn_id // 5
        ax_idx = nn_id % 5

        # Position plots
        pos_fig, pos_axs = pos_figs[fig_idx]
        pos_axs[ax_idx].plot(steps[: len(actuator_positions[nn_id])], actuator_positions[nn_id])
        pos_axs[ax_idx].set_ylabel(f"{nn_id_to_joint.get(nn_id, f'nn_id {nn_id}')}\nPosition")
        pos_axs[ax_idx].grid(True)

        # Velocity plots
        vel_fig, vel_axs = vel_figs[fig_idx]
        vel_axs[ax_idx].plot(steps[: len(actuator_velocities[nn_id])], actuator_velocities[nn_id])
        vel_axs[ax_idx].set_ylabel(f"{nn_id_to_joint.get(nn_id, f'nn_id {nn_id}')}\nVelocity")
        vel_axs[ax_idx].grid(True)

    # Set labels for x-axis on the bottom subplot
    for figs in [pos_figs, vel_figs]:
        for fig, axs in figs:
            axs[-1].set_xlabel("Steps")
            fig.tight_layout()

    # Save the figures
    for i, (fig, _) in enumerate(pos_figs):
        fig.savefig(osp.join(output_dir, f"command_positions_group{i + 1}.pdf"))
        plt.close(fig)

    for i, (fig, _) in enumerate(vel_figs):
        fig.savefig(osp.join(output_dir, f"command_velocities_group{i + 1}.pdf"))
        plt.close(fig)


def plot_vector_data(data: dict, key: str, output_dir: str) -> None:
    """Plot data that is a list of vectors over time."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    steps = np.arange(len(data[key]))

    if key in ["pos_diff", "vel_obs", "prev_action"]:
        vector_len = len(data[key][0]) if data[key] else 0
        # These have nn_id as the position in each vector
        nn_id_to_joint = {a.nn_id: a.joint_name for a in actuator_list}

        # Create 4 figures with 5 subplots each
        fig1, axs1 = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
        fig2, axs2 = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
        fig3, axs3 = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
        fig4, axs4 = plt.subplots(5, 1, figsize=(10, 15), sharex=True)

        figs = [(fig1, axs1), (fig2, axs2), (fig3, axs3), (fig4, axs4)]

        # Extract data for each nn_id over time
        nn_id_data: dict[int, list[float]] = {i: [] for i in range(vector_len)}

        for step_data in data[key]:
            for i in range(vector_len):
                nn_id_data[i].append(float(step_data[i]) if i < len(step_data) else np.nan)

        # Plot each nn_id data
        for nn_id in range(min(20, vector_len)):
            fig_idx = nn_id // 5
            ax_idx = nn_id % 5

            fig, axs = figs[fig_idx]
            axs[ax_idx].plot(steps[: len(nn_id_data[nn_id])], nn_id_data[nn_id][: len(steps)])
            axs[ax_idx].set_ylabel(f"{nn_id_to_joint.get(nn_id, f'nn_id {nn_id}')}")
            axs[ax_idx].grid(True)

        # Set labels for x-axis on the bottom subplot
        for fig, axs in figs:
            axs[-1].set_xlabel("Steps")
            fig.suptitle(f"{key} over time")
            fig.tight_layout()

        # Save the figures
        for i, (fig, _) in enumerate(figs):
            fig.savefig(osp.join(output_dir, f"{key}_group{i + 1}.pdf"))
            plt.close(fig)

    elif key in ["imu_accel", "imu_gyro", "euler_angles"]:
        # Plot IMU acceleration or gyroscope data
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract x, y, and z components
        x_data = [float(step_data[0]) for step_data in data[key]]
        y_data = [float(step_data[1]) for step_data in data[key]]
        z_data = [float(step_data[2]) for step_data in data[key]]

        # Plot the data
        ax.plot(steps[: len(x_data)], x_data[: len(steps)], label="X")
        ax.plot(steps[: len(y_data)], y_data[: len(steps)], label="Y")
        ax.plot(steps[: len(z_data)], z_data[: len(steps)], label="Z")

        component_type = "Acceleration" if key == "imu_accel" else "Gyroscope" if key == "imu_gyro" else "Euler Angles"
        ax.set_ylabel(f"IMU {component_type}")
        ax.set_xlabel("Steps")
        ax.legend()
        ax.grid(True)

        fig.suptitle(f"IMU {component_type} over time")
        fig.tight_layout()
        fig.savefig(osp.join(output_dir, f"{key}.pdf"))
        plt.close(fig)

    elif key == "imu_obs":
        # For backward compatibility: imu_obs has shape (6,) with first 3 being accel and last 3 being gyro
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

        # Extract accel and gyro data
        accel_x = [float(step_data[0]) for step_data in data[key]]
        accel_y = [float(step_data[1]) for step_data in data[key]]
        accel_z = [float(step_data[2]) for step_data in data[key]]

        gyro_x = [float(step_data[3]) for step_data in data[key]]
        gyro_y = [float(step_data[4]) for step_data in data[key]]
        gyro_z = [float(step_data[5]) for step_data in data[key]]

        # Plot accel data
        ax1.plot(steps[: len(accel_x)], accel_x[: len(steps)], label="X")
        ax1.plot(steps[: len(accel_y)], accel_y[: len(steps)], label="Y")
        ax1.plot(steps[: len(accel_z)], accel_z[: len(steps)], label="Z")
        ax1.set_ylabel("Acceleration")
        ax1.legend()
        ax1.grid(True)

        # Plot gyro data
        ax2.plot(steps[: len(gyro_x)], gyro_x[: len(steps)], label="X")
        ax2.plot(steps[: len(gyro_y)], gyro_y[: len(steps)], label="Y")
        ax2.plot(steps[: len(gyro_z)], gyro_z[: len(steps)], label="Z")
        ax2.set_ylabel("Gyroscope")
        ax2.set_xlabel("Steps")
        ax2.legend()
        ax2.grid(True)

        # Calculate magnitudes and remove top 100
        magnitudes = np.sqrt(
            np.array(accel_x) ** 2
            + np.array(accel_y) ** 2
            + np.array(accel_z) ** 2
            + np.array(gyro_x) ** 2
            + np.array(gyro_y) ** 2
            + np.array(gyro_z) ** 2
        )
        top_100_indices = np.argsort(magnitudes)[-100:]
        mask = np.ones(len(magnitudes), dtype=bool)
        mask[top_100_indices] = False

        # Plot filtered data
        ax3.plot(steps[mask], np.array(accel_x)[mask], label="X")
        ax3.plot(steps[mask], np.array(accel_y)[mask], label="Y")
        ax3.plot(steps[mask], np.array(accel_z)[mask], label="Z")
        ax3.set_ylabel("Filtered Acceleration")
        ax3.legend()
        ax3.grid(True)

        fig.suptitle("IMU Observations over time")
        fig.tight_layout()
        fig.savefig(osp.join(output_dir, "imu_obs.pdf"))
        plt.close(fig)

    elif key in ["controller_cmd"]:
        # Handle controller_cmd data
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract x and y data
        x_data = [float(step_data[0]) for step_data in data[key]]
        y_data = [float(step_data[1]) for step_data in data[key]]

        # Plot x and y data
        ax.plot(steps[: len(x_data)], x_data[: len(steps)], label="X")
        ax.plot(steps[: len(y_data)], y_data[: len(steps)], label="Y")
        ax.set_ylabel("Controller Command")
        ax.set_xlabel("Steps")
        ax.legend()
        ax.grid(True)

        fig.suptitle("Controller Commands over time")
        fig.tight_layout()
        fig.savefig(osp.join(output_dir, f"{key}.pdf"))
        plt.close(fig)

    elif key == "phase":
        # phase has shape (2,) from phase_vec =
        # np.array([np.cos(self.phase), np.sin(self.phase)])
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        # Extract cos and sin components
        cos_phase = [float(step_data[0]) for step_data in data[key]]
        sin_phase = [float(step_data[1]) for step_data in data[key]]

        # Plot cos component
        ax1.plot(steps[: len(cos_phase)], cos_phase[: len(steps)])
        ax1.set_ylabel("cos(phase)")
        ax1.grid(True)

        # Plot sin component
        ax2.plot(steps[: len(sin_phase)], sin_phase[: len(steps)])
        ax2.set_ylabel("sin(phase)")
        ax2.set_xlabel("Steps")
        ax2.grid(True)

        fig.suptitle("Phase over time")
        fig.tight_layout()
        fig.savefig(osp.join(output_dir, "phase.pdf"))
        plt.close(fig)

    elif key in ["imu_gravity", "projected_gravity"]:
        # projected_gravity has shape (3,) with x, y, z components
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract x, y, and z components
        x_data = [float(step_data[0]) for step_data in data[key]]
        y_data = [float(step_data[1]) for step_data in data[key]]
        z_data = [float(step_data[2]) for step_data in data[key]]

        # Plot the data
        ax.plot(steps[: len(x_data)], x_data[: len(steps)], label="X")
        ax.plot(steps[: len(y_data)], y_data[: len(steps)], label="Y")
        ax.plot(steps[: len(z_data)], z_data[: len(steps)], label="Z")

        ax.set_ylabel("Projected Gravity")
        ax.set_xlabel("Steps")
        ax.legend()
        ax.grid(True)

        fig.suptitle("Projected Gravity over time")
        fig.tight_layout()
        fig.savefig(osp.join(output_dir, "projected_gravity.pdf"))
        plt.close(fig)

    elif key == "loop_overrun_time":
        non_zero_overruns = [t for t in data[key] if t > 0]
        if non_zero_overruns:
            STATS["avg_loop_overrun_time"] = np.mean(non_zero_overruns)
            STATS["max_loop_overrun_time"] = np.max(non_zero_overruns)
        else:
            STATS["avg_loop_overrun_time"] = 0.0
            STATS["max_loop_overrun_time"] = 0.0

        fig, ax = plt.subplots(figsize=(10, 6))
        # Extract x and y data
        x_data = data[key]

        # Plot x and y data
        ax.plot(steps[: len(x_data)], x_data[: len(steps)], label="X")
        ax.set_ylabel("Loop Overrun Time (s)")
        ax.set_xlabel("Steps")
        ax.legend()
        ax.grid(True)

        fig.suptitle("Per Loop Overrun over time")
        fig.tight_layout()
        fig.savefig(osp.join(output_dir, f"{key}.pdf"))
        plt.close(fig)
   
    elif key == "quat":
        # quat has shape (4,) with w, x, y, z components
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 15), sharex=True)

        # Extract w, x, y, z components
        w_data = [float(step_data[0]) for step_data in data[key]]
        x_data = [float(step_data[1]) for step_data in data[key]]
        y_data = [float(step_data[2]) for step_data in data[key]]
        z_data = [float(step_data[3]) for step_data in data[key]]

        # Plot w, x, y, z components
        ax1.plot(steps[: len(w_data)], w_data[: len(steps)], label="w")
        ax2.plot(steps[: len(x_data)], x_data[: len(steps)], label="x")
        ax3.plot(steps[: len(y_data)], y_data[: len(steps)], label="y")
        ax4.plot(steps[: len(z_data)], z_data[: len(steps)], label="z")

        ax1.set_ylabel("w")
        ax2.set_ylabel("x")
        ax3.set_ylabel("y")
        ax4.set_ylabel("z")
        ax4.set_xlabel("Steps")
        ax1.grid(True)
        ax2.grid(True)
        ax3.grid(True)
        ax4.grid(True)

        fig.suptitle("Quaternion over time")
        fig.tight_layout()
        fig.savefig(osp.join(output_dir, f"{key}.pdf"))
        plt.close(fig)
   

    else:
        raise ValueError(f"Unknown key, add this plot {key}")


def calculate_imu_update_rate(data: dict, output_dir: str) -> None:
    """Calculate running average frequency of unique IMU data points.
    
    Args:
        data: Deployment data dictionary containing timestamps and IMU data
        output_dir: Directory to save the plot
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get timestamps and sensor data
    timestamps = data["timestamp"]
    
    # Check which IMU data are available
    imu_keys = []
    for key in ["quat", "imu_accel", "imu_gyro"]:
        if key in data and len(data[key]) > 0:
            imu_keys.append(key)
    
    if not imu_keys:
        logger.warning("No IMU data found for update rate calculation")
        return
    
    # Initialize tracking variables
    prev_values = {key: None for key in imu_keys}
    change_timestamps = []
    
    # Find timestamps where any of the IMU data changes
    for i in range(len(timestamps)):
        change_detected = False
        
        for key in imu_keys:
            if i < len(data[key]):
                current_value = tuple(float(x) for x in data[key][i])
                
                if prev_values[key] is None or current_value != prev_values[key]:
                    prev_values[key] = current_value
                    change_detected = True
        
        if change_detected:
            change_timestamps.append(timestamps[i])
    
    # Calculate time differences between changes
    if len(change_timestamps) <= 1:
        logger.warning("Not enough unique IMU data points to calculate update rate")
        return
    
    time_diffs = np.diff(change_timestamps)
    
    # Calculate frequencies (1/time_diff) in Hz
    frequencies = 1.0 / time_diffs
    
    # Calculate running average of frequencies
    running_avg_freq = np.cumsum(frequencies) / np.arange(1, len(frequencies) + 1)
    
    # Calculate overall average frequency
    avg_frequency = np.mean(frequencies)
    
    # Plot running average frequency
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(1, len(running_avg_freq) + 1), running_avg_freq)
    ax.set_ylabel("Running Average Frequency (Hz)")
    ax.set_xlabel("Number of Changes")
    ax.set_title(f"Running Average of IMU Update Frequency (Average: {avg_frequency:.2f} Hz)")
    ax.grid(True)
    
    fig.tight_layout()
    fig.savefig(osp.join(output_dir, "imu_update_frequency.pdf"))
    plt.close(fig)
    
    # logger.info(f"IMU update rate analysis: {len(change_timestamps)} unique data points")
    # logger.info(f"Average IMU update frequency: {avg_frequency:.2f} Hz")

    STATS["avg_imu_freq"] = avg_frequency

def plot_deployment_data(data: dict, output_dir: str) -> None:
    """Plot all deployment data."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot command data
    plot_command_data(data, output_dir)

    logger.warning("Plotting %s", str(data.keys()))

    STATS["deployment_length"] = max(data["timestamp"]) - min(data["timestamp"])
    
    # Calculate IMU update rate
    calculate_imu_update_rate(data, output_dir)

    for key in data.keys():
        if key in ["command", "timestamp", "model_name"]:
            continue
        else:
            plot_vector_data(data, key, output_dir)

    # Log STATS values after all plots completed (and stats calculated)
    logger.info("-" * 50)
    logger.info("DEPLOYMENT STATISTICS:")
    logger.info(f"Model name: {data['model_name']}")
    logger.info(f"Deployment length: {STATS['deployment_length']:.2f} seconds")
    logger.info(f"Average per loop overrun time: {STATS['avg_loop_overrun_time']:.2f} s")
    logger.info(f"Maximum per loop overrun time: {STATS['max_loop_overrun_time']:.2f} s")
    logger.info(f"Maximum commanded velocity: {STATS['max_commanded_velocity']:.2f} deg/s")
    if STATS["avg_imu_freq"] > 0:
        logger.info(f"Average IMU update frequency: {STATS['avg_imu_freq']:.2f} Hz")
    logger.info("-" * 50)

    logger.info("All plots saved to %s/", output_dir)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Plot K-Bot deployment logs.")
    parser.add_argument("--date", type=str, help="Date in format YYYYMMDD (e.g., 20250422)")
    parser.add_argument(
        "--type",
        type=str,
        choices=["sim", "real-check", "real-deploy"],
        help="Log type (sim, real-check, or real-deploy)",
    )

    args = parser.parse_args()

    if args.date and args.type:
        # User provided date and type via command line
        deployment_data = find_deployment_file(args.date, args.type)
    else:
        raise ValueError("Invalid date or type provided")

    if deployment_data:
        data, filename, data_dir = deployment_data
        # Create output directory in the same location as the data directory
        # Use the file prefix (without extension) as the folder name
        file_prefix = os.path.splitext(filename)[0]
        output_dir = os.path.join(data_dir, file_prefix)

        logger.info("Generating plots for %s...", filename)
        plot_deployment_data(data, output_dir)
    else:
        logger.error("No data to plot. Exiting.")
