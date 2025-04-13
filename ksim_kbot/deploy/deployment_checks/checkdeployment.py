"""Module for checking and visualizing deployment data from K-Bot."""
import glob
import os
import os.path as osp
import pickle
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


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


def load_latest_deployment() -> tuple[dict, str] | None:
    """Load the latest deployment pickle file."""
    # Get the directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Find all pickle files matching the pattern
    pkl_files = glob.glob(os.path.join(current_dir, "sim_*.pkl"))

    if not pkl_files:
        print("No deployment pickle files found.")
        return None

    # Extract timestamps and sort files
    file_timestamps = []
    for file in pkl_files:
        # Extract the timestamp part from the filename
        filename = os.path.basename(file)
        timestamp_str = filename.split("_")[1].split(".")[0]  # Extract YYYYMMDD-HHMMSS

        # Convert to datetime object for comparison
        timestamp = datetime.strptime(timestamp_str, "%Y%m%d-%H%M%S")
        file_timestamps.append((file, timestamp))

    # Sort by timestamp (newest first)
    def sort_by_timestamp(item: tuple[str, datetime]) -> datetime:
        return item[1]

    file_timestamps.sort(key=sort_by_timestamp, reverse=True)

    # Get the latest file
    latest_file = file_timestamps[0][0]
    latest_filename = os.path.basename(latest_file)
    print(f"Loading latest deployment file: {latest_filename}")

    # Load the pickle file
    with open(latest_file, "rb") as f:
        data = pickle.load(f)

    return data, latest_filename


def plot_command_data(data: dict, output_dir: str = "plots") -> None:
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
    actuator_positions = {i: [] for i in range(20)}
    actuator_velocities = {i: [] for i in range(20)}

    for step_data in data["command"]:
        # Each step_data is a list of actuator commands
        for actuator_data in step_data:
            actuator_id = actuator_data["actuator_id"]
            # Find the nn_id for this actuator_id
            nn_id = next((a.nn_id for a in actuator_list if a.actuator_id == actuator_id), None)
            if nn_id is not None:
                actuator_positions[nn_id].append(actuator_data["position"])
                actuator_velocities[nn_id].append(float(actuator_data["velocity"]))

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
        fig.savefig(osp.join(output_dir, f"command_positions_group{i+1}.pdf"))
        plt.close(fig)

    for i, (fig, _) in enumerate(vel_figs):
        fig.savefig(osp.join(output_dir, f"command_velocities_group{i+1}.pdf"))
        plt.close(fig)


def plot_vector_data(data: dict, key: str, output_dir: str = "plots") -> None:
    """Plot data that is a list of vectors over time."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    steps = np.arange(len(data[key]))
    vector_len = len(data[key][0]) if data[key] else 0

    if key in ["pos_diff", "vel_obs", "prev_action"]:
        # These have nn_id as the position in each vector
        nn_id_to_joint = {a.nn_id: a.joint_name for a in actuator_list}

        # Create 4 figures with 5 subplots each
        fig1, axs1 = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
        fig2, axs2 = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
        fig3, axs3 = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
        fig4, axs4 = plt.subplots(5, 1, figsize=(10, 15), sharex=True)

        figs = [(fig1, axs1), (fig2, axs2), (fig3, axs3), (fig4, axs4)]

        # Extract data for each nn_id over time
        nn_id_data = {i: [] for i in range(vector_len)}

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
            fig.savefig(osp.join(output_dir, f"{key}_group{i+1}.pdf"))
            plt.close(fig)

    elif key == "imu_obs":
        # imu_obs has shape (6,) with first 3 being accel and last 3 being gyro
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

    elif key == "cmd":
        # controller_cmd has shape (2,) with first being x, second being y
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
        fig.savefig(osp.join(output_dir, "controller_cmd.pdf"))
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


def plot_deployment_data(data: dict, output_dir: str = "plots") -> None:
    """Plot all deployment data."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot command data
    plot_command_data(data, output_dir)

    # Plot other vector data
    for key in ["pos_diff", "vel_obs", "imu_obs", "cmd", "prev_action", "phase"]:
        if key in data:
            plot_vector_data(data, key, output_dir)

    print(f"All plots saved to {output_dir}/")


if __name__ == "__main__":
    deployment_data = load_latest_deployment()
    if deployment_data:
        data, filename = deployment_data
        print(data)
        # For 'command', it is a list over time, with each item being a list
        # representing actuator data. Plot different subplots for each actuator.
        # There are likely 20 actuators.
        # For 'pos_diff', 'vel_obs', and 'prev_action', each is a list similar
        # to 'command', but each entry is a single list. Use the nn_id to map
        # positions to actuator names for labeling.
        # For 'imu_obs', it is a list of lists with shape (6,) per time instance.
        # The first three values are accelerometer data, and the last three are
        # gyroscope data.
        # For 'controller_cmd', it is a list of lists with shape (2,) per time
        # instance. The first value is x, and the second is y.
        # For 'phase', it is a list of lists with shape (2,) per time instance,
        # derived from phase_vec = np.array([np.cos(self.phase), np.sin(self.phase)]).flatten().

        # Create output directory based on pickle filename (without extension)
        output_dir = os.path.join("plots", os.path.splitext(filename)[0])

        # Plot the deployment data
        plot_deployment_data(data, output_dir)
