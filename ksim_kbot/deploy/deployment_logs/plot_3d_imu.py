"""Module for visualizing IMU orientation data from K-Bot deployments using 3D plots."""

import argparse
import glob
import logging
import os
import pickle
from datetime import datetime
from typing import Dict

import colorlogging
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)
colorlogging.configure()


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


def quaternion_to_rotation_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to rotation matrix using scipy.

    Args:
        w: Scalar part of quaternion
        x: First vector component of quaternion
        y: Second vector component of quaternion
        z: Third vector component of quaternion

    Returns:
        3x3 rotation matrix
    """
    # Convert to scipy's quaternion format [x, y, z, w]
    quat = np.array([x, y, z, w])

    # Create rotation object and get matrix
    rotation = R.from_quat(quat)

    # Return the rotation matrix
    return rotation.as_matrix()


def quaternion_to_euler(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Convert quaternion to Euler angles (roll, pitch, yaw) in degrees.

    Args:
        w: Scalar part of quaternion
        x: First vector component of quaternion
        y: Second vector component of quaternion
        z: Third vector component of quaternion

    Returns:
        Array of Euler angles in degrees [roll, pitch, yaw]
    """
    # Convert to scipy's quaternion format [x, y, z, w]
    quat = np.array([x, y, z, w])

    # Create rotation object and get Euler angles in degrees
    rotation = R.from_quat(quat)
    euler = rotation.as_euler("xyz", degrees=True)

    return euler


def plot_3d_imu(data: Dict, output_dir: str, hz: int, skip_factor: int) -> None:
    """Plot the 3D IMU data and save as video.

    Args:
        data: Dictionary containing quaternion data in data["quat"]
        output_dir: Directory to save the output video
        hz: Frequency for video playback in Hz (frames per second)
        skip_factor: Factor to skip frames by (e.g., 2 means use every other frame)
    """
    # Check if quaternion data exists
    if "quat" not in data or not data["quat"]:
        logger.error("No quaternion data found in the provided data")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(10, 8), dpi=80)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("World_X")
    ax.set_ylabel("World_Y")
    ax.set_zlabel("World_Z")
    ax.set_title("IMU Orientation")

    # Initial axis vectors
    axis_x = np.array([1, 0, 0])
    axis_y = np.array([0, 1, 0])
    axis_z = np.array([0, 0, 1])

    # Get quaternion data and apply skip factor
    quat_data = data["quat"][::skip_factor]
    frame_count = len(quat_data)
    logger.info("Using %d quaternion frames (skipping every %d frames)", frame_count, skip_factor)

    # Log first frame Euler angles
    if frame_count > 0:
        w, x, y, z = quat_data[0]
        euler = quaternion_to_euler(w, x, y, z)
        logger.info(
            "Initial orientation (Euler angles xyz): Roll=%.2f°, Pitch=%.2f°, Yaw=%.2f°", euler[0], euler[1], euler[2]
        )

    # Initialize the lines once with empty data
    (line_x,) = ax.plot([0, 0], [0, 0], [0, 0], "r-", linewidth=2, label="X (Red)")
    (line_y,) = ax.plot([0, 0], [0, 0], [0, 0], "g-", linewidth=2, label="Y (Green)")
    (line_z,) = ax.plot([0, 0], [0, 0], [0, 0], "b-", linewidth=2, label="Z (Blue)")

    # Add axis labels at the end of each line
    text_x = ax.text(0, 0, 0, "", color="red")
    text_y = ax.text(0, 0, 0, "", color="green")
    text_z = ax.text(0, 0, 0, "", color="blue")

    # Add legend
    ax.legend(loc="upper right")

    # Title text for frame count display
    title_text = ax.set_title("IMU Orientation (from Quat)")

    # Define update function for animation
    def update(frame: int) -> tuple:
        # Get quaternion for this frame
        w, x, y, z = quat_data[frame]

        # Convert quaternion to rotation matrix
        rotation_matrix = quaternion_to_rotation_matrix(w, x, y, z)

        # Get Euler angles for logging (every 10th frame to avoid excessive logging)
        if frame % 10 == 0:
            euler = quaternion_to_euler(w, x, y, z)
            logger.info("Frame %d orientation: Roll=%.2f°, Pitch=%.2f°, Yaw=%.2f°", frame, euler[0], euler[1], euler[2])

        # Rotate axes
        rotated_x = rotation_matrix @ axis_x
        rotated_y = rotation_matrix @ axis_y
        rotated_z = rotation_matrix @ axis_z

        # Scale up length if needed
        scale = 0.8
        rotated_x *= scale
        rotated_y *= scale
        rotated_z *= scale

        # Update lines
        line_x.set_data([0, rotated_x[0]], [0, rotated_x[1]])
        line_x.set_3d_properties([0, rotated_x[2]], "z")

        line_y.set_data([0, rotated_y[0]], [0, rotated_y[1]])
        line_y.set_3d_properties([0, rotated_y[2]], "z")

        line_z.set_data([0, rotated_z[0]], [0, rotated_z[1]])
        line_z.set_3d_properties([0, rotated_z[2]], "z")

        # Update text positions
        text_x.set_position((rotated_x[0], rotated_x[1]))
        text_x.set_3d_properties(rotated_x[2], "z")
        text_x.set_text("X")

        text_y.set_position((rotated_y[0], rotated_y[1]))
        text_y.set_3d_properties(rotated_y[2], "z")
        text_y.set_text("Y")

        text_z.set_position((rotated_z[0], rotated_z[1]))
        text_z.set_3d_properties(rotated_z[2], "z")
        text_z.set_text("Z")

        # Update title
        title_text.set_text(f"IMU Orientation (from Quat) - Frame {frame}/{frame_count - 1}")

        return line_x, line_y, line_z, text_x, text_y, text_z, title_text

    logger.info("Creating animation with %d frames at %d Hz", frame_count, hz)

    # Create animation
    ani = FuncAnimation(
        fig,
        update,
        frames=frame_count,
        interval=1000 / hz,  # ms between frames
        blit=True,  # Use blitting for better performance
    )

    # Save as MP4
    video_path = os.path.join(output_dir, "imu_visualization.mp4")
    logger.info("Saving video to %s", video_path)

    # Set up writer
    writer = animation.FFMpegWriter(
        fps=hz,
        metadata=dict(artist="K-Bot Deployment"),
        bitrate=800,
        codec="h264",
        extra_args=["-preset", "ultrafast", "-crf", "30"],
    )

    # Save animation
    ani.save(video_path, writer=writer)
    logger.info("Video saved to %s", video_path)

    # Close the plot
    plt.close(fig)


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
    parser.add_argument(
        "--hz", type=int, default=10, help="Frequency (frames per second) for the output video (default: 30)"
    )
    parser.add_argument(
        "--skip", type=int, default=2, help="Skip factor for frames (e.g., 2 uses every other frame, default: 1)"
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
        plot_3d_imu(data, output_dir, args.hz, args.skip)
    else:
        logger.error("No data to plot. Exiting.")
