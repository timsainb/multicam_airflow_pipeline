import sys
import logging

logger = logging.getLogger(__name__)
logger.info(f"Python interpreter binary location: {sys.executable}")

# general imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import tempfile
import os
import sys
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
import json
import re
import scipy

# function specific imports
from scipy.signal import medfilt
from scipy.stats import circmean


# load skeleton
from multicamera_airflow_pipeline.tim_240731.skeletons.defaults import (
    dataset_info,
    parents_dict,
    keypoint_info,
    keypoints,
    keypoints_order,
    kpt_dict,
)


class ContinuousVariables:
    def __init__(
        self,
        continuous_features_output_directory,
        coordinates_egocentric_filename,
        coordinates_arena_filename,
        framerate,
        recompute_completed=False,
        kpt_dict=kpt_dict,
    ):
        self.continuous_features_output_directory = continuous_features_output_directory
        self.coordinates_arena_filename = coordinates_arena_filename
        self.coordinates_egocentric_filename = coordinates_egocentric_filename
        self.recompute_completed = recompute_completed
        self.framerate = framerate
        self.kpt_dict = kpt_dict

    def check_completed(self):
        return (self.continuous_features_output_directory / "continuous_features.pickle").exists()

    def run(
        self,
        speed_kernel_size_ms=150,
        acceleration_kernel_size_ms=150,
        heading_kernel_size_ms=33,
        angular_velocity_kernel_size_ms=150,
        angular_acceleration_kernel_size_ms=150,
        spine_curvature_kernel_size_ms=150,
        limb_velocity_kernel_size_ms=150,
        limb_acceleration_kernel_size_ms=150,
        limb_correlation_window_size_ms=1000,
        acceleration_keypoints_kernel_size_ms=150,
        wall_positions_x=[-200, 200],  # wall positions, assumes arena to be rectangular
        wall_positions_y=[-200, 200],
    ):

        if (self.recompute_completed == False) and self.check_completed():
            logger.info("Continuous variables already computed and saved")
            return

        logger.info("Loading coordinates egocentric and arena aligned")
        self.coordinates_egocentric = load_memmap_from_filename(
            self.coordinates_egocentric_filename
        )
        self.coordinates_arena = load_memmap_from_filename(self.coordinates_arena_filename)

        logger.info("Computing continuous features")

        (feature_df, centroids) = compute_continuous_features(
            coordinates_egocentric=self.coordinates_egocentric,
            coordinates_arena=self.coordinates_arena,
            framerate=self.framerate,
            kpt_dict=self.kpt_dict,  # dictionary mapping keypoint names to indices
            speed_kernel_size_ms=speed_kernel_size_ms,
            acceleration_kernel_size_ms=acceleration_kernel_size_ms,
            heading_kernel_size_ms=heading_kernel_size_ms,
            angular_velocity_kernel_size_ms=angular_velocity_kernel_size_ms,
            angular_acceleration_kernel_size_ms=angular_acceleration_kernel_size_ms,
            spine_curvature_kernel_size_ms=spine_curvature_kernel_size_ms,
            limb_velocity_kernel_size_ms=limb_velocity_kernel_size_ms,
            limb_acceleration_kernel_size_ms=limb_acceleration_kernel_size_ms,
            limb_correlation_window_size_ms=limb_correlation_window_size_ms,
            acceleration_keypoints_kernel_size_ms=acceleration_keypoints_kernel_size_ms,
            wall_positions_x=wall_positions_x,  # wall positions, assumes arena to be rectangular
            wall_positions_y=wall_positions_y,
        )

        # save results
        feature_df.to_pickle(
            self.continuous_features_output_directory / "continuous_features.pickle"
        )
        # save centroids as npy
        np.save(self.continuous_features_output_directory / "centroids.npy", centroids)
        # plot computed features
        fig, axs = plot_continuous_features(
            self.coordinates_egocentric,
            self.coordinates_arena,
            centroids=centroids,
            feature_df=feature_df,
            kpt_dict=kpt_dict,
        )
        # save plot to output directory
        fig.savefig(self.continuous_features_output_directory / "continuous_features.png")


def compute_continuous_features(
    coordinates_egocentric,
    coordinates_arena,
    framerate,
    kpt_dict,  # dictionary mapping keypoint names to indices
    speed_kernel_size_ms=150,
    acceleration_kernel_size_ms=150,
    heading_kernel_size_ms=33,
    angular_velocity_kernel_size_ms=150,
    angular_acceleration_kernel_size_ms=150,
    spine_curvature_kernel_size_ms=150,
    limb_velocity_kernel_size_ms=150,
    limb_acceleration_kernel_size_ms=150,
    limb_correlation_window_size_ms=1000,
    acceleration_keypoints_kernel_size_ms=150,
    wall_positions_x=[-200, 200],  # wall positions, assumes arena to be rectangular
    wall_positions_y=[-200, 200],
):
    """
    Compute continuous features from the coordinates of the animal in the arena.
    Parameters

    """

    def ensure_odd(kernel_size):
        return kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    speed_kernel_size_frames = int(speed_kernel_size_ms * framerate / 1000)
    acceleration_kernel_size_frames = int(acceleration_kernel_size_ms * framerate / 1000)
    heading_kernel_size_frames = ensure_odd(int(heading_kernel_size_ms * framerate / 1000))
    angular_velocity_kernel_size_frames = int(angular_velocity_kernel_size_ms * framerate / 1000)
    angular_acceleration_kernel_size_frames = int(
        angular_acceleration_kernel_size_ms * framerate / 1000
    )
    spine_curvature_kernel_size_frames = ensure_odd(
        int(spine_curvature_kernel_size_ms * framerate / 1000)
    )
    limb_velocity_kernel_size_frames = int(limb_velocity_kernel_size_ms * framerate / 1000)
    limb_acceleration_kernel_size_frames = int(limb_acceleration_kernel_size_ms * framerate / 1000)
    limb_correlation_window_size_frames = int(limb_correlation_window_size_ms * framerate / 1000)
    acceleration_keypoints_kernel_size_frames = int(
        acceleration_keypoints_kernel_size_ms * framerate / 1000
    )

    # compute the current centroids
    centroids = np.median(coordinates_arena, axis=1)

    # compute current body positions
    maximum_body_position = np.max(coordinates_arena[:, :, 2], axis=1)
    minimum_body_position = np.min(coordinates_arena[:, :, 2], axis=1)
    median_body_position = np.median(coordinates_arena[:, :, 2], axis=1)

    ### Speed
    # Calculate differences in position
    delta_positions = np.diff(centroids[:, :2], axis=0)

    # Calculate speed in mm / s
    speed = np.linalg.norm(delta_positions, axis=1) * framerate
    speed = np.concatenate([speed, [0]])

    # smooth
    speed = speeds_mm_per_s = scipy.ndimage.uniform_filter1d(speed, size=speed_kernel_size_frames)

    ### Acceleration
    acceleration = np.concatenate([[0], np.diff(speed)])
    acceleration = scipy.ndimage.uniform_filter1d(
        acceleration, size=acceleration_kernel_size_frames
    )

    ### vertical speed
    # Calculate differences in position
    vertical_delta_positions = np.diff(maximum_body_position, axis=0)

    # Calculate speed in mm / s
    vertical_speed = vertical_delta_positions * framerate
    vertical_speed = np.concatenate([vertical_speed, [0]])

    # smooth
    vertical_speed = speeds_mm_per_s = scipy.ndimage.uniform_filter1d(
        vertical_speed, size=speed_kernel_size_frames
    )

    ### Acceleration
    vertical_acceleration = np.concatenate([[0], np.diff(vertical_speed)])
    vertical_acceleration = scipy.ndimage.uniform_filter1d(
        vertical_acceleration, size=acceleration_kernel_size_frames
    )

    ### heading
    spine_keypoints_front = ["nose_tip", "forehead", "spine_high"]
    spine_keypoints_back = ["spine_mid", "spine_low", "tail_base"]
    spine_keypoint_front_indices = [kpt_dict[i] for i in spine_keypoints_front]
    spine_keypoint_back_indices = [kpt_dict[i] for i in spine_keypoints_back]
    # Assume 'nose_tip' is the first point (index 0) and 'tail_base' is the last point (index 5)
    # These can be adjusted based on the actual indexing of your points
    head_point = np.mean(
        coordinates_arena[:, spine_keypoint_front_indices, :], axis=1
    )  # 'nose_tip' for all time steps
    tail_point = np.mean(
        coordinates_arena[:, spine_keypoint_back_indices, :], axis=1
    )  # 'tail_base' for all time steps
    # Calculate the direction vector from the tail to the head
    direction_vectors = head_point - tail_point
    # Calculate the heading angle based on the direction vector
    # Note: This uses the centroid-to-head vector for heading calculation. Adjust as needed.
    angles_spine = np.arctan2(direction_vectors[:, 1], direction_vectors[:, 0])
    angles_spine = median_filter_angles(angles_spine, kernel_size=heading_kernel_size_frames)

    ### angular velocity
    angular_velocity = np.concatenate([(np.diff(np.unwrap(angles_spine)) * framerate), [0]])
    angular_velocity = scipy.ndimage.uniform_filter1d(
        angular_velocity, size=angular_velocity_kernel_size_frames
    )

    ### angular acceleration
    angular_acceleration = np.diff(np.concatenate([[0], angular_velocity]))
    angular_acceleration = scipy.ndimage.uniform_filter1d(
        angular_acceleration, size=angular_acceleration_kernel_size_frames
    )

    ### curvature of spine
    spine_keypoints = [
        "nose_tip",
        "forehead",
        "spine_high",
        "spine_mid",
        "spine_low",
        "tail_base",
    ]
    spine_keypoint_indices = [kpt_dict[i] for i in spine_keypoints]
    spine_points = coordinates_arena[:, spine_keypoint_indices]
    spine_curvature = compute_spine_curvature(spine_points)
    spine_curvature = scipy.signal.medfilt(
        spine_curvature, kernel_size=spine_curvature_kernel_size_frames
    )

    ### distance to wall
    # Compute distances to the nearest wall in x and y directions
    dist_to_nearest_wall_x = np.minimum(
        np.min(np.abs((coordinates_arena[:, :, 0] - wall_positions_x[0])), axis=1),
        np.min(np.abs((coordinates_arena[:, :, 0] - wall_positions_x[1])), axis=1),
    )
    dist_to_nearest_wall_y = np.minimum(
        np.min(np.abs((coordinates_arena[:, :, 1] - wall_positions_y[0])), axis=1),
        np.min(np.abs((coordinates_arena[:, :, 1] - wall_positions_y[1])), axis=1),
    )
    # The distance to the nearest wall is the minimum of the two distances
    dist_to_nearest_wall_mm = np.minimum(dist_to_nearest_wall_x, dist_to_nearest_wall_y)

    ### limb velocity
    left_fore_paw_speed = scipy.ndimage.uniform_filter1d(
        (
            # np.linalg.norm(
            np.concatenate(
                [
                    np.abs(
                        np.diff(coordinates_egocentric[:, kpt_dict["left_fore_paw"], 0], axis=0)
                    ),
                    [0],
                ]
            )
            #    axis=1,
            # )
            * framerate
        ),
        size=limb_velocity_kernel_size_frames,
    )
    right_fore_paw_speed = scipy.ndimage.uniform_filter1d(
        (
            # np.linalg.norm(
            np.concatenate(
                [
                    np.abs(
                        np.diff(
                            coordinates_egocentric[:, kpt_dict["right_fore_paw"], 0],
                            axis=0,
                        )
                    ),
                    [0],
                ]
            )
            #    axis=1,
            # )
            * framerate
        ),
        size=limb_velocity_kernel_size_frames,
    )
    left_hind_paw_speed = scipy.ndimage.uniform_filter1d(
        (
            # np.linalg.norm(
            np.concatenate(
                [
                    np.abs(
                        np.diff(
                            coordinates_egocentric[:, kpt_dict["left_hind_paw_front"], 0],
                            axis=0,
                        )
                    ),
                    [0],
                ]
            )
            #    axis=1,
            # )
            * framerate
        ),
        size=limb_velocity_kernel_size_frames,
    )
    right_hind_paw_speed = scipy.ndimage.uniform_filter1d(
        (
            # np.linalg.norm(
            np.concatenate(
                [
                    np.abs(
                        np.diff(
                            coordinates_egocentric[:, kpt_dict["right_hind_paw_front"], 0],
                            axis=0,
                        )
                    ),
                    [0],
                ]
            )
            #    axis=1,
            # )
            * framerate
        ),
        size=limb_velocity_kernel_size_frames,
    )
    mean_paw_speed = np.mean(
        np.stack(
            [
                left_fore_paw_speed,
                right_fore_paw_speed,
                left_hind_paw_speed,
                right_hind_paw_speed,
            ]
        ),
        axis=0,
    )

    ### limb acceleration
    limb_acceleration = scipy.ndimage.uniform_filter1d(
        np.diff(np.concatenate([[0], mean_paw_speed])),
        size=limb_acceleration_kernel_size_frames,
    )

    ### limb correlation
    forepaw_correlations = rolling_correlation(
        coordinates_egocentric[:, kpt_dict["left_fore_paw"], 0],
        coordinates_egocentric[:, kpt_dict["right_fore_paw"], 0],
        window=limb_correlation_window_size_frames,
    )

    hindpaw_correlations = rolling_correlation(
        coordinates_egocentric[:, kpt_dict["left_hind_paw_front"], 0],
        coordinates_egocentric[:, kpt_dict["right_hind_paw_front"], 0],
        window=limb_correlation_window_size_frames,
    )

    left_side_correlations = rolling_correlation(
        coordinates_egocentric[:, kpt_dict["left_hind_paw_front"], 0],
        coordinates_egocentric[:, kpt_dict["left_fore_paw"], 0],
        window=limb_correlation_window_size_frames,
    )

    right_side_correlations = rolling_correlation(
        coordinates_egocentric[:, kpt_dict["right_hind_paw_front"], 0],
        coordinates_egocentric[:, kpt_dict["right_fore_paw"], 0],
        window=limb_correlation_window_size_frames,
    )
    # compute mean correlation for sides and front/back
    mean_side_to_side_correlation = (right_side_correlations + left_side_correlations) / 2
    mean_front_to_back_correlation = (forepaw_correlations + hindpaw_correlations) / 2

    ### velocity of individual keypoints along the spine
    haunch_keypoints = [
        "nose_tip",
        "spine_high",
        "spine_mid",
        "spine_low",
        "tail_base",
        "left_shoulder",
        "right_shoulder",
    ]
    haunch_keypoints_indices = [kpt_dict[i] for i in haunch_keypoints]

    velocity_keypoints_kernel_size_frames = 9
    velocity_keypoints = (
        np.sum(
            np.diff(coordinates_egocentric[:, haunch_keypoints_indices], axis=0) ** 2,
            axis=2,
        )
        / framerate
    )
    # smooth
    velocity_keypoints = scipy.ndimage.uniform_filter1d(
        velocity_keypoints, size=velocity_keypoints_kernel_size_frames, axis=0
    )
    velocity_keypoints = np.concatenate([velocity_keypoints, [velocity_keypoints[-1]]])

    median_velocity_keypoints = np.median(velocity_keypoints, axis=1)

    ### acceleration of individual keypoints
    acceleration_keypoints = np.diff(velocity_keypoints, axis=0)
    acceleration_keypoints = scipy.ndimage.uniform_filter1d(
        acceleration_keypoints, size=acceleration_keypoints_kernel_size_frames, axis=0
    )
    acceleration_keypoints = np.concatenate([acceleration_keypoints, [acceleration_keypoints[-1]]])

    median_acceleration_keypoints = np.median(acceleration_keypoints, axis=1)

    ### nose to tail distance
    distance_tail_to_nose = np.sqrt(
        np.sum(
            (
                coordinates_arena[:, kpt_dict["nose_tip"]]
                - coordinates_arena[:, kpt_dict["tail_base"]]
            )
            ** 2,
            axis=1,
        )
    )

    # create a dataframe of features
    feature_df = pd.DataFrame(
        {
            "dist_to_nearest_wall_mm": dist_to_nearest_wall_mm,
            "maximum_body_position": maximum_body_position,
            "minimum_body_position": minimum_body_position,
            "median_body_position": median_body_position,
            "speed": speed,
            "vertical_speed": vertical_speed,
            "acceleration": acceleration,
            "vertical_acceleration": vertical_acceleration,
            "angles_spine": angles_spine,
            "angular_velocity": angular_velocity,
            "angular_acceleration": angular_acceleration,
            "spine_curvature": spine_curvature,
            "left_fore_paw_speed": left_fore_paw_speed,
            "right_fore_paw_speed": right_fore_paw_speed,
            "left_hind_paw_speed": left_hind_paw_speed,
            "right_hind_paw_speed": right_hind_paw_speed,
            "mean_paw_speed": mean_paw_speed,
            "limb_acceleration": limb_acceleration,
            "forepaw_correlations": forepaw_correlations,
            "hindpaw_correlations": hindpaw_correlations,
            "mean_front_to_back_correlation": mean_front_to_back_correlation,
            "left_side_correlations": left_side_correlations,
            "right_side_correlations": right_side_correlations,
            "mean_side_to_side_correlation": mean_side_to_side_correlation,
            "median_velocity_keypoints": median_velocity_keypoints,
            "median_acceleration_keypoints": median_acceleration_keypoints,
            "distance_tail_to_nose": distance_tail_to_nose,
        }
    )

    # save results
    return feature_df, centroids


def median_filter_angles(angles, kernel_size=3):
    # Convert angles to complex numbers
    complex_representation = np.exp(1j * angles)

    # Apply median filter to real and imaginary parts separately
    filtered_real = medfilt(np.real(complex_representation), kernel_size=kernel_size)
    filtered_imag = medfilt(np.imag(complex_representation), kernel_size=kernel_size)

    # Recombine filtered real and imaginary parts
    filtered_complex = filtered_real + 1j * filtered_imag

    # Convert filtered complex numbers back to angles
    filtered_angles = np.angle(filtered_complex)

    return filtered_angles


def load_memmap_from_filename(filename):
    # Extract the metadata from the filename
    parts = filename.name.rsplit(".", 4)  # Split the filename into parts
    dtype_str = parts[-3]  # Get the dtype part of the filename
    shape_str = parts[-2]  # Get the shape part of the filename
    shape = tuple(map(int, shape_str.split("x")))  # Convert shape string to a tuple of integers
    # Load the array using numpy memmap
    array = np.memmap(filename, dtype=dtype_str, mode="r", shape=shape)
    return array


def compute_spine_curvature(spine_points, degree=2):

    curvatures = np.zeros(len(spine_points))
    for pi, points in enumerate(tqdm(spine_points, leave=False, desc="computing curvature")):
        # Calculate first derivatives (dx, dy)
        dx = np.diff(points[:, 0])
        dy = np.diff(points[:, 1])

        # Calculate second derivatives (d2x, d2y)
        d2x = np.diff(dx)
        d2y = np.diff(dy)

        # Calculate differences in arc length (ds)
        ds = np.sqrt(dx[:-1] ** 2 + dy[:-1] ** 2)

        # Prevent division by zero in curvature calculation
        ds[ds == 0] = 1e-6

        # Calculate curvature using the formula for discrete points
        curvature = (dx[:-1] * d2y - dy[:-1] * d2x) / (ds**3)

        curvatures[pi] = np.median(curvature)
    return curvatures


def rolling_correlation(arr1, arr2, window=100):
    """Compute rolling correlation between two arrays with a given window size."""
    # Convert arrays to pandas Series for convenience
    s1 = pd.Series(arr1)
    s2 = pd.Series(arr2)
    # Use pandas rolling with window size and compute correlation
    return np.nan_to_num(s1.rolling(window=window).corr(s2).values)


def plot_continuous_features(
    coordinates_egocentric,
    coordinates_arena,
    centroids,
    feature_df,
    kpt_dict,
    start=0,
    stop=3000,
):
    keypoints = bodyparts = np.array(list(kpt_dict.keys()))
    fig, axs = plt.subplots(
        22,
        1,
        figsize=(10, 24),
        gridspec_kw={
            "height_ratios": [
                3,
                3,
                3,
                1,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                1,
                1,
                1,
                1,
                1,
                2,
                1,
                1,
                1,
                1,
                1,
                1,
            ]
        },
    )

    x_pts = np.arange(start, stop)
    ax = axs[0]
    ax.set_ylabel("Aligned Keypoints \n (Anterior Posterior)", rotation=0, va="center", ha="right")
    ax.matshow(coordinates_egocentric[start:stop, :, 0].T, aspect="auto", cmap="coolwarm")
    ax.set_yticks(np.arange(len(bodyparts)))
    ax.set_yticklabels(bodyparts, fontsize=6)
    ax.set_xticks([])
    ax = axs[1]
    ax.set_ylabel("Aligned Keypoints \n (Dorsal Ventral)", rotation=0, va="center", ha="right")
    ax.matshow(coordinates_egocentric[start:stop, :, 1].T, aspect="auto", cmap="coolwarm")
    ax.set_yticks(np.arange(len(bodyparts)))
    ax.set_yticklabels(bodyparts, fontsize=6)
    ax.set_xticks([])

    ax = axs[2]
    ax.set_ylabel("Aligned Keypoints \n (Medial Lateral)", rotation=0, va="center", ha="right")
    ax.matshow(coordinates_egocentric[start:stop, :, 2].T, aspect="auto", cmap="coolwarm")
    ax.set_yticks(np.arange(len(bodyparts)))
    ax.set_yticklabels(bodyparts, fontsize=6)
    ax.set_xticks([])

    ax = axs[3]
    ax.set_ylabel("Centroids\n (mm)", rotation=0, va="center", ha="right")
    ax.plot(
        x_pts,
        centroids[start:stop, 0],
        lw=2,
    )
    ax.plot(x_pts, centroids[start:stop, 1])
    ax.set_xlim([start, stop])
    ax.set_xticks([])

    ax = axs[4]
    ax.set_ylabel("Dist. to wall \n (mm)", rotation=0, va="center", ha="right")
    ax.plot(x_pts, feature_df["dist_to_nearest_wall_mm"].values[start:stop], lw=2, color="k")
    ax.set_xlim([start, stop])
    ax.set_xticks([])

    ax = axs[5]
    ax.set_ylabel("Dist. from floor \n (mm)", rotation=0, va="center", ha="right")
    ax.plot(x_pts, feature_df["maximum_body_position"].values[start:stop], label="head")
    ax.plot(x_pts, feature_df["minimum_body_position"].values[start:stop], label="feet")
    ax.plot(x_pts, feature_df["median_body_position"].values[start:stop], label="centroid")
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax.set_xlim([start, stop])

    ax = axs[6]
    ax.set_ylabel("Speed \n (mm/s)", rotation=0, va="center", ha="right")
    ax.plot(x_pts, feature_df["speed"].values[start:stop], lw=2, color="k")
    ax.set_xlim([start, stop])
    ax.set_xticks([])

    ax = axs[7]
    ax.set_ylabel("Acc. \n (mm/s$^2$)", rotation=0, va="center", ha="right")
    ax.plot(x_pts, feature_df["acceleration"].values[start:stop], lw=2, color="k")
    ax.set_xlim([start, stop])
    ax.set_xticks([])
    ax.axhline(0, color="k", ls="dashed")

    ax = axs[8]
    ax.set_ylabel("Vert. Speed \n (mm/s)", rotation=0, va="center", ha="right")
    ax.plot(x_pts, feature_df["vertical_speed"].values[start:stop], lw=2, color="k")
    ax.set_xlim([start, stop])
    ax.set_xticks([])

    ax = axs[9]
    ax.set_ylabel("Vert. Acc. \n (mm/s$^2$)", rotation=0, va="center", ha="right")
    ax.plot(x_pts, feature_df["vertical_acceleration"].values[start:stop], lw=2, color="k")
    ax.set_xlim([start, stop])
    ax.set_xticks([])
    ax.axhline(0, color="k", ls="dashed")

    ax = axs[10]
    ax.set_ylabel("Heading \n (rad.)", rotation=0, va="center", ha="right")
    ax.scatter(
        x_pts,
        feature_df["angles_spine"].values[start:stop],
        s=1,
        c=feature_df["angular_velocity"].values[start:stop],
        cmap="coolwarm",
    )
    ax.axhline(0, color="k", ls="dashed")
    ax.set_xlim([start, stop])

    ax = axs[11]
    ax.set_ylabel("Angular vel. \n (rad./s)", rotation=0, va="center", ha="right")
    ax.plot(x_pts, feature_df["angular_velocity"].values[start:stop], color="k", lw=2)
    ax.set_xlim([start, stop])
    ax.axhline(0, color="k", ls="dashed")

    ax = axs[12]
    ax.set_ylabel("Angular acc. \n (rad./s)", rotation=0, va="center", ha="right")
    ax.plot(x_pts, feature_df["angular_acceleration"].values[start:stop], color="k", lw=2)
    ax.set_xlim([start, stop])
    ax.axhline(0, color="k", ls="dashed")

    ax = axs[13]
    ax.set_ylabel("Spine curvature \n (1/mm)", rotation=0, va="center", ha="right")
    ax.plot(x_pts, feature_df["spine_curvature"].values[start:stop], color="k", lw=2)
    ax.set_xlim([start, stop])
    ax.axhline(0, color="k", ls="dashed")

    ax = axs[14]
    ax.set_ylabel("tail-to-nose distance \n (mm)", rotation=0, va="center", ha="right")
    ax.plot(x_pts, feature_df["distance_tail_to_nose"].values[start:stop], color="k", lw=2)
    ax.set_xlim([start, stop])

    ax = axs[15]
    ax.set_ylabel("Paw pos.", rotation=0, va="center", ha="right")
    ax.plot(
        x_pts,
        coordinates_egocentric[start:stop, kpt_dict["left_fore_paw"], 0],
        color="blue",
        label="left_fore_paw",
    )
    ax.plot(
        x_pts,
        coordinates_egocentric[start:stop, kpt_dict["right_fore_paw"], 0],
        color="red",
        label="right_fore_paw",
    )
    ax.plot(
        x_pts,
        coordinates_egocentric[start:stop, kpt_dict["left_hind_paw_front"], 0],
        color="blue",
        ls="dashed",
        label="left_hind_paw_front",
    )
    ax.plot(
        x_pts,
        coordinates_egocentric[start:stop, kpt_dict["right_hind_paw_front"], 0],
        color="red",
        ls="dashed",
        label="right_hind_paw_front",
    )
    ax.set_xlim([start, stop])
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

    ax = axs[16]
    ax.set_ylabel("Limb speeds \n (mm/s)", rotation=0, va="center", ha="right")
    ax.plot(
        x_pts,
        feature_df["left_fore_paw_speed"].values[start:stop],
        color="blue",
        label="left_fore_paw",
    )
    ax.plot(
        x_pts,
        feature_df["right_fore_paw_speed"].values[start:stop],
        color="red",
        label="right_fore_paw",
    )
    ax.plot(
        x_pts,
        feature_df["left_hind_paw_speed"].values[start:stop],
        color="blue",
        ls="dashed",
        label="left_hind_paw_front",
    )
    ax.plot(
        x_pts,
        feature_df["right_hind_paw_speed"].values[start:stop],
        color="red",
        ls="dashed",
        label="right_hind_paw_front",
    )
    ax.plot(
        x_pts,
        feature_df["mean_paw_speed"].values[start:stop],
        color="black",
        lw=3,
        label="avg.",
    )
    ax.set_xlim([start, stop])

    ax = axs[17]
    ax.set_ylabel("Limb acc \n (mm/s$^2$)", rotation=0, va="center", ha="right")
    ax.plot(x_pts, feature_df["limb_acceleration"].values[start:stop], lw=3, color="k")
    ax.axhline(0, color="k", ls="dashed")
    ax.set_xlim([start, stop])

    ax = axs[18]
    ax.set_ylabel("Limb correlation (ML)", rotation=0, va="center", ha="right")
    ax.plot(x_pts, feature_df["forepaw_correlations"].values[start:stop], label="forepaw")
    ax.plot(x_pts, feature_df["hindpaw_correlations"].values[start:stop], label="hindpaw")
    ax.plot(
        x_pts,
        feature_df["mean_front_to_back_correlation"].values[start:stop],
        color="k",
        lw=3,
        label="mean",
    )
    ax.set_xlim([start, stop])
    ax.axhline(0, color="k", ls="dashed")
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

    ax = axs[19]
    ax.set_ylabel("Limb corr. (AP)", rotation=0, va="center", ha="right")
    ax.plot(
        x_pts, feature_df["left_side_correlations"].values[start:stop], label="left", color="blue"
    )
    ax.plot(
        x_pts, feature_df["right_side_correlations"].values[start:stop], label="right", color="red"
    )
    ax.plot(
        feature_df["mean_side_to_side_correlation"].values[start:stop],
        color="k",
        lw=3,
        label="mean",
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax.axhline(0, color="k", ls="dashed")
    ax.set_xlim([start, stop])

    ax = axs[20]
    ax.set_ylabel("Keypoint velocity", rotation=0, va="center", ha="right")
    # for i in range(feature_df["velocity_keypoints"].values.shape[1]):
    #    ax.plot(feature_df["velocity_keypoints"].values[start:stop, i], color="grey", lw=1)
    ax.plot(
        x_pts,
        feature_df["median_velocity_keypoints"].values[start:stop],
        color="k",
        lw=3,
        label="mean",
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax.set_xlim([start, stop])

    ax = axs[21]
    ax.set_ylabel("Keypoint acceleration", rotation=0, va="center", ha="right")
    # for i in range(feature_df["acceleration_keypoints"].values.shape[1]):
    #    ax.plot(feature_df["acceleration_keypoints"].values[start:stop, i], color="grey", lw=1)
    ax.plot(
        x_pts,
        feature_df["median_acceleration_keypoints"].values[start:stop],
        color="k",
        lw=3,
        label="mean",
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax.axhline(0, color="k", ls="dashed")
    ax.set_xlim([start, stop])

    for ax in axs.flatten():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    # fig.align_labels()
    # fig.tight_layout()

    # Manually adjust the y-axis labels to be left aligned and positioned at the figure's leftmost part
    for ax in axs:
        ax.tick_params(
            top=False,
            bottom=False,
            left=False,
            right=False,
            labelleft=True,
            labelbottom=True,
        )

        ax.get_yaxis().set_label_coords(-0.025, 0.5)
        ax.yaxis.tick_right()
        ax.set_xticks([])

    return fig, axs
