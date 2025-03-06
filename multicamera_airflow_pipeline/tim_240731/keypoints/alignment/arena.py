import sys
import logging

logging.basicConfig(level=logging.DEBUG)
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

# function specific imports
from sklearn.decomposition import PCA
import scipy.stats
import cv2
from scipy.spatial.transform import Rotation as R
from sklearn.linear_model import LinearRegression

# load skeleton
from multicamera_airflow_pipeline.tim_240731.skeletons.defaults import (
    dataset_info,
    parents_dict,
    keypoint_info,
    keypoints,
    keypoints_order,
    kpt_dict,
)


class ArenaAligner:
    def __init__(
        self,
        predictions_3d_file,  # Path to the 3D predictions file (size normalized mmap)
        arena_alignment_output_directory,
        batch_size=100000,
        flip_z=False,
        reorder_dims=[0, 1, 2],
        plot_steps=True,
        recompute_completed=False,
    ):
        """Aligns the 3D predictions to the arena by rotating the data to align the floor with the feet.
        The algorithm first fits a rectangle to the data, then rotates the data to align the floor with the feet.
        The algorithm then fits a linear regression to the data to align the floor with the feet.
        The algorithm then fits a final rectangle to the data, and rotates the data to align the floor with the feet.
        The algorithm then checks whether the z axis is flipped (in the original data, the floor is the furthest object in the z plane).
        If the z axis is flipped, the data is flipped.
        The final aligned data is saved to the arena_alignment_output_directory.

        Parameters
        ----------
        predictions_3d_file : str
            Path to the 3D predictions file (size normalized mmap)
        arena_alignment_output_directory : str
            Path to the output directory
        batch_size : int, optional
            Number of samples to process at once, by default 100000
        flip_z : bool, optional
            Whether to flip the z axis, by default False. Use flip_z if the floor is the furthest object in the z plane.
        reorder_dims : list, optional
            Order of the dimensions, by default [0, 1, 2]. Use reorder_dims to reorder the dimensions
            (this will be based on which camera is the 'master' in the calibration)
        plot_steps : bool, optional
            Whether to plot the steps, by default True
        recompute_completed : bool, optional
            Whether to recompute the completed file, by default False
        """
        self.batch_size = batch_size
        self.recompute_completed = recompute_completed
        self.predictions_3d_file = Path(predictions_3d_file)
        self.arena_alignment_output_directory = Path(arena_alignment_output_directory)
        self.plot_steps = plot_steps
        self.flip_z = flip_z
        self.reorder_dims = np.array(reorder_dims)

    def check_completed(self):
        return (self.arena_alignment_output_directory / "completed.txt").exists()

    def run(self):

        if self.check_completed() & (self.recompute_completed == False):
            logger.info("Arena alignment already completed, skipping")
            return

        # load unaligned 3D predictions
        self.predictions_3D_mmap = load_memmap_from_filename(self.predictions_3d_file)

        # get the number of data points
        self.n_data = len(self.predictions_3D_mmap)
        self.n_batches = int(np.ceil(self.n_data / self.batch_size))
        logger.info(f"\t Data length: {self.n_data}")

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Convert the temporary directory path to a Path object
            tmpdir_path = Path(tmpdirname)
            logger.info(f"Temporary directory created: {tmpdir_path}")
            # initialize output mmaps
            temp_coordinates_file, temp_centroids_file = self.initialize_output(tmpdir_path)

            # grab a subset of the coordinates that will fit into memory
            coordinates = sample_evenly(self.predictions_3D_mmap, max_samples=self.batch_size)
            coordinates = coordinates[:, :, self.reorder_dims]
            if self.flip_z:
                coordinates[:, :, 2] *= -1

            if np.any(np.isnan(coordinates)):
                logger.info(f"\t prop nans: {np.mean(np.isnan(coordinates))}")
                coordinates = generate_initial_positions(coordinates)

            # depth should initially be up is positive
            original_depth_sample = np.median(coordinates[::100, :, 2], axis=1)

            # subtract out the floor
            minimum_body_position = np.min(coordinates[:, :, 2], axis=1)
            median_coordinates = np.median(coordinates[:, :, 2])
            coordinates[:, :, 2] -= median_coordinates
            minimum_body_position = np.min(coordinates[:, :, 2], axis=1)

            # fill in coordinates_output with predictions_3D_mmap in batches of 100k samples
            for batch in range(self.n_batches):
                batch_start = batch * self.batch_size
                batch_end = (batch + 1) * self.batch_size
                batch = np.array(
                    self.predictions_3D_mmap[batch_start:batch_end, :, self.reorder_dims]
                )
                if np.any(np.isnan(batch)):
                    logger.info(f"\t prop nans batch: {np.mean(np.isnan(batch))}")
                    batch = generate_initial_positions(batch)
                batch[:, :, 2] -= median_coordinates
                self.coordinates_output[batch_start:batch_end] = batch

            # compute the current centroids
            centroids = np.median(coordinates, axis=1)

            if np.all(centroids == 0):
                logger.info("\t All zeros, skipping")
                return

            if self.plot_steps:
                fig, ax = plt.subplots(figsize=(20, 2))
                ax.plot(centroids[::10, 0])
                ax.plot(centroids[::10, 1])
                ax.plot(centroids[::10, 2])
                # save to spikesorting_output_directory
                plt.savefig(self.arena_alignment_output_directory / "0-initial_centroids.png")
                plt.close()

            # detect outliers
            x_min, x_max = np.percentile(centroids[:, 0], (5, 95))
            y_min, y_max = np.percentile(centroids[:, 1], (5, 95))
            x_min -= (x_max - x_min) / 5
            x_max += (x_max - x_min) / 5
            y_min -= (y_max - y_min) / 5
            y_max += (y_max - y_min) / 5
            m = ()
            outlier_mask = np.array(
                (centroids[:, 0] > x_min)
                & (centroids[:, 0] < x_max)
                & (centroids[:, 1] > y_min)
                & (centroids[:, 1] < y_max)
                & (minimum_body_position < np.median(minimum_body_position) + 40)
                & (minimum_body_position > np.median(minimum_body_position) - 40)
            )

            if self.plot_steps:
                fig, axs = plt.subplots(ncols=2, figsize=(5, 2))
                axs[0].scatter(
                    centroids[::10, 0],
                    centroids[::10, 1],
                    c=outlier_mask[::10],
                    vmin=0,
                    vmax=1,
                    cmap="coolwarm",
                    s=1,
                )
                axs[0].axvline(x_min, color="green")
                axs[0].axvline(x_max, color="green")
                axs[0].axhline(y_min, color="green")
                axs[0].axhline(y_max, color="green")
                axs[1].scatter(
                    centroids[::10, 0],
                    centroids[::10, 2],
                    c=outlier_mask[::10],
                    s=1,
                    cmap="coolwarm",
                )
                # save to spikesorting_output_directory
                plt.savefig(self.arena_alignment_output_directory / "1-outlier_mask.png")
                plt.close()

            # fit a rectangle to the data
            points = centroids[outlier_mask][::10, :2].astype("float32")
            center, size, angle = cv2.minAreaRect(np.array(points))
            # Rotate points
            rotated = rotate_points(
                np.reshape(
                    coordinates[:, :, :2],
                    (np.product(np.shape(coordinates)[:2]), 2),
                ),
                center,
                angle,
            )
            coordinates[:, :, :2] = rotated.reshape(coordinates[:, :, :2].shape)

            # rotate points in coordinates_output memmap in batches of 100k samples
            #   following the fit rectangle
            for batch in range(self.n_batches):
                batch_start = batch * self.batch_size
                batch_end = (batch + 1) * self.batch_size
                batch = np.array(self.coordinates_output[batch_start:batch_end])
                # TODO:
                rotated_batch = rotate_points(
                    np.reshape(
                        batch[:, :, :2],
                        (np.product(np.shape(batch)[:2]), 2),
                    ),
                    center,
                    angle,
                )
                batch[:, :, :2] = rotated_batch.reshape(batch[:, :, :2].shape)
                self.coordinates_output[batch_start:batch_end] = batch

            # compute the current centroids
            centroids = np.median(coordinates, axis=1)
            minimum_body_position = np.min(coordinates[:, :, 2], axis=1)

            # detect outliers
            x_min, x_max = np.percentile(centroids[:, 0], (5, 95))
            y_min, y_max = np.percentile(centroids[:, 1], (5, 95))
            m = ()
            outlier_mask = np.array(
                (centroids[:, 0] > x_min)
                & (centroids[:, 0] < x_max)
                & (centroids[:, 1] > y_min)
                & (centroids[:, 1] < y_max)
                # & (minimum_body_position <10)
                # & (minimum_body_position > -10)
            )

            if self.plot_steps:
                fig, axs = plt.subplots(ncols=2, figsize=(5, 2))
                axs[0].scatter(
                    centroids[::10, 0],
                    centroids[::10, 1],
                    c=outlier_mask[::10],
                    vmin=0,
                    vmax=1,
                    cmap="coolwarm",
                    s=1,
                )
                axs[0].axvline(x_min, color="green")
                axs[0].axvline(x_max, color="green")
                axs[0].axhline(y_min, color="green")
                axs[0].axhline(y_max, color="green")
                axs[1].scatter(
                    centroids[::10, 0],
                    centroids[::10, 2],
                    c=outlier_mask[::10],
                    s=1,
                    cmap="coolwarm",
                )
                # save to spikesorting_output_directory
                plt.savefig(self.arena_alignment_output_directory / "2-outlier_mask.png")
                plt.close()

                fig, axs = plt.subplots(ncols=2, figsize=(5, 2))
                axs[0].scatter(
                    centroids[outlier_mask][::10, 0],
                    centroids[outlier_mask][::10, 1],
                    c=minimum_body_position[outlier_mask][::10],
                    vmin=-10,
                    vmax=10,
                )
                axs[1].scatter(
                    centroids[outlier_mask][::10, 0],
                    centroids[outlier_mask][::10, 2],
                    c=centroids[outlier_mask][::10, 1],
                )
                # save to spikesorting_output_directory
                plt.savefig(self.arena_alignment_output_directory / "3-outlier_mask.png")
                plt.close()

            # fit (another) rectangle to the data
            points = centroids[outlier_mask][::10, :2].astype("float32")
            center, size, angle = cv2.minAreaRect(np.array(points))

            # Rotate points
            rotated = rotate_points(
                np.reshape(
                    coordinates[:, :, :2],
                    (np.product(np.shape(coordinates)[:2]), 2),
                ),
                center,
                angle,
            )
            coordinates[:, :, :2] = rotated.reshape(coordinates[:, :, :2].shape)

            # rotate points in coordinates_output memmap in batches of 100k samples
            #   following the fit rectangle
            for batch in range(self.n_batches):
                batch_start = batch * self.batch_size
                batch_end = (batch + 1) * self.batch_size
                batch = np.array(self.coordinates_output[batch_start:batch_end])
                rotated_batch = rotate_points(
                    np.reshape(
                        batch[:, :, :2],
                        (np.product(np.shape(batch)[:2]), 2),
                    ),
                    center,
                    angle,
                )
                batch[:, :, :2] = rotated_batch.reshape(batch[:, :, :2].shape)
                self.coordinates_output[batch_start:batch_end] = batch

            # compute the current centroids
            centroids = np.median(coordinates, axis=1)
            minimum_body_position = np.min(coordinates[:, :, 2], axis=1)

            if self.plot_steps:
                fig, axs = plt.subplots(ncols=2, figsize=(5, 2))
                axs[0].scatter(
                    centroids[outlier_mask][::10, 0],
                    centroids[outlier_mask][::10, 1],
                    c=minimum_body_position[outlier_mask][::10],
                    vmin=-10,
                    vmax=10,
                )
                axs[1].scatter(
                    centroids[outlier_mask][::10, 0],
                    centroids[outlier_mask][::10, 2],
                    c=centroids[outlier_mask][::10, 1],
                )
                # save to spikesorting_output_directory
                plt.savefig(self.arena_alignment_output_directory / "4-outlier_mask.png")
                plt.close()

            ## fix remaining misalignment by rotating to keep the floor aligned to the feet
            # threshold out from regression any jumping or outliers
            # fit model
            model = LinearRegression()
            model.fit(centroids[outlier_mask, :2], minimum_body_position[outlier_mask])
            # Assuming centroids is your array of points and model is your trained LinearRegression model
            a, b = model.coef_  # Coefficients from the linear regression
            # Normal vector of the plane
            normal_vector = np.array([a, b, -1])
            # Angle between the normal vector and the z-axis
            cos_theta = -1 / np.sqrt(a**2 + b**2 + 1)
            sin_theta = np.sqrt(1 - cos_theta**2)
            theta = np.arctan2(sin_theta, cos_theta)
            # Rotation axis (cross product of normal_vector and z-axis (0, 0, 1))
            rotation_axis = np.cross(normal_vector, np.array([0, 0, 1]))
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)  # Normalize the axis
            # Construct the rotation matrix
            rot = R.from_rotvec(rotation_axis * theta)
            rotation_matrix = rot.as_matrix()
            # Apply the rotation to the centroids
            rotated = np.dot(
                rotation_matrix,
                np.reshape(
                    coordinates,
                    (np.product(np.shape(coordinates)[:2]), 3),
                ).T,
            ).T
            coordinates = rotated.reshape(coordinates.shape)
            coordinates = coordinates.astype(np.float32)

            # TODO: rotate points in centroids_output memmap in batches of 100k samples
            #   the fit linear regression
            for batch in range(self.n_batches):
                batch_start = batch * self.batch_size
                batch_end = (batch + 1) * self.batch_size
                batch = np.array(self.coordinates_output[batch_start:batch_end])
                rotated_batch = np.dot(
                    rotation_matrix,
                    np.reshape(
                        batch,
                        (np.product(np.shape(batch)[:2]), 3),
                    ).T,
                ).T
                batch = rotated_batch.reshape(batch.shape)
                batch = batch.astype(np.float32)
                self.coordinates_output[batch_start:batch_end] = batch

            # check whether z axis is flipped (in the original data, the floor is the furthest object in the z plane)
            depth_sample = np.median(coordinates[::100, :, 2], axis=1)
            z_axis_correlation = scipy.stats.pearsonr(
                depth_sample, original_depth_sample
            ).statistic
            z_axis_is_flipped = np.sign(z_axis_correlation) == -1
            if z_axis_is_flipped:
                if self.flip_z == False:
                    coordinates[:, :, 2] *= -1

                    # rotate points in coordinates_output memmap in batches of 100k samples
                    #   following the fit rectangle
                    for batch in range(self.n_batches):
                        batch_start = batch * self.batch_size
                        batch_end = (batch + 1) * self.batch_size
                        batch = np.array(self.coordinates_output[batch_start:batch_end])
                        batch[:, :, 2] *= -1
                        self.coordinates_output[batch_start:batch_end] = batch

            # compute the current centroids
            centroids = np.median(coordinates, axis=1)

            # TODO: add centroids to coordinates_output memmap in batches of 100k samples
            for batch in range(self.n_batches):
                batch_start = batch * self.batch_size
                batch_end = batch + 1 * self.batch_size
                batch = np.array(self.coordinates_output[batch_start:batch_end])
                centroids_batch = np.median(batch, axis=1)
                self.centroids_output[batch_start:batch_end] = centroids_batch

            # fit a final rectangle to the data
            points = centroids[outlier_mask][::100, :2].astype("float32")
            center, size, angle = cv2.minAreaRect(np.array(points))

            # Rotate points
            rotated = rotate_points(
                np.reshape(
                    coordinates[:, :, :2],
                    (np.product(np.shape(coordinates)[:2]), 2),
                ),
                center,
                angle,
            )
            coordinates[:, :, :2] = rotated.reshape(coordinates[:, :, :2].shape)

            # compute the current centroids
            centroids = np.median(coordinates, axis=1)
            # compute current body positions
            maximum_body_position = np.max(coordinates[:, :, 2], axis=1)
            minimum_body_position = np.min(coordinates[:, :, 2], axis=1)
            median_body_position = np.median(coordinates[:, :, 2], axis=1)
            mean_floor = np.median(minimum_body_position)

            coordinates[:, :, 2] -= mean_floor  # set floor to zero
            # compute the current centroids
            centroids = np.median(coordinates, axis=1)
            # compute current body positions
            maximum_body_position = np.max(coordinates[:, :, 2], axis=1)
            minimum_body_position = np.min(coordinates[:, :, 2], axis=1)
            median_body_position = np.median(coordinates[:, :, 2], axis=1)

            # rotate points in coordinates_output memmap in batches of 100k samples
            #   following the fit rectangle
            for batch in range(self.n_batches):
                batch_start = batch * self.batch_size
                batch_end = (batch + 1) * self.batch_size
                batch = np.array(self.coordinates_output[batch_start:batch_end])
                # TODO:
                rotated_batch = rotate_points(
                    np.reshape(
                        batch[:, :, :2],
                        (np.product(np.shape(batch)[:2]), 2),
                    ),
                    center,
                    angle,
                )
                batch[:, :, :2] = rotated_batch.reshape(batch[:, :, :2].shape)
                batch[:, :, 2] -= mean_floor  # set floor to zero
                self.coordinates_output[batch_start:batch_end] = batch

            if self.plot_steps:

                fig, axs = plt.subplots(ncols=3, figsize=(8, 2))
                axs[0].scatter(
                    centroids[::10, 0],
                    centroids[::10, 1],
                    c=minimum_body_position[::10],
                    s=1,
                    vmin=-10,
                    vmax=10,
                )
                axs[1].scatter(centroids[::10, 0], centroids[::10, 2], c=centroids[::10, 1], s=1)
                axs[1].set_ylim([-100, 100])
                axs[2].hist(minimum_body_position.flatten(), bins=np.linspace(-20, 20, 40))
                # save to spikesorting_output_directory
                plt.savefig(self.arena_alignment_output_directory / "4-final_alignment.png")

                plt.close()

                # move the final centroids and coordinates to the output directory
                shutil.move(
                    temp_coordinates_file,
                    self.arena_alignment_output_directory / temp_coordinates_file.name,
                )
                shutil.move(
                    temp_centroids_file,
                    self.arena_alignment_output_directory / temp_centroids_file.name,
                )

        # save completed file
        with open(self.arena_alignment_output_directory / "completed.txt", "w") as f:
            f.write("completed")

    def initialize_output(self, tmpdir_path):

        # initialize
        keypoints_3d_shape = self.predictions_3D_mmap.shape
        centroids_shape = (keypoints_3d_shape[0], keypoints_3d_shape[2])
        mmap_dtype = "float32"
        keypoints_3d_dtype = mmap_dtype
        keypoints_3d_shape_str = "x".join(map(str, keypoints_3d_shape))
        centroids_shape_str = "x".join(map(str, centroids_shape))

        # save output coordinates and centroids to these files
        coordinates_file = (
            tmpdir_path / f"coordinates_aligned.{keypoints_3d_dtype}.{keypoints_3d_shape_str}.mmap"
        )

        centroids_file = tmpdir_path / f"centroids.{keypoints_3d_dtype}.{centroids_shape_str}.mmap"

        self.coordinates_output = np.memmap(
            coordinates_file,
            dtype=keypoints_3d_dtype,
            mode="w+",
            shape=keypoints_3d_shape,
        )

        self.centroids_output = np.memmap(
            centroids_file,
            dtype=keypoints_3d_dtype,
            mode="w+",
            shape=centroids_shape,
        )
        return coordinates_file, centroids_file


def generate_initial_positions(positions):
    init_positions = np.zeros_like(positions)
    for k in range(positions.shape[1]):
        ix = np.nonzero(~np.isnan(positions[:, k, 0]))[0]
        for i in range(positions.shape[2]):
            init_positions[:, k, i] = np.interp(
                np.arange(positions.shape[0]), ix, positions[:, k, i][ix]
            )
    return init_positions


def rotate_points(points, center, angle):
    # Convert angle from degrees to radians
    angle_rad = np.deg2rad(180 - angle)

    # Rotation matrix
    rotation_matrix = np.array(
        [[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]]
    )

    # Translate points to origin
    translated_points = points - center

    # Rotate points
    rotated_points = np.dot(translated_points, rotation_matrix.T)

    # Translate points back
    # rotated_points += center

    return rotated_points


def load_memmap_from_filename(filename):
    # Extract the metadata from the filename
    parts = filename.name.rsplit(".", 4)  # Split the filename into parts
    dtype_str = parts[-3]  # Get the dtype part of the filename
    shape_str = parts[-2]  # Get the shape part of the filename
    shape = tuple(map(int, shape_str.split("x")))  # Convert shape string to a tuple of integers
    # Load the array using numpy memmap
    array = np.memmap(filename, dtype=dtype_str, mode="r", shape=shape)
    return array


def sample_evenly(data, max_samples=100000):
    num_samples = data.shape[0]
    step = max(1, num_samples // max_samples)
    sampled_indices = np.arange(0, num_samples, step)
    sampled_data = data[sampled_indices]
    return sampled_data
