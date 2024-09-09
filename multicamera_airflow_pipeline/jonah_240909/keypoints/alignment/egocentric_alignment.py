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
    skeleton_info,
    parents_dict,
    keypoint_info,
    keypoints,
    keypoints_order,
    kpt_dict,
)


class EgocentricAligner:
    def __init__(
        self,
        predictions_3d_file,
        egocentric_alignment_output_directory,
        batch_size=20000,
        distance_from_median_thresh=50,
        plot_steps=True,
        recompute_completed=False,
        alignment_method="rigid",
    ):
        self.predictions_3d_file = Path(predictions_3d_file)
        self.batch_size = batch_size
        self.egocentric_alignment_output_directory = Path(egocentric_alignment_output_directory)
        self.distance_from_median_thresh = distance_from_median_thresh
        self.plot_steps = plot_steps

        logger.info(f"Output directory: {self.egocentric_alignment_output_directory}")
        self.egocentric_alignment_output_directory.mkdir(parents=True, exist_ok=True)

        if alignment_method not in ["rigid", "nonrigid"]:
            raise ValueError("alignment_method must be 'rigid' or 'nonrigid'")
        self.alignment_method = alignment_method
        self.recompute_completed = recompute_completed

    def check_completed(self):
        return (self.egocentric_alignment_output_directory / "completed.txt").exists()

    def run(self):
        logger.info(f"Running alignment method: {self.alignment_method}")

        # check if completed
        if self.check_completed() & (self.recompute_completed is False):
            logger.info("Alignment already completed. Skipping.")
            return

        # load unaligned 3D predictions
        self.predictions_3D_mmap = load_memmap_from_filename(self.predictions_3d_file)

        # get the number of data points
        self.n_data = len(self.predictions_3D_mmap)
        self.n_batches = int(np.ceil(self.n_data / self.batch_size))
        logger.info(f"\t Data length: {self.n_data}")

        # ensure that there are no nans in the input
        assert np.any(np.isnan(self.predictions_3D_mmap)) == False, "nans in poses"

        keypoints = np.array(list(kpt_dict.keys()))
        if self.plot_steps:
            plot_keypoints(
                self.predictions_3D_mmap[0],
                keypoints,
                save_location=self.egocentric_alignment_output_directory
                / "poses_before_alignment.png",
            )

        if self.alignment_method == "rigid":
            align_poses = align_poses_rigid
        elif self.alignment_method == "nonrigid":
            align_poses = align_poses_nonrigid

        logger.info(f"initializing output")

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Convert the temporary directory path to a Path object
            tmpdir_path = Path(tmpdirname)
            temp_aligned_poses_file = self.initialize_output(tmpdir_path)

            # prepopulate batch with unmodified data
            for batch in tqdm(range(self.n_batches), desc="performing alignment"):
                batch_start = self.batch_size * batch
                batch_end = self.batch_size * (batch + 1)
                coordinates = np.array(self.predictions_3D_mmap)[batch_start:batch_end]
                aligned_poses = align_poses(
                    coordinates,
                )
                self.aligned_poses_mmap[batch_start:batch_end] = aligned_poses

            if self.plot_steps:
                plot_keypoints(
                    self.aligned_poses_mmap[0],
                    keypoints,
                    save_location=self.egocentric_alignment_output_directory
                    / "poses_after_alignment.png",
                )

            # move the final temp_coordinates_file to the output directory
            shutil.move(
                temp_aligned_poses_file,
                self.egocentric_alignment_output_directory / temp_aligned_poses_file.name,
            )

        # save completed file
        with open(self.egocentric_alignment_output_directory / "completed.txt", "w") as f:
            f.write("completed")

    def initialize_output(self, tmpdir_path):
        # initialize
        mmap_dtype = "float32"
        keypoints_3d_shape = self.predictions_3D_mmap.shape
        keypoints_3d_dtype = mmap_dtype
        keypoints_3d_shape_str = "x".join(map(str, keypoints_3d_shape))

        aligned_poses_file = (
            tmpdir_path
            / f"egocentric_alignment_rigid.{keypoints_3d_dtype}.{keypoints_3d_shape_str}.mmap"
        )

        self.aligned_poses_mmap = np.memmap(
            aligned_poses_file,
            dtype=keypoints_3d_dtype,
            mode="w+",
            shape=keypoints_3d_shape,
        )

        return aligned_poses_file


def load_memmap_from_filename(filename):
    # Extract the metadata from the filename
    parts = filename.name.rsplit(".", 4)  # Split the filename into parts
    dtype_str = parts[-3]  # Get the dtype part of the filename
    shape_str = parts[-2]  # Get the shape part of the filename
    shape = tuple(map(int, shape_str.split("x")))  # Convert shape string to a tuple of integers
    # Load the array using numpy memmap
    array = np.memmap(filename, dtype=dtype_str, mode="r", shape=shape)
    return array


def generate_initial_positions(positions):
    init_positions = np.zeros_like(positions)
    for k in range(positions.shape[1]):
        ix = np.nonzero(~np.isnan(positions[:, k, 0]))[0]
        for i in range(positions.shape[2]):
            init_positions[:, k, i] = np.interp(
                np.arange(positions.shape[0]), ix, positions[:, k, i][ix]
            )
    return init_positions


def align_poses_rigid(poses):
    """This function performs egocentric alignment of pose data of the shape
    (timepoints, keypoints, 3), so that a front keypoint, such as the top of the spine,
    and a back keypoint, such as the bottom of the spine, are aligned along the x axis
    Then, rotate the side keypoint to be aligned with the y axis, perpendicular to
    the front and side keypoints
    """

    kpt_dict = {
        "nose_tip": 0,
        "left_ear": 1,
        "right_ear": 2,
        "left_eye": 3,
        "right_eye": 4,
        "throat": 5,
        "forehead": 6,
        "left_shoulder": 7,
        "right_shoulder": 8,
        "left_elbow": 9,
        "right_elbow": 10,
        "left_wrist": 11,
        "right_wrist": 12,
        "left_hind_paw_front": 13,
        "right_hind_paw_front": 14,
        "left_hind_paw_back": 15,
        "right_hind_paw_back": 16,
        "left_knee": 17,
        "right_knee": 18,
        "tail_base": 19,
        "spine_low": 20,
        "spine_mid": 21,
        "spine_high": 22,
        "left_fore_paw": 23,
        "right_fore_paw": 24,
    }

    front_keypoints = [
        "nose_tip",
        "left_ear",
        "right_ear",
        "left_eye",
        "right_eye",
        "throat",
        "forehead",
        "left_shoulder",
        "right_shoulder",
        "right_elbow",
        "left_elbow",
        "left_fore_paw",
        "right_fore_paw",
        "spine_high",
    ]
    back_keypoints = [
        "spine_mid",
        "spine_low",
        "tail_base",
        "left_knee",
        "right_knee",
        "left_hind_paw_front",
        "right_hind_paw_front",
        "left_hind_paw_back",
        "right_hind_paw_back",
    ]
    side_keypoints = [
        "left_knee",
        "right_knee",
        "left_shoulder",
        "right_shoulder",
        "throat",
        "right_elbow",
        "left_elbow",
        "left_fore_paw",
        "right_fore_paw",
        "left_hind_paw_front",
        "right_hind_paw_front",
        "left_hind_paw_back",
        "right_hind_paw_back",
        "nose_tip",
    ]

    front_keypoint_indices = [kpt_dict[i] for i in front_keypoints]
    back_keypoint_indices = [kpt_dict[i] for i in back_keypoints]
    side_keypoint_indices = [kpt_dict[i] for i in side_keypoints]
    # ensure indices are array and not list
    front_keypoint_indices = np.array(front_keypoint_indices)
    back_keypoint_indices = np.array(back_keypoint_indices)
    side_keypoint_indices = np.array(side_keypoint_indices)

    # aligned_poses = np.copy(poses)
    # Compute the vector from spine bottom to spine top
    vectors = np.mean(poses[:, front_keypoint_indices], axis=1) - np.mean(
        poses[:, back_keypoint_indices], axis=1
    )

    # Normalize the vector
    norm_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    # Compute the axis of rotation
    axes = np.cross(norm_vectors, np.tile([1, 0, 0], (norm_vectors.shape[0], 1)))
    axis_lengths = np.linalg.norm(axes, axis=1, keepdims=True) + 1e-10

    # Normalize axes for valid rotations
    axes /= axis_lengths

    # Compute rotation matrices
    Ks = np.zeros((poses.shape[0], 3, 3))
    Ks[:, 0, 1] = -axes[:, 2]
    Ks[:, 1, 0] = axes[:, 2]
    Ks[:, 0, 2] = axes[:, 1]
    Ks[:, 2, 0] = -axes[:, 1]
    Ks[:, 1, 2] = -axes[:, 0]
    Ks[:, 2, 1] = axes[:, 0]

    # Compute the angles between norm_vectors and the x-axis
    angles = np.arccos(norm_vectors[:, 0])

    # Compute the rotation matrix to align this vector with the x-axis
    rotation_matrices = np.tile(np.eye(3), (poses.shape[0], 1, 1))
    sin_angles = np.sin(angles)
    cos_angles = np.cos(angles)
    rotation_matrices = [
        np.eye(3) + sin_angles[i] * Ks[i] + (1 - cos_angles[i]) * np.dot(Ks[i], Ks[i])
        for i in range(len(sin_angles))
    ]

    # Apply the rotation matrix to all keypoints
    aligned_poses = np.einsum(
        "ikl,ijl->ijk",
        poses - np.expand_dims(np.mean(poses[:, back_keypoint_indices], axis=1), 1),
        rotation_matrices,
    ).transpose((0, 2, 1))
    # Subtract the spine_bottom position to set it as origin (0,0,0) and ensure y and z are 0 for the spine line
    aligned_poses -= (
        np.expand_dims(np.mean(aligned_poses[:, front_keypoint_indices], axis=1), 1)
        + np.expand_dims(np.mean(aligned_poses[:, back_keypoint_indices], axis=1), 1)
    ) / 2

    ### ROTATE
    # rotate to align side
    # Assuming the spine is aligned along the x-axis, and we want the left shoulder
    # to be perpendicular to the x-axis along the y-axis.
    side_keypoint_positions = np.mean(aligned_poses[:, side_keypoint_indices], axis=1)

    # Project the left shoulder position onto the yz-plane
    projections = np.zeros_like(side_keypoint_positions)
    projections[:, 1:] = side_keypoint_positions[:, 1:]

    # Normalize the projection vector
    norm_projections = projections / np.linalg.norm(projections, axis=1, keepdims=True)

    # Compute the rotation angle to align the projection with the y-axis
    dot_product = np.dot(norm_projections, [0, 1, 0])
    angles = np.arccos(dot_product)

    # Determine the direction of the rotation (clockwise or counterclockwise)
    # by using the sign of the z-coordinate of the projection
    angles[projections[:, 2] > 0] = -angles[projections[:, 2] > 0]

    # Rotation matrix around the x-axis
    sin_angles = np.sin(angles)
    cos_angles = np.cos(angles)
    rotation_matrices = np.zeros((len(angles), 3, 3))
    rotation_matrices[:, 0, 0] = 1
    rotation_matrices[:, 1, 1] = cos_angles
    rotation_matrices[:, 1, 2] = -sin_angles
    rotation_matrices[:, 2, 1] = sin_angles
    rotation_matrices[:, 2, 2] = cos_angles

    # Apply the rotation matrix to all keypoints
    rotated_poses = np.einsum("ijk,ikl->ijl", aligned_poses, rotation_matrices.transpose(0, 2, 1))
    # switch up to ceiling
    rotated_poses[:, :, 1] *= -1
    return rotated_poses


def plot_keypoints(kpts, keypoints, extent=50, save_location=None):
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(5, 5))
    for _, link_info in skeleton_info.items():
        point1, point2 = link_info["link"]
        idx1 = np.where(keypoints == point1)[0][0]
        idx2 = np.where(keypoints == point2)[0][0]
        color = tuple(int(c) for c in link_info["color"])
        x = kpts[:, 0]
        y = kpts[:, 1]
        z = kpts[:, 2]
        axs[0, 0].plot(
            [x[idx1], x[idx2]],
            [y[idx1], y[idx2]],
            c=np.array(color) / 255.0,
            lw=2,
        )
        axs[0, 0].scatter(x, y, color="k", s=1)

        axs[1, 0].plot(
            [x[idx1], x[idx2]],
            [z[idx1], z[idx2]],
            c=np.array(color) / 255.0,
            lw=2,
        )
        axs[1, 0].scatter(x, z, color="k", s=1)

        axs[0, 1].plot(
            [z[idx1], z[idx2]],
            [y[idx1], y[idx2]],
            c=np.array(color) / 255.0,
            lw=2,
        )

        axs[0, 1].scatter(z, y, color="k", s=1)

    axs[0, 0].grid(visible=True)
    axs[0, 1].grid(visible=True)
    axs[1, 0].grid(visible=True)
    axs[1, 1].axis("off")

    axs[0, 0].set_xlim([np.nanmean(x) - extent, np.nanmean(x) + extent])
    axs[0, 0].set_ylim([np.nanmean(y) - extent, np.nanmean(y) + extent])

    axs[1, 0].set_xlim([np.nanmean(x) - extent, np.nanmean(x) + extent])
    axs[1, 0].set_ylim([np.nanmean(z) - extent, np.nanmean(z) + extent])

    axs[0, 1].set_xlim([np.nanmean(z) - extent, np.nanmean(z) + extent])
    axs[0, 1].set_ylim([np.nanmean(y) - extent, np.nanmean(y) + extent])
    if save_location is not None:
        plt.savefig(save_location, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def align_poses_nonrigid(poses):
    """This function performs egocentric alignment of pose data of the shape
    (timepoints, keypoints, 3), so that a front keypoint, such as the top of the spine,
    and a back keypoint, such as the bottom of the spine, are aligned along the x axis
    Then, rotate the side keypoint to be aligned with the y axis, perpendicular to
    the front and side keypoints
    """

    kpt_dict = {
        "nose_tip": 0,
        "left_ear": 1,
        "right_ear": 2,
        "left_eye": 3,
        "right_eye": 4,
        "throat": 5,
        "forehead": 6,
        "left_shoulder": 7,
        "right_shoulder": 8,
        "left_elbow": 9,
        "right_elbow": 10,
        "left_wrist": 11,
        "right_wrist": 12,
        "left_hind_paw_front": 13,
        "right_hind_paw_front": 14,
        "left_hind_paw_back": 15,
        "right_hind_paw_back": 16,
        "left_knee": 17,
        "right_knee": 18,
        "tail_base": 19,
        "spine_low": 20,
        "spine_mid": 21,
        "spine_high": 22,
        "left_fore_paw": 23,
        "right_fore_paw": 24,
    }

    front_keypoints = [
        "spine_high",
    ]
    back_keypoints = [
        "tail_base",
    ]

    front_keypoint_indices = [kpt_dict[i] for i in front_keypoints]
    back_keypoint_indices = [kpt_dict[i] for i in back_keypoints]

    # standardization functions
    def vector_to_angle(V):
        y, x = V[..., 1], V[..., 0] + 1e-10
        angles = (np.arctan(y / x) + (x > 0) * np.pi) % (2 * np.pi) - np.pi
        return angles

    def angle_to_rotation_matrix(h, keypoint_dim=3):
        m = np.tile(np.eye(keypoint_dim), (*h.shape, 1, 1))
        m[..., 0, 0] = np.cos(h)
        m[..., 1, 1] = np.cos(h)
        m[..., 0, 1] = -np.sin(h)
        m[..., 1, 0] = np.sin(h)
        return m

    def standardize_poses(poses, front_keypoint_indices, back_keypoint_indices):
        poses = poses - poses.mean(1, keepdims=True)
        front = np.mean(poses[:, front_keypoint_indices, :2], axis=1)
        back = np.mean(poses[:, back_keypoint_indices, :2], axis=1)
        angle = vector_to_angle(front - back)
        rot = angle_to_rotation_matrix(angle, keypoint_dim=poses.shape[-1])
        return poses @ rot

    # ensure indices are array and not list
    front_keypoint_indices = np.array(front_keypoint_indices)
    back_keypoint_indices = np.array(back_keypoint_indices)

    #
    rotated_poses = standardize_poses(poses, front_keypoint_indices, back_keypoint_indices)

    return rotated_poses
