from joblib import Parallel, delayed
import numpy as np
from copy import deepcopy
import sys
import pandas as pd
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
import glob
import numpy as np
import joblib, os, h5py
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import logging
import tempfile
import shutil
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.info(f"Python interpreter binary location: {sys.executable}")

from multicamera_airflow_pipeline.tim_240731.skeletons.defaults import (
    dataset_info,
    parents_dict,
    keypoint_info,
    skeleton_info,
    keypoints,
    keypoints_order,
    kpt_dict,
    default_template_bone_length_mean,
    default_template_bone_length_std,
    default_hierarchy,
)

default_kpt_dict = kpt_dict


class SizeNormalizer:

    def __init__(
        self,
        predictions_3d_file,
        size_norm_output_directory,
        template_bone_length_mean=default_template_bone_length_mean,
        template_bone_length_std=default_template_bone_length_std,
        kpt_dict=default_kpt_dict,
        hierarchy=default_hierarchy,
        rigid_bones=False,
        root_joint="spine_base",
        subsample=None,
        n_jobs=10,
        plot_steps=True,
        recompute_completed=False,
    ):

        self.predictions_3d_file = Path(predictions_3d_file)
        self.kpt_dict = kpt_dict
        self.size_norm_output_directory = Path(size_norm_output_directory)
        self.template_bone_length_mean = template_bone_length_mean
        self.template_bone_length_std = template_bone_length_std
        self.hierarchy = hierarchy
        self.rigid_bones = rigid_bones
        self.root_joint = root_joint
        self.recompute_completed = recompute_completed
        self.subsample = subsample
        self.n_jobs = n_jobs
        self.plot_steps = plot_steps

    def load_predictions_3d(self):
        # ensure that triangulation / gimbal has been completed
        assert (
            len(list(self.predictions_3d_file.parent.glob("*completed.log"))) > 0
        ), "Predictions not completed"

        self.predictions_3D_mmap = load_memmap_from_filename(self.predictions_3d_file)

    def check_completed(self):
        return (self.size_norm_output_directory / "completed.log").exists()

    def run(self):

        # skip if completed
        if self.check_completed() & (self.recompute_completed == False):
            logger.info(f"Size normalization already completed, skipping")
            return

        logger.info(f"Starting size normalization")
        self.load_predictions_3d()

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Convert the temporary directory path to a Path object
            tmpdir_path = Path(tmpdirname)
            tmp_size_norm_file, tmp_size_norm_angles_file = self.initialize_output_folder(
                tmpdir_path
            )

            keypoints = np.array(list(self.kpt_dict.keys()))

            keypoints_to_index = {kpt: i for i, kpt in enumerate(keypoints)}
            # TODO batch this, costs a lot of memory...
            kpts = np.array(self.predictions_3D_mmap)

            # add a 'spine base' to set as the
            spine_high_idx = np.where(np.array(keypoints) == "spine_high")[0][0]
            spine_mid_idx = np.where(np.array(keypoints) == "spine_mid")[0][0]
            spine_base_pos = np.expand_dims(
                (kpts[:, spine_mid_idx] + kpts[:, spine_high_idx]) / 2, 1
            )
            kpts = np.concatenate([kpts, spine_base_pos], axis=1)
            keypoints_to_index["spine_base"] = len(keypoints)

            # initialize positions (fill in any nans, shouldn't be needed with gimbal)
            kpts = generate_initial_positions(kpts)

            # convert keypoints to a dictionary
            kpts_dict = {}
            for key, k_index in keypoints_to_index.items():
                kpts_dict[key] = kpts[:, k_index]
            kpts_dict["joints"] = list(keypoints_to_index.keys())
            kpts = kpts_dict

            # assign hierarchy
            kpts["hierarchy"] = self.hierarchy
            kpts["root_joint"] = self.root_joint

            # compute bone lengths and stds
            kpts = get_bone_lengths(
                kpts,
                self.template_bone_length_mean,
                self.template_bone_length_std,
            )

            # create a skeleton to recompute positions from angles
            kpts = generate_kpt_offsets_and_skeleton(kpts, self.hierarchy)

            # calculate joint angles and add them in as kpts[joint+'_angles']
            kpts = calculate_joint_angles_parallel(
                kpts,
                root_joint=self.root_joint,
                samples_to_calculate=self.subsample,
                n_jobs=self.n_jobs,
            )

            # compute new sizes
            kpts = size_normalize(
                kpts,
                template_bone_length_mean=self.template_bone_length_mean,
                hierarchy=self.hierarchy,
                root_joint=self.root_joint,
                rigid_bones=self.rigid_bones,
            )

            # convert back to array
            recomputed_keypoints = np.stack(
                [kpts[f"recomputed_{joint}"] for joint in keypoints], axis=1
            )

            # convert angles back to array
            recomputed_angles = np.stack(
                [kpts[f"recomputed_angles_{joint}"] for joint in keypoints], axis=1
            )
            recomputed_angles[:, :, 2] = (
                recomputed_angles[:, :, 2]
                + np.expand_dims(kpts[f"{self.root_joint}_angles"][:, 2], -1)
            ) % (2 * np.pi)
            recomputed_angles[:, :, 2][recomputed_angles[:, :, 2] > np.pi] -= 2 * np.pi

            self.size_norm_mmap[:] = recomputed_keypoints
            self.size_norm_mmap_angles[:] = recomputed_angles

            # move outputs to self.size_norm_output_directory
            shutil.move(
                tmp_size_norm_file, self.size_norm_output_directory / tmp_size_norm_file.name
            )
            shutil.move(
                tmp_size_norm_angles_file,
                self.size_norm_output_directory / tmp_size_norm_angles_file.name,
            )
        # mark as completed
        (self.size_norm_output_directory / "completed.log").touch()

    def initialize_output_folder(self, tmpdir_path):

        self.size_norm_output_directory.mkdir(exist_ok=True, parents=True)
        # initialize
        mmap_dtype = "float32"
        keypoints_3d_shape = self.predictions_3D_mmap.shape
        keypoints_3d_dtype = mmap_dtype
        keypoints_3d_shape_str = "x".join(map(str, keypoints_3d_shape))
        size_norm_file = (
            tmpdir_path / f"size_norm.{keypoints_3d_dtype}.{keypoints_3d_shape_str}.mmap"
        )

        size_norm_angles_file = (
            tmpdir_path / f"size_norm_angles.{keypoints_3d_dtype}.{keypoints_3d_shape_str}.mmap"
        )

        self.size_norm_mmap = np.memmap(
            size_norm_file,
            dtype=keypoints_3d_dtype,
            mode="w+",
            shape=keypoints_3d_shape,
        )

        self.size_norm_mmap_angles = np.memmap(
            size_norm_angles_file,
            dtype=keypoints_3d_dtype,
            mode="w+",
            shape=keypoints_3d_shape,
        )

        return size_norm_file, size_norm_angles_file


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


# remove jittery keypoints by applying a median filter along each axis
def median_filter(kpts, window_size=3):

    import copy

    filtered = copy.deepcopy(kpts)

    from scipy.signal import medfilt

    # apply median filter to get rid of poor keypoints estimations
    for joint in filtered["joints"]:
        joint_kpts = filtered[joint]
        xs = joint_kpts[:, 0]
        ys = joint_kpts[:, 1]
        zs = joint_kpts[:, 2]
        xs = medfilt(xs, window_size)
        ys = medfilt(ys, window_size)
        zs = medfilt(zs, window_size)
        filtered[joint] = np.stack([xs, ys, zs], axis=-1)

    return filtered


# general rotation matrices
def get_R_x(theta):
    R = np.array(
        [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]
    )
    return R


def get_R_y(theta):
    R = np.array(
        [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
    )
    return R


def get_R_z(theta):
    R = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    )
    return R


def get_R_z_vectorized(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R = np.array(
        [
            [cos_theta, -sin_theta, np.zeros_like(theta)],
            [sin_theta, cos_theta, np.zeros_like(theta)],
            [np.zeros_like(theta), np.zeros_like(theta), np.ones_like(theta)],
        ]
    )
    return R


def get_R_y_vectorized(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R = np.array(
        [
            [cos_theta, np.zeros_like(theta), sin_theta],
            [np.zeros_like(theta), np.ones_like(theta), np.zeros_like(theta)],
            [-sin_theta, np.zeros_like(theta), cos_theta],
        ]
    )
    return R


def get_R_x_vectorized(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R = np.array(
        [
            [np.ones_like(theta), np.zeros_like(theta), np.zeros_like(theta)],
            [np.zeros_like(theta), cos_theta, -sin_theta],
            [np.zeros_like(theta), sin_theta, cos_theta],
        ]
    )
    return R


# calculate rotation matrix to take A vector to B vector
def Get_R(A, B):

    # get unit vectors
    uA = A / np.sqrt(np.sum(np.square(A)))
    uB = B / np.sqrt(np.sum(np.square(B)))

    # get products
    dotprod = np.sum(uA * uB)
    crossprod = np.sqrt(np.sum(np.square(np.cross(uA, uB))))  # magnitude

    # get new unit vectors
    u = uA
    v = uB - dotprod * uA
    v = v / np.sqrt(np.sum(np.square(v)))
    w = np.cross(uA, uB)
    w = w / np.sqrt(np.sum(np.square(w)))

    # get change of basis matrix
    C = np.array([u, v, w])

    # get rotation matrix in new basis
    R_uvw = np.array([[dotprod, -crossprod, 0], [crossprod, dotprod, 0], [0, 0, 1]])

    # full rotation matrix
    R = C.T @ R_uvw @ C
    # print(R)
    return R


# Same calculation as above using a different formalism
def Get_R2(A, B):

    # get unit vectors
    uA = A / np.sqrt(np.sum(np.square(A)))
    uB = B / np.sqrt(np.sum(np.square(B)))

    v = np.cross(uA, uB)
    s = np.sqrt(np.sum(np.square(v)))
    c = np.sum(uA * uB)

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    R = np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)

    return R


# decomposes given R matrix into rotation along each axis. In this case Rz @ Ry @ Rx
def Decompose_R_ZYX(R):

    # decomposes as RzRyRx. Note the order: ZYX <- rotation by x first
    thetaz = np.arctan2(R[1, 0], R[0, 0])
    thetay = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    thetax = np.arctan2(R[2, 1], R[2, 2])

    return thetaz, thetay, thetax


def Decompose_R_ZXY(R):

    # decomposes as RzRXRy. Note the order: ZXY <- rotation by y first
    thetaz = np.arctan2(-R[0, 1], R[1, 1])
    thetay = np.arctan2(-R[2, 0], R[2, 2])
    thetax = np.arctan2(R[2, 1], np.sqrt(R[2, 0] ** 2 + R[2, 2] ** 2))

    return thetaz, thetay, thetax


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


def standardize_poses(poses, indices_egocentric):
    poses = poses - poses.mean(1, keepdims=True)
    front = poses[:, indices_egocentric[1], :2]
    back = poses[:, indices_egocentric[0], :2]
    angle = vector_to_angle(front - back)
    rot = angle_to_rotation_matrix(angle, keypoint_dim=poses.shape[-1])
    return poses @ rot


def get_joint_rotations(joint_name, joints_hierarchy, joints_offsets, frame_rotations, frame_pos):
    _invR = np.eye(3)
    for i, parent_name in enumerate(joints_hierarchy[joint_name]):
        _r_angles = frame_rotations[parent_name]
        R = get_R_z(_r_angles[0]) @ get_R_x(_r_angles[1]) @ get_R_y(_r_angles[2])
        _invR = _invR @ R.T

    b = _invR @ (frame_pos[joint_name] - frame_pos[joints_hierarchy[joint_name][0]])

    _R = Get_R2(joints_offsets[joint_name], b)
    tz, ty, tx = Decompose_R_ZXY(_R)
    joint_rs = np.array([tz, tx, ty])

    return joint_rs


import scipy.stats as stats


def remap_to_normal_distribution(x, loc=0, scale=1):
    # Calculate the percentiles ranks of the points in x
    percentile_ranks = stats.rankdata(x, "average") / len(x) * 100

    # Convert these percentiles into the values of a normal distribution with mean=0 and std=1
    normal_values = stats.norm.ppf(percentile_ranks / 100.0, loc=loc, scale=scale)

    return normal_values


import copy


def get_bone_lengths(
    kpts,
    template_bone_length_mean,
    template_bone_length_std,
    show_plot=True,
    prefix="",
    hierarchy=None,
    clip_low_pct=5,
    clip_high_pct=95,
):
    if hierarchy == None:
        hierarchy = kpts["hierarchy"]

    bone_lengths_median = {}
    bone_stds = {}
    bone_bounds = {}
    bone_mads = {}
    bone_lengths = {}
    for ji, joint in enumerate(
        tqdm(kpts["joints"], desc="calculating joint lengths", leave=False)
    ):
        if joint == kpts["root_joint"]:
            continue

        parent = hierarchy[joint][0]
        joint_kpts = kpts[f"{prefix}{joint}"]
        parent_kpts = kpts[f"{prefix}{parent}"]

        _bone = joint_kpts - parent_kpts
        _bone_lengths = np.sqrt(np.sum(np.square(_bone), axis=-1))

        # get the median length
        bone_lengths_median[joint] = np.nanmedian(_bone_lengths)
        min_, max_ = np.percentile(_bone_lengths, (clip_low_pct, clip_high_pct))
        # _bone_lengths[_bone_lengths > max_] = max_
        # _bone_lengths[_bone_lengths < min_] = min_
        bone_stds[joint] = np.nanstd(_bone_lengths)
        bone_mads[joint] = np.nanmedian(np.abs(bone_lengths_median[joint] - _bone_lengths))
        bone_bounds[joint] = (min_, max_)

        # remap to a normal distribution
        _bone_lengths = remap_to_normal_distribution(
            _bone_lengths,
            loc=template_bone_length_mean[joint],
            scale=template_bone_length_std[joint],
        )
        # remove any outliers outside of 3 std
        max_ = template_bone_length_mean[joint] + template_bone_length_std[joint] * 3
        _bone_lengths[_bone_lengths > max_] = max_
        min_ = template_bone_length_mean[joint] - template_bone_length_std[joint] * 3
        _bone_lengths[_bone_lengths < min_] = min_

        # save all bone lengths for later use
        bone_lengths[joint] = _bone_lengths

    kpts["bone_lengths_median"] = bone_lengths_median
    kpts["bone_lengths"] = bone_lengths
    kpts["bone_stds"] = bone_stds
    kpts["bone_bounds"] = bone_bounds
    return kpts


def generate_kpt_offsets_and_skeleton(kpts, hierarchy):
    # this defines a generic skeleton to which we can apply rotations to
    body_lengths = kpts["bone_lengths"]
    # define skeleton offset directions
    offset_directions = {}

    for joint in kpts["joints"]:
        if joint == kpts["root_joint"]:
            offset_directions[joint] = np.array([0, 0, 0])
        else:
            offset_directions[joint] = np.array([1, 0, 0])

    kpts["offset_directions"] = offset_directions

    # create skeleton
    base_skeleton = {kpts["root_joint"]: np.array([0, 0, 0])}
    for joint, parents in hierarchy.items():
        if joint == kpts["root_joint"]:
            continue
        base_skeleton[joint] = (
            offset_directions[joint] * kpts["bone_lengths_median"][joint]
        )  # + base_skeleton[parents[0]]
    kpts["base_skeleton"] = base_skeleton
    kpts["normalization"] = 1
    return kpts


# Refactored function to calculate joint angles for a single frame
def calculate_frame_joint_angles(
    joints, hierarchy, offset_directions, frame_pos, root_joint="spine_base"
):
    root_position, root_rotation = get_root_position_and_rotation(frame_pos)
    frame_rotations = {root_joint: root_rotation}

    for joint in joints:
        frame_pos[joint] -= root_position

    max_connected_joints = max(len(hierarchy[joint]) for joint in joints)

    for depth in range(1, max_connected_joints + 1):
        for joint in joints:
            if len(hierarchy[joint]) == depth:
                joint_rs = get_joint_rotations(
                    joint,
                    hierarchy,
                    offset_directions,
                    frame_rotations,
                    frame_pos,
                )
                frame_rotations[joint] = joint_rs

    return {joint: frame_rotations[joint] for joint in joints}


# Parallelized version of the calculate_joint_angles function
def calculate_joint_angles_parallel(
    kpts, root_joint="spine_base", samples_to_calculate=None, n_jobs=-1
):
    for joint in kpts["joints"]:
        kpts[joint + "_angles"] = []

    if samples_to_calculate is None:
        samples_to_calculate = range(kpts[root_joint].shape[0])

    # Parallel computation

    # results = Parallel(n_jobs=n_jobs)(
    results = Parallel(n_jobs=1)(
        delayed(calculate_frame_joint_angles)(
            joints=kpts["joints"],
            hierarchy=kpts["hierarchy"],
            offset_directions=kpts["offset_directions"],
            frame_pos={joint: kpts[joint][framenum] for joint in kpts["joints"]},
            root_joint=root_joint,
        )
        for framenum in tqdm(samples_to_calculate, desc="calculating angles")
    )

    # Update kpts with calculated angles
    for framenum, frame_result in enumerate(results):
        for joint in kpts["joints"]:
            kpts[joint + "_angles"].append(frame_result[joint])

    # Convert joint angles list to numpy arrays
    for joint in kpts["joints"]:
        kpts[joint + "_angles"] = np.array(kpts[joint + "_angles"])

    return kpts


# calculate the rotation of the root joint with respect to the world coordinates
def get_root_position_and_rotation(
    frame_pos, root_joint="spine_base", root_define_joints=["spine_high", "spine_mid"]
):
    # root position is saved directly
    root_position = frame_pos[root_joint]

    # calculate unit vectors of root joint
    root_u = frame_pos[root_define_joints[0]] - frame_pos[root_joint]
    root_u = root_u / np.sqrt(np.sum(np.square(root_u)))
    root_v = frame_pos[root_define_joints[1]] - frame_pos[root_joint]
    root_v = root_v / np.sqrt(np.sum(np.square(root_v)))
    root_w = np.cross(root_u, root_v)

    # Make the rotation matrix
    C = np.array([root_u, root_v, root_w]).T
    thetaz, thetay, thetax = Decompose_R_ZXY(C)
    root_rotation = np.array([thetaz, thetax, thetay])

    return root_position, root_rotation


def get_temp_hierarchy(_j, hierarchy, root_joint):
    hierarchy_temp = hierarchy[_j].copy()
    if _j != root_joint:
        hierarchy_temp.insert(0, _j)
    hierarchy_temp = hierarchy_temp[::-1]
    return hierarchy_temp


def get_rotation_chain(joint, hierarchy_temp, frame_rotations, root_joint="spine_base"):
    # this code assumes ZXY rotation order
    R = np.eye(3)
    for parent in hierarchy_temp:
        angles = frame_rotations[parent]
        _R = get_R_z(angles[0]) @ get_R_x(angles[1]) @ get_R_y(angles[2])
        R = R @ _R

    return R


def size_normalize(
    kpts,
    template_bone_length_mean,
    hierarchy,
    root_joint="spine_base",
    samples_to_calculate=None,
    rigid_bones=True,
):
    if samples_to_calculate is None:
        samples_to_calculate = range(kpts[root_joint].shape[0])

    kpts[f"recomputed_{root_joint}"] = kpts[f"{root_joint}"]
    for _j in tqdm(kpts["hierarchy"], desc="computing new keypoint positions", leave=False):
        if _j == root_joint:
            continue

        # get hierarchy of how the joint connects back to root joint
        kpt_hierarchy = hierarchy[_j]

        # get the current position of the parent joint
        parent = kpts["hierarchy"][_j][0]
        r1 = kpts[f"recomputed_{parent}"]

        # get rotation relative to parent
        hierarchy_temp = get_temp_hierarchy(_j, hierarchy, root_joint)

        R = [np.eye(3) for framenum in samples_to_calculate]
        for parent in hierarchy_temp:
            angles = kpts[parent + "_angles"]
            R_zs = get_R_z_vectorized(angles[: len(samples_to_calculate), 0]).transpose((2, 0, 1))
            R_xs = get_R_x_vectorized(angles[: len(samples_to_calculate), 1]).transpose((2, 0, 1))
            R_ys = get_R_y_vectorized(angles[: len(samples_to_calculate), 2]).transpose((2, 0, 1))
            R = np.einsum("...ij,...jk,...kl,...lm->...im", R, R_zs, R_xs, R_ys)

        if rigid_bones:
            bone_length = np.zeros((len(samples_to_calculate), 3))
            bone_length[:, 0] = template_bone_length_mean[_j]
        else:
            bone_length = np.zeros((len(samples_to_calculate), 3))
            bone_length[:, 0] = kpts["bone_lengths"][_j][: len(samples_to_calculate)]

        # add bone length
        r2 = np.array(
            [
                r1_i + R_i @ bone_length_i
                for r1_i, R_i, bone_length_i in zip(
                    tqdm(r1, desc="adding bone length", leave=False), R, bone_length
                )
            ]
        )
        kpts[f"recomputed_{_j}"] = r2
        # get the angles
        kpts[f"recomputed_angles_{_j}"] = np.array([Decompose_R_ZYX(i) for i in R])

    return kpts
