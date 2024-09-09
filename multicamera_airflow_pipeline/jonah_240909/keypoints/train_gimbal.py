import sys
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import joblib
import numpy as np
import joblib, json, os, h5py
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import scipy.stats
from tqdm.autonotebook import tqdm
import time
import copy
import jax
import networkx as nx
import gimbal.mcmc3d_full
import gimbal
import jax.numpy as jnp
import jax.random as jr
import multicam_calibration as mcc

jax.config.update("jax_enable_x64", False)
from tensorflow_probability.substrates.jax.distributions import VonMisesFisher as VMF
from gimbal.fit import em_step
from jax import lax, jit
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
from jax.lib import xla_bridge

logger.info(f"Python interpreter binary location: {sys.executable}")
logger.info(f"Python version: {sys.version}")
logger.info(f"JAX version: {jax.__version__}")
logger.info(f"Backend: {xla_bridge.get_backend().platform}")
logger.info(f"JAX devices: {jax.devices()}")

# load skeleton
try:
    from multicamera_airflow_pipeline.tim_240731.skeletons.sainburg25pt import dataset_info
except:

    # Add the directory containing the file to the system path
    sys.path.append(
        "/n/groups/datta/tim_sainburg/projects/multicamera_airflow_pipeline/multicamera_airflow_pipeline/tim_240731/skeletons/sainburg25pt.py"
    )
    # Now import the dataset_info dictionary
    from sainburg25pt import dataset_info
keypoint_info = dataset_info["keypoint_info"]
default_keypoints = [keypoint_info[i]["name"] for i in keypoint_info.keys()]
default_keypoints = np.array(default_keypoints)

# load skeleton
from multicamera_airflow_pipeline.tim_240731.skeletons.defaults import (
    dataset_info,
    parents_dict,
    keypoint_info,
    keypoints,
    keypoints_order,
    kpt_dict,
    gimbal_skeleton,
)

default_keypoints = keypoints
skeleton = gimbal_skeleton


class GimbalTrainer:
    def __init__(
        self,
        gimbal_output_directory,
        calibration_folder,  # in jarvis format
        predictions_3d_directory,
        samplerate,
        num_iters_train=1000,
        num_states=50,
        indices_egocentric=[0, 2],
        max_inlier_var=100,  # the greatest expected variance for a bone
        conf_sigmoid_gain=20,
        conf_sigmoid_center=0.5,
        thresh_bad_keypoints_mads=10,
        constant_inlier_variance=25,
        outlier_thresh_mm=150,  # distance of any keypoint from median
        obs_outlier_variance=1e6,
        pos_dt_variance=1,
        num_leapfrog_steps=5,
        step_size=0.2,
        training_subsample_frames=None,
        plot_triangulation_error=False,
        plot_joint_directions=True,
        plot_fit_likelihood=True,
        plot_inference_likelihood=False,
        remove_samples_with_nans=False,
        recompute_completed=False,
        keypoints=default_keypoints,
    ):
        """
        Parameters
        ----------
        gimbal_output_directory : Path
            Directory to save the gimbal output files.
        calibration_folder : Path
            Directory containing the calibration files.
        predictions_3d_directory : Path
            Directory containing the 3D predictions.
        samplerate : int
            The samplerate of the video
        num_iters_train : int
            Number of iterations to train the gimbal model.
        num_states : int
            Number of states in the gimbal model.
        indices_egocentric : list
            Indices of the egocentric joints (relative to keypoints, this should be
            the base of the spine and the top of the spine).
        max_inlier_var : int
            The greatest expected variance for a bone.
        conf_sigmoid_gain : int
            The gain of the sigmoid function for confidence.
        conf_sigmoid_center : float
            The center of the sigmoid function for confidence.
        thresh_bad_keypoints_mads : int
            Threshold for bad keypoints in MADS.
        constant_inlier_variance : int
            The constant inlier variance.
        outlier_thresh_mm : int
            The distance of any keypoint from the median.
        obs_outlier_variance : int
            The outlier variance. (gimbal model).
        pos_dt_variance : int
            The position variance. (gimbal model).
        num_leapfrog_steps : int
            Number of leapfrog steps (gimbal model).
        step_size : float
            The step size. (gimbal model).
        training_subsample_frames : int
            Number of frames to subsample for training. (defaults to 30 minutes).
        plot_triangulation_error : bool
            Whether to plot the triangulation error.
        plot_joint_directions : bool
            Whether to plot the joint directions.
        plot_fit_likelihood : bool
            Whether to plot the fit likelihood.
        plot_inference_likelihood : bool
            Whether to plot the inference likelihood.
        remove_samples_with_nans : bool
            Whether to remove samples with NaNs.
        recompute_completed : bool
            Whether to recompute the completed status.
        """

        # set parameters
        self.samplerate = samplerate
        self.predictions_3d_directory = Path(predictions_3d_directory)
        self.calibration_folder = Path(calibration_folder)
        self.gimbal_output_directory = Path(gimbal_output_directory)
        self.num_iters_train = num_iters_train
        self.num_states = num_states
        self.indices_egocentric = indices_egocentric
        self.max_inlier_var = max_inlier_var
        self.conf_sigmoid_gain = conf_sigmoid_gain
        self.conf_sigmoid_center = conf_sigmoid_center
        self.thresh_bad_keypoints_mads = thresh_bad_keypoints_mads
        self.constant_inlier_variance = constant_inlier_variance
        self.outlier_thresh_mm = outlier_thresh_mm
        self.obs_outlier_variance = obs_outlier_variance
        self.pos_dt_variance = pos_dt_variance
        self.num_leapfrog_steps = num_leapfrog_steps
        self.step_size = step_size
        self.plot_triangulation_error = plot_triangulation_error
        self.plot_joint_directions = plot_joint_directions
        self.plot_fit_likelihood = plot_fit_likelihood
        self.plot_inference_likelihood = plot_inference_likelihood
        self.remove_samples_with_nans = remove_samples_with_nans
        self.recompute_completed = recompute_completed
        self.gimbal_params_file = self.gimbal_output_directory / "gimbal_params.p"
        self.gimbal_output_directory.mkdir(parents=True, exist_ok=True)
        self.keypoints = keypoints
        self.training_subsample_frames = training_subsample_frames

    def check_completed(self):
        return self.gimbal_params_file.exists()

    def load_calibration_data(self):
        self.all_extrinsics, self.all_intrinsics, self.camera_names = mcc.load_calibration(
            self.calibration_folder,
            load_format="jarvis",
        )
        self.cameras = copy.deepcopy(self.camera_names)

    def load_predictions(self):

        # ensure that triangulation completed
        assert (self.predictions_3d_directory / "triangulation_completed.log").exists()

        confidences_2d_file = list(self.predictions_3d_directory.glob("confidences_2d*.mmap"))[0]
        confidences_3d_file = list(self.predictions_3d_directory.glob("confidences_3d*.mmap"))[0]
        predictions_2d_file = list(self.predictions_3d_directory.glob("predictions_2d*.mmap"))[0]
        predictions_3d_file = list(self.predictions_3d_directory.glob("predictions_3d*.mmap"))[0]
        reprojection_errors_file = list(
            self.predictions_3d_directory.glob("reprojection_errors*.mmap")
        )[0]
        # load confidences and predictions
        self.confidences_2D_mmap = load_memmap_from_filename(confidences_2d_file)
        self.predictions_2D_mmap = load_memmap_from_filename(predictions_2d_file)
        self.confidences_3D_mmap = load_memmap_from_filename(confidences_3d_file)
        self.predictions_3D_mmap = load_memmap_from_filename(predictions_3d_file)
        self.reprojection_errors_mmap = load_memmap_from_filename(reprojection_errors_file)

        # load a sample into memory for training
        if self.training_subsample_frames is None:
            self.training_subsample_frames = self.samplerate * 60 * 30
        self.confidences_2D = confidences = np.array(
            self.confidences_2D_mmap[: self.training_subsample_frames]
        )
        self.positions_2D = np.array(self.predictions_2D_mmap[: self.training_subsample_frames])
        self.confidences_3D = np.array(self.confidences_3D_mmap[: self.training_subsample_frames])
        self.positions_3D = np.array(self.predictions_3D_mmap[: self.training_subsample_frames])
        self.reprojection_errors = np.array(
            self.reprojection_errors_mmap[: self.training_subsample_frames]
        )

    def run(self):
        # skip if finished
        if self.check_completed() and (self.recompute_completed == False):
            logger.info(f"Gimbal training already completed, skipping")
            return

        logger.info(f"Running gimbal training on {self.gimbal_output_directory}")

        logger.info(f"Loading calibration data")
        self.load_calibration_data()

        logger.info(f"Loading predictions")
        self.load_predictions()

        # remove outliers
        logger.info(f"Removing outliers")
        outlier_pts = np.sqrt(
            np.sum(
                (self.positions_3D - np.expand_dims(np.median(self.positions_3D, axis=1), 1)) ** 2,
                axis=(2),
            )
        )
        self.positions_3D[outlier_pts > self.outlier_thresh_mm] = np.nan

        if self.remove_samples_with_nans:
            mask = (~np.isnan(self.positions_3D)).all((1, 2)) == False
            logger.info(f"masking {np.sum(mask)} timepoints")
            self.positions_3D[mask] = np.nan

        # fill in nans
        logger.info(f"Filling in NaNs")
        for cami in range(self.positions_2D.shape[1]):
            self.positions_2D[:, cami] = generate_initial_positions(self.positions_2D[:, cami])
        self.positions_3D = generate_initial_positions(self.positions_3D)

        poses = self.positions_3D
        bodyparts = self.keypoints
        camera_calibration_order = self.cameras
        camera_calibration_order = list(np.array(camera_calibration_order).astype(str))
        bodyparts = list(np.array(bodyparts).astype(str))

        logger.info(f"Standardizing data")
        # determine node order
        keypoints = use_bodyparts = bodyparts
        keypoints = np.array(keypoints)
        use_bodyparts_idx = np.array([bodyparts.index(bp) for bp in use_bodyparts])
        edges = np.array(get_edges(use_bodyparts, skeleton))
        node_order, parents = build_node_hierarchy(use_bodyparts, skeleton, "spine_low")
        edges = np.argsort(node_order)[edges]
        keypoints_reorder = np.array(keypoints)[node_order]

        # rearrage to fit node order
        poses = poses[:, node_order]
        confidence = self.confidences_2D[:, :, node_order]
        # observations = self.positions_2D[:, :, node_order]

        # standardize poses
        # poses = poses[(~np.isnan(poses)).all((1, 2))]
        poses_standard = standardize_poses(poses, self.indices_egocentric)

        # get triangulation error
        logger.info(f"Estimating triangulation error & computing variance")
        median_error = np.nanmedian(self.reprojection_errors, axis=0)

        # save fig of skeleton_distances to gimbal_output_directory
        fig = plot_skeleton_distances(skeleton, keypoints, node_order, poses_standard)
        fig.savefig(self.gimbal_output_directory / "skeleton_distances.jpg", format="jpg")
        plt.close()

        # compute the variance on position
        obs_inlier_variance = np.minimum(median_error**2, self.max_inlier_var)
        obs_inlier_variance[np.isnan(obs_inlier_variance)] = self.max_inlier_var

        # get radii and directions of poses
        radii = np.sqrt(((poses - poses[:, parents]) ** 2).sum(-1))
        directions = compute_directions(poses_standard, parents)

        if self.plot_joint_directions:
            # plot directions
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.matshow(directions[:10000, :, 1].T, aspect="auto", cmap="twilight")
            plt.yticks(range(len(use_bodyparts)), [use_bodyparts[i] for i in node_order])
            # save to gimbal_output_directory
            plt.savefig(self.gimbal_output_directory / "joint_directions.jpg")
            plt.close()

        logger.info(f"Running model fit")
        # fit gimbal model
        key = jr.PRNGKey(1)
        dirs = jnp.array(directions)
        em_output = em_movMF(key, directions[:, 1:], self.num_states, self.num_iters_train)
        lls, E_zs, pis_em, mus_em, kappas_em = map(np.array, em_output)
        kappas_root = np.zeros((self.num_states, 1))
        mu_root = (np.arange(dirs.shape[-1]) == 0).astype(float)
        mus_root = np.tile(mu_root, (self.num_states, 1, 1))
        kappas_em = np.concatenate([kappas_root, kappas_em], axis=1)
        mus_em = np.concatenate([mus_root, mus_em], axis=1)

        if self.plot_fit_likelihood:
            fig, axs = plt.subplots(1, 2)
            axs[0].plot(lls)
            axs[0].set_xlabel("Iters")
            axs[0].set_ylabel("Log likelihood")
            axs[1].bar(range(self.num_states), pis_em)
            axs[1].set_xlabel("States")
            axs[1].set_ylabel("State probability")
            fig.set_size_inches((9, 3))
            plt.tight_layout()
            plt.savefig(self.gimbal_output_directory / "fit_likelihood.jpg")
            plt.close()

        # mask out joints that are very far away from their parent joint
        med_radii = np.median(radii, axis=0)
        mad_radii = scipy.stats.median_abs_deviation(radii, axis=0)
        # todo, this could be bidirectional, right now the 'base' will never be affected
        bad_samples_mask = radii > (med_radii + mad_radii * self.thresh_bad_keypoints_mads)
        poses[bad_samples_mask] = np.nan
        poses_standard[bad_samples_mask] = np.nan
        # recompute radii and directions of poses
        radii = np.sqrt(((poses - poses[:, parents]) ** 2).sum(-1))
        directions = compute_directions(poses_standard, parents)

        # save gimbal parameters
        params = {
            "obs_inlier_variance": obs_inlier_variance,
            "indices_egocentric": self.indices_egocentric,
            "parents": parents,
            "radii": np.nanmedian(radii, axis=0),
            "radii_std": np.nanstd(radii, axis=0),
            "kappas": kappas_em,
            "mus": mus_em,
            "pis": pis_em,
        }
        joblib.dump(params, self.gimbal_params_file)


def get_edges(use_bodyparts, skeleton):
    """Represent the skeleton as a list of index-pairs.

    Parameters
    -------
    use_bodyparts: list
        Bodypart names

    skeleton: list
        Pairs of bodypart names as tuples (bodypart1,bodypart2)

    Returns
    -------
    edges: list
        Pairs of indexes representing the enties of `skeleton`
    """
    edges = []
    if len(skeleton) > 0:
        if isinstance(skeleton[0][0], int):
            edges = skeleton
        else:
            assert use_bodyparts is not None, fill(
                "If skeleton edges are specified using bodypart names, "
                "`use_bodyparts` must be specified"
            )

            for bp1, bp2 in skeleton:
                if bp1 in use_bodyparts and bp2 in use_bodyparts:
                    edges.append([use_bodyparts.index(bp1), use_bodyparts.index(bp2)])
    return edges


def build_node_hierarchy(bodyparts, skeleton, root_node):
    """
    Define a rooted hierarchy based on the edges of a spanning tree.

    Parameters
    ----------
    bodyparts: list of str
        Ordered list of node names.

    skeleton: list of tuples
        Edges of the spanning tree as pairs of node names.

    root_node: str
        The desired root node of the hierarchy

    Returns
    -------
    node_order: array of shape (num_nodes,)
        Integer array specifying an ordering of nodes in which parents
        precede children (i.e. a topological ordering).

    parents: array of shape (num_nodes,)
        Child-parent relationships using the indexes from `node_order`,
        such that `parent[i]==j` when `node_order[j]` is the parent of
        `node_order[i]`.

    Raises
    ------
    ValueError
        The edges in `skeleton` do not define a spanning tree.
    """
    G = nx.Graph()
    G.add_nodes_from(bodyparts)
    G.add_edges_from(skeleton)

    if not nx.is_tree(G):
        cycles = list(nx.cycle_basis(G))
        raise ValueError(
            "The skeleton does not define a spanning tree, "
            "as it contains the following cycles: {}".format(cycles)
        )

    if not nx.is_connected(G):
        raise ValueError(
            "The skeleton does not define a spanning tree, "
            "as it contains multiple connected components."
        )

    node_order = list(nx.dfs_preorder_nodes(G, root_node))
    parents = np.zeros(len(node_order), dtype=int)

    for i, j in skeleton:
        i, j = node_order.index(i), node_order.index(j)
        if i < j:
            parents[j] = i
        else:
            parents[i] = j

    node_order = np.array([bodyparts.index(n) for n in node_order])
    return node_order, parents


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


# fit gimbal
def compute_directions(poses, parents):
    dirs = poses - poses[:, parents]
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-16
    return dirs[~np.isnan(dirs).any((1, 2))]


def fit_gimbal_model(dirs, num_states, num_iters):

    key = jr.PRNGKey(1)
    dirs = jnp.array(dirs)

    em_output = gimbal.fit.em_movMF(key, dirs[:, 1:], num_states, num_iters)
    lls, E_zs, pis_em, mus_em, kappas_em = map(np.array, em_output)

    kappas_root = np.zeros((num_states, 1))
    mu_root = (np.arange(dirs.shape[-1]) == 0).astype(float)
    mus_root = np.tile(mu_root, (num_states, 1, 1))

    kappas_em = np.concatenate([kappas_root, kappas_em], axis=1)
    mus_em = np.concatenate([mus_root, mus_em], axis=1)

    return lls, pis_em, mus_em, kappas_em


def generate_gimbal_params(
    camera_matrices,
    fitted_params,
    obs_outlier_variance,
    pos_dt_variance,
    num_leapfrog_steps=5,
    step_size=1e-1,
    dtype="float32",
):

    num_cameras, num_joints = fitted_params["obs_inlier_variance"].shape

    params = {
        "obs_outlier_probability": jnp.zeros((num_cameras, num_joints), dtype),
        "obs_outlier_location": jnp.zeros((num_cameras, num_joints, 2), dtype),
        "obs_outlier_variance": jnp.ones((num_cameras, num_joints), dtype) * obs_outlier_variance,
        "obs_inlier_location": jnp.zeros((num_cameras, num_joints, 2), dtype),
        "obs_inlier_variance": jnp.array(fitted_params["obs_inlier_variance"], dtype),
        "camera_matrices": jnp.array(camera_matrices, dtype),
        "pos_radius": jnp.array(fitted_params["radii"], dtype),
        "pos_radial_variance": jnp.array([1e8, *fitted_params["radii_std"][1:] ** 2], dtype),
        "parents": jnp.array(fitted_params["parents"]),
        "pos_dt_variance": jnp.ones(num_joints, dtype) * pos_dt_variance,
        "state_probability": jnp.array(fitted_params["pis"], dtype),
        "state_directions": jnp.array(fitted_params["mus"], dtype),
        "state_concentrations": jnp.array(fitted_params["kappas"], dtype),
        "crf_keypoints": fitted_params["indices_egocentric"][::-1],
        "crf_abscissa": jnp.zeros(3, dtype).at[0].set(1),
        "crf_normal": jnp.zeros(3, dtype).at[2].set(1),
        "crf_axes": jnp.eye(3).astype(dtype),
        "state_transition_count": jnp.ones(len(fitted_params["pis"]), dtype),
        "step_size": step_size,
        "num_leapfrog_steps": num_leapfrog_steps,
    }
    return gimbal.mcmc3d_full.initialize_parameters(params)


def generate_initial_positions(positions):
    init_positions = np.zeros_like(positions)
    for k in range(positions.shape[1]):
        ix = np.nonzero(~np.isnan(positions[:, k, 0]))[0]
        for i in range(positions.shape[2]):
            init_positions[:, k, i] = np.interp(
                np.arange(positions.shape[0]), ix, positions[:, k, i][ix]
            )
    return init_positions


def generate_outlier_probs(
    confidence, outlier_prob_bounds=[1e-3, 1 - 1e-6], conf_sigmoid_center=0.3, conf_sigmoid_gain=20
):

    outlier_p = jax.nn.sigmoid((conf_sigmoid_center - confidence) * conf_sigmoid_gain)
    return jnp.clip(outlier_p, *outlier_prob_bounds)


def em_movMF(key, xs, num_states, num_iters):
    """
    xs should not pass in joint j=0, so we don't have to worry about it being undefined
    xs: shape (num_timesteps, num_joints, dim)
    """
    num_timesteps, num_joints, dim = xs.shape

    # Initialize parameters
    pis = jnp.ones(num_states) / num_states

    # Initialize kappas and mus
    key, key_1, key_2 = jr.split(key, 3)

    kappas = jnp.maximum(jr.normal(key_1, shape=(num_states, num_joints)) + 5, 0)

    rs = jnp.nansum(xs, axis=0)  # Shape (num_joints, dim)
    mus_mle = rs / jnp.linalg.norm(rs, axis=-1, keepdims=True)
    mus = VMF(mus_mle[None, :, :], jnp.ones((num_states, num_joints)) * 5).sample(
        seed=key_2
    )  # Sample with low concentration

    assert jnp.all(~jnp.isnan(xs))
    assert jnp.all(~jnp.isnan(mus))
    assert jnp.all(~jnp.isnan(kappas)) and jnp.all(kappas >= 0)

    init_carry = carry = (
        jnp.empty((num_timesteps, num_states)),
        pis,
        mus,
        kappas,
        jnp.empty(num_iters),
        xs,
    )
    # Use a Python for loop with tqdm for progress tracking
    for _ in tqdm(range(num_iters), desc="training GIMBAL"):
        carry = em_step(_, carry)
    E_zs, pis, mus, kappas, lls, _ = carry
    return lls, E_zs, pis, mus, kappas


def load_memmap_from_filename(filename):
    # Extract the metadata from the filename
    parts = filename.name.rsplit(".", 4)  # Split the filename into parts
    dtype_str = parts[-3]  # Get the dtype part of the filename
    shape_str = parts[-2]  # Get the shape part of the filename
    shape = tuple(map(int, shape_str.split("x")))  # Convert shape string to a tuple of integers
    # Load the array using numpy memmap
    array = np.memmap(filename, dtype=dtype_str, mode="r", shape=shape)
    return array


def plot_skeleton_distances(skeleton, keypoints, node_order, poses_standard):
    # get distances for each bone
    distance_list = []
    for si, (kpt1, kpt2) in enumerate(skeleton):
        ix1 = np.where(np.array(keypoints)[node_order] == kpt1)[0][0]
        ix2 = np.where(np.array(keypoints)[node_order] == kpt2)[0][0]
        a = poses_standard[:, ix1]
        b = poses_standard[:, ix2]
        distances = np.linalg.norm(a - b, axis=1)
        distance_list.append(np.median(distances))
    fig, axs = plt.subplots(ncols=6, nrows=4, figsize=(16, 4))
    for si, (kpt1, kpt2) in enumerate(np.array(skeleton)[np.argsort(distance_list)]):
        ix1 = np.where(np.array(keypoints)[node_order] == kpt1)[0][0]
        ix2 = np.where(np.array(keypoints)[node_order] == kpt2)[0][0]
        a = poses_standard[:, ix1]
        b = poses_standard[:, ix2]
        distances = np.linalg.norm(a - b, axis=1)
        ax = axs.flatten()[si]
        ax.hist(distances, bins=np.linspace(0, 50))
        ax.set_title(f"{kpt1}-{kpt2}")
        ax.set_yticks([])
    plt.tight_layout()
    return fig
