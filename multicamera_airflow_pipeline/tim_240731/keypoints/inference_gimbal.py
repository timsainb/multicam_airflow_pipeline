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
from scipy.signal import medfilt

jax.config.update("jax_enable_x64", False)
from tensorflow_probability.substrates.jax.distributions import VonMisesFisher as VMF
from gimbal.fit import em_step
from jax import lax, jit
import logging

logger = logging.getLogger(__name__)
from jax.lib import xla_bridge

logger.info(f"Python interpreter binary location: {sys.executable}")
logger.info(f"Python version: {sys.version}")
logger.info(f"JAX version: {jax.__version__}")
logger.info(f"Backend: {xla_bridge.get_backend().platform}")
logger.info(f"JAX devices: {jax.devices()}")


# load skeleton
from multicamera_airflow_pipeline.tim_240731.skeletons.defaults import (
    dataset_info,
    parents_dict,
    keypoint_info,
    keypoints,
    keypoints_order,
    kpt_dict,
)

from .train_gimbal import (
    generate_gimbal_params,
    get_edges,
    build_node_hierarchy,
    generate_initial_positions,
    # vector_to_angle,
    # angle_to_rotation_matrix,
    standardize_poses,
    compute_directions,
    # fit_gimbal_model,
    load_memmap_from_filename,
    generate_initial_positions,
    generate_outlier_probs,
    # em_movMF,
    skeleton,
)

from multicamera_airflow_pipeline.tim_240731.skeletons.defaults import (
    dataset_info,
    parents_dict,
    keypoint_info,
    keypoints,
    keypoints_order,
    kpt_dict,
)

default_kpt_dict = kpt_dict


class GimbalInferencer:
    def __init__(
        self,
        gimbal_output_directory,
        calibration_folder,
        predictions_3d_directory,
        num_iters_inference=250,
        n_initialization_epochs=100,
        num_states=50,
        indices_egocentric=[0, 2],
        max_inlier_var=100,  # the greatest expected variance for a bone
        thresh_bad_keypoints_mads=10,
        constant_inlier_variance=25,
        outlier_thresh_mm=150,  # distance of any keypoint from median
        obs_outlier_variance=1e6,
        pos_dt_variance=1,
        num_leapfrog_steps=5,
        step_size=0.1,
        conf_sigmoid_gain=20,
        conf_sigmoid_center=0.1,
        batch_size=200000,
        distance_from_median_thresh=50,
        kpt_dict=default_kpt_dict,
        recompute_completed=False,
        testing=False,
    ):
        self.gimbal_output_directory = Path(gimbal_output_directory)
        self.predictions_3d_directory = Path(predictions_3d_directory)
        self.calibration_folder = Path(calibration_folder)
        self.num_iters_inference = num_iters_inference
        self.n_initialization_epochs = n_initialization_epochs
        self.num_states = num_states
        self.indices_egocentric = indices_egocentric
        self.max_inlier_var = max_inlier_var
        self.thresh_bad_keypoints_mads = thresh_bad_keypoints_mads
        self.constant_inlier_variance = constant_inlier_variance
        self.outlier_thresh_mm = outlier_thresh_mm
        self.obs_outlier_variance = obs_outlier_variance
        self.pos_dt_variance = pos_dt_variance
        self.num_leapfrog_steps = num_leapfrog_steps
        self.step_size = step_size
        self.conf_sigmoid_gain = conf_sigmoid_gain
        self.conf_sigmoid_center = conf_sigmoid_center
        self.batch_size = batch_size
        self.distance_from_median_thresh = distance_from_median_thresh
        self.kpt_dict = kpt_dict
        self.recompute_completed = recompute_completed
        self.testing = testing

        if self.testing:
            logger.warning("Testing mode is on. Only running on first 10k samples.")

        self.gimbal_params_file = Path(gimbal_output_directory) / "gimbal_params.p"

        if self.gimbal_params_file.exists() == False:
            raise FileNotFoundError(
                f"Could not find gimbal_params_file: {self.gimbal_params_file}"
            )

    def run(self):
        if (self.recompute_completed == False) and self.check_completed():
            logger.info("Gimbal inference already completed. Skipping.")
            return
        self.load_calibration_data()
        self.load_predictions()
        self.load_gimbal_params()

        # get camera matrices in gimbal format
        self.camera_matrices = np.array(
            [
                mcc.get_projection_matrix(extrinsics, intrinsics)
                for extrinsics, intrinsics in zip(self.all_extrinsics, self.all_intrinsics)
            ]
        )

        # get fitting parameters
        self.params = generate_gimbal_params(
            self.camera_matrices,
            self.fitted_params,
            obs_outlier_variance=self.obs_outlier_variance,
            pos_dt_variance=self.pos_dt_variance,
            num_leapfrog_steps=self.num_leapfrog_steps,
            step_size=self.step_size,
            dtype="float32",
        )

        # use all bodyparts
        self.use_bodyparts = list(np.array(keypoints).astype(str))

        # determine node order
        use_bodyparts_idx = np.array([self.use_bodyparts.index(bp) for bp in self.use_bodyparts])
        edges = np.array(get_edges(self.use_bodyparts, skeleton))
        self.node_order, self.parents = build_node_hierarchy(
            self.use_bodyparts, skeleton, "spine_low"
        )
        edges = np.argsort(self.node_order)[edges]
        self.total_samples = len(self.predictions_3D_mmap)
        self.n_keypoints = len(self.use_bodyparts)
        self.n_batches = int(np.ceil(self.total_samples / self.batch_size))

        # initialize output mmaps
        self.initialize_output()

        # save the gimbal keypoints order
        keypoints_order_gimbal = np.array([self.use_bodyparts[i] for i in self.node_order])
        np.save(
            self.gimbal_output_directory / "keypoints_order_gimbal.npy",
            keypoints_order_gimbal,
        )

        self.gimbal_kpt_indices = np.array([self.kpt_dict[i] for i in keypoints_order_gimbal])

        # run inference for each batch
        for batch in tqdm(range(self.n_batches), desc="batch"):
            self.infer_batch(batch)

        # create a log file to indicate completion
        with open(self.gimbal_output_directory / "completed.log", "w") as f:
            f.write("completed")

    def infer_batch(self, batch):

        batch_start = self.batch_size * batch
        batch_end = self.batch_size * (batch + 1)
        batch_failed = False

        # load batch into memory
        confidences_2D = confidences = np.array(self.confidences_2D_mmap[batch_start:batch_end])
        positions_2D = np.array(self.predictions_2D_mmap[batch_start:batch_end])
        # confidences_3D = np.array(self.confidences_3D_mmap[batch_start:batch_end])
        positions_3D = np.array(self.predictions_3D_mmap[batch_start:batch_end])
        reprojection_errors = np.array(self.reprojection_errors_mmap[batch_start:batch_end])
        confidences_2D.shape

        # re-order 2d
        confidences_2D = confidences_2D  # [:, camera_reorder_2d]
        positions_2D = positions_2D  # [:, camera_reorder_2d]

        # remove outliers
        outlier_pts = np.sqrt(
            np.sum(
                (positions_3D - np.expand_dims(np.median(positions_3D, axis=1), 1)) ** 2,
                axis=(2),
            )
        )
        positions_3D[outlier_pts > self.outlier_thresh_mm] = np.nan

        # initialize poses
        poses = generate_initial_positions(positions_3D)
        del positions_3D

        # rearrage to fit node order
        poses = poses[:, self.node_order]
        confidence = confidences[:, :, self.node_order]
        observations = positions_2D[:, :, self.node_order]
        reprojection_errors = reprojection_errors  # [:, camera_reorder_2d]

        # fill confidence of nans to lowest value
        confidence[np.any(np.isnan(positions_2D), axis=-1)] = 1e-10
        # fill in nans in positions 2d
        for cami in range(observations.shape[1]):
            observations[:, cami] = generate_initial_positions(observations[:, cami])

        # standardize poses
        poses_standard = standardize_poses(poses, self.indices_egocentric)
        # get radii and directions of poses
        radii = np.sqrt(((poses - poses[:, self.parents]) ** 2).sum(-1))
        # directions = compute_directions(poses_standard, self.parents)
        # mask out joints that are very far away from their parent joint
        med_radii = np.median(radii, axis=0)
        mad_radii = scipy.stats.median_abs_deviation(radii, axis=0)
        # todo, this could be bidirectional, right now the 'base' will never be affected
        bad_samples_mask = radii > (med_radii + mad_radii * self.thresh_bad_keypoints_mads)
        poses[bad_samples_mask] = np.nan
        poses_standard[bad_samples_mask] = np.nan
        # recompute radii and directions of poses
        radii = np.sqrt(((poses - poses[:, self.parents]) ** 2).sum(-1))
        # directions = compute_directions(poses_standard, self.parents)

        # initialize positions (fill in any nans)
        init_positions = generate_initial_positions(poses)

        # compute a median filter over the data
        median_init_positions = medfilt(init_positions, kernel_size=(11, 1, 1))
        # compute the distance from median for outliers
        distance_from_median = np.sqrt(
            np.sum((median_init_positions - init_positions) ** 2, axis=-1)
        )
        # threshold and re-interpolate
        init_positions[distance_from_median > self.distance_from_median_thresh] = np.nan
        init_positions = generate_initial_positions(init_positions)

        # calculate probabilies that datapoint is noise
        outlier_prob = generate_outlier_probs(
            confidence,
            outlier_prob_bounds=[1e-6, 1 - 1e-6],
            conf_sigmoid_center=self.conf_sigmoid_center,
            conf_sigmoid_gain=self.conf_sigmoid_gain,  # 20
        )

        # remove out of frame predictions
        observations[observations < 0] = 0
        observations[observations > 2000] = 2000

        # initialize gimbal state
        init_positions = jnp.array(init_positions, "float32")
        observations = jnp.array(observations, "float32")
        outlier_prob = jnp.array(outlier_prob, "float32")
        samples = gimbal.mcmc3d_full.initialize(
            jr.PRNGKey(0), self.params, observations, outlier_prob, init_positions
        )
        # print(self.params)

        ## inference over the entire video
        # average positions over some timeframe
        positions_sum = np.zeros_like(init_positions)
        positions_mean = np.zeros_like(init_positions)
        tot = 0
        log_likelihood_history = []
        difference_from_baseline_history = []
        pbar = tqdm(range(self.num_iters_inference), leave=False, desc="inference")
        for itr in pbar:
            random_number = jr.PRNGKey(itr)
            samples = gimbal.mcmc3d_full.step(
                random_number, self.params, observations, outlier_prob, samples
            )
            log_likelihood = samples["log_probability"].item()
            log_likelihood_history.append(log_likelihood)
            difference_from_baseline = np.mean(np.abs(init_positions - samples["positions"]))
            # skip if the gimbal isn't working
            if difference_from_baseline < 1e-9:
                if itr > 5:
                    print("no difference between input and gimbal output")
                    batch_failed = True
                    break
            if np.isnan(log_likelihood):
                if itr > 5:
                    print("Loss is NAN")
                    batch_failed = True
                    break

            difference_from_baseline_history.append(difference_from_baseline)

            # wait until training begins to stabilize before averaging over gimbal samples
            if itr > self.n_initialization_epochs:
                positions_sum += np.array(samples["positions"])
                tot += 1
                positions_mean = positions_sum / tot

                fig, axs = plt.subplots(ncols=2, figsize=(10, 3))
                axs[0].plot(samples["positions"][:1000, 0, 0])
                axs[0].plot(positions_mean[:1000, 0, 0])
                axs[0].plot(init_positions[:1000, 0, 0])

                axs[1].plot(samples["positions"][:100, 0, 0])
                axs[1].plot(positions_mean[:100, 0, 0])
                axs[1].plot(init_positions[:100, 0, 0])
                plt.show()

            pbar.set_description(
                "ll={:.2f}, diff={:.2f}".format(log_likelihood, difference_from_baseline)
            )

        if batch_failed == False:
            # save output to gimbal
            self.gimbal_output[batch_start:batch_end] = positions_mean[
                :, np.argsort(self.gimbal_kpt_indices)
            ]
            self.gimbal_success[batch_start:batch_end] = 1

    def initialize_output(self):

        self.gimbal_output_directory.mkdir(exist_ok=True, parents=True)
        # initialize output
        mmap_dtype = "float32"
        keypoints_3d_shape = (self.total_samples, self.n_keypoints, 3)
        keypoints_3d_dtype = mmap_dtype
        keypoints_3d_shape_str = "x".join(map(str, keypoints_3d_shape))

        gimbal_file = (
            self.gimbal_output_directory
            / f"gimbal.{keypoints_3d_dtype}.{keypoints_3d_shape_str}.mmap"
        )

        self.gimbal_output = np.memmap(
            gimbal_file,
            dtype=keypoints_3d_dtype,
            mode="w+",
            shape=keypoints_3d_shape,
        )

        gimbal_success_shape = (self.total_samples, 1)
        gimbal_success_str = "x".join(map(str, gimbal_success_shape))

        gimbal_success_file = (
            self.gimbal_output_directory / f"gimbal_success.bool.{gimbal_success_str}.mmap"
        )
        self.gimbal_success = np.memmap(
            gimbal_success_file,
            dtype="bool",
            mode="w+",
            shape=gimbal_success_shape,
        )

        # prepopulate batch with unmodified data
        for batch in tqdm(range(self.n_batches), desc="prepopulating output"):
            batch_start = self.batch_size * batch
            batch_end = self.batch_size * (batch + 1)
            self.gimbal_output[batch_start:batch_end] = np.array(
                self.predictions_3D_mmap[batch_start:batch_end]
            )
            self.gimbal_success[batch_start:batch_end] = 0

    def load_calibration_data(self):
        self.all_extrinsics, self.all_intrinsics, self.camera_names = mcc.load_calibration(
            self.calibration_folder,
            load_format="jarvis",
        )
        self.cameras = copy.deepcopy(self.camera_names)

    def load_predictions(self):
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

        if self.testing:
            # subset data to first 10k points
            self.confidences_2D_mmap = self.confidences_2D_mmap[:10000]
            self.predictions_2D_mmap = self.predictions_2D_mmap[:10000]
            self.confidences_3D_mmap = self.confidences_3D_mmap[:10000]
            self.predictions_3D_mmap = self.predictions_3D_mmap[:10000]
            self.reprojection_errors_mmap = self.reprojection_errors_mmap[:10000]

    def load_gimbal_params(self):
        # load params
        self.fitted_params = joblib.load(self.gimbal_params_file)
        if self.constant_inlier_variance is not None:
            self.fitted_params["obs_inlier_variance"] = (
                np.ones_like(self.fitted_params["obs_inlier_variance"])
                * self.constant_inlier_variance
            )

        if self.testing:
            logger.info("Loading test good gimbal params")
            good_params = joblib.load(
                Path(
                    "/n/groups/datta/tim_sainburg/projects/24-04-22-neuropixels-recordings/data/keypoints/mmpose-predictions/M04002/gimbal_params.p"
                )
            )
            logger.info("Loading new test")
            # good_params = joblib.load(
            #    Path(
            #        "/n/groups/datta/tim_sainburg/projects/24-04-22-neuropixels-recordings/notebooks/keypoints/test_gimbal_params.p"
            #    )
            # )
            # self.fitted_params["obs_inlier_variance"] = good_params["obs_inlier_variance"]
            # self.fitted_params["radii"] = good_params["radii"]
            # self.fitted_params["radii_std"] = good_params["radii_std"]

            # self.fitted_params["kappas"] = good_params["kappas"]

            # self.fitted_params["mus"] = good_params["mus"]
            # self.fitted_params["pis"] = good_params["pis"]
            # self.fitted_params = good_params

    def check_completed(self):
        # check if completed.log exists
        return (self.gimbal_output_directory / "completed.log").exists()
