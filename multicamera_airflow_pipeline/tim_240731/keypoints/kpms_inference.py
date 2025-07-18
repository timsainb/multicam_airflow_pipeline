import jax
from jax.lib import xla_bridge
import jaxlib
import chex
import keypoint_moseq as kpms

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from tqdm.auto import tqdm
import h5py
import multicam_calibration as mcc
import sys
import logging
import tempfile
import shutil
import os

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

logger.info(f"Python interpreter binary location: {sys.executable}")


from scipy.signal import savgol_filter


class KPMSInferencer:
    def __init__(
        self,
        egocentric_alignment_file,
        kpms_output_directory,
        recompute_completed,
        kpms_project_directory,
        kpms_model_name,
        kpms_model_iteration,
        samplerate_hz_recording,
        samplerate_hz_kpms,
        batch_size=300000,
        savgol_window_length=25,
        savgol_polyorder=10,
        **kwargs,
    ):
        self.egocentric_alignment_file = Path(egocentric_alignment_file)
        self.batch_size = batch_size
        self.recompute_completed = recompute_completed
        self.savgol_window_length = savgol_window_length
        self.savgol_polyorder = savgol_polyorder
        self.kpms_project_directory = Path(kpms_project_directory)
        self.kpms_output_directory = Path(kpms_output_directory)
        self.kpms_model_name = kpms_model_name
        self.kpms_model_iteration = kpms_model_iteration

        # TODO: resample to correct framerate if not matching the expected framerate
        if samplerate_hz_recording != samplerate_hz_kpms:
            raise NotImplementedError(
                f"Resampling not implemented yet, kpms trained at {samplerate_hz_kpms}Hz"
            )

    def run(self):
        # load data
        self.egocentric_coordinates = load_memmap_from_filename(self.egocentric_alignment_file)
        self.n_frames = len(self.egocentric_coordinates)

        # load config
        self.config = lambda: kpms.load_config(str(self.kpms_project_directory))
        self.latent_dim = self.config()["ar_hypparams"]["latent_dim"]

        # load model
        self.model, _, _, _ = kpms.load_checkpoint(
            str(self.kpms_project_directory),
            self.kpms_model_name,
            iteration=self.kpms_model_iteration,
        )

        # create a temporary file to save the predictions
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Convert the temporary directory path to a Path object
            tmpdir_path = Path(tmpdirname)

            # prepare the output files
            (
                tmp_syllable_file,
                tmp_latent_state_file,
                # tmp_centroid_file,
                # tmp_heading_file,
            ) = self.create_output_files(tmpdir_path)

            # NOTE: chunks are based on the length of videos, so if a video is e.g. 120 minutes long
            #  this can be a large memory requirement
            logger.info("Running kpms over chunks")
            chunks = np.arange(0, self.n_frames, self.batch_size)
            for start_frame in tqdm(chunks, desc="chunk"):
                self.infer_batch(start_frame)

            # Check if the destination file exists and remove it if it does
            def move_and_overwrite(src, dst_dir):
                dst = os.path.join(dst_dir, os.path.basename(src))
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)

            # move the final temp_coordinates_file to the output directory
            self.kpms_output_directory.mkdir(parents=True, exist_ok=True)
            move_and_overwrite(tmp_syllable_file, self.kpms_output_directory)
            move_and_overwrite(tmp_latent_state_file, self.kpms_output_directory)

        # write that we are completed
        self.kpms_completed()

    def infer_batch(self, start_frame):

        # Load the chunk of egocentric coordinates
        # end_frame = np.min([self.n_frames, start_frame + self.expected_frames_per_video])
        chunk_start = int(start_frame)
        chunk_end = int(start_frame + self.batch_size)
        egocentric_coordinates_chunk = self.egocentric_coordinates[chunk_start:chunk_end, :, :]

        # Apply Savitzky-Golay filter to smooth the data
        egocentric_coordinates_savgol = savgol_filter_3d_fast(
            egocentric_coordinates_chunk,
            window_length=self.savgol_window_length,
            polyorder=self.savgol_polyorder,
        )

        # format for KPMS
        coordinates = {
            "chunk": egocentric_coordinates_savgol,
        }
        confidences = {"chunk": np.ones(egocentric_coordinates_savgol.shape[:-1])}

        # format for kpms
        new_data, new_metadata = kpms.format_data(coordinates, confidences, **self.config())

        # apply model
        results = kpms.apply_model(
            self.model,
            new_data,
            new_metadata,
            str(self.kpms_project_directory),
            self.kpms_model_name,
            num_iters=self.kpms_model_iteration,
            ar_only=True,
            save_results=False,
            # parallel_message_passing=True,
            **self.config(),
        )

        # extract results
        chunk_syllable_ids = results["chunk"]["syllable"]
        chunk_latent_state = results["chunk"]["latent_state"]
        # chunk_centroid = results["chunk"]["centroid"]
        # chunk_heading = results["chunk"]["heading"]

        # update memmaps
        self.syllables[chunk_start:chunk_end, 0] = chunk_syllable_ids
        self.latent[chunk_start:chunk_end, :] = chunk_latent_state

    def create_output_files(self, tmpdir_path):
        logger.info("Creating output files")
        # set the shape and dtype of arrays
        syllable_shape = (self.n_frames, 1)
        syllable_dtype = "int32"
        syllable_shape_str = "x".join(map(str, syllable_shape))

        latent_shape = (self.n_frames, self.latent_dim)
        latent_dtype = "float32"
        latent_shape_str = "x".join(map(str, latent_shape))

        tmp_syllable_file = tmpdir_path / f"syllables.{syllable_dtype}.{syllable_shape_str}.mmap"
        tmp_latent_state_file = tmpdir_path / f"latent.{latent_dtype}.{latent_shape_str}.mmap"

        # create memmap to write to
        self.syllables = np.memmap(
            tmp_syllable_file,
            dtype=np.int32,
            mode="w+",
            shape=syllable_shape,
        )

        self.latent = np.memmap(
            tmp_latent_state_file,
            dtype=np.float32,
            mode="w+",
            shape=latent_shape,
        )

        return tmp_syllable_file, tmp_latent_state_file

    def kpms_completed(self):
        self.kpms_output_directory.mkdir(parents=True, exist_ok=True)
        with open(self.kpms_output_directory / "kpms_completed.log", "w") as f:
            f.write("KPMS completed")


def savgol_filter_3d_fast(X, window_length=50, polyorder=3):
    """
    Smoothes 3D time series data for multiple keypoints using a Savitzky-Golay filter.

    Parameters:
    X (numpy.ndarray): 3D array representing the time series data with shape (# time samples, # keypoints, 3).
    window_length (int): The length of the filter window (i.e., the number of coefficients). Must be a positive odd integer.
    polyorder (int): The order of the polynomial used to fit the samples. Must be less than `window_length`.

    Returns:
    numpy.ndarray: Smoothed 3D time series data with the same shape as input.
    """
    # Ensure correct input dimensions
    n_samples, n_keypoints, n_coords = X.shape

    # Reshape data to apply filter across all keypoints and coordinates simultaneously
    X_reshaped = X.reshape(n_samples, n_keypoints * n_coords)

    # Apply the Savitzky-Golay filter to each time point across all dimensions in one go
    smoothed_X_reshaped = savgol_filter(X_reshaped, window_length, polyorder, axis=0)

    # Reshape back to the original shape
    smoothed_X = smoothed_X_reshaped.reshape(n_samples, n_keypoints, n_coords)

    return smoothed_X


def load_memmap_from_filename(filename):
    # Extract the metadata from the filename
    parts = filename.name.rsplit(".", 4)  # Split the filename into parts
    dtype_str = parts[-3]  # Get the dtype part of the filename
    shape_str = parts[-2]  # Get the shape part of the filename
    shape = tuple(map(int, shape_str.split("x")))  # Convert shape string to a tuple of integers
    # Load the array using numpy memmap
    array = np.memmap(filename, dtype=dtype_str, mode="r", shape=shape)
    return array
