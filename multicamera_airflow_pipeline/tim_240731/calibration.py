import glob, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from vidio.read import OpenCVReader
import multicam_calibration as mcc
import pathlib
from pathlib import Path
import subprocess
from pathlib import PosixPath
import tempfile
import shutil
import pandas as pd
from tqdm.auto import tqdm
import sys
import logging

logger = logging.getLogger(__name__)
logger.info(f"Python interpreter binary location: {sys.executable}")


class Calibrator:
    def __init__(
        self,
        calibration_video_directory,
        calibration_output_directory,
        camera_names=None,
        board_shape=(5, 7),
        square_size=12.5,
        n_frames_to_sample=2000,
        n_jobs=10,
        sampling_method="subsample",
        verbose=True,
        video_output_directory=None,
        fps=None,
        recompute_completed=False,
        cameras_to_use=None,
    ):
        self.calibration_video_directory = Path(calibration_video_directory)
        self.camera_names = camera_names
        self.calibration_output_directory = Path(calibration_output_directory)
        self.sampling_method = sampling_method
        self.fps = fps
        self.verbose = verbose
        self.board_shape = board_shape
        self.square_size = square_size
        self.n_frames_to_sample = n_frames_to_sample
        self.n_jobs = n_jobs
        self.video_output_directory = video_output_directory
        self.recompute_completed = recompute_completed
        # if there are simultaneous recordings with different subsets of cameras
        #  (e.g. to and bottom chronic rigs, specific which cameras we're currently using)
        self.cameras_to_use = cameras_to_use

    def check_if_completed(self):
        # check if calibration has already been completed
        if (self.calibration_output_directory / "gimbal" / "camera_params.h5").exists():
            return True
        else:
            return False

    def find_videos(self):
        # list videos for each camera (if there are multiple)
        try:
            self.video_paths = {
                camera: [
                    list(Path(f"{self.calibration_video_directory}").glob(f"*{camera}.mp4"))[0]
                ]
                for camera in self.camera_names
            }
        except:
            self.video_paths = {
                camera: [
                    list(Path(f"{self.calibration_video_directory}").glob(f"*{camera}.0.mp4"))[0]
                ]
                for camera in self.camera_names
            }
        for camera in self.camera_names:
            additional_videos = list(
                Path(f"{self.calibration_video_directory}").glob(f"{camera}.*.mp4")
            )
            additional_videos = [
                i for i in additional_videos if i.stem.split(".")[-1].startswith("0") == False
            ]
            srt = np.argsort([int(i.stem.split(".")[-1]) for i in additional_videos])
            additional_videos = np.array(additional_videos)[srt]
            self.video_paths[camera] = self.video_paths[camera] + list(additional_videos)
            self.video_paths[camera] = [str(i) for i in self.video_paths[camera]]

    def run(self):

        if self.recompute_completed == False:
            logger.info(f"Checking if calibration is already completed")
            if self.check_if_completed():
                return

        if self.camera_names is None:
            logger.info(f"Retrieving camera names")
            # get camera names (video format should be {camera_name}.{serial}.{frame}.mp4)
            self.camera_names = [
                i.stem.split(".")[-2] for i in list(self.calibration_video_directory.glob("*.mp4"))
            ]
        # logger report camera names
        logger.info(f"Camera names: {self.camera_names}")
        assert len(self.camera_names) > 0, "No camera names found"

        logger.info(f"Finding videos")
        self.find_videos()
        # logger report number of videos for each camera
        for camera in self.camera_names:
            logger.info(f"Found {len(self.video_paths[camera])} videos for {camera}")

        # create a temporary file joining together all of the calibration videos for each camera
        use_temp = False
        if self.video_output_directory is None:
            use_temp = True
            tmpdir = tempfile.mkdtemp()
            self.video_output_directory = tmpdir

        # store new videos in a temporary directory
        logger.info(f"Concatenating videos")
        output_paths = []
        # create a temporary super video
        for camera in self.camera_names:
            output_path = Path(self.video_output_directory) / f"{camera}.mp4"
            concatenate_videos_ffmpeg(
                videos=self.video_paths[camera], save_path=output_path, fps=self.fps
            )
            output_paths.append(output_path)
        self.video_paths = output_paths

        # make sure all videos are the same length
        logger.info(f"Checking video lengths")
        video_lengths = np.array([get_video_len(video_path) for video_path in self.video_paths])
        assert np.all(
            video_lengths == video_lengths[0]
        ), f"Videos are not the same length: {video_lengths}"

        n_frames = int(np.mean(video_lengths))
        n_vids = len(self.video_paths)

        assert n_frames > 0, "No frames found in videos"

        if self.verbose:
            print(f"Found {n_frames} frames in each video")

        if self.sampling_method == "subsample":
            # subsample the number of frames to detect on
            subsampled_frames = np.unique(
                (np.linspace(1, n_frames - 1, self.n_frames_to_sample)).astype(int)
            )
            aligned_frame_ixs = np.stack([subsampled_frames] * n_vids).T
        elif self.sampling_method == "first_n":
            subsampled_frames = np.arange(self.n_frames_to_sample)
            aligned_frame_ixs = np.stack([subsampled_frames] * n_vids).T
        else:
            aligned_frame_ixs = np.stack([np.arange(n_frames)] * n_vids).T

        # detect calibration object in each video
        logger.info(f"Detecting calibration object")
        all_calib_uvs, all_img_sizes = mcc.run_calibration_detection(
            [i.as_posix() for i in self.video_paths],
            mcc.detect_chessboard,
            n_workers=self.n_jobs,
            aligned_frame_ixs=aligned_frame_ixs,
            detection_options=dict(board_shape=self.board_shape, scale_factor=0.5),
            overwrite=True,
        )

        if self.verbose:
            # display a table with the detections shared between camera pairs
            mcc.summarize_detections(all_calib_uvs)

            # plot corner-match scores for each frame
            fig = mcc.plot_chessboard_qc_data(self.video_paths)

        # initial calibration
        logger.info(f"Initial calibration")
        calib_objpoints = mcc.generate_chessboard_objpoints(self.board_shape, self.square_size)

        all_extrinsics, all_intrinsics, calib_poses = mcc.calibrate(
            all_calib_uvs,
            all_img_sizes,
            calib_objpoints,
            root=0,
            n_samples_for_intrinsics=100,
        )

        if self.verbose:
            (
                fig,
                mean_squared_error,
                reprojections,
                transformed_reprojections,
            ) = mcc.plot_residuals(
                all_calib_uvs, all_extrinsics, all_intrinsics, calib_objpoints, calib_poses
            )
            fig.savefig(
                self.calibration_output_directory / "initial_calibration_result.jpg", format="jpg"
            )

        # bundle adjustment
        logger.info(f"Bundle adjustment")
        (
            adj_extrinsics,
            adj_intrinsics,
            adj_calib_poses,
            use_frames,
            result,
        ) = mcc.bundle_adjust(
            all_calib_uvs,
            all_extrinsics,
            all_intrinsics,
            calib_objpoints,
            calib_poses,
            n_frames=2000,
            ftol=1e-4,
        )

        # save
        logger.info(f"Saving calibration")
        self.calibration_output_directory.mkdir(parents=True, exist_ok=True)
        # save for JARVIS
        jarvis_save_path = self.calibration_output_directory / "jarvis" / f"CalibrationParameters/"
        jarvis_save_path.mkdir(parents=True, exist_ok=True)
        mcc.save_calibration(
            adj_extrinsics,
            adj_intrinsics,
            self.camera_names,
            jarvis_save_path,
            save_format="jarvis",
        )
        # save for GIMBAL
        gimbal_save_path = self.calibration_output_directory / "gimbal" / f"camera_params.h5"
        (self.calibration_output_directory / "gimbal").mkdir(parents=True, exist_ok=True)
        mcc.save_calibration(
            all_extrinsics,
            all_intrinsics,
            self.camera_names,
            gimbal_save_path.as_posix(),
            save_format="gimbal",
        )

        if self.verbose:
            (
                fig,
                median_error,
                reprojections,
                transformed_reprojections,
            ) = mcc.plot_residuals(
                all_calib_uvs[:, use_frames],
                adj_extrinsics,
                adj_intrinsics,
                calib_objpoints,
                adj_calib_poses,
            )
            fig.savefig(self.calibration_output_directory / "calibration_result.jpg", format="jpg")
            print(f"Median Error: {median_error}")

        if use_temp:
            shutil.rmtree(tmpdir)


def get_video_len(video_path):
    # Note, this function will not work unless videos are properly muxed
    reader = OpenCVReader(str(video_path))
    n_frames = len(reader)
    reader.close()
    return n_frames


def concatenate_videos_ffmpeg(videos, save_path, fps=None):
    """
    Concatenate a list of videos into a single video using ffmpeg.

    Parameters:
    - videos (list): A list of paths to video files.
    - save_path (str): The path where the concatenated video should be saved.

    Returns:
    - None
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        list_file_path = PosixPath(temp_dir) / "temp_list.txt"

        # Create a list file for ffmpeg inside the temporary directory
        with open(list_file_path, "w") as f:
            for video in videos:
                f.write(f"file '{video}'\n")

        # Use ffmpeg to concatenate
        cmd = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file_path),
        ]
        if fps is not None:
            cmd += [
                "-vf",
                f"fps={fps}",
                "-c:a",
            ]
        else:
            cmd += ["-c"]
        cmd += [
            "copy",
            str(save_path),
        ]
        print(" ".join(cmd))

        subprocess.run(cmd)
