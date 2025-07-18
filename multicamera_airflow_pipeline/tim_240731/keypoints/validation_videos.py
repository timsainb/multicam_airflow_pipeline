import logging
import os
from pathlib import Path
import subprocess
import sys

import cv2
import h5py
import matplotlib.pyplot as plt
import multicam_calibration as mcc
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from tqdm.auto import tqdm

# load skeleton
from multicamera_airflow_pipeline.tim_240731.skeletons.defaults import (
    dataset_info,
)

logging.info("Python interpreter binary location:", sys.executable)
logger = logging.getLogger(__name__)


class KeypointVideoCreator:
    def __init__(
        self,
        predictions_2d_directory,
        predictions_triang_directory,
        camera_calibration_directory,
        raw_video_directory,
        output_directory_keypoint_vids,
        max_frames=2400,
        predictions_gimbal_directory=None,
        recompute_completed=False,
    ):
        """
        Class to take the output of the keypoint pipeline and create videos with keypoints overlaid.

        Parameters:
        -----------
        predictions_2d_directory : str
            Path to the directory containing the 2D keypoint predictions.

        predictions_triang_directory : str
            Path to the directory containing the triangulated 3D keypoint predictions.

        camera_calibration_directory : str
            Path to the directory containing the camera calibration files.

        raw_video_directory: str
            Path to the directory containing the raw video files.

        output_directory_keypoint_vids : str
            Path to the directory where the output keypoint videos will be saved.

        max_frames : int
            Length of the output video in frames.

        recompute_completed : bool
            If True, the keypoint videos will be recreated even if they already exist in the output directory.
        """
        self.predictions_2d_directory = Path(predictions_2d_directory)
        self.predictions_triang_directory = Path(predictions_triang_directory)
        self.camera_calibration_directory = Path(camera_calibration_directory)
        if predictions_gimbal_directory is None:
            self.predictions_gimbal_directory = None
        else:
            self.predictions_gimbal_directory = Path(predictions_gimbal_directory)
        self.raw_video_directory = Path(raw_video_directory)
        self.output_directory_keypoint_vids = Path(output_directory_keypoint_vids)
        self.max_frames = max_frames
        self.recompute_completed = recompute_completed

        # Initialize keypoint and skeleton information from dataset_info
        self.keypoint_info = dataset_info["keypoint_info"]
        self.skeleton_info = dataset_info["skeleton_info"]
        self.keypoint_idx = {value["name"]: key for key, value in self.keypoint_info.items()}

        # Order of keypoints as predicted
        self.keypoints = np.array(
            [self.keypoint_info[i]["name"] for i in self.keypoint_info.keys()]
        )
        self.n_keypoints = len(self.keypoints)

    def check_if_validation_vids_exist(self):
        # checks a log saved in the output directory to see if the validation videos have already been created

        if (self.output_directory_keypoint_vids / "keypoint_videos_completed.log").exists():
            return True
        else:
            return False

    def set_completed(self):
        with open(self.output_directory_keypoint_vids / "keypoint_videos_completed.log", "w") as f:
            f.write("Keypoint vids completed")

    def load_video_filenames(self):
        # grab all the video files
        self.video_files = {}
        for camera in self.cameras:
            self.video_files[camera] = list(self.raw_video_directory.glob(f"*{camera}*.mp4"))
            if len(self.video_files[camera]) == 0:
                raise ValueError(f"No video files found for camera {camera}")
            elif len(self.video_files[camera]) > 1:
                # TODO: Add logic to select the correct video file
                # raise ValueError(f"Multiple video files found for camera {camera}")
                self.video_files[camera] = list(self.raw_video_directory.glob(f"*{camera}.0.mp4"))[
                    0
                ]
            else:
                self.video_files[camera] = self.video_files[camera][0]

    def load_2D_prediction_filenames(self):
        # grab all the predictions 2D h5 files
        predictions_2d_files = list(self.predictions_2d_directory.glob("*.h5"))
        cam = [i.stem.split(".")[1] for i in predictions_2d_files]
        frame = [int(i.stem.split(".")[2]) for i in predictions_2d_files]
        self.recording_predictions = pd.DataFrame(
            {"camera": cam, "frame": frame, "file": predictions_2d_files}
        )
        # Filter out cameras not present in data
        self.recording_predictions = self.recording_predictions[
            self.recording_predictions.camera.isin(self.cameras)
        ]

    def load_triang_prediction_filenames(self):
        predictions_triang_files = list(
            self.predictions_triang_directory.glob("predictions_3d*.mmap")
        )
        confidences_triang_files = list(
            self.predictions_triang_directory.glob("confidences_3d*.mmap")
        )

        assert len(predictions_triang_files) == len(confidences_triang_files) == 1
        self.triang_predictions_file = predictions_triang_files[0]
        self.triang_confidences_file = confidences_triang_files[0]

        if self.predictions_gimbal_directory is None:
            self.gimbal_predictions_file = None
        else:
            predictions_gimbal_files = list(
                self.predictions_gimbal_directory.glob("gimbal.*.mmap")
            )

            if len(predictions_gimbal_files) > 0:
                self.gimbal_predictions_file = predictions_gimbal_files[0]
            else:
                self.gimbal_predictions_file = None

    def load_2D_predictions(self):
        # Load up to max_frames of 2D predictions
        self.predictions_2d = {}
        for camera in self.cameras:
            keypoint_coords = []
            keypoint_conf = []
            detection_coords = []
            frames_loaded = 0
            while frames_loaded < self.max_frames:
                # Get the next file
                next_file = (
                    self.recording_predictions[
                        (self.recording_predictions.camera == camera)
                        & (self.recording_predictions.frame >= frames_loaded)
                    ]
                    .sort_values("frame")
                    .iloc[0]
                )
                # Load the predictions
                with h5py.File(next_file.file, "r") as file:
                    _keypoint_coords = np.array(
                        file["keypoint_coords"]
                    )  # shape: (n_frames, n_keypoints, 2)
                    _keypoint_conf = np.array(file["keypoint_conf"])
                    _detection_coords = np.array(file["detection_coords"])
                _keypoint_conf[_keypoint_conf > 1] = 1
                keypoint_conf.append(_keypoint_conf)
                keypoint_coords.append(_keypoint_coords)
                detection_coords.append(_detection_coords)
                frames_loaded += len(_keypoint_coords)

            self.predictions_2d[camera] = {
                "keypoint_coords": np.concatenate(keypoint_coords, axis=0).squeeze(),
                "keypoint_conf": np.concatenate(keypoint_conf, axis=0).squeeze(),
                "detection_coords": np.concatenate(detection_coords, axis=0).squeeze(),
            }

    def load_triang_reproj_predictions(self):
        # Load max_frames of 2D reprojections
        self.predictions_triang = {}
        keypoint_coords = load_memmap_from_filename(
            self.triang_predictions_file
        )  # shape: (n_frames, n_keypoints, 3)

        keypoint_conf = load_memmap_from_filename(self.triang_confidences_file)
        for iCamera, camera in enumerate(self.cameras):
            these_coords_3D = keypoint_coords[: self.max_frames, :, :]
            these_confs = keypoint_conf[: self.max_frames, :]

            # Reproject the coords into 2D
            extrinsics = self.all_extrinsics[iCamera]
            camera_matrix, dist_coefs = self.all_intrinsics[iCamera]
            these_coords_2D = mcc.project_points(
                these_coords_3D,
                extrinsics=extrinsics,
                camera_matrix=camera_matrix,
                dist_coefs=dist_coefs,
            )
            self.predictions_triang[camera] = {
                "keypoint_coords": these_coords_2D,
                "keypoint_conf": these_confs,
            }
        if self.gimbal_predictions_file is not None:
            self.predictions_gimbal = {}
            gimbal_coords = load_memmap_from_filename(self.gimbal_predictions_file)
            for iCamera, camera in enumerate(self.cameras):
                these_coords_3D = gimbal_coords[: self.max_frames, :, :]
                these_confs = keypoint_conf[: self.max_frames, :]

                # Reproject the coords into 2D
                extrinsics = self.all_extrinsics[iCamera]
                camera_matrix, dist_coefs = self.all_intrinsics[iCamera]
                these_coords_2D = mcc.project_points(
                    these_coords_3D,
                    extrinsics=extrinsics,
                    camera_matrix=camera_matrix,
                    dist_coefs=dist_coefs,
                )
                self.predictions_gimbal[camera] = {
                    "keypoint_coords": these_coords_2D,
                    "keypoint_conf": these_confs,
                }

    def create_2D_keypoint_conf_plots(self):
        # import pdb; pdb.set_trace()
        all_kp_confs = np.stack(
            [self.predictions_2d[cam]["keypoint_conf"] for cam in self.cameras], axis=-1
        )
        max_kp_confs_per_frame = np.max(all_kp_confs, axis=-1)
        plt.figure()
        plt.matshow(max_kp_confs_per_frame.T, aspect="auto", cmap="PiYG", vmin=0, vmax=1)
        cbar = plt.colorbar()
        cbar.set_label("Max keypoint confidence")
        cbar.set_ticks([0, 0.5, 1])
        plt.ylabel("Keypoint")
        plt.xlabel("Frame")
        plt.title(
            f"Max keypoint confidences across cameras\n{self.recording_predictions.iloc[0].file.stem}"
        )
        plt.savefig(self.output_directory_keypoint_vids / "max_keypoint_confs.png")
        plt.close()

    def create_2D_keypoint_videos(self):
        for camera in self.cameras:
            keypoint_coords = self.predictions_2d[camera]["keypoint_coords"]
            keypoint_conf = self.predictions_2d[camera]["keypoint_conf"]
            bbox_coords = self.predictions_2d[camera]["detection_coords"]

            generate_keypoint_video(
                output_directory=self.output_directory_keypoint_vids,
                video_path=self.video_files[camera],
                keypoint_coords=keypoint_coords,
                keypoint_conf=keypoint_conf,
                keypoint_info=dataset_info["keypoint_info"],
                vid_suffix="with_2D_keypoints",
                detection_coords=bbox_coords,
                skeleton_info=dataset_info["skeleton_info"],
                max_frames=self.max_frames,
            )

    def create_reproj_triang_keypoint_videos(self):
        for camera in self.cameras:
            keypoint_coords = self.predictions_triang[camera]["keypoint_coords"]
            keypoint_conf = self.predictions_triang[camera]["keypoint_conf"]

            generate_keypoint_video(
                output_directory=self.output_directory_keypoint_vids,
                video_path=self.video_files[camera],
                keypoint_coords=keypoint_coords,
                keypoint_conf=keypoint_conf,
                keypoint_info=dataset_info["keypoint_info"],
                vid_suffix="with_triang_keypoints",
                skeleton_info=dataset_info["skeleton_info"],
                max_frames=self.max_frames,
            )

            if self.predictions_gimbal is not None:
                keypoint_coords = self.predictions_gimbal[camera]["keypoint_coords"]
                keypoint_conf = self.predictions_gimbal[camera]["keypoint_conf"]

                generate_keypoint_video(
                    output_directory=self.output_directory_keypoint_vids,
                    video_path=self.video_files[camera],
                    keypoint_coords=keypoint_coords,
                    keypoint_conf=keypoint_conf,
                    keypoint_info=dataset_info["keypoint_info"],
                    vid_suffix="with_gimbal_keypoints",
                    skeleton_info=dataset_info["skeleton_info"],
                    max_frames=self.max_frames,
                )

    def crop_and_stitch_2D_keypoint_videos(self):
        bbox_coords_by_camera = {
            camera: self.predictions_2d[camera]["detection_coords"] for camera in self.cameras
        }
        crop_and_stich_vids(
            output_directory=self.output_directory_keypoint_vids,
            single_vid_suffix="with_2D_keypoints",  # Suffix to identify the single videos to be stitched together
            bbox_coords_by_camera=bbox_coords_by_camera,
            bbox_crop_size=(400, 400),
            max_frames=self.max_frames,
        )

    def crop_and_stitch_reproj_triang_keypoint_videos(self):
        # Use the centroid of the triang'd kps as the center of the bbox
        detection_coords_by_camera = {}
        for camera in self.cameras:
            reproj_coords = self.predictions_triang[camera]["keypoint_coords"]
            centroids = np.nanmean(reproj_coords, axis=1)
            centroids = nan_to_preceding(centroids)
            detection_coords_by_camera[camera] = centroids

        crop_and_stich_vids(
            output_directory=self.output_directory_keypoint_vids,
            single_vid_suffix="with_triang_keypoints",  # Suffix to identify the single videos to be stitched together
            detection_coords_by_camera=detection_coords_by_camera,
            bbox_crop_size=(400, 400),
            max_frames=self.max_frames,
        )

        if self.predictions_gimbal is not None:
            # Use the centroid of the gimbal'd kps as the center of the bbox
            detection_coords_by_camera = {}
            for camera in self.cameras:
                reproj_coords = self.predictions_gimbal[camera]["keypoint_coords"]
                centroids = np.nanmean(reproj_coords, axis=1)
                centroids = nan_to_preceding(centroids)
                detection_coords_by_camera[camera] = centroids

            crop_and_stich_vids(
                output_directory=self.output_directory_keypoint_vids,
                single_vid_suffix="with_gimbal_keypoints",  # Suffix to identify the single videos to be stitched together
                detection_coords_by_camera=detection_coords_by_camera,
                bbox_crop_size=(400, 400),
                max_frames=self.max_frames,
            )

    def run(self):
        # Check if already completed
        if not self.recompute_completed:
            if self.check_if_validation_vids_exist():
                logger.info("Triangulation already exists")
                return

        # Create output dir if needed
        if not self.output_directory_keypoint_vids.exists():
            self.output_directory_keypoint_vids.mkdir(parents=True)

        # Load calibration + camera info
        self.all_extrinsics, self.all_intrinsics, camera_names = mcc.load_calibration(
            self.camera_calibration_directory,
            load_format="jarvis",
        )
        self.cameras = camera_names
        self.n_cameras = len(self.cameras)
        logging.info(f"\t n_cameras {self.n_cameras}")

        # Load the video filenames
        self.load_video_filenames()

        # Load the 2D predictions
        self.load_2D_prediction_filenames()
        self.load_2D_predictions()

        logger.info("Creating 2D video")
        # Create the 2D keypoint videos
        self.create_2D_keypoint_conf_plots()
        self.create_2D_keypoint_videos()
        self.crop_and_stitch_2D_keypoint_videos()

        # Load the triangulated 3D predictions
        self.load_triang_prediction_filenames()
        self.load_triang_reproj_predictions()

        logger.info("Creating 3D video")
        # Create the triangulated keypoint videos
        self.create_reproj_triang_keypoint_videos()
        self.crop_and_stitch_reproj_triang_keypoint_videos()

        logger.info("Creating 3D video")
        # Compress all the videos
        # (Runs at ~10 fps --> adds another 7200 frames / 10 fps = 720s = 12 minutes x 6 vids = ~1 hr)

        logging.info("Compressing videos")
        for vid in self.output_directory_keypoint_vids.glob("*.mp4"):
            compressed_vid = vid.with_name(vid.stem + "_compressed.mp4")
            compress_vid_via_ffmpeg(
                vid,
                compressed_vid,
                crf=23,
                preset="fast",  # runs at ~10 fps (which is kinda slow) but compresses ~10x which is nice for speeding up subsequent downloads of the QC vids.
                recompute_completed=self.recompute_completed,
            )

            # Remove the original
            # TODO: fix error os.remove(vid) FileNotFoundError
            os.remove(vid)

            # Rename the compressed file
            compressed_vid.rename(vid)

        # Mark as completed
        logging.info("Marking as completed")
        self.set_completed()

        return


def compress_vid_via_ffmpeg(
    input_vid, output_vid, crf=23, preset="fast", recompute_completed=False
):
    """
    Compresses a video file using ffmpeg.

    Parameters:
    -----------
    input_vid : Path
        Path to the input video file.

    output_vid : Path
        Path to the output video file.

    crf : int
        Constant Rate Factor (CRF) value for the video compression. Lower values result in higher quality videos.

    preset : str
        Preset for the video compression. Options are: 'ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium',
    """

    # Check if the output file already exists
    if output_vid.exists() and not recompute_completed:
        logging.info(f"Output file already exists: {output_vid}")
        return

    # Run the ffmpeg command for video compression
    command = [
        "/n/app/ffmpeg/3.3.3/ffmpeg",
        "-y",
        "-i",
        str(input_vid),
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        str(output_vid),
    ]
    logging.info(f"Compressing video: {input_vid}")
    subprocess.Popen(command).wait()
    return


def generate_keypoint_video(
    output_directory: Path,
    video_path: Path,
    keypoint_coords: np.ndarray,
    keypoint_conf: np.ndarray,  # New parameter for keypoint confidence
    keypoint_info: dict,
    skeleton_info: dict,
    vid_suffix: str,
    detection_coords: np.ndarray = None,
    max_frames=None,
):
    """
    Generates a video with keypoint predictions overlaid on the original video frames.

    Parameters:
    -----------
    output_directory : Path
        Directory where the output video will be saved.

    video_path : Path
        Path to the input video file.

    keypoint_coords : np.ndarray
        Array of shape (#frames, #keypoints, 2) containing the coordinates of keypoints for each frame.

    keypoint_conf : np.ndarray
        Array of shape (#frames, #keypoints) containing the confidence values (0-1) for each keypoint in each frame.

    keypoint_info : dict
        Dictionary containing information about the keypoints. Each key in the dictionary represents a keypoint ID, and the
        value is another dictionary with the following structure:
        {
            'name': str,       # Keypoint name
            'id': int,         # Keypoint ID
            'color': list,     # RGB color for the keypoint [R, G, B]
            'type': str,       # Keypoint type (e.g., 'upper', 'lower')
            'swap': str        # Name of the corresponding left/right keypoint to be swapped (for symmetry)
        }

    skeleton_info : dict
        Dictionary containing information about the skeleton. Each key in the dictionary represents a skeleton link ID, and
        the value is another dictionary with the following structure:
        {
            'link': tuple,     # Tuple containing the names of the two keypoints that form the link
            'id': int,         # Link ID
            'color': list      # RGB color for the link [R, G, B]
        }

    vid_suffix : str
        Suffix to add to the video file name. Ie "with_2D_keypoints" or "with_3D_keypoints"

    Returns:
    --------
    None
        The function saves the output video with keypoints and skeletons overlaid to the specified output directory.

    Raises:
    -------
    ValueError
        If the input video cannot be opened.

    Example:
    --------
    output_directory = Path('/output/directory')
    video_path = Path('/path/to/video.mp4')
    keypoint_coords = np.load('keypoint_coords.npy')  # Load your keypoints array
    keypoint_conf = np.load('keypoint_conf.npy')  # Load your keypoint confidence array
    keypoint_info = {
        0: {'name': 'nose_tip', 'id': 0, 'color': [120, 184, 181], 'type': 'upper', 'swap': ''},
        # Add other keypoints as needed
    }
    skeleton_info = {
        0: {'link': ('tail_base', 'spine_low'), 'id': 0, 'color': [173, 160, 183]},
        # Add other links as needed
    }

    generate_keypoint_video(output_directory, video_path, keypoint_coords, keypoint_conf, keypoint_info, skeleton_info)
    """

    # Open the input video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create the VideoWriter object
    output_path = output_directory / (video_path.stem + "_" + vid_suffix + ".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    frame_idx = 0
    if total_frames < 0 and max_frames is None:
        raise ValueError(
            "Could not determine total number of frames in the video -- please specify max_frames."
        )
    elif total_frames < 0:
        total_frames = max_frames
    elif max_frames is not None:
        total_frames = np.min([max_frames, total_frames])

    logging.info(f"Total frames: {total_frames}")

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Create an overlay for drawing
            overlay = frame.copy()

            # Draw keypoints
            for kp_idx, kp_info in keypoint_info.items():
                if frame_idx < len(keypoint_coords) and kp_idx < keypoint_coords.shape[1]:
                    x, y = keypoint_coords[frame_idx, kp_idx]
                    if np.isnan(x) or np.isnan(y):
                        continue
                    conf = keypoint_conf[frame_idx, kp_idx]
                    color = tuple(kp_info["color"])
                    alpha = conf  # Alpha value is based on the confidence (0-1)
                    if conf > 0:  # Only draw if confidence is greater than 0
                        overlay = cv2.circle(
                            overlay,
                            (int(x), int(y)),
                            radius=4,
                            color=color,
                            thickness=-1,
                        )

            # Apply the overlay with alpha blending for keypoints
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Draw skeleton
            for link_info in skeleton_info.values():
                kp1_name, kp2_name = link_info["link"]
                kp1_id = next(
                    (kp["id"] for kp in keypoint_info.values() if kp["name"] == kp1_name),
                    None,
                )
                kp2_id = next(
                    (kp["id"] for kp in keypoint_info.values() if kp["name"] == kp2_name),
                    None,
                )

                if kp1_id is not None and kp2_id is not None:
                    if (
                        frame_idx < len(keypoint_coords)
                        and kp1_id < keypoint_coords.shape[1]
                        and kp2_id < keypoint_coords.shape[1]
                    ):
                        x1, y1 = keypoint_coords[frame_idx, kp1_id]
                        x2, y2 = keypoint_coords[frame_idx, kp2_id]
                        if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
                            continue
                        kp1_conf = keypoint_conf[frame_idx, kp1_id]
                        kp2_conf = keypoint_conf[frame_idx, kp2_id]
                        color = tuple(link_info["color"])
                        alpha = min(
                            kp1_conf, kp2_conf
                        )  # Alpha value is the minimum confidence of the link
                        if (
                            kp1_conf > 0 and kp2_conf > 0 and not np.isnan(x1)
                        ):  # Only draw if both confidence values are greater than 0
                            overlay = cv2.line(
                                overlay,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                color=color,
                                thickness=2,
                            )

            # Apply the overlay with alpha blending for skeleton
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Find centroid of bounding box
            # x1, y1, x2, y2 = detection_coords[frame_idx, 0, :]
            # centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            # overlay = cv2.circle(
            #     overlay, centroid, radius=4, color=(0, 255, 0), thickness=-1
            # )

            # Draw the detection bounding box on the frame
            # if frame_idx < len(detection_coords):
            #     x1, y1, x2, y2 = detection_coords[frame_idx,0,:]
            #     overlay = cv2.rectangle(
            #         overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
            #     )

            # # Apply the overlay
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Write the frame with keypoints and skeletons to the output video
            out.write(frame)
            frame_idx += 1
            pbar.update(1)
            if max_frames and frame_idx >= max_frames:
                break

    # Release video objects
    cap.release()
    out.release()
    logging.info(f"Video saved to: {output_path}")
    return


def crop_and_stich_vids(
    output_directory: Path,
    single_vid_suffix: str,
    bbox_coords_by_camera: dict[np.ndarray] = None,
    detection_coords_by_camera: dict[np.ndarray] = None,
    bbox_crop_size=(400, 400),
    max_frames=None,
):
    """
    Take keypoint videos and crop the mouse out, and stitch together the cropped videos into one row.

    Parameters:
    -----------
    output_directory : Path
        Directory where the single videos will be found + output video will be saved.

    single_vid_suffix : str
        Suffix to identify the single videos to be stitched together.

    bbox_coords_by_camera : dict or None
        Dictionary containing the bounding box coordinates for each camera. The keys are camera names and the values are
        numpy arrays of shape (#frames, 4) containing the bounding box coordinates (x1, y1, x2, y2) for each frame.
        If None, must provide detection coordinates instead, which wil be treated as centroids.

    detection_coords_by_camera : dict or None
        Dictionary containing the detection coordinates for each camera. The keys are camera names and the values are
        numpy arrays of shape (#frames, 4) containing the detection (ie centroid) coordinates (x, y) for each frame.
        If None, must provide bbox coordinates instead, which will be used to infer a centroid + crop
        (the bboxes from mmpose aren't uniform size, so we infer centroid + crop to standard size).

    """

    assert (
        bbox_coords_by_camera is not None or detection_coords_by_camera is not None
    ), "Must provide either bbox or detection coordinates."
    assert (
        bbox_coords_by_camera is None or detection_coords_by_camera is None
    ), "Must provide either bbox or detection coordinates, not both."

    out_vids = list(output_directory.glob(f"*{single_vid_suffix}.mp4"))
    timestamp, cam, vid_suffix = out_vids[0].stem.split(".")
    stitched_vid_name = ".".join([timestamp, "stitched", vid_suffix, ".mp4"])

    # Get the total number of frames to use
    tmp_cap = cv2.VideoCapture(str(out_vids[0]))
    total_frames = int(tmp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 0 and max_frames is None:
        raise ValueError(
            "Could not determine total number of frames in the video -- please specify max_frames."
        )
    elif total_frames < 0:
        total_frames = max_frames
    elif max_frames is not None:
        total_frames = np.min([max_frames, total_frames])
    tmp_cap.release()

    # Calculate bbox centroids for the cropping
    bbox_centroids_by_camera = {}
    if bbox_coords_by_camera is not None:
        for vid in out_vids:
            recording_id, camera, frame, ext = os.path.basename(vid).split(".")
            detn_coords = bbox_coords_by_camera[camera]
            bbox_centroids_by_camera[camera] = np.array(
                [[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in detn_coords]
            )
            # Apply median filter smoothing to reduce jitter
            bbox_centroids_by_camera[camera] = median_filter(
                bbox_centroids_by_camera[camera], size=(12, 1)
            )
    elif detection_coords_by_camera is not None:
        for vid in out_vids:
            recording_id, camera, frame, ext = os.path.basename(vid).split(".")
            bbox_centroids_by_camera[camera] = median_filter(
                detection_coords_by_camera[camera], size=(12, 1)
            )

    # Open the output video
    out_vid_path = output_directory / Path(stitched_vid_name)
    logging.info(f"Output video path: {out_vid_path}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_frame_size = (bbox_crop_size[0] * len(out_vids), bbox_crop_size[1])
    out = cv2.VideoWriter(str(out_vid_path), fourcc, 30, output_frame_size)

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        # Open the input videos
        cap_by_camera = {}
        for vid in out_vids:
            cap = cv2.VideoCapture(str(vid))
            cap_by_camera[os.path.basename(vid).split(".")[1]] = cap

        frame_idx = 0
        while True:
            frames = []
            for camera, cap in cap_by_camera.items():
                # Read the frame
                ret, frame = cap.read()
                if not ret:
                    break

                # Crop the frame
                x, y = bbox_centroids_by_camera[camera][frame_idx]
                x1, y1 = x - bbox_crop_size[0] // 2, y - bbox_crop_size[1] // 2
                x2, y2 = x + bbox_crop_size[0] // 2, y + bbox_crop_size[1] // 2
                frame = frame[int(y1) : int(y2), int(x1) : int(x2)]
                frames.append(frame)

            if not ret:
                break

            # Stitch the frames together
            stitched_frame = np.zeros(
                (bbox_crop_size[1], bbox_crop_size[0] * len(out_vids), 3),
                dtype=np.uint8,
            )
            for i, frame in enumerate(frames):
                stitched_frame[
                    0 : frame.shape[0], i * frame.shape[1] : (i + 1) * frame.shape[1]
                ] = frame

            # Write the stitched frame to the output video
            out.write(stitched_frame)

            # Loop control
            frame_idx += 1
            pbar.update(1)
            if max_frames and frame_idx >= max_frames:
                break


def load_memmap_from_filename(filename):
    # Extract the metadata from the filename
    parts = filename.name.rsplit(".", 4)  # Split the filename into parts
    dtype_str = parts[-3]  # Get the dtype part of the filename
    shape_str = parts[-2]  # Get the shape part of the filename
    shape = tuple(map(int, shape_str.split("x")))  # Convert shape string to a tuple of integers
    # Load the array using numpy memmap
    array = np.memmap(filename, dtype=dtype_str, mode="r", shape=shape)
    return array


def nan_to_preceding(arr):
    # Make a copy of the array to avoid modifying the original array
    result = arr.copy()

    # Iterate over each element in the first dimension
    for i in range(1, arr.shape[0]):
        mask = np.isnan(result[i])  # Identify the NaN values
        result[i][mask] = result[i - 1][mask]  # Replace NaNs with preceding values

    return result
