import numpy as np
import sys
from pathlib import Path
import os
import h5py
from tqdm.auto import tqdm
import logging
import shutil
import tempfile
import cv2
from datetime import datetime
import torch
from datetime import datetime, timedelta
import signal
import multiprocessing
import traceback
import textwrap
import subprocess

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.info(f"Python interpreter binary location: {sys.executable}")


class Inferencer2D:
    def __init__(
        self,
        recording_directory,  # folder containing videos
        output_directory_predictions,  # where to save video predictions
        pose_estimator_config,
        pose_estimator_checkpoint,
        detector_config,
        detector_checkpoint,
        tensorrt_model_directory="",
        tensorrt_rtmdetection_model_name="",
        tensorrt_rtmpose_model_name="",
        n_keypoints=25,
        n_animals=1,
        detection_interval=1,
        expected_video_length_frames=None,
        use_motpy=True,
        n_motpy_tracks=3,
        use_tensorrt=False,
        ignore_log_files=False,
        recompute_completed=False,
    ):

        self.n_keypoints = n_keypoints
        self.n_animals = n_animals
        self.recording_directory = Path(recording_directory)
        self.output_directory_predictions = Path(output_directory_predictions)
        self.output_directory_predictions.mkdir(parents=True, exist_ok=True)
        self.detection_interval = detection_interval
        self.expected_video_length_frames = expected_video_length_frames
        self.use_motpy = use_motpy
        self.n_motpy_tracks = n_motpy_tracks
        self.pose_estimator_config = Path(pose_estimator_config)
        self.pose_estimator_checkpoint = Path(pose_estimator_checkpoint)
        self.detector_config = Path(detector_config)
        self.detector_checkpoint = Path(detector_checkpoint)
        self.use_tensorrt = use_tensorrt
        self.tensorrt_model_directory = Path(tensorrt_model_directory)
        self.tensorrt_rtmdetection_model_name = tensorrt_rtmdetection_model_name
        self.tensorrt_rtmpose_model_name = tensorrt_rtmpose_model_name
        self.recompute_completed = recompute_completed
        self.ignore_log_files = ignore_log_files

        # get the device
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            self.device_name = torch.cuda.get_device_name(0).replace(" ", "_")
            logger.info(f"Using CUDA device: {self.device_name}")
        else:
            raise ValueError("CUDA is not available on this machine.")

        logger.info(f"Using Tensorrt: {self.use_tensorrt}")
        # tensorrt uses different models
        if self.use_tensorrt:
            logger.info(f"Using tensorrt, loading models")
            self.tensorrt_detection_model_path = (
                self.tensorrt_model_directory
                / self.tensorrt_rtmdetection_model_name
                / self.device_name
            )
            self.tensorrt_pose_estimator_path = (
                self.tensorrt_model_directory / self.tensorrt_rtmpose_model_name / self.device_name
            )

            assert (self.tensorrt_detection_model_path / "output_tensorrt.jpg").exists()
            assert (self.tensorrt_pose_estimator_path / "output_tensorrt.jpg").exists()
        else:
            self.tensorrt_detection_model_path = Path("")
            self.tensorrt_pose_estimator_path = Path("")

        # self.detector = Detector(detection_model_path)
        # self.pose_estimator = PoseDetector(pose_estimator_path)
        logger.info(f"Init completed")

    def check_completed(self):
        return (self.output_directory_predictions / "completed.log").exists()

    def run(self):

        logger.info(f"Running 2D inference over videos")

        if self.recompute_completed == False:
            if self.check_completed():
                logger.info(f"Video processing completed, quitting")
                return
            else:
                logger.info(f"Video processing incomplete, running")

        self.all_videos = list(self.recording_directory.glob("*.mp4"))
        self.all_videos = [v for v in self.all_videos if "azure" not in v.stem]

        logger.info(f"Processing {len(self.all_videos)} videos")
        assert len(self.all_videos) > 0, f"No videos found in {self.recording_directory}"

        for video_path in tqdm(self.all_videos, desc="processing videos"):

            output_h5_file = self.output_directory_predictions / f"{video_path.stem}.h5"
            if output_h5_file.exists() and not self.recompute_completed:
                # logger.info(f"Completed, skipping {video_path}")
                continue

            # if a log file exists, then this video is being processed by another job
            #   read the current time from the log file, and if it has been more than 1 day, then reprocess
            #   this accounts for the case where a job is killed and the log file is not deleted.
            log_file = self.output_directory_predictions / f"{video_path.stem}.log"
            if self.ignore_log_files == False:
                if (self.recompute_completed == False) and (log_file.exists()):
                    with open(log_file, "r") as f:
                        log_time = datetime.strptime(f.readline().strip(), "%Y-%m-%d %H:%M:%S.%f")
                    if datetime.now() - log_time < timedelta(hours=1):
                        logger.info(f"Log file exists from {log_time}, skipping {log_file}")
                        continue
            # create a log with the name of this video, to indicate that it is being processed
            #   the log will have a timestamp
            with open(log_file, "w") as f:
                f.write(f"{datetime.now()}\n")

            # initially save output to temp file
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_h5_file:
                temp_h5_path = temp_h5_file.name

                # update the signal handler for the current video log file
                update_log_file_signal_handler(log_file)

                if self.use_tensorrt == False:
                    try:
                        logger.info(f"Processing {video_path}")
                        # do prediction
                        predict_video(
                            video_path=video_path,
                            output_h5_file=temp_h5_path,
                            tensorrt_detection_model_path=self.tensorrt_detection_model_path,
                            tensorrt_pose_estimator_path=self.tensorrt_pose_estimator_path,
                            pose_estimator_config=self.pose_estimator_config,
                            pose_estimator_checkpoint=self.pose_estimator_checkpoint,
                            detector_config=self.detector_config,
                            detector_checkpoint=self.detector_checkpoint,
                            n_keypoints=self.n_keypoints,
                            detection_interval=self.detection_interval,
                            n_animals=self.n_animals,
                            use_motpy=self.use_motpy,
                            n_motpy_tracks=self.n_motpy_tracks,
                            use_tensorrt=self.use_tensorrt,
                            total_frames=self.expected_video_length_frames,
                        )
                        # when completed, move to output file
                        shutil.copy(temp_h5_path, output_h5_file)
                        if os.path.exists(temp_h5_path):
                            os.remove(temp_h5_path)
                        # remove the log file
                        if os.path.exists(log_file):
                            os.remove(log_file)
                    except Exception as e:
                        # Catch any exception and clean up
                        print(f"Exception caught: {e}", file=sys.stderr)
                        if os.path.exists(log_file):
                            os.remove(log_file)
                        if os.path.exists(temp_h5_path):
                            os.remove(temp_h5_path)
                        raise e
                else:
                    # use subprocess to run the prediction, so that if we get a core dump from tensorrt, we can move on
                    #    to the next video without crashing the entire process
                    python_script = textwrap.dedent(
                        f"""
                    import sys
                    # print python executable
                    print(sys.executable)
                    from multicamera_airflow_pipeline.jonah_241112.keypoints.predict_2D import predict_video
                    predict_video(
                        video_path="{video_path.as_posix()}",
                        output_h5_file="{temp_h5_path}",
                        tensorrt_detection_model_path="{self.tensorrt_detection_model_path.as_posix()}",
                        tensorrt_pose_estimator_path="{self.tensorrt_pose_estimator_path.as_posix()}",
                        pose_estimator_config="{self.pose_estimator_config.as_posix()}",
                        pose_estimator_checkpoint="{self.pose_estimator_checkpoint.as_posix()}",
                        detector_config="{self.detector_config.as_posix()}",
                        detector_checkpoint="{self.detector_checkpoint.as_posix()}",
                        n_keypoints={self.n_keypoints},
                        detection_interval={self.detection_interval},
                        n_animals={self.n_animals},
                        use_motpy={self.use_motpy},
                        n_motpy_tracks={self.n_motpy_tracks},
                        use_tensorrt={self.use_tensorrt},
                        total_frames={self.expected_video_length_frames},
                    )"""
                    )
                    # print(sys.executable)
                    # Run the process and capture the output
                    command = f"module load cuda/11.7 && {sys.executable} -c '{python_script}'"
                    process = subprocess.Popen(
                        # ["python", "-c", python_script],
                        # [sys.executable, "-c", python_script],
                        command,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )

                    stdout, stderr = process.communicate()

                    # Check the exit code and handle errors
                    if process.returncode != 0:
                        error_log_file = (
                            self.output_directory_predictions / f"{video_path.stem}.error.log"
                        )
                        logger.info(
                            f"Video failed {video_path}, writing to log file {error_log_file}"
                        )
                        with open(error_log_file, "w") as f:
                            # Write both stdout and stderr to the file
                            f.write("Standard Output:\n")
                            f.write(stdout)
                            f.write("\n\nStandard Error:\n")
                            f.write(stderr)
                    else:
                        # when completed, move to output file
                        shutil.copy(temp_h5_path, output_h5_file)
                    if os.path.exists(temp_h5_path):
                        os.remove(temp_h5_path)
                    # remove the log file
                    if os.path.exists(log_file):
                        os.remove(log_file)

        logger.info(f"Completed processing {len(self.all_videos)} videos")

        # if the number of videos matches the number of h5 files, then we are done
        #  we do this in case there is a second job (e.g. local vs remote) working on the same data
        n_h5_predictions = len(list(self.output_directory_predictions.glob("*.h5")))
        logger.info(f"Completed {n_h5_predictions} out of {len(self.all_videos)} videos")
        if n_h5_predictions == len(self.all_videos):
            # save a file completed.log in self.output_directory_predictions to indicate completion
            with open(self.output_directory_predictions / "completed.log", "w") as f:
                f.write("completed")
            logger.info(
                f"Completed {n_h5_predictions} out of {len(self.all_videos)} videos, written completed.log"
            )
        else:
            logger.info("Not all videos completed, not writing completed.log")


def predict_video(
    video_path,
    output_h5_file,
    # detector,
    # pose_estimator,
    tensorrt_detection_model_path,
    tensorrt_pose_estimator_path,
    pose_estimator_config,
    pose_estimator_checkpoint,
    detector_config,
    detector_checkpoint,
    n_keypoints,
    detection_interval=5,
    n_animals=1,
    total_frames=None,
    use_motpy=True,
    n_motpy_tracks=3,
    use_tensorrt=False,
    copy_video_locally=False,
):
    from motpy import Detection, MultiObjectTracker
    from mmpose.apis import inference_topdown
    from mmdet.apis import inference_detector
    import torch

    video_path = Path(video_path)
    if not video_path.exists():
        raise ValueError(f"Video file {video_path} does not exist")

    if use_tensorrt:
        detector, pose_estimator = load_models_tensorrt(
            tensorrt_detection_model_path, tensorrt_pose_estimator_path
        )
    else:
        detector, pose_estimator = load_models(
            pose_estimator_config, pose_estimator_checkpoint, detector_config, detector_checkpoint
        )

    # TODO copy the video over locally to tmp (using tempfile) to avoid issues with remote file systems
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        if copy_video_locally:
            logger.info("Copying video to local file system")
            shutil.copyfile(video_path, temp_file.name)
            local_video_path = temp_file.name
        else:
            local_video_path = video_path.as_posix()

        # get video
        video = cv2.VideoCapture(local_video_path)
        if total_frames is None:
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 0:
            logger.info(f"Manually counting frames for {video_path}")
            # count frames manually
            total_frames = 0
            with tqdm(desc="counting frames") as pbar:
                while True:
                    ret, frame = video.read()
                    if not ret:
                        break
                    total_frames += 1
                    pbar.update(1)
            logger.info(f"Total frames: {total_frames}")
            # release and reallocate video
            video.release()
            video = cv2.VideoCapture(local_video_path)

        total_frames = int(total_frames)
        n_animals = int(n_animals)
        # allocate arrays
        detection_coords = np.zeros((total_frames, n_animals, 4))
        detection_conf = np.zeros((total_frames, n_animals, 1))
        keypoint_coords = np.zeros((total_frames, n_animals, n_keypoints, 2))
        keypoint_conf = np.zeros((total_frames, n_animals, n_keypoints))

        # initialize motpy tracking
        if use_motpy:
            tracker = MultiObjectTracker(dt=0.1)
            animal_ids = {str(i): i for i in range(n_animals)}
            bboxes_motpy = np.zeros((n_animals, 4))
            confs_motpy = np.zeros((n_animals, 1))
            detection_changes = np.zeros((total_frames, n_animals, 1))
            missing_detections = np.zeros((total_frames, n_animals, 1))

            assert n_motpy_tracks >= n_animals

        n_frames = 0
        for frame_id in tqdm(range(total_frames), leave=False, desc="frames", miniters=1000):
            success, frame = video.read()
            if not success:
                break
            if frame_id % detection_interval == 0:
                if use_tensorrt:
                    bboxes, _, _ = detector(frame)
                    conf = bboxes[:n_motpy_tracks, 4]
                    bboxes = bboxes[:n_motpy_tracks, :4]
                else:
                    result = inference_detector(detector, frame)
                    bboxes = result.pred_instances["bboxes"][:n_motpy_tracks].cpu().numpy()
                    conf = result.pred_instances["scores"][:n_motpy_tracks].cpu().numpy()

            if use_motpy:
                # attach current tracked frame to previous frames
                tracker.step(
                    detections=[
                        Detection(box=bboxes[i], score=conf[i]) for i in range(n_motpy_tracks)
                    ]
                )
                tracks = tracker.active_tracks()
                track_ids = {i.id: j for j, i in enumerate(tracks)}
                track_ids_idx = list(track_ids.keys())
                track_score_rank = np.argsort([i.score for i in tracks])[::-1]

                # get tracking point for each individual animal
                for animal_id in list(animal_ids.keys()):
                    # if the id is still being tracked, keep it
                    if animal_id in track_ids:
                        track = tracks[track_ids[animal_id]]
                    else:
                        tracking_missed = True
                        for rank in track_score_rank:
                            if track_ids_idx[rank] not in animal_ids:
                                track = tracks[rank]

                                # add the new id
                                animal_ids[track.id] = animal_ids[animal_id]
                                detection_changes[frame_id, animal_ids[animal_id]] = 1
                                # remove the previous id
                                del animal_ids[animal_id]
                                animal_id = track.id
                                tracking_missed = False
                                break

                    if not tracking_missed:
                        bboxes_motpy[animal_ids[animal_id]] = track.box
                        confs_motpy[animal_ids[animal_id]] = track.score
                    else:
                        missing_detections[frame_id, animal_ids[animal_id]] = 1

                detection_coords[frame_id] = bboxes_motpy
                detection_conf[frame_id] = confs_motpy
                if use_tensorrt:
                    poses = pose_estimator(frame, bboxes_motpy.astype(int))
                    keypoint_coords[frame_id] = poses[:n_animals, :, :2]
                    keypoint_conf[frame_id] = poses[:n_animals, :, 2]
                else:
                    predictions = inference_topdown(pose_estimator, img=frame, bboxes=bboxes_motpy)
                    for i in range(n_animals):
                        keypoint_coords[frame_id, i, :, :2] = predictions[
                            i
                        ].pred_instances.keypoints[0]
                        keypoint_conf[frame_id, i] = predictions[i].pred_instances.keypoint_scores[
                            0
                        ]

            else:
                # poses = pose_detector(frame, bboxes[:n_animals, :4].astype(int))
                predictions = inference_topdown(pose_estimator, img=frame, bboxes=bboxes)
                detection_coords[frame_id] = bboxes[:n_animals]
                detection_conf[frame_id] = conf[:n_animals]
                keypoint_coords[frame_id, i, :, :2] = predictions[i].pred_instances.keypoints[0]
                keypoint_conf[frame_id, i] = predictions[i].pred_instances.keypoint_scores[0]
            n_frames += 1

        # stop reading the video
        video.release()

    # generate the output h5 file
    with h5py.File(output_h5_file, "w") as h5f:
        h5f["detection_coords"] = detection_coords[:n_frames]
        h5f["detection_conf"] = detection_conf[:n_frames]
        h5f["keypoint_coords"] = keypoint_coords[:n_frames]
        h5f["keypoint_conf"] = keypoint_conf[:n_frames]
        if use_motpy:
            h5f["detection_changes"] = detection_changes[: (frame_id + 1)]
            h5f["missing_detections"] = missing_detections[: (frame_id + 1)]


def update_log_file_signal_handler(log_file):
    """Utility to update the signal handler for the current log file.

    This function exists in case the prediction is interrupted by e.g. tensorrt failing.
    """

    def handle_termination_signal(signum, frame):
        print("Termination signal caught, deleting log file.", file=sys.stderr)
        if os.path.exists(log_file):
            os.remove(log_file)
        logger.info("Termination signal caught, deleting log file.")
        sys.exit(1)

    # Register the signal handler for the current video log file
    signal.signal(signal.SIGABRT, handle_termination_signal)
    signal.signal(signal.SIGTERM, handle_termination_signal)
    signal.signal(signal.SIGINT, handle_termination_signal)


def load_models(
    pose_estimator_config, pose_estimator_checkpoint, detector_config, detector_checkpoint
):
    from mmdet.apis import init_detector
    from mmpose.utils import adapt_mmdet_pipeline
    from mmpose.apis import init_model as init_pose_estimator

    pose_estimator_config = Path(pose_estimator_config)
    pose_estimator_checkpoint = Path(pose_estimator_checkpoint)
    detector_config = Path(detector_config)
    detector_checkpoint = Path(detector_checkpoint)

    pose_estimator = init_pose_estimator(
        pose_estimator_config.as_posix(),
        pose_estimator_checkpoint.as_posix(),
        device="cuda",
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))),
    )
    detector = init_detector(
        detector_config.as_posix(), detector_checkpoint.as_posix(), device="cuda"
    )
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    logger.info(f"models loaded (not tensorrt)")
    return detector, pose_estimator


def load_models_tensorrt(tensorrt_detection_model_path, tensorrt_pose_estimator_path):
    import tensorrt
    from mmdeploy_runtime import Detector, PoseDetector

    tensorrt_detection_model_path = Path(tensorrt_detection_model_path)
    tensorrt_pose_estimator_path = Path(tensorrt_pose_estimator_path)
    # create object detector
    detector = Detector(model_path=tensorrt_detection_model_path.as_posix(), device_name="cuda")
    pose_estimator = PoseDetector(
        model_path=tensorrt_pose_estimator_path.as_posix(), device_name="cuda"
    )
    logger.info(f"tensorrt models loaded")
    return detector, pose_estimator
