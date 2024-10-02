import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
import yaml
import logging
import sys
import cv2

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.info(f"Python interpreter binary location: {sys.executable}")


class CameraSynchronizer:
    def __init__(
        self,
        recording_directory,
        output_directory,
        samplerate,
        trigger_pin=2,
        recompute_completed=False,
    ):
        """
        Parameters
        ----------
        recording_directory : Path
            Path to the directory containing the recording data.
        output_location : Path
            Path to the directory where the synchronized data will be saved.
        trigger_pin : int
            GPIO pin number used to trigger the recording.
        """
        self.recording_directory = Path(recording_directory)
        self.output_directory = Path(output_directory)
        self.trigger_pin = trigger_pin
        self.samplerate = samplerate
        # get the expected interval between frames (in microseconds)
        self.isi_uS = 1 / self.samplerate * 1000 * 1000
        # get recording start time
        self.recording_start = datetime.strptime(
            self.recording_directory.name, "%y-%m-%d-%H-%M-%S-%f"
        )
        self.recompute_completed = recompute_completed

    def check_completed(self):
        if (self.output_directory / "camera_sync.csv").exists():
            return True
        else:
            return False

    def load_config(self):
        # config
        try:
            config_file = list(self.recording_directory.glob("*.recording_config.yaml"))[0]
        except:
            raise FileNotFoundError("No recording config file found in the recording directory.")
        self.config = yaml.load(config_file.read_text(), Loader=yaml.FullLoader)

    def estimate_total_frames_from_metadata(self):
        first_camera = self.metadata_csvs_df.iloc[0].camera
        camera_metadata_csvs = self.metadata_csvs_df[
                self.metadata_csvs_df.camera == first_camera
            ].sort_values(by="frame")
        final_metadata_csv = camera_metadata_csvs.iloc[-1].csv_loc
        final_metadata_df = pd.read_csv(final_metadata_csv, header=0)
        final_start_frame = camera_metadata_csvs.iloc[0].frame
        final_n_frames = final_metadata_df.shape[0]
        total_frames = final_start_frame + final_n_frames
        return total_frames

    def make_fictive_triggerdata(self):
        self.trigger_times = np.round(
            np.arange(
                0, self.estimate_total_frames_from_metadata() * self.isi_uS, self.isi_uS
            ),
            0,
        ).astype(int)
        self.trigger_states = np.zeros(len(self.trigger_times)).astype(int)

    def load_triggerdata(self):

        # load triggerdata
        trig_files = list(self.recording_directory.glob("*.triggerdata.csv"))
        if len(trig_files) == 0:
            self.make_fictive_triggerdata()
            logger.info("No triggerdata file found. Creating fictive triggerdata.")
        else:
            triggerdata_csv = trig_files[0]
            times, pins, states = np.loadtxt(triggerdata_csv, delimiter=",", skiprows=1).T
            self.trigger_times = times[pins == self.trigger_pin]
            self.trigger_states = states[pins == self.trigger_pin].astype(int)

        # ensure that no frames have beeen skipped in the microcontroller trigger
        if np.any(np.diff(self.trigger_times) / self.isi_uS > 1.5):
            max_skip = np.max(np.diff(self.trigger_times) / self.isi_uS)
            logger.info(f"Skipped frames in microcontroller trigger: {max_skip}")
            raise ValueError(f"Skipped frames in microcontroller trigger: {max_skip}")

        # get recording length in hours
        self.recording_length_hours = round(len(self.trigger_times) / self.samplerate / 60 / 60, 5)  # not used

        # create a pandas dataframe populated by frame number, arduino time, and trigger state
        self.frame_df = pd.DataFrame(
            {"trigger_times": self.trigger_times, "trigger_states": self.trigger_states}
        )

    def load_metadata(self):
        # get the camera metadata csvs
        metadata_csvs = list(self.recording_directory.glob(f"*.metadata.csv"))
        camera = [i.stem.split(".")[1] for i in metadata_csvs]
        frame = [i.stem.split(".")[2] for i in metadata_csvs]
        self.metadata_csvs_df = pd.DataFrame(
            {
                "camera": camera,
                "frame": np.array(frame).astype(int),
                "csv_loc": metadata_csvs,
            }
        )
        self.cameras = self.metadata_csvs_df.camera.unique()

    def check_if_correct_fps(self):
        # checks if self.samplerate matches the samplerate of videos
        logger.info("Checking if correct FPS")
        # get the videos in recording directory
        videos = list(self.recording_directory.glob("*.mp4"))
        if len(videos) == 0:
            raise FileNotFoundError("No videos found in the recording directory.")
        # get the fps of the videos
        fps = []
        for video in videos:
            cap = cv2.VideoCapture(str(video))
            fps.append(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
        # check if fps matches the expected fps, within 1 fps
        if np.all(np.abs(np.array(fps) - self.samplerate) < 1):
            return True
        else:
            logger.info(f"Expected FPS: {self.samplerate}, Found FPS: {fps}")
            return False

    def run(self):

        # check if sync already completed
        if self.recompute_completed == False:
            if self.check_completed():
                logger.info("Sync already completed")
                return

        assert self.check_if_correct_fps(), "Incorrect FPS detected"

        # load the config and triggerdata files
        logger.info("Loading video config, metadata, and triggerdata")
        self.load_config()
        self.load_metadata()
        self.load_triggerdata()

        # get the frame indexes
        change_m = (
            self.frame_df.trigger_states.values[1:] != self.frame_df.trigger_states.values[:-1]
        )
        change_times_s = self.frame_df[:-1][change_m].trigger_times.values / 1000000
        change_times_f = change_times_s * self.samplerate

        # for each camera, get the frame times
        for camera in self.cameras:
            logger.info(f"\t aligning frames for camera: {camera}")
            # find relevant metadata files for camera
            camera_metadata_csvs = self.metadata_csvs_df[
                self.metadata_csvs_df.camera == camera
            ].sort_values(by="frame")

            self.frame_df[f"{camera}_frames"] = -1
            # loop over metadata files to align frames to arduino trigger
            camera_frame_idx = 0
            pulse_frame_idx = 0
            for metadata_csv in camera_metadata_csvs.csv_loc.values:

                # get the hardware timestamp from the metadata csv
                # camera_hw_timestamps = np.loadtxt(metadata_csv, delimiter=",", skiprows=1)[:, 1]
                metadata_df = pd.read_csv(metadata_csv, header=0)
                camera_hw_timestamps = metadata_df["frame_timestamp"].values
                camera_hw_timestamps = camera_hw_timestamps / 1e3  # Convert from nanoseconds to microseconds

                # estimate the dropped frames
                frame_indexes, dropped_frames = estimate_frame_indexes(camera_hw_timestamps)

                frame_indexes += pulse_frame_idx
                self.frame_df.loc[frame_indexes, f"{camera}_frames"] = (
                    camera_frame_idx + np.arange(len(frame_indexes))
                )

                # set the current true pulse number
                pulse_frame_idx = frame_indexes[-1] + 1

                # set the current camera frame number
                camera_frame_idx += len(frame_indexes)

        # Estimate the frame times
        self.frame_df["datetime_est"] = [
            (self.recording_start + timedelta(seconds=i / 1000000)).timestamp()
            for i in self.frame_df["trigger_times"].values
        ]

        # save
        self.frame_df.to_csv(self.output_directory / "camera_sync.csv")


def estimate_frame_indexes(timestamps):
    """
    Estimate the frame indexes from a sequence of timestamps (useful to detecting
    dropped frames). This function assumes that the first frame corresponds to index 0,
    i.e. the first frame was not dropped.

    Parameters
    ----------
    timestamps : np.ndarray
        Array of timestamps (in microseconds) for each frame in the video.

    Returns
    -------
    frame_indexes : np.ndarray
        Array of indices of captured frames.

    dropped_frames : np.ndarray
        Array of indices of dropped frames.


    Examples
    --------
    >>> timestamps = np.array([0, 33333, 66666, 133333, 166666])
    >>> detect_dropped_frames(timestamps)
    (array([3]), array([0, 1, 2, 4, 5]))
    """
    # Estimate number of periods between frames
    time_diffs = np.diff(timestamps)
    quantized_diffs = np.rint(time_diffs / np.median(time_diffs))

    # Estimate frame indexes
    frame_indexes = np.cumsum(quantized_diffs).astype(int)
    frame_indexes = np.insert(frame_indexes, 0, 0)

    # Estimate dropped frames
    dropped_frames = np.setdiff1d(np.arange(frame_indexes.max() + 1), frame_indexes)
    return frame_indexes, dropped_frames
