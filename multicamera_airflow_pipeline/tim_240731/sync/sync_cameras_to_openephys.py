from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pandas as pd
from open_ephys.analysis import Session
import spikeinterface as si
import spikeinterface.extractors as se
from datetime import datetime, timedelta
import logging
import sys

logger = logging.getLogger(__name__)
logger.info(f"Python interpreter binary location: {sys.executable}")


class OpenEphysSynchronizer:
    def __init__(
        self,
        camera_sync_file,
        ephys_recording_path,
        ephys_sync_output_directory,
        npx_samplerate,
        camera_samplerate,
        recompute_completed=False,
        search_window_s=60 * 5,
        frame_window=1000,
    ):

        # csv file containing the camera sync data outputted from sync_cameras
        self.camera_sync_file = Path(camera_sync_file)
        # the folder containing the openephys recording
        self.ephys_recording_path = Path(ephys_recording_path)
        # usually base_folder/{subject}/{recording}/
        self.ephys_sync_output_directory = Path(ephys_sync_output_directory)

        self.recompute_completed = recompute_completed
        self.npx_samplerate = npx_samplerate
        self.camera_samplerate = camera_samplerate
        self.search_window_s = (
            search_window_s  # range in time to search for a matching search window in
        )
        self.frame_window = frame_window  # size of window in frames to search for match

    def check_if_camera_sync_exists(self):
        if self.camera_sync_file.exists():
            return True
        else:
            return False

    def load_openephys_recording(self):
        self.session = Session(self.ephys_recording_path.as_posix())
        # get the first recording in the first recordnode
        # TODO: support multiple recordings if needed
        #    (for now we should only have 1 recording per folder)
        self.recording = self.session.recordnodes[0].recordings[0]

        # get stream info
        stream_names, stream_numbers = se.OpenEphysBinaryRecordingExtractor.get_streams(
            self.ephys_recording_path
        )
        # only subset open ephys streams
        self.stream_names = [i for i in stream_names if "Neuropix-PXI" in i]

    def load_camera_sync(self):
        if self.check_if_camera_sync_exists() == False:
            raise FileNotFoundError("Camera sync file not found.")

        self.camera_sync_df = pd.read_csv(self.camera_sync_file)

    def load_sample_numbers_and_states(self, stream_name):
        # load sample numbers and states
        npx_sample_numbers_npy = list(
            Path(self.recording.directory).glob(
                f"**/events/{stream_name.split('#')[1]}/TTL/sample_numbers.npy"
            )
        )[0]
        npx_states_npy = list(
            Path(self.recording.directory).glob(
                f"**/events/{stream_name.split('#')[1]}/TTL/states.npy"
            )
        )[0]
        npx_words_npy = list(
            Path(self.recording.directory).glob(
                f"**/events/{stream_name.split('#')[1]}/TTL/full_words.npy"
            )
        )[0]
        npx_full_words = np.load(npx_words_npy)
        npx_sample_numbers = np.load(npx_sample_numbers_npy)
        npx_states = np.load(npx_states_npy) == 1
        return npx_sample_numbers, npx_states, npx_full_words

    def determine_recording_creation_time(self):
        # get the expected start and end time of this recording
        with open(Path(self.recording.directory) / "sync_messages.txt", "r") as file:
            dattime_ms = int(file.readline().split(": ")[1])
            self.recording_created = datetime.fromtimestamp(dattime_ms / 1000)

    def run(self):
        self.load_camera_sync()
        self.load_openephys_recording()
        self.determine_recording_creation_time()

        # synchronize each stream individually (e.g. for different probes)
        for stream_name in self.stream_names:
            self.synchronize_stream(stream_name)

    def remap_npx_samples(self, stream_name, npx_sample_numbers):
        # remap npx sample numbers into the sample the index of the recording
        # the continuous data has a sample number for each index, which does not start at 0
        # we want to get the 0 indexed values
        continuous_sample_numbers_npy = list(
            Path(self.recording.directory).glob(
                f"**/continuous/{stream_name.split('#')[1]}/sample_numbers.npy"
            )
        )[0]
        continuous_sample_numbers = np.load(continuous_sample_numbers_npy, mmap_mode="r+")
        exhaustive_search = False
        if exhaustive_search:
            sample_idx = dict(
                zip(continuous_sample_numbers, np.arange(len(continuous_sample_numbers)))
            )
            npx_sample_numbers_remapped = [sample_idx[i] for i in npx_sample_numbers]
        else:
            sample_idx_start = np.where(npx_sample_numbers[0] == continuous_sample_numbers)[0][0]
            npx_sample_numbers_remapped = (
                npx_sample_numbers - npx_sample_numbers[0] + sample_idx_start
            )
        return npx_sample_numbers_remapped

    def synchronize_stream(self, stream_name):

        logger.info(f"Syncing {stream_name}")

        # create directory for each stream
        (self.ephys_sync_output_directory / stream_name).mkdir(exist_ok=True, parents=True)

        # skip LFP streams
        if "-LFP" in stream_name:
            return

        # define the output file
        ephys_alignment_file = (
            self.ephys_sync_output_directory / stream_name / "ephys_alignment.mmap"
        )
        incomplete_file_name = (
            self.ephys_sync_output_directory / stream_name / "ephys_alignment.incomplete.mmap"
        )
        if self.recompute_completed == False and ephys_alignment_file.exists():
            print(f"Skipping {stream_name} because it already exists")
            return

        npx_sample_numbers, npx_states, npx_full_words = self.load_sample_numbers_and_states(
            stream_name
        )

        # remap sample numbers into zero indexed range
        npx_sample_numbers_remapped = self.remap_npx_samples(stream_name, npx_sample_numbers)

        # detect state changes
        try:
            npx_frame_states, npx_frame_numbers = get_npx_state_changes(
                npx_sample_numbers_remapped,
                npx_full_words,
                self.npx_samplerate,
                self.camera_samplerate,
            )
        except AssertionError:
            print("\t Unable to get frame states from npx data")
            return

        # estimate the time of each state change
        npx_datetimes = np.array(
            [
                ((self.recording_created + timedelta(seconds=i / self.npx_samplerate)).timestamp())
                for i in npx_sample_numbers
            ]
        )

        # Ensure all frame skips match expected samplerate
        frames_skipped = (
            np.diff(npx_sample_numbers_remapped) / self.npx_samplerate * self.camera_samplerate
        )
        frames_skipped_int = np.round(frames_skipped).astype(int)
        test = np.abs((frames_skipped_int - frames_skipped))
        assert np.max(np.abs(test)) < 0.1

        # Ensure all frame skips match expected samplerate
        frames_skipped = (
            np.diff(npx_sample_numbers_remapped) / self.npx_samplerate * self.camera_samplerate
        )
        frames_skipped_int = np.round(frames_skipped).astype(int)
        test = np.abs((frames_skipped_int - frames_skipped))
        assert np.max(np.abs(test)) < 0.1

        camera_frame_states = self.camera_sync_df.trigger_states.values
        camera_frame_numbers = self.camera_sync_df.index.values
        camera_datetimes = self.camera_sync_df.datetime_est.values

        # get the signals to align
        signal1_frames = camera_frame_numbers
        signal1_states = camera_frame_states
        signal1_timestamps = camera_datetimes
        signal2_frames = npx_frame_numbers
        signal2_states = npx_frame_states
        signal2_timestamps = npx_datetimes

        # create a window of search_window_s to look for a match
        search_window = [
            signal2_timestamps[0] - self.search_window_s,
            signal2_timestamps[0] + self.search_window_s,
        ]
        search_mask = (signal1_timestamps >= search_window[0]) & (
            signal1_timestamps <= search_window[1]
        )
        search_start = np.where(search_mask)[0][0]

        # drag a sliding window to find when the random state matches
        correlations = sliding_correlation(
            signal2_states[: self.frame_window],
            signal1_states[search_mask],
        )

        # ensure we found a match
        if np.max(correlations) < 0.9:
            print(f"\t NO CORRELATION FOUND {np.max(correlations)}")
            return

        # grab the point of match
        peak_correlation = np.argmax(correlations)
        start_position = search_start + peak_correlation

        # create a figure to show the match, save to ephys_sync_output_directory as jpg
        fig, axs = plt.subplots(ncols=2, figsize=(10, 1))
        axs[0].plot(correlations)
        axs[1].plot(signal1_states[start_position : start_position + 100])
        axs[1].plot(signal2_states[:100])
        correlation_figure_file = (
            self.ephys_sync_output_directory / stream_name / "correlation.jpg"
        )
        plt.savefig(correlation_figure_file)
        plt.close()

        matches = [
            state == signal1_states[position + start_position]
            for position, state in enumerate(signal2_states)
        ]

        logger.info(f"\t creating ephys alignment mmap")

        # create the mmap
        ephys_alignment_file.parent.mkdir(exist_ok=True, parents=True)
        memmap_array = np.memmap(
            incomplete_file_name,
            dtype="int32",
            mode="w+",
            shape=len(camera_frame_states),
        )
        for i in camera_frame_numbers:
            memmap_array[i] = -1
            if i - start_position < 0:
                continue
            else:
                if i - start_position < len(signal2_frames):
                    memmap_array[i] = signal2_frames[i - start_position]

        # move the incomplete file to the complete file
        incomplete_file_name.rename(ephys_alignment_file)

        # ensure memmap array is properly synchronized
        start = 10000
        stop = 10100
        fig, ax = plt.subplots(figsize=(5, 1))
        ax.plot(camera_frame_states[start:stop])
        ax.plot(
            npx_frame_states[
                (npx_frame_numbers >= memmap_array[start])
                & (npx_frame_numbers <= memmap_array[stop])
            ]
        )
        alignment_figure_file = self.ephys_sync_output_directory / stream_name / "alignment.jpg"
        plt.savefig(alignment_figure_file)
        plt.close()


def get_npx_state_changes(npx_sample_numbers, npx_states, npx_samplerate, camera_samplerate):
    # determine how many frames have been skipped for each state change
    frames_skipped = np.diff(npx_sample_numbers) / npx_samplerate * camera_samplerate
    frames_skipped_int = np.round(frames_skipped).astype(int)
    assert (
        np.any(np.abs((frames_skipped_int - frames_skipped)) > 0.05) == False
    ), "Unable to calculate frames skipped from NPX"

    # populate each frame with zeros and ones
    initial_state = npx_states[0]
    npx_frame_states = np.zeros(np.sum(frames_skipped_int), dtype=bool)
    cur = 0
    for i, state in zip(tqdm(frames_skipped_int, leave=False), npx_states):
        npx_frame_states[cur : cur + i] = state
        cur += i

    # get frame number for each frame
    frame_numbers = np.zeros(len(npx_frame_states) + 50)
    frame = 0
    for skipped, current_frame, next_frame in zip(
        tqdm(frames_skipped_int, leave=False), npx_sample_numbers[:-1], npx_sample_numbers[1:]
    ):
        frame_numbers[frame] = current_frame
        if skipped > 1:
            # TODO figure out why this hack is needed...
            # HACK = min([0, len(frame_numbers) - (frame + skipped + 1)])
            frame_numbers[frame : frame + skipped + 1] = np.linspace(
                current_frame, next_frame, skipped + 1  # + HACK
            ).astype(int)
        frame += skipped
    frame_numbers = frame_numbers[:frame]
    return npx_frame_states, frame_numbers


def sliding_correlation(arr1, arr2, preceding_zeros=100, disable_tqdm=False):
    """
    Compute the sliding correlation between two arrays, accounting for NaNs.
    """
    len1, len2 = arr1.shape[0], arr2.shape[0]
    correlations = np.zeros(len2 - len1 + 1)

    correlations = np.zeros(len2 - len1 + 1)
    for i in tqdm(range(len2 - len1 + 1), leave=False, desc="doing correlation"):
        # Flatten the arrays for correlation and mask NaNs

        mask = ~np.isnan(arr1) & ~np.isnan(arr2[i : i + len1])

        if np.any(mask):
            # Calculate correlation only for non-NaN values
            correlation = np.corrcoef(arr1[mask], arr2[i : i + len1][mask])[0, 1]
        else:
            correlation = np.nan

        correlations[i] = correlation

    return correlations
