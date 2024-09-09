import sys
import logging

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
import torch
import re

# spikesorter specific
import spikeinterface.extractors as se
import spikeinterface as si
from spikeinterface import qualitymetrics
import kilosort
from spikeinterface import exporters
from spikeinterface.preprocessing import bandpass_filter, phase_shift, common_reference
from spikeinterface.widgets import plot_traces
from spikeinterface.sorters import run_sorter

logger.info(f"SI version {si.__version__}")
logger.info(f"SI location {si.__file__}")
logger.info(f"KS version {kilosort.__version__}")
logger.info(f"KS file {kilosort.__file__}")


class SpikeSorter:
    def __init__(
        self,
        ephys_recording_directory,
        spikesorting_output_directory,
        amplitude_cutoff_thresh=0.1,
        isi_violations_ratio_thresh=1,
        presence_ratio_thresh=0.9,
        n_jobs=-1,
        rm_channels=[],
        plot_samples=True,
        recompute_completed=False,
    ):
        self.ephys_recording_directory = Path(ephys_recording_directory)
        self.spikesorting_output_directory = Path(spikesorting_output_directory)
        self.amplitude_cutoff_thresh = amplitude_cutoff_thresh
        self.isi_violations_ratio_thresh = isi_violations_ratio_thresh
        self.presence_ratio_thresh = presence_ratio_thresh
        self.rm_channels = rm_channels
        self.n_jobs = n_jobs
        self.plot_samples = plot_samples
        self.recompute_completed = recompute_completed

    def run(self):

        logger.info(f"Running spikesorting on {self.ephys_recording_directory}")
        logger.info(f"Output directory: {self.spikesorting_output_directory}")

        # find streams
        logger.info("Finding streams")
        self.find_streams()

        # sort streams
        for stream_name in self.stream_names:
            self.sort_stream(stream_name)

    def find_streams(self):
        stream_names, stream_numbers = se.OpenEphysBinaryRecordingExtractor.get_streams(
            self.ephys_recording_directory
        )
        # exclude LFP and grab only neuropixels streams
        self.stream_names = [i for i in stream_names if "Neuropix-PXI" in i and "LFP" not in i]

    def sort_stream(self, stream_name):
        logger.info(f"Sorting stream {stream_name}")
        torch.cuda.empty_cache()

        stream_alphanumeric = re.sub(r"\W+", "-", stream_name)

        if self.check_completed(stream_alphanumeric) & (self.recompute_completed is False):
            logger.info(f"Stream {stream_name} already sorted. Skipping.")
            return

        stream_output_directory = self.spikesorting_output_directory / stream_alphanumeric
        (stream_output_directory).mkdir(
            parents=True, exist_ok=True
        )

        recording = se.read_openephys(
            folder_path=self.ephys_recording_directory, stream_name=stream_name
        )

        # get probe info
        probe_dataframe = recording.get_probe().to_dataframe()
        probe_dataframe.to_pickle(self.ephys_recording_directory / "probe_dataframe.pickle")

        channel_ids = recording.channel_ids

        # preprocess
        # TODO: ensure global is by shank and not across shanks
        recording_rm_chans = recording.remove_channels(remove_channel_ids=self.rm_channels)
        recording_bandpass = bandpass_filter(recording_rm_chans, freq_min=300.0, freq_max=6000.0)
        recording_shift = phase_shift(recording=recording_bandpass)
        recording_car = common_reference(recording_shift, operator="median", reference="global")

        if self.plot_samples:
            sample_trace = recording_car.get_traces(
                start_frame=0, end_frame=30000, channel_ids=[channel_ids[0]]
            )
            fig, ax = plt.subplots(figsize=(20, 2))
            ax.plot(sample_trace[:, 0])
            # save to spikesorting_output_directory
            plt.savefig(
                stream_output_directory / "sample_trace.png"
            )
            plt.close()

            # here we use static plot using matplotlib backend
            fig, axs = plt.subplots(ncols=3, figsize=(20, 4))

            plot_traces(recording, backend="matplotlib", clim=(-5000, 5000), ax=axs[0])
            plot_traces(recording_bandpass, backend="matplotlib", clim=(-1000, 1000), ax=axs[1])
            plot_traces(recording_car, backend="matplotlib", clim=(-1000, 1000), ax=axs[2])
            for i, label in enumerate(("unfiltered", "bandpass", "final")):
                axs[i].set_title(label)
            # save to spikesorting_output_directory
            plt.savefig(
                stream_output_directory / "preprocessing.png"
            )
            plt.close()

        # we can estimate the noise on the scaled traces (microV) or on the raw one (which is in our case int16).
        noise_levels_microV = si.get_noise_levels(recording_car, return_scaled=True)
        np.save(
            self.spikesorting_output_directory / stream_alphanumeric / "noise_levels_channels.npy",
            noise_levels_microV,
        )

        # create and delete a temporary directory for saving preprocessed data
        # tempdir = tempfile.TemporaryDirectory()
        # temp_path = Path(tempdir.name)
        # use a temporary directory to save the preprocessed data
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            logger.info("Temporary directory:", temp_path)
            # save processed file locally
            job_kwargs = dict(n_jobs=self.n_jobs, chunk_duration="5s", progress_bar=True)
            rec = recording_car.save(
                folder=temp_path / (self.spikesorting_output_directory.name + stream_alphanumeric),
                format="binary",
                **job_kwargs,
            )

            # run sort
            sorting = self.run_sort(rec, stream_alphanumeric)
            sorting.save(
                folder=stream_output_directory / "sort_result"
            )

            # analyze results
            self.run_analysis(sorting, rec, stream_output_directory)

    def run_analysis(self, sorting, rec, stream_output_directory):
        # postprocessing
        analyzer = si.create_sorting_analyzer(
            sorting,
            rec,
            sparse=True,
            format="memory",
            # folder=output_folder / "post_results"
        )
        # grab spikes
        analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
        # grab waveforms from spikes
        job_kwargs = dict(n_jobs=self.n_jobs, chunk_duration="1s", progress_bar=True)
        analyzer.compute("waveforms", ms_before=1.5, ms_after=2.0, **job_kwargs)
        # estimate templatesz
        analyzer.compute("templates", operators=["average", "median", "std"])
        # noise levels
        analyzer.compute("noise_levels")
        # correlogram (this takes a while)
        analyzer.compute("correlograms")
        # get unit locations
        analyzer.compute("unit_locations")
        # get spike amplitudes
        analyzer.compute("spike_amplitudes", **job_kwargs)
        # estimate similarity
        analyzer.compute("template_similarity")

        # if stream_output_directory / "analyzer" already exists, remove it
        if (stream_output_directory / "analyzer").exists():
            logger.info("Removing existing analyzer directory")
            shutil.rmtree(stream_output_directory / "analyzer")
        # save analyzer results
        analyzer_saved = analyzer.save_as(
            folder=stream_output_directory / "analyzer",
            format="binary_folder",
        )
        # compute quality metrics
        metric_names = [
            "firing_rate",
            "presence_ratio",
            "snr",
            "isi_violation",
            "amplitude_cutoff",
        ]
        metrics = qualitymetrics.compute_quality_metrics(analyzer, metric_names=metric_names)
        # threshold 'good' units
        condition1 = f"(amplitude_cutoff < {self.amplitude_cutoff_thresh})"
        condition2 = f"(isi_violations_ratio < {self.isi_violations_ratio_thresh})"
        condition3 = f"(presence_ratio > {self.presence_ratio_thresh})"
        our_query = f"{condition1} & {condition2} & {condition3}"
        logger.info(our_query)
        keep_units = metrics.query(our_query)
        keep_unit_ids = keep_units.index.values
        logger.info(len(keep_unit_ids))

        # if stream_output_directory / "analyzer_clean" already exists, remove it
        if (stream_output_directory / "analyzer_clean").exists():
            logger.info(f"Removing existing analyzer_clean directory")
            shutil.rmtree(stream_output_directory / "analyzer_clean")  
            
        # save a 'clean' version of the analyzer results
        analyzer_clean = analyzer.select_units(
            keep_unit_ids,
            folder=stream_output_directory / "analyzer_clean",
            format="binary_folder",
        )

    def run_sort(self, rec, stream_alphanumeric):
        # see https://github.com/SpikeInterface/spikeinterface/blob/b6c2c91ecfc71dfa44213eaa30bf8dacf2da72a7/src/spikeinterface/sorters/external/kilosort4.py#L12
        # TODO, add these options to the general config
        sorting = run_sorter(
            sorter_name="kilosort4",
            recording=rec,
            output_folder=self.spikesorting_output_directory / stream_alphanumeric,
            remove_existing_folder=True,
            verbose=True,
            # torch_device="cuda",
            skip_kilosort_preprocessing=False,
            do_CAR=False,
            do_correction=True,
            batch_size=60000,
            nblocks=1,
            templates_from_data=False,
            n_templates=6,  # only if templates_from_data is true
        )
        return sorting

    def check_completed(self, stream_alphanumeric):
        logger.info(
            f"Checking if stream {stream_alphanumeric} is already sorted at {self.spikesorting_output_directory / stream_alphanumeric}"
        )
        if (self.spikesorting_output_directory / stream_alphanumeric / "sort_result").exists():
            return True
        return False
