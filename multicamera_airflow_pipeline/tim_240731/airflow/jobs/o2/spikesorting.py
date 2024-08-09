from datetime import datetime, timedelta
import pandas as pd
import requests
from io import BytesIO
from pathlib import Path
from multicamera_airflow_pipeline.tim_240731.interface.o2 import O2Runner
from datetime import datetime
import textwrap
import inspect
import time
import yaml

import logging

logger = logging.getLogger(__name__)


def convert_minutes_to_hms(minutes_float):
    # Convert minutes to total seconds
    total_seconds = int(minutes_float * 60)

    # Extract hours, minutes, and seconds using divmod
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Format as HH:MM:SS
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def check_spikesorting_completion(spikesorting_output_directory, n_ephys_streams_expected):
    n_sorts_found = len(list(spikesorting_output_directory.glob("*/sort_result")))
    if n_sorts_found == n_ephys_streams_expected:
        return True
    else:
        return False


def spikesorting(
    recording_row,
    job_directory,
    output_directory,
    config_file,
):
    n_ephys_streams_expected = recording_row.n_ephys_streams

    # load config
    config_file = Path(config_file)
    config = yaml.safe_load(open(config_file, "r"))

    # where the video data is located
    recording_directory = (
        Path(recording_row.calibration_location_on_o2) / recording_row.calibration_id
    )

    # where to save output
    spikesorting_output_directory = (
        output_directory / "spikesorting" / recording_row.video_recording_id
    )
    spikesorting_output_directory.mkdir(parents=True, exist_ok=True)
    current_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    remote_job_directory = job_directory / current_datetime_str

    # check if sync successfully completed
    if config["spikesorting"]["recompute_completed"] == False:
        if check_spikesorting_completion(spikesorting_output_directory, n_ephys_streams_expected):
            logger.info("spikesorting completed, quitting")
            return
        else:
            logger.info("spikesorting incomplete, starting")

    duration_requested = convert_minutes_to_hms(
        recording_row.duration_m * config["o2"]["spikesorting"]["o2_runtime_multiplier"]
    )
    duration_requested

    samplerate = recording_row.samplerate

    ephys_recording_directory = Path(recording_row.ephys_location_on_o2) / recording_row.ephys_id

    params = {
        "spikesorting_output_directory": spikesorting_output_directory.as_posix(),
        "ephys_recording_directory": ephys_recording_directory.as_posix(),
    }

    # create the job runner
    runner = O2Runner(
        job_name_prefix=f"{recording_row.video_recording_id}_spikesorting",
        remote_job_directory=remote_job_directory,
        conda_env="/n/groups/datta/tim_sainburg/conda_envs/kilosort4",
        o2_username=recording_row.username,
        o2_server="login.o2.rc.hms.harvard.edu",
        job_params=params,
        o2_n_cpus=config["o2"]["spikesorting"]["o2_n_cpus"],
        o2_memory=config["o2"]["spikesorting"]["o2_memory"],
        o2_time_limit=duration_requested,
        o2_queue=config["o2"]["spikesorting"]["o2_queue"],
        o2_exclude=config["o2"]["spikesorting"]["o2_exclude"],
        o2_qos=config["o2"]["spikesorting"]["o2_qos"],
        o2_gres=config["o2"]["spikesorting"]["o2_gres"],
    )

    runner.python_script = textwrap.dedent(
        f"""
    # load params
    import yaml
    params_file = "{runner.remote_job_directory / f"{runner.job_name}.params.yaml"}"
    config_file = "{config_file.as_posix()}"

    params = yaml.safe_load(open(params_file, 'r'))
    config = yaml.safe_load(open(config_file, 'r'))

    # grab sync cameras function
    from multicamera_airflow_pipeline.tim_240731.ephys.spikesorting_ks4 import SpikeSorter
    spikesorter = SpikeSorter(
        ephys_recording_directory= params["ephys_recording_directory"],
        spikesorting_output_directory = params["spikesorting_output_directory"],
        **config["spikesorting"]
    )
    spikesorter.run()


    """
    )

    print(runner.python_script)

    runner.run()

    # wait until the job is finished
    # 10000/60/24 = roughly 1 week
    for i in range(10000):
        # check job status every n seconds
        status = runner.check_job_status()
        if status:
            break
        time.sleep(60)

    # check if sync successfully completed
    if check_spikesorting_completion(spikesorting_output_directory, n_ephys_streams_expected):
        logger.info("spikesorting completed successfully")
    else:
        raise ValueError("spikesorting did not complete successfully.")
