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
import os
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)


def convert_minutes_to_hms(minutes_float):
    # Convert minutes to total seconds
    total_seconds = int(minutes_float * 60)

    # Extract hours, minutes, and seconds using divmod
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Format as HH:MM:SS
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def check_ephys_sync_completion(output_directory_ephys_sync, n_expected_streams):
    logger.info(
        f"Searching for {n_expected_streams} ephys sync streams in {output_directory_ephys_sync}"
    )
    # TODO check if number of ephys streams matches expected
    ephys_streams_found = list(output_directory_ephys_sync.glob("*/ephys_alignment.mmap"))
    n_aligned_ephys_streams = len(ephys_streams_found)
    logger.info(f"Found {n_aligned_ephys_streams} streams")
    if int(n_aligned_ephys_streams) >= int(n_expected_streams):
        logger.info(f"Found the following streams: {n_aligned_ephys_streams}")
        for stream in ephys_streams_found:
            logger.info(stream)
        return True

    return False


def sync_cameras_to_openephys(
    recording_row,
    job_directory,
    output_directory,
    config_file,
):

    try:
        # Check if value is None or NaN
        if recording_row.ephys_id is None or (
            isinstance(recording_row.ephys_id, float) and np.isnan(recording_row.ephys_id)
        ):
            logger.info("No ephys_id found, skipping spikesorting")
            return
        # Convert to string
        recording_row.ephys_id = str(recording_row.ephys_id)
    except Exception as e:
        logger.warning(
            f"Invalid ephys_id value: {recording_row.ephys_id}, error: {e}, skipping spikesorting"
        )
        return

    # load config
    config_file = Path(config_file)
    config = yaml.safe_load(open(config_file, "r"))

    # the number of streams expected to have been synced
    #   used to determine if we can skip this step
    n_expected_streams = recording_row.n_ephys_streams

    # load config
    config_file = Path(config_file)
    config = yaml.safe_load(open(config_file, "r"))

    # where the video data is located
    recording_directory = (
        Path(recording_row.video_location_on_o2) / recording_row.video_recording_id
    )

    # where to save output
    output_directory_ephys_sync = (
        output_directory / "openephys_sync" / recording_row.video_recording_id
    )
    logger.info(f"Creating output directory: {output_directory_ephys_sync}")
    output_directory_ephys_sync.mkdir(parents=True, exist_ok=True)
    os.chmod(output_directory_ephys_sync.as_posix(), 0o2775)
    current_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    remote_job_directory = job_directory / current_datetime_str

    # check if sync successfully completed
    if config["sync_ephys"]["recompute_completed"] == False:
        if check_ephys_sync_completion(output_directory_ephys_sync, n_expected_streams):
            logger.info(f"Streams found: {n_expected_streams}, quitting")
            return
        else:
            logger.info("No streams matched")

    samplerate = recording_row.samplerate

    recording_row.ephys_location_on_o2

    camera_sync_file = (
        output_directory / "camera_sync" / recording_row.video_recording_id / "camera_sync.csv"
    )
    assert camera_sync_file.exists()

    ephys_recording_path = Path(recording_row.ephys_location_on_o2) / recording_row.ephys_id
    assert ephys_recording_path.exists()

    params = {
        "recording_directory": recording_directory.as_posix(),
        "ephys_sync_output_directory": output_directory_ephys_sync.as_posix(),
        "samplerate": samplerate,
        "camera_sync_file": camera_sync_file.as_posix(),
        "ephys_recording_path": ephys_recording_path.as_posix(),
    }

    duration_requested = convert_minutes_to_hms(
        recording_row.duration_m * config["o2"]["sync_ephys"]["o2_runtime_multiplier"]
    )
    duration_requested

    # create the job runner
    runner = O2Runner(
        job_name_prefix=f"{recording_row.video_recording_id}_ephys_sync",
        remote_job_directory=remote_job_directory,
        conda_env=config["o2"]["sync_ephys"]["conda_env"],
        o2_username=recording_row.username,
        o2_server="login.o2.rc.hms.harvard.edu",
        job_params=params,
        o2_n_cpus=config["o2"]["sync_ephys"]["o2_n_cpus"],
        o2_memory=config["o2"]["sync_ephys"]["o2_memory"],
        o2_time_limit=duration_requested,
        o2_queue=config["o2"]["sync_ephys"]["o2_queue"],
    )

    runner.python_script = textwrap.dedent(
        f"""
    # load params
    import yaml
    params_file = "{runner.remote_job_directory / f"{runner.job_name}.params.yaml"}"
    config_file = "{config_file.as_posix()}"
    import os; os.umask(0o002)
    params = yaml.safe_load(open(params_file, 'r'))
    config = yaml.safe_load(open(config_file, 'r'))

    # grab sync cameras function
    from multicamera_airflow_pipeline.tim_240731.sync.sync_cameras_to_openephys import OpenEphysSynchronizer
    synchronizer = OpenEphysSynchronizer(
        camera_sync_file =params["camera_sync_file"] ,
        ephys_recording_path = params["ephys_recording_path"],
        ephys_sync_output_directory =  params["ephys_sync_output_directory"],
        camera_samplerate = params["samplerate"],
        **config["sync_ephys"]
    )
    synchronizer.run()
    """
    )

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
    if check_ephys_sync_completion(output_directory_ephys_sync, n_expected_streams):
        logger.info("Ephys sync completed successfully")
    else:
        raise ValueError("Ephys sync did not complete successfully.")
