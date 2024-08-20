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


def sync_cameras(
    recording_row,
    job_directory,
    output_directory,
    config_file,
):

    # where the video data is located
    recording_directory = (
        Path(recording_row.video_location_on_o2) / recording_row.video_recording_id
    )
    assert (
        recording_directory.exists()
    ), f"Recording directory {recording_directory} does not exist"

    # load config
    config_file = Path(config_file)
    config = yaml.safe_load(open(config_file, "r"))

    # where to save output
    output_directory_camera_sync = (
        output_directory / "camera_sync" / recording_row.video_recording_id
    )
    logger.info("Starting sync cameras")

    # check if sync is already completed
    if config["sync_cameras"]["recompute_completed"] == False:
        if check_camera_sync_completion(output_directory_camera_sync):
            logger.info("Camera sync already completed, quitting")
            return
        else:
            logger.info("Camera sync not completed, running")

    output_directory_camera_sync.mkdir(parents=True, exist_ok=True)
    current_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    remote_job_directory = job_directory / current_datetime_str

    # get camera info
    samplerate = recording_row.samplerate
    # trigger_pin = recording_row.trigzger_pin
    params = {
        "recording_directory": recording_directory.as_posix(),
        "output_directory_camera_sync": output_directory_camera_sync.as_posix(),
        "samplerate": samplerate,
        # "trigger_pin": trigger_pin,
    }

    # specify the duration of the job (here, it should be short)
    #  in config, set o2_runtime_multiplier to choose how long the job should
    #  run relative to recording duration
    duration_requested = convert_minutes_to_hms(
        recording_row.duration_m * config["o2"]["sync_cameras"]["o2_runtime_multiplier"]
    )

    # create the job runner
    runner = O2Runner(
        job_name_prefix=f"{recording_row.video_recording_id}_ephys_sync",
        remote_job_directory=remote_job_directory,
        conda_env=config["o2"]["sync_cameras"]["conda_env"],
        o2_username=recording_row.username,
        o2_server="login.o2.rc.hms.harvard.edu",
        job_params=params,
        o2_n_cpus=config["o2"]["sync_cameras"]["o2_n_cpus"],
        o2_memory=config["o2"]["sync_cameras"]["o2_memory"],
        o2_time_limit=duration_requested,
        o2_queue=config["o2"]["sync_cameras"]["o2_queue"],
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
    from multicamera_airflow_pipeline.tim_240731.sync.sync_cameras import CameraSynchronizer
    synchronizer = CameraSynchronizer(
        recording_directory=params["recording_directory"],
        output_directory=params["output_directory_camera_sync"],
        samplerate=params["samplerate"],  # camera sample rate
        **config["sync_cameras"],
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
    if check_camera_sync_completion(output_directory_camera_sync):
        logger.info("Camera sync completed successfully")
    else:
        raise ValueError("Camera sync did not complete successfully.")


def check_camera_sync_completion(output_directory):
    return (output_directory / "camera_sync.csv").exists()
