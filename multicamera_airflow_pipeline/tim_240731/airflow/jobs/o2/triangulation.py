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


def check_triangulation_completion(output_directory_triangulation):
    return (output_directory_triangulation / "triangulation_completed.log").exists()


def triangulation(
    recording_row,
    job_directory,
    output_directory,
    config_file,
):

    # load config
    config_file = Path(config_file)
    config = yaml.safe_load(open(config_file, "r"))

    # where the video data is located
    recording_directory = (
        Path(recording_row.calibration_location_on_o2) / recording_row.calibration_id
    )

    # where to save output
    output_directory_triangulation = (
        output_directory / "triangulation" / recording_row.video_recording_id
    )
    output_directory_triangulation.mkdir(parents=True, exist_ok=True)
    current_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    remote_job_directory = job_directory / current_datetime_str

    # check if sync successfully completed
    if config["triangulation"]["recompute_completed"] == False:
        if check_triangulation_completion(output_directory_triangulation):
            logger.info("triangulation completed, quitting")
            return
        else:
            logger.info("triangulation incomplete, starting")

    predictions_2d_directory = (
        output_directory / "2D_predictions" / recording_row.video_recording_id
    )

    expected_frames_per_video = recording_row.samplerate * 60 * recording_row.max_video_duration_m

    camera_sync_file = (
        output_directory / "camera_sync" / recording_row.video_recording_id / "camera_sync.csv"
    )

    assert camera_sync_file.exists()

    duration_requested = convert_minutes_to_hms(
        recording_row.duration_m * config["o2"]["triangulation"]["o2_runtime_multiplier"]
    )
    duration_requested

    camera_calibration_directory = (
        output_directory
        / "camera_calibration"
        / recording_row.calibration_id
        / "jarvis"
        / "CalibrationParameters"
    )

    # if the calibration completed successfully, the camera calibration directory should exist
    assert camera_calibration_directory.exists()

    params = {
        "predictions_2d_directory": predictions_2d_directory.as_posix(),
        "output_directory_triangulation": output_directory_triangulation.as_posix(),
        "camera_sync_file": camera_sync_file.as_posix(),
        "expected_frames_per_video": expected_frames_per_video,
        "camera_calibration_directory": camera_calibration_directory.as_posix(),
    }

    # create the job runner
    runner = O2Runner(
        job_name_prefix=f"{recording_row.video_recording_id}_triangulation",
        remote_job_directory=remote_job_directory,
        conda_env=config["o2"]["triangulation"]["conda_env"],
        o2_username=recording_row.username,
        o2_server="login.o2.rc.hms.harvard.edu",
        job_params=params,
        o2_n_cpus=config["o2"]["triangulation"]["o2_n_cpus"],
        o2_memory=config["o2"]["triangulation"]["o2_memory"],
        o2_time_limit=duration_requested,
        o2_queue=config["o2"]["triangulation"]["o2_queue"],
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
    from multicamera_airflow_pipeline.tim_240731.keypoints.triangulation import Triangulator 
    triangulator = Triangulator(
        predictions_2d_directory = params['predictions_2d_directory'],
        output_directory_triangulation = params['output_directory_triangulation'],
        camera_sync_file = params['camera_sync_file'],
        expected_frames_per_video = params['expected_frames_per_video'],
        camera_calibration_directory = params['camera_calibration_directory'],
        n_frames=None,
        **config["triangulation"]
    )
    triangulator.run()
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
    if check_triangulation_completion(output_directory_triangulation):
        logger.info("triangulation completed successfully")
    else:
        raise ValueError("triangulation did not complete successfully.")
