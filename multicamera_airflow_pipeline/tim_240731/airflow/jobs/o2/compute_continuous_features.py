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


def check_continuous_features_completion(continuous_features_output_directory):
    return (continuous_features_output_directory / "continuous_features.pickle").exists()


def compute_continuous_features(
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
    continuous_features_output_directory = (
        output_directory / "continuous_features" / recording_row.video_recording_id
    )
    #    create output directory
    logger.info(f"Creating output directory: {continuous_features_output_directory}")
    continuous_features_output_directory.mkdir(parents=True, exist_ok=True)
    os.chmod(continuous_features_output_directory.as_posix(), 0o2775)
    current_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    remote_job_directory = job_directory / current_datetime_str

    # check if sync successfully completed
    if config["continuous_features"]["recompute_completed"] == False:
        if check_continuous_features_completion(continuous_features_output_directory):
            logger.info("continuous_features completed, quitting")
            return
        else:
            logger.info("continuous_features incomplete, starting")

    egocentric_alignment_file = list(
        (
            output_directory / "egocentric_alignment" / recording_row.video_recording_id / "rigid"
        ).glob("egocentric_alignment_rigid.*.mmap")
    )[0]

    assert egocentric_alignment_file.exists()

    arena_alignment_file = list(
        (output_directory / "arena_alignment" / recording_row.video_recording_id).glob(
            "coordinates_aligned.*.mmap"
        )
    )[0]

    assert arena_alignment_file.exists()

    duration_requested = convert_minutes_to_hms(
        recording_row.duration_m * config["o2"]["continuous_features"]["o2_runtime_multiplier"]
    )
    duration_requested

    samplerate = recording_row.samplerate

    params = {
        "continuous_features_output_directory": continuous_features_output_directory.as_posix(),
        "coordinates_egocentric_filename": egocentric_alignment_file.as_posix(),
        "coordinates_arena_filename": arena_alignment_file.as_posix(),
        "samplerate": samplerate,
    }

    # create the job runner
    runner = O2Runner(
        job_name_prefix=f"{recording_row.video_recording_id}_continuous_features",
        remote_job_directory=remote_job_directory,
        conda_env=config["o2"]["continuous_features"]["conda_env"],
        o2_username=recording_row.username,
        o2_server="login.o2.rc.hms.harvard.edu",
        job_params=params,
        o2_n_cpus=config["o2"]["continuous_features"]["o2_n_cpus"],
        o2_memory=config["o2"]["continuous_features"]["o2_memory"],
        o2_time_limit=duration_requested,
        o2_queue=config["o2"]["continuous_features"]["o2_queue"],
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
    from multicamera_airflow_pipeline.tim_240731.keypoints.continuous_variables import ContinuousVariables # run rigid alignment
    continuous_features_estimator = ContinuousVariables(
        continuous_features_output_directory= params["continuous_features_output_directory"],
        coordinates_egocentric_filename = params["coordinates_egocentric_filename"],
        coordinates_arena_filename = params["coordinates_arena_filename"],
        framerate = params["samplerate"],
        **config["continuous_features"]
    )
    continuous_features_estimator.run()


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
    if check_continuous_features_completion(continuous_features_output_directory):
        logger.info("continuous_features completed successfully")
    else:
        raise ValueError("continuous_features did not complete successfully.")
