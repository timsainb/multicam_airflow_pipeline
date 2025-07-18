from datetime import datetime, timedelta
import pandas as pd
import requests
from io import BytesIO
from pathlib import Path
from multicamera_airflow_pipeline.jonah_241112.interface.o2 import O2Runner
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


def check_arena_alignment_completion(arena_alignment_output_directory):
    return (arena_alignment_output_directory / "completed.txt").exists()


def arena_alignment(
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
    arena_alignment_output_directory = (
        output_directory / "arena_alignment" / recording_row.video_recording_id
    )
    arena_alignment_output_directory.mkdir(parents=True, exist_ok=True)
    current_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    remote_job_directory = job_directory / current_datetime_str

    # check if sync successfully completed
    if config["arena_alignment"]["recompute_completed"] == False:
        if check_arena_alignment_completion(arena_alignment_output_directory):
            logger.info("arena_alignment completed, quitting")
            return
        else:
            logger.info("arena_alignment not complete, starting")

    predictions_3d_file = list(
        (output_directory / "size_normalization" / recording_row.video_recording_id).glob(
            "size_norm.*.mmap"
        )
    )[0]

    assert predictions_3d_file.exists()

    duration_requested = convert_minutes_to_hms(
        recording_row.duration_m * config["o2"]["arena_alignment"]["o2_runtime_multiplier"]
    )
    duration_requested

    params = {
        "predictions_3d_file": predictions_3d_file.as_posix(),
        "arena_alignment_output_directory": arena_alignment_output_directory.as_posix(),
    }

    # create the job runner
    runner = O2Runner(
        job_name_prefix=f"{recording_row.video_recording_id}_arena_alignment",
        remote_job_directory=remote_job_directory,
        conda_env=config["o2"]["arena_alignment"]["conda_env"],
        o2_username=recording_row.username,
        o2_server="login.o2.rc.hms.harvard.edu",
        job_params=params,
        o2_n_cpus=config["o2"]["arena_alignment"]["o2_n_cpus"],
        o2_memory=config["o2"]["arena_alignment"]["o2_memory"],
        o2_time_limit=duration_requested,
        o2_queue=config["o2"]["arena_alignment"]["o2_queue"],
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
    from multicamera_airflow_pipeline.jonah_241112.keypoints.alignment.arena import ArenaAligner 
    arena_aligner = ArenaAligner(
        predictions_3d_file = params['predictions_3d_file'],
        arena_alignment_output_directory = params['arena_alignment_output_directory'],
        **config["arena_alignment"]
    )
    arena_aligner.run()
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
    if check_arena_alignment_completion(arena_alignment_output_directory):
        logger.info("arena alignment completed successfully")
    else:
        raise ValueError("arena alignment did not complete successfully.")
