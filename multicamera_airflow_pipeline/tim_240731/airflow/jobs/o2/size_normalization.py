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


def check_size_normalization_completion(output_directory_size_normalization):
    return (output_directory_size_normalization / "completed.log").exists()


def size_normalization(recording_row, job_directory, output_directory, config_file):

    # load config
    config_file = Path(config_file)
    config = yaml.safe_load(open(config_file, "r"))

    # where the video data is located
    recording_directory = (
        Path(recording_row.calibration_location_on_o2) / recording_row.calibration_id
    )

    # check if an id
    if "size_normalization_id" in config["size_normalization"]:
        size_normalization_id = config["size_normalization"]["size_normalization_id"]
    else:
        size_normalization_id = None

    # where to save output
    if size_normalization_id is None:
        output_directory_size_normalization = (
            output_directory / "size_normalization" / recording_row.video_recording_id
        )
    else:
        output_directory_size_normalization = (
            output_directory
            / "size_normalization"
            / size_normalization_id
            / recording_row.video_recording_id
        )
    logger.info(f"Creating output directory: {output_directory_size_normalization}")
    output_directory_size_normalization.mkdir(parents=True, exist_ok=True)
    os.chmod(output_directory_size_normalization.as_posix(), 0o2775)
    current_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    remote_job_directory = job_directory / current_datetime_str

    # check if sync successfully completed
    if config["size_normalization"]["recompute_completed"] == False:
        if check_size_normalization_completion(output_directory_size_normalization):
            logger.info("size_normalization completed, quitting")
            return
        else:
            logger.info("size_normalization incomplete, starting")

    predictions_3d_file = list(
        (output_directory / "gimbal" / recording_row.video_recording_id).glob("gimbal.*.mmap")
    )[0]

    assert predictions_3d_file.exists()

    duration_requested = convert_minutes_to_hms(
        recording_row.duration_m * config["o2"]["size_normalization"]["o2_runtime_multiplier"]
    )
    duration_requested

    params = {
        "size_norm_output_directory": output_directory_size_normalization.as_posix(),
        "predictions_3d_file": predictions_3d_file.as_posix(),
    }

    # create the job runner
    runner = O2Runner(
        job_name_prefix=f"{recording_row.video_recording_id}_p",
        remote_job_directory=remote_job_directory,
        conda_env=config["o2"]["size_normalization"]["conda_env"],
        o2_username=recording_row.username,
        o2_server="login.o2.rc.hms.harvard.edu",
        job_params=params,
        o2_n_cpus=config["o2"]["size_normalization"]["o2_n_cpus"],
        o2_memory=config["o2"]["size_normalization"]["o2_memory"],
        o2_time_limit=duration_requested,
        o2_queue=config["o2"]["size_normalization"]["o2_queue"],
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
    from multicamera_airflow_pipeline.tim_240731.keypoints.size_norm import SizeNormalizer 
    size_normalizer = SizeNormalizer(
        size_norm_output_directory = params['size_norm_output_directory'],
        predictions_3d_file = params['predictions_3d_file'],
        **config["size_normalization"]
    )
    size_normalizer.run()
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
    if check_size_normalization_completion(output_directory_size_normalization):
        logger.info("size norm completed successfully")
    else:
        raise ValueError("size norm did not complete successfully.")
