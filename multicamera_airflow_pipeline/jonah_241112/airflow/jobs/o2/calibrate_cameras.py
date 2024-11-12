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


def check_calibration_completion(output_directory_camera_calibration):
    if (output_directory_camera_calibration / "gimbal" / "camera_params.h5").exists():
        return True
    else:
        return False


def calibrate_cameras(
    recording_row,
    job_directory,
    output_directory,
    config_file,
):
    logger.info("Starting camera calibration airflow function")
    logger.info(f"output_directory: {output_directory}")
    # load config
    config_file = Path(config_file)
    config = yaml.safe_load(open(config_file, "r"))

    # where the video data is located
    recording_directory = (
        Path(recording_row.calibration_location_on_o2) / recording_row.calibration_id
    )

    assert (
        recording_directory.exists()
    ), f"Recording directory {recording_directory} does not exist"

    # where to save output
    output_directory_camera_calibration = (
        output_directory / "camera_calibration" / recording_row.calibration_id
    )
    output_directory_camera_calibration.mkdir(parents=True, exist_ok=True)
    current_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    remote_job_directory = job_directory / current_datetime_str

    # check if sync successfully completed
    logger.info("Checking for calibration completion")
    if config["camera_calibration"]["recompute_completed"] == False:
        if check_calibration_completion(output_directory_camera_calibration):
            logger.info("Calibration completed, quitting")
            return
        else:
            logger.info("Calibration incomplete, starting")

    params = {
        "calibration_video_directory": recording_directory.as_posix(),
        "output_directory_camera_calibration": output_directory_camera_calibration.as_posix(),
    }

    # create the job runner
    runner = O2Runner(
        job_name_prefix=f"{recording_row.calibration_id}_calibration",
        remote_job_directory=remote_job_directory,
        conda_env=config["o2"]["camera_calibration"]["conda_env"],
        o2_username=recording_row.username,
        o2_server="login.o2.rc.hms.harvard.edu",
        job_params=params,
        o2_n_cpus=config["o2"]["camera_calibration"]["o2_n_cpus"],
        o2_memory=config["o2"]["camera_calibration"]["o2_memory"],
        o2_time_limit=config["o2"]["camera_calibration"]["o2_time_limit"],
        o2_queue=config["o2"]["camera_calibration"]["o2_queue"],
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
    from multicamera_airflow_pipeline.tim_240731.calibration import Calibrator 
    camera_calibrator = Calibrator(
        calibration_video_directory = params["calibration_video_directory"],
        calibration_output_directory = params["output_directory_camera_calibration"],
        **config["camera_calibration"]
    )
    camera_calibrator.run()
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
    if check_calibration_completion(output_directory_camera_calibration):
        logger.info("camera calibration completed successfully")
    else:
        raise ValueError("camera calibration did not complete successfully.")
