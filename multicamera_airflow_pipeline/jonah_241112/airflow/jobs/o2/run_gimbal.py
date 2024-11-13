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


def check_gimbal_completion(gimbal_output_directory):
    return (gimbal_output_directory / "completed.log").exists()


def run_gimbal(
    recording_row,
    job_directory,
    output_directory,
    config_file,
):
    # training and inference
    # load config
    config_file = Path(config_file)
    config = yaml.safe_load(open(config_file, "r"))

    # where the video data is located
    recording_directory = (
        Path(recording_row.calibration_location_on_o2) / recording_row.calibration_id
    )

    # where to save output
    gimbal_output_directory = output_directory / "gimbal" / recording_row.video_recording_id
    gimbal_output_directory.mkdir(parents=True, exist_ok=True)
    current_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    remote_job_directory = job_directory / current_datetime_str

    # check if sync successfully completed
    if config["gimbal"]["recompute_completed"] == False:
        if check_gimbal_completion(gimbal_output_directory):
            logger.info("gimbal completed, quitting")
            return
        else:
            logger.info("gimbal incomplete, starting")

    calibration_folder = (
        output_directory
        / "camera_calibration"
        / recording_row.calibration_id
        / "jarvis"
        / "CalibrationParameters"
    )

    assert calibration_folder.exists()

    predictions_3d_directory = (
        output_directory / "triangulation" / recording_row.video_recording_id
    )

    assert predictions_3d_directory.exists()

    duration_requested = convert_minutes_to_hms(
        recording_row.duration_m * config["o2"]["gimbal"]["o2_runtime_multiplier"]
    )
    duration_requested

    framerate = recording_row.samplerate

    params = {
        "gimbal_output_directory": gimbal_output_directory.as_posix(),
        "calibration_folder": calibration_folder.as_posix(),
        "predictions_3d_directory": predictions_3d_directory.as_posix(),
        "samplerate": framerate,
    }

    # create the job runner
    runner = O2Runner(
        job_name_prefix=f"{recording_row.video_recording_id}_gimbal",
        remote_job_directory=remote_job_directory,
        conda_env=config["o2"]["gimbal"]["conda_env"],
        o2_username=recording_row.username,
        o2_server="login.o2.rc.hms.harvard.edu",
        job_params=params,
        o2_n_cpus=config["o2"]["gimbal"]["o2_n_cpus"],
        o2_memory=config["o2"]["gimbal"]["o2_memory"],
        o2_time_limit=duration_requested,
        o2_queue=config["o2"]["gimbal"]["o2_queue"],
        o2_exclude=config["o2"]["gimbal"]["o2_exclude"],
        o2_qos=config["o2"]["gimbal"]["o2_qos"],
        o2_gres=config["o2"]["gimbal"]["o2_gres"],
        modules_to_load=["gcc/9.2.0", "cuda/12.1"],
    )

    runner.python_script = textwrap.dedent(
        f"""
    # load params
    import yaml
    params_file = "{runner.remote_job_directory / f"{runner.job_name}.params.yaml"}"
    config_file = "{config_file.as_posix()}"

    params = yaml.safe_load(open(params_file, 'r'))
    config = yaml.safe_load(open(config_file, 'r'))

    from multicamera_airflow_pipeline.jonah_241112.keypoints.train_gimbal import GimbalTrainer 
    from multicamera_airflow_pipeline.jonah_241112.keypoints.inference_gimbal import GimbalInferencer 
    # train gimbal
    gimbal_trainer = GimbalTrainer(
        gimbal_output_directory=params["gimbal_output_directory"],
        calibration_folder=params["calibration_folder"],
        predictions_3d_directory=params["predictions_3d_directory"],
        samplerate=params["samplerate"],
        **config["gimbal"]["train"]
    )
    gimbal_trainer.run()

    # inference gimbal
    gimbal_inferencer = GimbalInferencer(
        gimbal_output_directory=params["gimbal_output_directory"],
        calibration_folder=params["calibration_folder"],
        predictions_3d_directory=params["predictions_3d_directory"],
        **config["gimbal"]["inference"]
    )
    gimbal_inferencer.run()

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
    if check_gimbal_completion(gimbal_output_directory):
        logger.info("gimbal completed successfully")
    else:
        raise ValueError("gimbal did not complete successfully.")
