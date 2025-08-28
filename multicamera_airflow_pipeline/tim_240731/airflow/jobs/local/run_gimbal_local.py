from datetime import datetime, timedelta
import pandas as pd
import requests
from io import BytesIO
from pathlib import Path
from multicamera_airflow_pipeline.tim_240731.interface.local import LocalRunner
import textwrap
import time
import yaml
import socket
import subprocess
import logging

# Initialize logger
t_logging = logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

hostname = socket.gethostname()


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


def get_gpu_memory_usage(gpu_id):
    """
    Returns the memory usage of a specified GPU.
    """
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            str(gpu_id),
        ],
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def get_cuda_visible_device_with_lowest_memory():
    """
    Finds the GPU with the lowest memory usage.
    """
    # Get the number of GPUs
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    gpus = result.stdout.strip().split("\n")
    num_gpus = len(gpus)

    lowest_memory_usage = None
    best_gpu = None

    for i in range(num_gpus):
        mem_usage = get_gpu_memory_usage(i)
        if lowest_memory_usage is None or mem_usage < lowest_memory_usage:
            lowest_memory_usage = mem_usage
            best_gpu = i

    return best_gpu


def run_gimbal_local(
    recording_row,
    job_directory,
    output_directory,
    config_file,
):
    # load config
    config_file = Path(config_file)
    config = yaml.safe_load(open(config_file, "r"))

    # where the calibration data is located
    recording_directory = (
        Path(recording_row.calibration_location_on_o2) / recording_row.calibration_id
    )

    # where to save gimbal outputs
    gimbal_output_directory = output_directory / "gimbal" / recording_row.video_recording_id
    logger.info(f"Creating output directory: {gimbal_output_directory}")
    gimbal_output_directory.mkdir(parents=True, exist_ok=True)
    current_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    current_job_directory = job_directory / current_datetime_str

    # skip if already done
    if config["gimbal"]["recompute_completed"] is False:
        if check_gimbal_completion(gimbal_output_directory):
            logger.info("gimbal completed, quitting")
            return
        else:
            logger.info("gimbal incomplete, starting")

    # ensure calibration and triangulation data exist
    calibration_folder = (
        output_directory
        / "camera_calibration"
        / recording_row.calibration_id
        / "jarvis"
        / "CalibrationParameters"
    )
    assert calibration_folder.exists(), f"Calibration folder {calibration_folder} not found"

    predictions_3d_directory = (
        output_directory / "triangulation" / recording_row.video_recording_id
    )
    assert (
        predictions_3d_directory.exists()
    ), f"Triangulation outputs {predictions_3d_directory} not found"

    # prepare parameters
    framerate = recording_row.samplerate
    params = {
        "gimbal_output_directory": gimbal_output_directory.as_posix(),
        "calibration_folder": calibration_folder.as_posix(),
        "predictions_3d_directory": predictions_3d_directory.as_posix(),
        "samplerate": framerate,
    }

    # choose conda environment and GPU
    conda_env = Path(config["gimbal_local"]["conda_env"])
    gpu_to_use = get_cuda_visible_device_with_lowest_memory()
    logger.info(f"Using GPU {gpu_to_use}")

    # create local runner
    runner = LocalRunner(
        job_name_prefix=f"{recording_row.video_recording_id}_gimbal",
        job_directory=current_job_directory,
        conda_env=conda_env,
        job_params=params,
        gpu_to_use=gpu_to_use,
    )

    # build the Python script to run
    runner.python_script = textwrap.dedent(
        f"""
        # load params
        import yaml
        params_file = "{runner.job_directory / f"{runner.job_name}.params.yaml"}"
        config_file = "{config_file.as_posix()}"
        import os; os.umask(0o002)
        params = yaml.safe_load(open(params_file, 'r'))
        config = yaml.safe_load(open(config_file, 'r'))

        from multicamera_airflow_pipeline.tim_240731.keypoints.train_gimbal import GimbalTrainer
        from multicamera_airflow_pipeline.tim_240731.keypoints.inference_gimbal import GimbalInferencer

        # train gimbal model
        gimbal_trainer = GimbalTrainer(
            gimbal_output_directory=params["gimbal_output_directory"],
            calibration_folder=params["calibration_folder"],
            predictions_3d_directory=params["predictions_3d_directory"],
            samplerate=params["samplerate"],
            **config["gimbal"]["train"]
        )
        gimbal_trainer.run()

        # run inference
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

    # verify completion
    if check_gimbal_completion(gimbal_output_directory):
        logger.info("gimbal completed successfully")
    else:
        # clean up logs on failure
        if gimbal_output_directory.exists():
            for f in gimbal_output_directory.glob("*.log"):
                if f.name == "completed.log":
                    continue
                f.unlink()
        raise ValueError("gimbal did not complete successfully.")
