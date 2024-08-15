from datetime import datetime, timedelta
import pandas as pd
import requests
from io import BytesIO
from pathlib import Path
from multicamera_airflow_pipeline.tim_240731.interface.local import LocalRunner
from datetime import datetime
import textwrap
import inspect
import time
import yaml
import socket
import subprocess

hostname = socket.gethostname()
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


def check_2d_completion(output_directory_predictions):
    return (output_directory_predictions / "completed.log").exists()


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
    Finds the GPU with the lowest memory usage and sets CUDA_VISIBLE_DEVICES to that GPU.
    """
    # Get the number of GPUs
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True
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


def predict_2d_local(
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
        Path(recording_row.video_location_on_o2) / recording_row.video_recording_id
    )

    # where to save output
    output_directory_predictions = (
        output_directory / "2D_predictions" / recording_row.video_recording_id
    )
    output_directory_predictions.mkdir(parents=True, exist_ok=True)
    current_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    current_job_directory = job_directory / current_datetime_str

    # check if sync successfully completed
    if config["prediction_2d"]["recompute_completed"] == False:
        if check_2d_completion(output_directory_predictions):
            logger.info("2d prediction completed, quitting")
            return
        else:
            logger.info("2d prediction incomplete, starting")

    # duration of video files (useful if videos are not properly muxxed)
    expected_video_length_frames = (
        recording_row.max_video_duration_m * 60 * recording_row.samplerate
    )

    # where tensorrt compiled models are saved (specific to GPU, so we compute on the fly)
    tensorrt_model_directory = output_directory / "tensorrt" / hostname
    tensorrt_model_directory.mkdir(parents=True, exist_ok=True)

    params = {
        "recording_directory": recording_directory.as_posix(),
        "output_directory_predictions": output_directory_predictions.as_posix(),
        "expected_video_length_frames": expected_video_length_frames,
        "tensorrt_model_directory": tensorrt_model_directory.as_posix(),
    }

    conda_env = Path(config["tensorrt_conversion_local"]["conda_env"])

    # Call the function
    gpu_to_use = get_cuda_visible_device_with_lowest_memory()
    logger.info(f"Using GPU {gpu_to_use}")

    # create the job runner
    runner = LocalRunner(
        job_name_prefix=f"{recording_row.video_recording_id}_2d_predictions",
        job_directory=current_job_directory,
        conda_env=conda_env,
        job_params=params,
        gpu_to_use=gpu_to_use,
    )

    if config["prediction_2d"]["use_tensorrt"]:
        runner.python_script = textwrap.dedent(
            f"""
        # load params
        import yaml
        params_file = "{runner.job_directory / f"{runner.job_name}.params.yaml"}"
        config_file = "{config_file.as_posix()}"

        params = yaml.safe_load(open(params_file, 'r'))
        config = yaml.safe_load(open(config_file, 'r'))
        
        config["tensorrt_conversion"]["conda_env"] = "{conda_env.as_posix()}"
        # convert models to tensorrt
        from multicamera_airflow_pipeline.tim_240731.keypoints.tensorrt import RTMModelConverter
        model_converter = RTMModelConverter(
            tensorrt_output_directory = params["tensorrt_model_directory"],
            is_local=True,
            **config["tensorrt_conversion"]
        )
        model_converter.run()
        
        # run predictions
        from multicamera_airflow_pipeline.tim_240731.keypoints.predict_2D import Inferencer2D
        camera_calibrator = Inferencer2D(
            recording_directory = params["recording_directory"],
            output_directory_predictions = params["output_directory_predictions"],
            expected_video_length_frames = params["expected_video_length_frames"],
            tensorrt_model_directory = params["tensorrt_model_directory"],
            **config["prediction_2d"]
        )
        camera_calibrator.run()
        """
        )
    else:
        runner.python_script = textwrap.dedent(
            f"""
        # load params
        import yaml
        import sys
        print(sys.executable)
        params_file = "{runner.job_directory / f"{runner.job_name}.params.yaml"}"
        config_file = "{config_file.as_posix()}"

        params = yaml.safe_load(open(params_file, 'r'))
        config = yaml.safe_load(open(config_file, 'r'))

        # grab sync cameras function
        from multicamera_airflow_pipeline.tim_240731.keypoints.predict_2D import Inferencer2D
        camera_calibrator = Inferencer2D(
            recording_directory = params["recording_directory"],
            output_directory_predictions = params["output_directory_predictions"],
            expected_video_length_frames = params["expected_video_length_frames"],
            tensorrt_model_directory = params["tensorrt_model_directory"],
            **config["prediction_2d"]
        )
        camera_calibrator.run()
        """
        )

    print(runner.python_script)

    runner.run()

    # check if sync successfully completed
    if check_2d_completion(output_directory_predictions):
        logger.info("2D prediction completed successfully")
    else:
        raise ValueError("2D prediction did not complete successfully.")
