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


def check_2d_completion(output_directory_predictions):
    return (output_directory_predictions / "completed.log").exists()


def predict_2d(
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

    assert (
        recording_directory.exists()
    ), f"Recording directory {recording_directory} does not exist"

    # where to save output
    output_directory_predictions = (
        output_directory / "2D_predictions" / recording_row.video_recording_id
    )
    output_directory_predictions.mkdir(parents=True, exist_ok=True)
    current_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    remote_job_directory = job_directory / current_datetime_str

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
    tensorrt_model_directory = output_directory / "tensorrt"

    params = {
        "recording_directory": recording_directory.as_posix(),
        "output_directory_predictions": output_directory_predictions.as_posix(),
        "expected_video_length_frames": int(expected_video_length_frames),
        "tensorrt_model_directory": tensorrt_model_directory.as_posix(),
    }

    duration_requested = convert_minutes_to_hms(
        recording_row.duration_m * config["o2"]["prediction_2d"]["o2_runtime_multiplier"]
    )

    # create the job runner
    runner = O2Runner(
        job_name_prefix=f"{recording_row.video_recording_id}_2d_predictions",
        remote_job_directory=remote_job_directory,
        conda_env=config["o2"]["prediction_2d"]["conda_env"],
        o2_username=recording_row.username,
        o2_server="login.o2.rc.hms.harvard.edu",
        job_params=params,
        o2_n_cpus=config["o2"]["prediction_2d"]["o2_n_cpus"],
        o2_memory=config["o2"]["prediction_2d"]["o2_memory"],
        o2_time_limit=duration_requested,
        o2_queue=config["o2"]["prediction_2d"]["o2_queue"],
        o2_exclude=config["o2"]["prediction_2d"]["o2_exclude"],
        o2_qos=config["o2"]["prediction_2d"]["o2_qos"],
        o2_gres=config["o2"]["prediction_2d"]["o2_gres"],
        modules_to_load=["gcc/9.2.0", "cuda/11.7"],
    )

    if config["prediction_2d"]["use_tensorrt"]:
        runner.python_script = textwrap.dedent(
            f"""
        # load params
        import yaml
        params_file = "{runner.remote_job_directory / f"{runner.job_name}.params.yaml"}"
        config_file = "{config_file.as_posix()}"

        params = yaml.safe_load(open(params_file, 'r'))
        config = yaml.safe_load(open(config_file, 'r'))

        # covert models to tensorrt
        from multicamera_airflow_pipeline.tim_240731.keypoints.tensorrt import RTMModelConverter
        model_converter = RTMModelConverter(
            tensorrt_output_directory = params["tensorrt_model_directory"],
            **config["tensorrt_conversion"]
        )
        model_converter.run()

        # grab sync cameras function
        from multicamera_airflow_pipeline.tim_240731.keypoints.predict_2D import Inferencer2D
        inferencer = Inferencer2D(
            recording_directory = params["recording_directory"],
            output_directory_predictions = params["output_directory_predictions"],
            expected_video_length_frames = params["expected_video_length_frames"],
            tensorrt_model_directory = params["tensorrt_model_directory"],
            **config["prediction_2d"]
        )
        inferencer.run()
        """
        )
    else:
        runner.python_script = textwrap.dedent(
            f"""
        # load params
        import yaml
        params_file = "{runner.remote_job_directory / f"{runner.job_name}.params.yaml"}"
        config_file = "{config_file.as_posix()}"

        params = yaml.safe_load(open(params_file, 'r'))
        config = yaml.safe_load(open(config_file, 'r'))

        # grab sync cameras function
        from multicamera_airflow_pipeline.tim_240731.keypoints.predict_2D import Inferencer2D
        inferencer = Inferencer2D(
            recording_directory = params["recording_directory"],
            output_directory_predictions = params["output_directory_predictions"],
            expected_video_length_frames = params["expected_video_length_frames"],
            tensorrt_model_directory = params["tensorrt_model_directory"],
            **config["prediction_2d"]
        )
        inferencer.run()
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
        if check_2d_completion(output_directory_predictions):
            logger.info("2D prediction already completed successfully, quitting")
            # cancels the current job
            runner.cancel()
            break
        time.sleep(60)

    # check if sync successfully completed
    if check_2d_completion(output_directory_predictions):
        logger.info("2D prediction completed successfully")
    else:
        # if output_directory_predictions exists, remove all .log files in that folder
        if output_directory_predictions.exists():
            for file in output_directory_predictions.glob("*.log"):
                # if the file is completed.log, don't remove it
                if file.name == "completed.log":
                    continue
                file.unlink()
        raise ValueError("2D prediction did not complete successfully.")

    # TODO: Update this o2 version of predict local that deals with tensorrt failures the same was as the local version does
