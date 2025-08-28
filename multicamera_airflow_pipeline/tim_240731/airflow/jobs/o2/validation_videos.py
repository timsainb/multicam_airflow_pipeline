from datetime import datetime
import logging
from pathlib import Path
import textwrap
import time
import os
import yaml

from multicamera_airflow_pipeline.tim_240731.interface.o2 import O2Runner

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


def check_keypoint_videos_completion(output_directory_val_vids):
    return (output_directory_val_vids / "keypoint_videos_completed.log").exists()


def validation_videos(
    recording_row,
    job_directory,
    output_directory,
    config_file,
):
    # Load config
    config_file = Path(config_file)
    config = yaml.safe_load(open(config_file, "r"))

    # Set where video data is located
    raw_video_directory = (
        Path(recording_row.video_location_on_o2) / recording_row.video_recording_id
    )

    # Where to save output
    output_directory_val_vids = (
        output_directory / "keypoint_validation_videos" / recording_row.video_recording_id
    )
    output_directory_val_vids.mkdir(parents=True, exist_ok=True)
    current_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    remote_job_directory = job_directory / current_datetime_str

    # check if sync successfully completed
    if not config["validation_videos"]["recompute_completed"]:
        if check_keypoint_videos_completion(output_directory_val_vids):
            logger.info("validation_videos completed, quitting")
            return
        else:
            logger.info("validation_videos incomplete, starting")

    predictions_2d_directory = (
        output_directory / "2D_predictions" / recording_row.video_recording_id
    )
    predictions_triang_directory = (
        output_directory / "triangulation" / recording_row.video_recording_id
    )

    camera_calibration_directory = (
        output_directory
        / "camera_calibration"
        / recording_row.calibration_id
        / "jarvis"
        / "CalibrationParameters"
    )
    assert camera_calibration_directory.exists()

    params = {
        "predictions_2d_directory": predictions_2d_directory.as_posix(),
        "predictions_triang_directory": predictions_triang_directory.as_posix(),
        "camera_calibration_directory": camera_calibration_directory.as_posix(),
        "raw_video_directory": raw_video_directory.as_posix(),
        "output_directory_keypoint_vids": output_directory_val_vids.as_posix(),
    }

    # create the job runner
    runner = O2Runner(
        job_name_prefix=f"{recording_row.video_recording_id}_validation_videos",
        remote_job_directory=remote_job_directory,
        conda_env=config["o2"]["validation_videos"]["conda_env"],
        o2_username=recording_row.username,
        o2_server="login.o2.rc.hms.harvard.edu",
        job_params=params,
        o2_n_cpus=config["o2"]["validation_videos"]["o2_n_cpus"],
        o2_memory=config["o2"]["validation_videos"]["o2_memory"],
        o2_time_limit=config["o2"]["validation_videos"]["o2_time_limit"],
        o2_queue=config["o2"]["validation_videos"]["o2_queue"],
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
    from multicamera_airflow_pipeline.tim_240731.keypoints.validation_videos import KeypointVideoCreator 
    creator = KeypointVideoCreator(
        predictions_2d_directory = params['predictions_2d_directory'],
        predictions_triang_directory = params['predictions_triang_directory'],
        raw_video_directory = params['raw_video_directory'],
        output_directory_keypoint_vids = params['output_directory_keypoint_vids'],
        camera_calibration_directory = params['camera_calibration_directory'],
        **config["validation_videos"]
    )
    creator.run()
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
    if check_keypoint_videos_completion(output_directory_val_vids):
        logger.info("validation_videos completed successfully")
    else:
        raise ValueError("validation_videos did not complete successfully.")
