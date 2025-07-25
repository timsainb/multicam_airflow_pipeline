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


def check_completion(kpms_output_directory):
    return (kpms_output_directory / "kpms_completed.log").exists()


def run_kpms(
    recording_row,
    job_directory,
    output_directory,
    config_file,
):

    output_directory = Path(output_directory)
    job_directory = Path(job_directory)
    config_file = Path(config_file)
    config = yaml.safe_load(open(config_file, "r"))

    # where to save output
    kpms_output_directory = (
        output_directory
        / "kpms"
        / config["kpms"]["syllable_id"]
        / recording_row.video_recording_id
    )
    kpms_output_directory.mkdir(parents=True, exist_ok=True)
    current_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    remote_job_directory = job_directory / current_datetime_str

    # check if sync successfully completed
    if config["kpms"]["recompute_completed"] == False:
        if check_completion(kpms_output_directory):
            logger.info("kpms completed, quitting")
            return
        else:
            logger.info("kpms incomplete, starting")

    duration_requested = convert_minutes_to_hms(
        recording_row.duration_m * config["o2"]["kpms"]["o2_runtime_multiplier"]
    )
    duration_requested

    framerate = recording_row.samplerate

    egocentric_alignment_file = list(
        (
            output_directory / "egocentric_alignment" / recording_row.video_recording_id / "rigid"
        ).glob("egocentric_alignment_rigid.*.mmap")
    )[0]

    params = {
        "kpms_output_directory": kpms_output_directory.as_posix(),
        "egocentric_alignment_file": egocentric_alignment_file.as_posix(),
        "samplerate_hz_recording": framerate,
    }

    # create the job runner
    runner = O2Runner(
        job_name_prefix=f"{recording_row.video_recording_id}_kpms",
        remote_job_directory=remote_job_directory,
        conda_env=config["o2"]["kpms"]["conda_env"],
        o2_username=recording_row.username,
        o2_server="login.o2.rc.hms.harvard.edu",
        job_params=params,
        o2_n_cpus=config["o2"]["kpms"]["o2_n_cpus"],
        o2_memory=config["o2"]["kpms"]["o2_memory"],
        o2_time_limit=duration_requested,
        o2_queue=config["o2"]["kpms"]["o2_queue"],
        o2_exclude=config["o2"]["kpms"]["o2_exclude"],
        o2_qos=config["o2"]["kpms"]["o2_qos"],
        o2_gres=config["o2"]["kpms"]["o2_gres"],
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

    from multicamera_airflow_pipeline.tim_240731.keypoints.kpms_inference import KPMSInferencer
    
    kpms_inf = KPMSInferencer(
        egocentric_alignment_file = params["egocentric_alignment_file"],
        kpms_output_directory = params["kpms_output_directory"],
        samplerate_hz_recording = params["samplerate_hz_recording"],
        **config["kpms"]
    )
    kpms_inf.run()



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
    if check_completion(kpms_output_directory):
        logger.info("kpms completed successfully")
    else:
        raise ValueError("kpms did not complete successfully.")
