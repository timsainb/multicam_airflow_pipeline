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
import os

# Initialize logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

hostname = socket.gethostname()


def convert_minutes_to_hms(minutes_float):
    # Convert minutes to total seconds
    total_seconds = int(minutes_float * 60)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def check_completion(kpms_output_directory):
    return (kpms_output_directory / "kpms_completed.log").exists()


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
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    gpus = result.stdout.strip().split("\n")
    lowest_memory_usage = None
    best_gpu = None
    for i in range(len(gpus)):
        mem = get_gpu_memory_usage(i)
        if lowest_memory_usage is None or mem < lowest_memory_usage:
            lowest_memory_usage = mem
            best_gpu = i
    return best_gpu


def run_kpms_local(
    recording_row,
    job_directory,
    output_directory,
    config_file,
):
    # load config
    output_directory = Path(output_directory)
    job_directory = Path(job_directory)
    config_file = Path(config_file)
    config = yaml.safe_load(open(config_file, "r"))

    # output path
    kpms_output_directory = (
        output_directory
        / "kpms"
        / config["kpms"]["syllable_id"]
        / recording_row.video_recording_id
    )
    logger.info(f"Creating output directory: {kpms_output_directory}")
    kpms_output_directory.mkdir(parents=True, exist_ok=True)
    os.chmod(kpms_output_directory.as_posix(), 0o2775)

    # job directory
    current_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    current_job_directory = job_directory / current_datetime_str

    # skip if already done
    if config["kpms"]["recompute_completed"] is False:
        if check_completion(kpms_output_directory):
            logger.info("kpms completed, quitting")
            return
        else:
            logger.info("kpms incomplete, starting")

    # prepare inputs
    framerate = recording_row.samplerate
    align_dir = (
        output_directory / "egocentric_alignment" / recording_row.video_recording_id / "rigid"
    )
    egocentric_alignment_file = list(align_dir.glob("egocentric_alignment_rigid.*.mmap"))[0]

    params = {
        "kpms_output_directory": kpms_output_directory.as_posix(),
        "egocentric_alignment_file": egocentric_alignment_file.as_posix(),
        "samplerate_hz_recording": framerate,
    }

    # conda env and GPU
    conda_env = Path(config["kpms_local"]["conda_env"])
    gpu_to_use = get_cuda_visible_device_with_lowest_memory()
    logger.info(f"Using GPU {gpu_to_use}")

    # create runner
    runner = LocalRunner(
        job_name_prefix=f"{recording_row.video_recording_id}_kpms",
        job_directory=current_job_directory,
        conda_env=conda_env,
        job_params=params,
        gpu_to_use=gpu_to_use,
    )

    # generate script
    runner.python_script = textwrap.dedent(
        f"""
        # load params
        import yaml
        params_file = "{runner.job_directory / f"{runner.job_name}.params.yaml"}"
        config_file = "{config_file.as_posix()}"
        import os; os.umask(0o002)
        params = yaml.safe_load(open(params_file, 'r'))
        config = yaml.safe_load(open(config_file, 'r'))

        from multicamera_airflow_pipeline.tim_240731.keypoints.kpms_inference import KPMSInferencer

        kpms_inf = KPMSInferencer(
            egocentric_alignment_file=params["egocentric_alignment_file"],
            kpms_output_directory=params["kpms_output_directory"],
            samplerate_hz_recording=params["samplerate_hz_recording"],
            **config["kpms"]
        )
        kpms_inf.run()
        """
    )

    print(runner.python_script)
    runner.run()

    # verify
    if check_completion(kpms_output_directory):
        logger.info("kpms completed successfully")
    else:
        # cleanup logs
        for f in kpms_output_directory.glob("*.log"):
            if f.name == "kpms_completed.log":
                continue
            f.unlink()
        raise ValueError("kpms did not complete successfully.")
