import sys
import logging

logger = logging.getLogger(__name__)
logger.info(f"Python interpreter binary location: {sys.executable}")

import sys
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import numpy as np
import joblib, os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import re
import subprocess
from pathlib import Path
from datetime import datetime
import tempfile
import shutil
import paramiko
import yaml


class O2Runner:
    """
    A class to run python scripts on O2.
    """

    def __init__(
        self,
        job_name_prefix,
        remote_job_directory,
        conda_env,
        job_params={},
        o2_username="tis697",
        o2_server="login.o2.rc.hms.harvard.edu",
        o2_n_cpus=1,
        o2_memory="16G",
        o2_time_limit="4:00:00",
        o2_queue="short",
        modules_to_load=["gcc/9.2.0"],
        o2_exclude=None,  # "compute-g-16-175,compute-g-16-176,compute-g-16-177,compute-g-16-194,compute-g-16-197"
        o2_qos=None,  # "gpuquad_qos"
        o2_gres=None,  # "gpu:1"
        do_not_submit=False,
    ):
        self.job_name_prefix = job_name_prefix
        self.remote_job_directory = Path(remote_job_directory)
        self.o2_n_cpus = o2_n_cpus
        self.o2_memory = o2_memory
        self.o2_time_limit = o2_time_limit
        self.o2_queue = o2_queue
        self.o2_username = o2_username
        self.o2_server = o2_server
        self.conda_env = conda_env
        self.job_params = job_params
        self.o2_exclude = o2_exclude
        self.o2_qos = o2_qos
        self.o2_gres = o2_gres
        self.modules_to_load = modules_to_load
        self.do_not_submit = do_not_submit  # don't actually submit
        self.slurm_job_id = None

        # determine a job id as the current timestamp
        self.job_datetime = datetime.now()
        self.job_datetime_str = self.job_datetime.strftime("%y-%m-%d-%G-%M-%S-%f")
        self.job_name = f"{self.job_name_prefix}_{self.job_datetime_str}"

        # where the script will be written on the remote server
        self.slurm_script_loc = self.remote_job_directory / f"{self.job_name}.sh"
        self.python_script_loc = self.remote_job_directory / f"{self.job_name}.py"
        self.params_loc = self.remote_job_directory / f"{self.job_name}.params.yaml"
        self.output_log = self.remote_job_directory / f"{self.job_name}.log"
        self.ssh = None

        self.establish_ssh_connection()

    def report_output_log(self):
        # read the output log and write it to the logger

        # check if the log file exists locally
        if not self.output_log.exists():
            return
        else:
            with self.output_log.open("r") as f:
                for line in f:
                    logger.info(line.strip())

    def run(self):
        # create the remote job directory
        logger.info(f"Creating remote job directory: {self.remote_job_directory}")
        # create_folder_on_remote(
        #    remote_path=self.remote_job_directory,
        #    username=self.o2_username,
        #    remote_server=self.o2_server,
        # )
        self.create_folder_on_remote(self.remote_job_directory)

        logger.info(f"Writing job files to remote directory: {self.remote_job_directory}")
        # write the python script
        self.write_python_script()
        # write the slurm script
        self.write_slurm_script()
        # write the params file
        self.write_params_file()
        # submit the job
        logger.info(f"Submitting job: {self.job_name}")
        if self.do_not_submit:
            logger.info("do_not_submit is True, not submitting job.")
        else:
            self.submit()

    def write_slurm_script(self):
        slurm_script = f"#!/usr/bin/env bash\n"
        slurm_script += f"#SBATCH --partition={self.o2_queue}\n"
        slurm_script += f"#SBATCH --job-name={self.job_name}\n"
        slurm_script += f"#SBATCH --cpus-per-task={self.o2_n_cpus}\n"
        slurm_script += f"#SBATCH --mem={self.o2_memory}\n"
        slurm_script += f"#SBATCH --time={self.o2_time_limit}\n"
        slurm_script += f"#SBATCH --output={self.output_log}\n\n"
        if self.o2_exclude is not None:
            slurm_script += f"#SBATCH --exclude={self.o2_exclude}\n"
        if self.o2_qos is not None:
            slurm_script += f"#SBATCH --qos={self.o2_qos}\n"
        if self.o2_gres is not None:
            slurm_script += f"#SBATCH --gres={self.o2_gres}\n"
        slurm_script += f"# Load the required modules\n"
        # slurm_script += f"module load gcc/9.2.0\n\n"
        for modules_to_load in self.modules_to_load:
            slurm_script += f"module load {modules_to_load}\n"
        slurm_script += f"source activate {self.conda_env}\n\n"
        slurm_script += f"python {self.python_script_loc}\n"

        # save the script to tmp, then move it to the correct location remotely
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(slurm_script)
            f.flush()
            # scp_file_to_remote(
            #    local_path=f.name,
            #    remote_path=self.slurm_script_loc,
            #    username=self.o2_username,
            #    remote_server=self.o2_server,
            # )
            self.copy_file_to_remote(local_path=f.name, remote_path=self.slurm_script_loc)

    def write_params_file(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            # save the params dict to a YAML file
            yaml.dump(self.job_params, f)
            # scp_file_to_remote(
            #    local_path=f.name,
            #    remote_path=self.params_loc,
            #    username=self.o2_username,
            #    remote_server=self.o2_server,
            # )
            self.copy_file_to_remote(local_path=f.name, remote_path=self.params_loc)

    def write_python_script(self):

        # save the script locally, then move it to the correct location remotely
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(self.python_script)
            f.flush()
            # scp_file_to_remote(
            #    local_path=f.name,
            #    remote_path=self.python_script_loc,
            #    username=self.o2_username,
            #    remote_server=self.o2_server,
            # )
            self.copy_file_to_remote(local_path=f.name, remote_path=self.python_script_loc)

    def create_folder_on_remote(self, remote_path):
        remote_path = Path(remote_path)
        # using paramiko self.ssh, create the folder on the remote server
        if self.ssh is None:
            raise ConnectionError("SSH connection is not established")
        try:
            logger.info(f"Creating remote directory: {remote_path.as_posix()}")

            # Execute the mkdir command to create the folder on the remote server
            stdin, stdout, stderr = self.ssh.exec_command(f"mkdir -p {remote_path.as_posix()}")

            # Check if there was any error
            error_message = stderr.read().decode().strip()
            if error_message:
                raise Exception(f"Error creating remote directory: {error_message.as_posix()}")

            logger.info(f"Successfully created remote directory: {remote_path.as_posix()}")
        except Exception as e:
            logger.error(f"Exception during directory creation: {str(e)}")
            raise

    def copy_file_to_remote(self, local_path, remote_path):
        local_path = Path(local_path)
        remote_path = Path(remote_path)
        if self.ssh is None:
            raise ConnectionError("SSH connection is not established")

        try:
            # Create an SFTP session from the SSH connection
            sftp = self.ssh.open_sftp()

            logger.info(f"Transferring {local_path} to {self.o2_server}:{remote_path.as_posix()}")
            sftp.put(local_path.as_posix(), remote_path.as_posix())

            logger.info(
                f"Successfully transferred {local_path.as_posix()} to {remote_path.as_posix()}"
            )

            # Close the SFTP session
            sftp.close()
        except Exception as e:
            logger.error(f"Exception during file transfer: {str(e)}")
            raise

    def establish_ssh_connection(self):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(self.o2_server, username=self.o2_username)

    def close_ssh_connection(self):
        self.ssh.close()
        self.ssh = None

    def submit(self):
        expected_slurm_output = "Submitted batch job "
        submit_command = f"sbatch {self.slurm_script_loc}"

        if self.ssh is None:
            raise ConnectionError("SSH connection is not established")

        # self.establish_ssh_connection()
        stdin, stdout, stderr = self.ssh.exec_command(submit_command)
        slurm_output = stdout.read().decode()
        # check if the job was submitted successfully
        if slurm_output[: len(expected_slurm_output)] != expected_slurm_output:
            raise Exception(f"Error submitting job: {slurm_output}")
        # grab job id
        self.slurm_job_id = slurm_output[len(expected_slurm_output) : -1]
        logger.info(f"Job submitted successfully with job id: {self.slurm_job_id}")
        # self.close_ssh_connection()

    def cancel(self):
        if self.ssh is None:
            raise ConnectionError("SSH connection is not established")
        if not self.slurm_job_id:
            logger.warning("No job id set, cannot cancel job")
            return
        cancel_command = f"scancel {self.slurm_job_id}"
        stdin, stdout, stderr = self.ssh.exec_command(cancel_command)
        logger.info(f"Job cancelled: {self.slurm_job_id}")
    
    def check_job_success(self):
        # job success is specific to the job type
        raise NotImplementedError

    def check_job_status(self):

        if self.ssh is None:
            raise ConnectionError("SSH connection is not established")

        if not self.slurm_job_id:
            raise ValueError("slurm_job_id is not set")

        logger.info(f"Checking job status: {self.slurm_job_id}")
        # check_command = f"sacct -j {self.slurm_job_id} --format=State --noheader"
        check_command = f"sacct -j {self.slurm_job_id} --format=JobID,State | grep -E '^[0-9]+ ' | awk '{{print $2}}'"
        # self.establish_ssh_connection()
        stdin, stdout, stderr = self.ssh.exec_command(check_command)
        slurm_output = stdout.read().decode()[:-1]
        job_state = slurm_output
        # self.close_ssh_connection()

        # Add custom handling based on the job state
        if job_state == "COMPLETED":
            logger.info("The job has finished successfully.")
            return True
        elif job_state == "PENDING":
            logger.info("The job is waiting to be scheduled.")
            return False
        elif job_state == "RUNNING":
            logger.info("The job is currently running.")
            return False
        elif job_state == "FAILED":
            logger.info("The job failed.")
            self.report_output_log()
            raise Exception("Job failed.")
        elif job_state == "CANCELLED":
            logger.info("The job was cancelled.")
            self.report_output_log()
            raise Exception("Job failed.")
        elif job_state == "CANCELLED+":
            logger.info("The job was cancelled.")
            self.report_output_log()
            raise Exception("Job failed.")
        elif job_state == "TIMEOUT":
            logger.info("The job has timed out.")
            self.report_output_log()
            raise Exception("Job failed.")
        elif job_state == "NODE_FAIL":
            logger.info("The job terminated due to node failure.")
            self.report_output_log()
            raise Exception("Job failed.")
        elif job_state == "OUT_OF_MEMORY":
            logger.info("The job was terminated due to exceeding memory limits.")
            self.report_output_log()
            raise Exception("Job failed.")
        elif job_state == "COMPLETING":
            logger.info("The job is in the process of completing.")
            return False
        elif job_state == "REQUEUED":
            logger.info("The job was requeued.")
            return False
        elif job_state == "RESIZING":
            logger.info("The job is being resized.")
            return False
        elif job_state == "SUSPENDED":
            logger.info("The job is suspended.")
            self.report_output_log()
            raise Exception("Job failed.")
        elif job_state == "SPECIAL_EXIT":
            logger.info("The job terminated with a special exit state.")
            self.report_output_log()
            raise Exception("Job failed.")
        else:
            logger.info(f"Unknown job state: {job_state}")
            return False


def create_folder_on_remote(
    remote_path, username="tis697", remote_server="login.o2.rc.hms.harvard.edu"
):
    command = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        f"{username}@{remote_server}",
        f"mkdir -p {remote_path}",
    ]
    logger.info("Creating folder: " + " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Error creating folder: {result.stderr}")


def scp_file_to_remote(
    local_path, remote_path, username="tis697", remote_server="login.o2.rc.hms.harvard.edu"
):
    # Configure SCP command with optimized options
    command = [
        "scp",
        "-T",  # Disable pseudo-terminal allocation
        "-o",
        "Compression=no",  # Disable compression
        "-o",
        "IPQoS=throughput",  # Optimize for throughput
        local_path,
        f"{username}@{remote_server}:{remote_path}",
    ]

    # Execute the SCP command
    logger.info("Sending file to remote: " + " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)

    # Check for errors in the SCP operation and raise an exception if any
    if result.returncode != 0:
        raise Exception(f"Error transferring file to remote: {result.stderr}")
