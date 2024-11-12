import sys
import logging

logger = logging.getLogger(__name__)
logger.info(f"Python interpreter binary location: {sys.executable}")

from pathlib import Path
from datetime import datetime
import yaml
import subprocess

class LocalRunner:
    """Class to run a job locally, with a specific conda environment"""

    def __init__(self, job_name_prefix, job_directory, conda_env, job_params={}, gpu_to_use=None):

        self.job_name_prefix = job_name_prefix
        # determine a job id as the current timestamp
        self.job_datetime = datetime.now()
        self.job_datetime_str = self.job_datetime.strftime("%y-%m-%d-%G-%M-%S-%f")
        self.job_name = f"{self.job_name_prefix}_{self.job_datetime_str}"
        self.job_directory = Path(job_directory)
        self.python_script = None
        self.conda_env = conda_env
        self.job_params = job_params
        self.gpu_to_use = gpu_to_use

        # where the script will be written on the remote server
        self.job_script_loc = self.job_directory / f"{self.job_name}.sh"
        self.python_script_loc = self.job_directory / f"{self.job_name}.py"
        self.params_loc = self.job_directory / f"{self.job_name}.params.yaml"
        self.output_log = self.job_directory / f"{self.job_name}.log"

    def run(self):
        logger.info(f"Creating remote job directory: {self.job_directory}")
        self.job_directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"Writing job script, python script, params")
        # write the job script
        self.write_job_script()
        # write the python script
        self.write_python_script()
        # write the params
        self.write_params()

        logger.info(f"Submitting job")
        self.submit()

    def write_job_script(self):
        job_script = f"#!/usr/bin/env bash\n"
        if self.gpu_to_use is not None:
            logger.info(f"Using GPU {self.gpu_to_use}")
            # set CUDA_DEVICE_ORDER to PCI_BUS_ID and CUDA_VISIBLE_DEVICES to the GPU to use
            job_script += "export CUDA_DEVICE_ORDER=PCI_BUS_ID\n"
            job_script += f"export CUDA_VISIBLE_DEVICES={self.gpu_to_use}\n"
        job_script += "source $(conda info --base)/etc/profile.d/conda.sh;\n"
        job_script += f"conda activate {self.conda_env}\n\n"
        job_script += f"python {self.python_script_loc}\n"
        with open(self.job_script_loc, "w") as file:
            file.write(job_script)

        # make the job script executable
        self.job_script_loc.chmod(0o755)

    def write_python_script(self):
        if self.python_script is None:
            raise ValueError("No python script provided")
        with open(self.python_script_loc, "w") as file:
            file.write(self.python_script)

    def write_params(self):
        with open(self.params_loc, "w") as file:
            yaml.dump(self.job_params, file)

    def submit(self):

        # subprocess log should output to self.output_log
        process = subprocess.Popen(
            self.job_script_loc.as_posix(),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            executable="/bin/bash",
        )
        # initialize output log
        with open(self.output_log, "w") as file:
            file.write("")

        # Read output line by line as it is produced
        for line in process.stdout:
            logger.info(line)
            # write to log file
            with open(self.output_log, "a") as file:
                file.write(line)

        # Wait for the subprocess to finish
        process.wait()
