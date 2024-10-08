# check if a tensorrt model exists for pose and detection
#
import torch
from pathlib import Path
from mmpose.registry import DATASETS
from mmpose.datasets.datasets.base import BaseCocoStyleDataset
import subprocess
import tempfile
import os
import shutil
import sys
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.info(f"Python interpreter binary location: {sys.executable}")


class RTMModelConverter:
    """
    Class to convert pose and detection models to tensorrt.
    """

    def __init__(
        self,
        path_to_rmpose_config,
        path_to_rmpose_checkpoint,
        path_to_rtmdetection_config,
        path_to_rtmdetection_checkpoint,
        tensorrt_output_directory,
        rtmdetection_model_name,
        rtmpose_model_name,
        skeleton_py_file="/n/groups/datta/tim_sainburg/projects/multicamera_airflow_pipeline/multicamera_airflow_pipeline/tim_240731/skeletons/sainburg25pt.py",
        conda_env="/n/groups/datta/tim_sainburg/conda_envs/mmdeploy",
        path_to_mmdeploy="/n/groups/datta/tim_sainburg/projects/mmdeploy/",
        path_to_demo_image_detection="/n/groups/datta/tim_sainburg/projects/24-01-05-multicamera_keypoints_mm2d/example_data/test_mouse.png",
        path_to_demo_image_pose="/n/groups/datta/tim_sainburg/projects/24-01-05-multicamera_keypoints_mm2d/example_data/test_mouse_cropped.png",
        path_to_mmdetection_config="/n/groups/datta/tim_sainburg/projects/mmdeploy/configs/mmdet/detection/detection_tensorrt_static-320x320.py",
        path_to_mmpose_config="/n/groups/datta/tim_sainburg/projects/mmdeploy/configs/mmpose/pose-detection_simcc_tensorrt_dynamic-256x256.py",
        is_local=False,
    ):
        self.path_to_rmpose_config = Path(path_to_rmpose_config)
        self.path_to_rmpose_checkpoint = Path(path_to_rmpose_checkpoint)
        self.path_to_rtmdetection_config = Path(path_to_rtmdetection_config)
        self.path_to_rtmdetection_checkpoint = Path(path_to_rtmdetection_checkpoint)
        self.path_to_mmdeploy = Path(path_to_mmdeploy)
        self.path_to_demo_image_detection = Path(path_to_demo_image_detection)
        self.path_to_demo_image_pose = Path(path_to_demo_image_pose)
        self.tensorrt_output_directory = Path(tensorrt_output_directory)
        self.rtmdetection_model_name = rtmdetection_model_name
        self.rtmpose_model_name = rtmpose_model_name
        self.conda_env = conda_env
        self.path_to_mmdetection_config = path_to_mmdetection_config
        self.path_to_mmpose_config = path_to_mmpose_config
        self.skeleton_py_file = Path(skeleton_py_file)
        self.is_local = is_local

        # get the device
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            self.device_name = torch.cuda.get_device_name(0).replace(" ", "_")
            logger.info(f"Using CUDA device: {self.device_name}")
        else:
            raise ValueError("CUDA is not available on this machine.")

        # get location of rtm models
        self.rtmdetection_output = (
            self.tensorrt_output_directory / rtmdetection_model_name / self.device_name
        )
        self.rtmdetection_output.mkdir(parents=True, exist_ok=True)
        self.rtmpose_output = (
            self.tensorrt_output_directory / rtmpose_model_name / self.device_name
        )
        self.rtmpose_output.mkdir(parents=True, exist_ok=True)

    def generate_sitecustomize_script(self):
        sitecustomize_script = "from mmpose.registry import DATASETS\n"
        sitecustomize_script += "from mmpose.datasets.datasets.base import BaseCocoStyleDataset\n"
        sitecustomize_script += f"skeleton_py_file = '{self.skeleton_py_file.as_posix()}'\n"
        sitecustomize_script += f"@DATASETS.register_module()\n"
        sitecustomize_script += f"class CoCo25pt(BaseCocoStyleDataset):\n"
        sitecustomize_script += f"\tMETAINFO: dict = dict(from_file=skeleton_py_file)\n"
        return sitecustomize_script

    def check_if_detector_tensorrt_exists(self):
        logger.info(f"Checking if tensorrt model exists at: {self.rtmdetection_output}")
        return (self.rtmdetection_output / "output_tensorrt.jpg").exists()

    def check_if_pose_tensorrt_exists(self):
        logger.info(f"Checking if tensorrt model exists at: {self.rtmpose_output}")
        return (self.rtmpose_output / "output_tensorrt.jpg").exists()

    def convert_pose_to_tensorrt(self):
        logger.info(f"Converting pose model to tensorrt. input: {self.rtmpose_output}")

        if self.check_if_pose_tensorrt_exists():
            logger.info("TensorRT pose model already exists.")
            return

        # generate the sitecustomize script
        sitecustomize_script = self.generate_sitecustomize_script()
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()

        # Path for the temporary sitecustomize.py file
        temp_sitecustomize_path = os.path.join(temp_dir, "sitecustomize.py")

        # Writing the sitecustomize script to the temporary file
        with open(temp_sitecustomize_path, "w") as file:
            file.write(sitecustomize_script)

        # warning: this will switch out current cuda module
        if self.is_local:
            # local (at least on peromoseq) needs to source conda first
            model_conversion_script = "source $(conda info --base)/etc/profile.d/conda.sh;\n"
            model_conversion_script += f"conda activate {self.conda_env};\n"
        else:
            model_conversion_script = f"module load gcc/9.2.0\n"
            model_conversion_script = f"module load cuda/11.7\n"
            model_conversion_script += f"source activate {self.conda_env};\n"
        # # Set PYTHONPATH to include the directory where sitecustomize.py is located
        model_conversion_script += f"export PYTHONPATH={temp_dir}:$PYTHONPATH;\n"
        model_conversion_script += (
            f"python {(self.path_to_mmdeploy / 'tools/deploy.py').as_posix()}"
        )
        model_conversion_script += f" {self.path_to_mmpose_config}"
        model_conversion_script += f" {self.path_to_rmpose_config}"
        model_conversion_script += f" {self.path_to_rmpose_checkpoint}"
        model_conversion_script += f" {self.path_to_demo_image_pose}"
        model_conversion_script += f" --work-dir { self.rtmpose_output}"
        model_conversion_script += f" --device cuda:0"
        # model_conversion_script += f" --log-level DEBUG"
        # model_conversion_script += f" --show"
        model_conversion_script += f" --dump-info"  # dump sdk info
        print(model_conversion_script)
        # Run the model conversion script
        process = subprocess.Popen(
            model_conversion_script,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            executable="/bin/bash" if self.is_local else None,
        )
        # Read output line by line as it is produced
        for line in process.stdout:
            print(line, end="")

        # Wait for the subprocess to finish
        process.wait()

        # Delete the temporary directory and all its contents
        shutil.rmtree(temp_dir)

        logger.info("Finished converting pose model to tensorrt.")
        logger.info(f"model output at: {self.rtmpose_output}")

    def convert_detection_to_tensorrt(self):
        logger.info(f"Converting detection model to tensorrt. input: {self.rtmdetection_output}")

        if self.check_if_detector_tensorrt_exists():
            logger.info("TensorRT detection model already exists.")
            return

        # generate the sitecustomize script
        sitecustomize_script = self.generate_sitecustomize_script()
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()

        # Path for the temporary sitecustomize.py file
        temp_sitecustomize_path = os.path.join(temp_dir, "sitecustomize.py")

        # Writing the sitecustomize script to the temporary file
        with open(temp_sitecustomize_path, "w") as file:
            file.write(sitecustomize_script)

        # warning: this will switch out current cuda module
        model_conversion_script = ""
        if self.is_local:
            # local (at least on peromoseq) needs to source conda first
            model_conversion_script += "source $(conda info --base)/etc/profile.d/conda.sh;\n"
            model_conversion_script += f"conda activate {self.conda_env};\n"
        else:
            model_conversion_script += f"module load gcc/9.2.0\n"
            model_conversion_script += f"module load cuda/11.7\n"
            model_conversion_script += f"source activate {self.conda_env};\n"
        # # Set PYTHONPATH to include the directory where sitecustomize.py is located
        model_conversion_script += f"export PYTHONPATH={temp_dir}:$PYTHONPATH;\n"
        model_conversion_script += (
            f"python {(self.path_to_mmdeploy / 'tools/deploy.py').as_posix()}"
        )
        model_conversion_script += f" {self.path_to_mmdetection_config}"
        model_conversion_script += f" {self.path_to_rtmdetection_config}"
        model_conversion_script += f" {self.path_to_rtmdetection_checkpoint}"
        model_conversion_script += f" {self.path_to_demo_image_detection}"
        model_conversion_script += f" --work-dir { self.rtmdetection_output}"
        model_conversion_script += f" --device cuda:0"
        # model_conversion_script += f" --log-level DEBUG"
        # model_conversion_script += f" --show"
        model_conversion_script += f" --dump-info"  # dump sdk info

        # Run the model conversion script
        print("Running model conversion script:")
        print(model_conversion_script)

        process = subprocess.Popen(
            model_conversion_script,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            # executable="/bin/bash" if self.is_local else None,
            executable="/bin/bash",
        )

        # Read output line by line as it is produced
        print("================================================================================")
        print("Output:")
        for line in process.stdout:
            print(line, end="")

        # Wait for the subprocess to finish
        process.wait()

        # Delete the temporary directory and all its contents
        shutil.rmtree(temp_dir)

        logger.info("Finished converting detection model to tensorrt.")
        logger.info(f"model output at: {self.rtmdetection_output}")

    def run(self):
        self.convert_detection_to_tensorrt()
        self.convert_pose_to_tensorrt()
