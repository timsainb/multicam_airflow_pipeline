import logging
from os.path import join
from pathlib import Path
import sys

import torch
import yaml

from multicamera_airflow_pipeline.tim_240731.keypoints.predict_2D import Inferencer2D

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.info(f"Python interpreter binary location: {sys.executable}")

print(torch.cuda.is_available())


config_file = "/n/groups/datta/Jonah/Local_code_groups/6cam_repos/multicam_airflow_pipeline/multicamera_airflow_pipeline/tim_240731/default_config.yaml"
config_file = Path(config_file)
config = yaml.safe_load(open(config_file, "r"))

for kw in [
    "use_tensorrt",
    "tensorrt_rtmdetection_model_name",
    "tensorrt_rtmpose_model_name",
]:
    if kw in config["prediction_2d"]:
        config["prediction_2d"].pop(kw)

rec_dir = "/n/groups/datta/Jonah/20240925_PBN_npx/raw_data/J07901/20240930_J07901_6cam_PBN/24-09-30-15-43-47-490092"
# output_directory_predictions = join(rec_dir, "tmp_2d_preds")
output_directory_predictions = "/n/groups/datta/kpts_pipeline/tim_240731/results/2D_predictions/24-09-30-15-43-47-490092"
expected_video_length_frames = 432017
tensorrt_model_directory = "/n/groups/datta/kpts_pipeline/tim_240731/results/tensorrt/"

inferencer = Inferencer2D(
    recording_directory=rec_dir,
    output_directory_predictions=output_directory_predictions,
    expected_video_length_frames=expected_video_length_frames,
    tensorrt_model_directory=tensorrt_model_directory,
    tensorrt_rtmdetection_model_name="rtmdet_tiny_8xb32-300e_coco_chronic_24-05-04-17-51-58_216661",
    tensorrt_rtmpose_model_name="rtmpose-m_8xb64-210e_ap10k-256x256_24-05-04-21-35-13_305524",
    use_tensorrt=True,
    **config["prediction_2d"],
)
inferencer.run()
