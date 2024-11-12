import sys
import numpy as np

# load skeleton
try:
    from multicamera_airflow_pipeline.jonah_241112.skeletons.weinreb15pt import (
        dataset_info,
        parents_dict,
    )
except:

    # Add the directory containing the file to the system path
    sys.path.append(
        "/n/groups/datta/tim_sainburg/projects/multicamera_airflow_pipeline/multicamera_airflow_pipeline/jonah_241112/skeletons/weinreb15pt.py"
    )
    # Now import the dataset_info dictionary
    from weinreb15pt import dataset_info, parents_dict
keypoint_info = dataset_info["keypoint_info"]
skeleton_info = dataset_info["skeleton_info"]
keypoints = [keypoint_info[i]["name"] for i in keypoint_info.keys()]
keypoints = np.array(keypoints)
keypoints_order = keypoints
kpt_dict = {j: i for i, j in enumerate(keypoints_order)}

# all mice will be projected into this template
default_template_bone_length_mean = {
    "spine_low": 20.5,
    "tail_base": 12.3,
    "spine_mid": 9.9,
    "spine_high": 9.9,
    "left_ear": 17.9,
    "right_ear": 17.9,
    "forehead": 18.6,
    "nose_tip": 19.3,
    "left_fore_paw": 4.8,
    "right_fore_paw": 4.8,
    "left_hind_paw_back": 14.6,
    "left_hind_paw_front": 13.9,
    "right_hind_paw_back": 14.6,
    "right_hind_paw_front": 13.9,
}

default_template_bone_length_std = {
    "tail_base": 0.9,
    "spine_low": 1.6,
    "spine_mid": 1.0,
    "spine_high": 1.0,
    "left_ear": 1.1,
    "right_ear": 1.1,
    "forehead": 1.6,
    "nose_tip": 1.3,
    "left_fore_paw": 1.0,
    "right_fore_paw": 1.0,
    "left_hind_paw_back": 1.3,
    "left_hind_paw_front": 1.6,
    "right_hind_paw_back": 1.3,
    "right_hind_paw_front": 1.6,
    
}
# path to get back to spine_base from each keypoint
default_hierarchy = {
    # main body
    "spine_base": [],
    "spine_mid": ["spine_base"],
    "spine_high": ["spine_base"],
    "spine_low": ["spine_mid", "spine_base"],
    "tail_base": ["spine_low", "spine_mid", "spine_base"],
    # head
    "forehead": ["spine_high", "spine_base"],
    "nose_tip": ["forehead", "spine_high", "spine_base"],
    "left_ear": ["forehead", "spine_high", "spine_base"],
    "right_ear": ["forehead", "spine_high", "spine_base"],
    # fore limbs
    "left_fore_paw": ["spine_high", "spine_base"],
    "right_fore_paw": ["spine_high", "spine_base"],
    # hind paws
    "left_hind_paw_back": ["spine_low", "spine_mid", "spine_base"],
    "right_hind_paw_back": ["spine_low", "spine_mid", "spine_base"],
    "left_hind_paw_front": [
        "left_hind_paw_back",
        "spine_low",
        "spine_mid",
        "spine_base",
    ],
    "right_hind_paw_front": [
        "right_hind_paw_back",
        "spine_low",
        "spine_mid",
        "spine_base",
    ],
}


gimbal_skeleton = [
    ["tail_base", "spine_low"],
    ["spine_low", "spine_mid"],
    ["spine_mid", "spine_high"],
    ["spine_high", "left_ear"],
    ["spine_high", "right_ear"],
    ["spine_high", "forehead"],
    ["forehead", "nose_tip"],
    ["left_hind_paw_back", "left_hind_paw_front"],
    ["spine_low", "left_hind_paw_back"],
    ["right_hind_paw_back", "right_hind_paw_front"],
    ["spine_low", "right_hind_paw_back"],
    ["spine_high", "left_fore_paw"],
    ["spine_high", "right_fore_paw"],
]
