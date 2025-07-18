import sys
import numpy as np

# load skeleton
try:
    from multicamera_airflow_pipeline.tim_240731.skeletons.sainburg25pt import (
        dataset_info,
        parents_dict,
    )
except:

    # Add the directory containing the file to the system path
    sys.path.append(
        "/n/groups/datta/tim_sainburg/projects/multicamera_airflow_pipeline/multicamera_airflow_pipeline/tim_240731/skeletons/sainburg25pt.py"
    )
    # Now import the dataset_info dictionary
    from sainburg25pt import dataset_info, parents_dict
keypoint_info = dataset_info["keypoint_info"]
skeleton_info = dataset_info["skeleton_info"]
keypoints = [keypoint_info[i]["name"] for i in keypoint_info.keys()]
keypoints = np.array(keypoints)
keypoints_order = keypoints
kpt_dict = {j: i for i, j in enumerate(keypoints_order)}

default_template_bone_length_mean = c57_template_bone_length_mean = {
    "nose_tip": 16.4,
    "left_ear": 15.1,
    "right_ear": 15.1,
    "left_eye": 7.1,
    "right_eye": 7.1,
    "throat": 16.0,
    "forehead": 18.3,
    "left_shoulder": 14.4,
    "right_shoulder": 14.4,
    "left_elbow": 10.9,
    "right_elbow": 10.9,
    "left_wrist": 7.7,
    "right_wrist": 7.7,
    "left_hind_paw_front": 12.5,
    "right_hind_paw_front": 12.5,
    "left_hind_paw_back": 13.7,
    "right_hind_paw_back": 13.7,
    "left_knee": 19.5,
    "right_knee": 19.5,
    "tail_base": 13.0,
    "spine_low": 20.6,
    "spine_mid": 9.8,
    "spine_high": 9.8,
    "left_fore_paw": 4.6,
    "right_fore_paw": 4.6,
}

default_template_bone_length_std = c57_template_bone_length_std = {
    "nose_tip": 1.2,
    "left_ear": 0.8,
    "right_ear": 0.8,
    "left_eye": 0.4,
    "right_eye": 0.4,
    "throat": 1.3,
    "forehead": 1.4,
    "left_shoulder": 1.3,
    "right_shoulder": 1.3,
    "left_elbow": 1.3,
    "right_elbow": 1.3,
    "left_wrist": 0.9,
    "right_wrist": 0.9,
    "left_hind_paw_front": 0.8,
    "right_hind_paw_front": 0.8,
    "left_hind_paw_back": 0.8,
    "right_hind_paw_back": 0.8,
    "left_knee": 1.2,
    "right_knee": 1.2,
    "tail_base": 0.7,
    "spine_low": 1.4,
    "spine_mid": 0.8,
    "spine_high": 0.8,
    "left_fore_paw": 0.6,
    "right_fore_paw": 0.6,
}

# all mice will be projected into this template
OLD_default_template_bone_length_mean = {
    "nose_tip": 19.3,
    "left_ear": 17.9,
    "right_ear": 17.9,
    "left_eye": 8.1,
    "right_eye": 8.1,
    "throat": 19.2,
    "forehead": 18.6,
    "left_shoulder": 15.7,
    "right_shoulder": 15.7,
    "left_elbow": 11.3,
    "right_elbow": 11.3,
    "left_wrist": 8.6,
    "right_wrist": 8.6,
    "left_hind_paw_front": 13.9,
    "right_hind_paw_front": 13.9,
    "left_hind_paw_back": 14.6,
    "right_hind_paw_back": 14.6,
    "left_knee": 21.0,
    "right_knee": 21.0,
    "tail_base": 12.3,
    "spine_low": 20.5,
    "spine_mid": 9.9,
    "spine_high": 9.9,
    "left_fore_paw": 4.8,
    "right_fore_paw": 4.8,
}
OLD_default_template_bone_length_std = {
    "nose_tip": 1.3,
    "left_ear": 1.1,
    "right_ear": 1.1,
    "left_eye": 0.9,
    "right_eye": 0.9,
    "throat": 1.6,
    "forehead": 1.6,
    "left_shoulder": 1.5,
    "right_shoulder": 1.5,
    "left_elbow": 2.5,
    "right_elbow": 2.5,
    "left_wrist": 1.6,
    "right_wrist": 1.6,
    "left_hind_paw_front": 1.6,
    "right_hind_paw_front": 1.6,
    "left_hind_paw_back": 1.3,
    "right_hind_paw_back": 1.3,
    "left_knee": 1.8,
    "right_knee": 1.8,
    "tail_base": 0.9,
    "spine_low": 1.6,
    "spine_mid": 1.0,
    "spine_high": 1.0,
    "left_fore_paw": 1.0,
    "right_fore_paw": 1.0,
}

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
    "left_eye": ["forehead", "spine_high", "spine_base"],
    "right_eye": ["forehead", "spine_high", "spine_base"],
    "throat": ["forehead", "spine_high", "spine_base"],
    # fore limbs
    "left_shoulder": ["spine_high", "spine_base"],
    "right_shoulder": ["spine_high", "spine_base"],
    "left_elbow": ["left_shoulder", "spine_high", "spine_base"],
    "right_elbow": ["right_shoulder", "spine_high", "spine_base"],
    "left_wrist": ["left_elbow", "left_shoulder", "spine_high", "spine_base"],
    "right_wrist": ["right_elbow", "right_shoulder", "spine_high", "spine_base"],
    "left_fore_paw": ["left_wrist", "left_elbow", "left_shoulder", "spine_high", "spine_base"],
    "right_fore_paw": ["right_wrist", "right_elbow", "right_shoulder", "spine_high", "spine_base"],
    # hind paws
    "left_knee": ["spine_low", "spine_mid", "spine_base"],
    "right_knee": ["spine_low", "spine_mid", "spine_base"],
    "left_hind_paw_back": ["left_knee", "spine_low", "spine_mid", "spine_base"],
    "right_hind_paw_back": ["right_knee", "spine_low", "spine_mid", "spine_base"],
    "left_hind_paw_front": [
        "left_hind_paw_back",
        "left_knee",
        "spine_low",
        "spine_mid",
        "spine_base",
    ],
    "right_hind_paw_front": [
        "right_hind_paw_back",
        "right_knee",
        "spine_low",
        "spine_mid",
        "spine_base",
    ],
}


gimbal_skeleton = [
    ["tail_base", "spine_low"],
    ["spine_low", "spine_mid"],
    ["spine_mid", "spine_high"],
    ["spine_high", "throat"],
    ["spine_high", "forehead"],
    # [b'throat', b'nose_tip'],
    ["nose_tip", "forehead"],
    ["spine_high", "left_shoulder"],
    ["left_shoulder", "left_elbow"],
    ["left_elbow", "left_wrist"],
    ["left_wrist", "left_fore_paw"],
    ["spine_high", "right_shoulder"],
    ["right_shoulder", "right_elbow"],
    ["right_elbow", "right_wrist"],
    ["right_wrist", "right_fore_paw"],
    # [b'nose_tip', b'left_eye'],
    ["left_eye", "forehead"],
    # [b'nose_tip', b'right_eye'],
    ["right_eye", "forehead"],
    # [b'left_eye', b'right_eye'],
    ["forehead", "right_ear"],
    ["forehead", "left_ear"],
    ["spine_low", "left_knee"],
    ["left_knee", "left_hind_paw_back"],
    ["left_hind_paw_back", "left_hind_paw_front"],
    ["spine_low", "right_knee"],
    ["right_knee", "right_hind_paw_back"],
    ["right_hind_paw_back", "right_hind_paw_front"],
]
