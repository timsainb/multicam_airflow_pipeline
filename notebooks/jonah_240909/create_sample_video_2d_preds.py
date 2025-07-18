from glob import glob
import os
from os.path import join
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np
from tqdm.auto import tqdm

from multicamera_airflow_pipeline.tim_240731.skeletons.sainburg25pt import dataset_info

print("Python interpreter binary location:", sys.executable)


def generate_keypoint_video(
    output_directory: Path,
    video_path: Path,
    keypoint_coords: np.ndarray,
    keypoint_conf: np.ndarray,  # New parameter for keypoint confidence
    keypoint_info: dict,
    skeleton_info: dict,
    max_frames=None,
):
    """
    Generates a video with keypoint predictions overlaid on the original video frames.
    
    Processes about ~30 frames per second. So 1 min of a 120 fps vid x 6 vids = ~30 min.

    Parameters:
    -----------
    output_directory : Path
        Directory where the output video will be saved.

    video_path : Path
        Path to the input video file.

    keypoint_coords : np.ndarray
        Array of shape (#frames, #keypoints, 2) containing the coordinates of keypoints for each frame.

    keypoint_conf : np.ndarray
        Array of shape (#frames, #keypoints) containing the confidence values (0-1) for each keypoint in each frame.

    keypoint_info : dict
        Dictionary containing information about the keypoints. Each key in the dictionary represents a keypoint ID, and the
        value is another dictionary with the following structure:
        {
            'name': str,       # Keypoint name
            'id': int,         # Keypoint ID
            'color': list,     # RGB color for the keypoint [R, G, B]
            'type': str,       # Keypoint type (e.g., 'upper', 'lower')
            'swap': str        # Name of the corresponding left/right keypoint to be swapped (for symmetry)
        }

    skeleton_info : dict
        Dictionary containing information about the skeleton. Each key in the dictionary represents a skeleton link ID, and
        the value is another dictionary with the following structure:
        {
            'link': tuple,     # Tuple containing the names of the two keypoints that form the link
            'id': int,         # Link ID
            'color': list      # RGB color for the link [R, G, B]
        }

    Returns:
    --------
    None
        The function saves the output video with keypoints and skeletons overlaid to the specified output directory.

    Raises:
    -------
    ValueError
        If the input video cannot be opened.

    Example:
    --------
    output_directory = Path('/output/directory')
    video_path = Path('/path/to/video.mp4')
    keypoint_coords = np.load('keypoint_coords.npy')  # Load your keypoints array
    keypoint_conf = np.load('keypoint_conf.npy')  # Load your keypoint confidence array
    keypoint_info = {
        0: {'name': 'nose_tip', 'id': 0, 'color': [120, 184, 181], 'type': 'upper', 'swap': ''},
        # Add other keypoints as needed
    }
    skeleton_info = {
        0: {'link': ('tail_base', 'spine_low'), 'id': 0, 'color': [173, 160, 183]},
        # Add other links as needed
    }

    generate_keypoint_video(output_directory, video_path, keypoint_coords, keypoint_conf, keypoint_info, skeleton_info)
    """

    # Open the input video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create the VideoWriter object
    output_path = output_directory / (video_path.stem + "_with_keypoints.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    frame_idx = 0
    if total_frames < 0 and max_frames is None:
        raise ValueError(
            "Could not determine total number of frames in the video -- please specify max_frames."
        )
    elif total_frames < 0:
        total_frames = max_frames
    elif max_frames is not None:
        total_frames = np.min([max_frames, total_frames])

    print(f"Total frames: {total_frames}")

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Create an overlay for drawing
            overlay = frame.copy()

            # Draw keypoints
            for kp_idx, kp_info in keypoint_info.items():
                if (
                    frame_idx < len(keypoint_coords)
                    and kp_idx < keypoint_coords.shape[1]
                ):
                    x, y = keypoint_coords[frame_idx, kp_idx]
                    conf = keypoint_conf[frame_idx, kp_idx]
                    color = tuple(kp_info["color"])
                    alpha = conf  # Alpha value is based on the confidence (0-1)
                    if conf > 0:  # Only draw if confidence is greater than 0
                        overlay = cv2.circle(
                            overlay,
                            (int(x), int(y)),
                            radius=4,
                            color=color,
                            thickness=-1,
                        )

            # Apply the overlay with alpha blending for keypoints
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Draw skeleton
            for link_info in skeleton_info.values():
                kp1_name, kp2_name = link_info["link"]
                kp1_id = next(
                    (
                        kp["id"]
                        for kp in keypoint_info.values()
                        if kp["name"] == kp1_name
                    ),
                    None,
                )
                kp2_id = next(
                    (
                        kp["id"]
                        for kp in keypoint_info.values()
                        if kp["name"] == kp2_name
                    ),
                    None,
                )

                if kp1_id is not None and kp2_id is not None:
                    if (
                        frame_idx < len(keypoint_coords)
                        and kp1_id < keypoint_coords.shape[1]
                        and kp2_id < keypoint_coords.shape[1]
                    ):
                        x1, y1 = keypoint_coords[frame_idx, kp1_id]
                        x2, y2 = keypoint_coords[frame_idx, kp2_id]
                        kp1_conf = keypoint_conf[frame_idx, kp1_id]
                        kp2_conf = keypoint_conf[frame_idx, kp2_id]
                        color = tuple(link_info["color"])
                        alpha = min(
                            kp1_conf, kp2_conf
                        )  # Alpha value is the minimum confidence of the link
                        if (
                            kp1_conf > 0 and kp2_conf > 0
                        ):  # Only draw if both confidence values are greater than 0
                            overlay = cv2.line(
                                overlay,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                color=color,
                                thickness=2,
                            )

            # Apply the overlay with alpha blending for skeleton
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Write the frame with keypoints and skeletons to the output video
            out.write(frame)
            frame_idx += 1
            pbar.update(1)
            if max_frames and frame_idx >= max_frames:
                break

    # Release video objects
    cap.release()
    out.release()
    print(f"Video saved to: {output_path}")


def main():
    sessions = [
        # "24-09-28-11-44-04-693209",
        "24-09-29-12-40-04-238868",
        "24-10-01-18-48-38-861115",
    ]
    recording_dir = (
        "/n/groups/datta/Jonah/20240925_PBN_npx/raw_data/J07901"  # path to raw videos
    )
    results_dir = "/n/groups/datta/kpts_pipeline/tim_240731/results"
    pred_2d_dir = join(results_dir, "2D_predictions")  # path to 2D kp predictions
    video_output_directory = Path(
        join(recording_dir, "tim_240731_keypoint_videos")
    )  # path to save output videos
    video_output_directory.mkdir(parents=True, exist_ok=True)
    max_frames = 120 * 60  # number of frames to process per video

    for session in sessions:
        prediction_files = glob(join(pred_2d_dir, session, f"{session}*.h5"))
        session_output_directory = video_output_directory / session
        session_output_directory.mkdir(parents=True, exist_ok=True)
        
        for h5_file in prediction_files:
            
            # Load the data
            with h5py.File(h5_file, "r") as file:
                print(list(file.keys()))
                keypoint_coords = np.array(file["keypoint_coords"])
                keypoint_conf = np.array(file["keypoint_conf"])
                # detection_conf = np.array(file["detection_conf"])
                # detection_coords = np.array(file["detection_coords"])
            keypoint_conf[keypoint_conf > 1] = 1  # not sure if this is the right way to fix this? Unhelpful discussion at https://github.com/open-mmlab/mmpose/issues/884

            # print(keypoint_coords.shape)
            # print(keypoint_conf.shape)
            # plt.figure()
            # plt.matshow(keypoint_conf[:, 0, :].T, aspect="auto")
            # plt.title(
            #     f'Keypoint confidence\n{session}\nCam {os.path.basename(h5_file).split(".")[1]}'
            # )

            # Find video
            # NB: may need to change this per user depending on how / where the videos are stored
            recording_id, camera, frame, ext = os.path.basename(h5_file).split(".")
            if frame == "0":
                # video_path = [i for i in recording_row.recording_path.glob(f"{camera}*.mp4") if i.stem.count('.') ==1][0]
                video_path = glob(
                    join(recording_dir, "**", session, f"*{camera}*.mp4")
                )[0]
            else:
                # video_path =  list(recording_row.recording_path.glob(f"{camera}.*.{frame}.mp4"))[0]
                video_path = glob(
                    join(recording_dir, "**", session, f"*{camera}.*.{frame}.mp4")
                )[0]

            generate_keypoint_video(
                output_directory=session_output_directory,
                video_path=Path(video_path),
                keypoint_coords=keypoint_coords[:, 0],
                keypoint_conf=keypoint_conf[:, 0],
                keypoint_info=dataset_info["keypoint_info"],
                skeleton_info=dataset_info["skeleton_info"],
                max_frames=max_frames,
            )


if __name__ == "__main__":
    main()
