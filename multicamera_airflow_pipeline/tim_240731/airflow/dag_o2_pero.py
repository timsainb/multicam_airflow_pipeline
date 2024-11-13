from datetime import datetime, timedelta
import pandas as pd
import requests
from io import BytesIO
from pathlib import Path
from datetime import datetime
import logging
import sys
import numpy as np

from multicamera_airflow_pipeline.tim_240731.airflow.jobs.o2 import (
    sync_cameras,
    sync_cameras_to_openephys,
    predict_2d,
    calibrate_cameras,
    spikesorting,
    triangulation,
    run_gimbal,
    size_normalization,
    arena_alignment,
    egocentric_alignment,
    compute_continuous_features,
)

from multicamera_airflow_pipeline.tim_240731.airflow.jobs.local.predict_2d_local import (
    predict_2d_local,
)

from multicamera_airflow_pipeline.tim_240731.airflow.jobs import await_2d_predictions

from airflow.decorators import task
from airflow.models.dag import DAG
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)
logger.info(f"Python interpreter binary location: {sys.executable}")


sync_cameras_task = task(sync_cameras.sync_cameras, pool="low_compute_pool")
predict_2d_task = task(predict_2d.predict_2d, pool="low_compute_pool")

calibrate_cameras_task = task(calibrate_cameras.calibrate_cameras, pool="low_compute_pool")
triangulation_task = task(triangulation.triangulation, pool="low_compute_pool")
run_gimbal_task = task(run_gimbal.run_gimbal, pool="low_compute_pool")
size_normalization_task = task(size_normalization.size_normalization, pool="low_compute_pool")
arena_alignment_task = task(arena_alignment.arena_alignment, pool="low_compute_pool")
egocentric_alignment_task = task(
    egocentric_alignment.egocentric_alignment, pool="low_compute_pool"
)
compute_continuous_features_task = task(
    compute_continuous_features.compute_continuous_features, pool="low_compute_pool"
)
await_2d_predictions_task = task(
    await_2d_predictions.await_2d_predictions, pool="low_compute_pool"
)
predict_2d_local_task = task(predict_2d_local, pool="local_gpu_pool")


def read_google_sheet(spreadsheet_url):
    response = requests.get(spreadsheet_url)
    recording_df = pd.read_csv(BytesIO(response.content))
    return recording_df


class AirflowDAG:
    def __init__(
        self,
        pipeline_name: str = "tim_240731",
        config_file: str = "/n/groups/datta/tim_sainburg/projects/multicamera_airflow_pipeline/multicamera_airflow_pipeline/tim_240731/default_config.yaml",
        output_directory: str = "/n/groups/datta/kpts_pipeline/tim_airflow_pero_241010/results",
        job_directory: str = "/n/groups/datta/kpts_pipeline/tim_airflow_pero_241010/jobs",
        spreadsheet_url: str = "https://docs.google.com/spreadsheet/ccc?key=14HIqUaSl_n-91hpAvmACY_iVY9nLKdlA6qklhxfZon0&output=csv&gid=785542790",
        default_args={
            "owner": "airflow",
            "depends_on_past": False,
            "start_date": datetime(2024, 1, 1),
            "email_on_failure": False,
            "email_on_retry": False,
            "retries": 1,
            "is_paused_upon_creation": False,
            "retry_delay": timedelta(minutes=60),
        },
        schedule_interval=timedelta(days=1),
    ):
        """This objects creates a new Airflow DAG for each row in a Google Sheet.
        Each DAG runs a pipeline to process video and ephys recordings. You can
        modify this class to add new tasks to the pipeline.
        """
        # where to save output data
        self.output_directory = Path(output_directory)
        # where to save jobs
        self.job_directory = Path(job_directory)
        self.config_file = Path(config_file)
        self.recording_df = read_google_sheet(spreadsheet_url)
        # remove missing data
        self.recording_df[
            np.any(
                self.recording_df[["date", "video_recording_id", "calibration_id"]].isnull().values
                == False,
                axis=1,
            )
        ]
        self.scheduling_interval = schedule_interval
        self.default_args = default_args
        self.pipeline_name = pipeline_name

    def generate_dags(self):

        for idx, recording_row in self.recording_df.iterrows():
            subject_id = recording_row["Subject"]
            video_recording_id = recording_row["video_recording_id"]
            dag_id = f"{self.pipeline_name}_{subject_id}_{video_recording_id}"
            logger.info(f"Attempting to create DAG {dag_id}")

            with DAG(
                dag_id=dag_id,
                default_args=self.default_args,
                description=f"DAG pipeline for {subject_id} {video_recording_id}",
                schedule_interval=timedelta(days=7),
                start_date=days_ago(1),
                catchup=False,
                is_paused_upon_creation=False,
            ) as generated_dag:

                logger.info(f"Starting creation of DAG {dag_id}")
                # define tasks
                synced_cams = sync_cameras_task(
                    recording_row,
                    self.job_directory,
                    self.output_directory,
                    self.config_file,
                )
                if recording_row["use_local"] == "FALSE":
                    predicted_2d = predict_2d_task(
                        recording_row,
                        self.job_directory,
                        self.output_directory,
                        self.config_file,
                    )
                else:
                    predicted_2d = predict_2d_local_task(
                        recording_row,
                        self.job_directory,
                        self.output_directory,
                        self.config_file,
                    )

                completed_2d = await_2d_predictions_task(
                    recording_row,
                    self.output_directory,
                    recheck_duration_s=60,
                    maximum_wait_time_s=604800,
                )

                calibrated = calibrate_cameras_task(
                    recording_row,
                    self.job_directory,
                    self.output_directory,
                    self.config_file,
                )

                triangulated = triangulation_task(
                    recording_row,
                    self.job_directory,
                    self.output_directory,
                    self.config_file,
                )
                gimbaled = run_gimbal_task(
                    recording_row,
                    self.job_directory,
                    self.output_directory,
                    self.config_file,
                )
                size_normed = size_normalization_task(
                    recording_row,
                    self.job_directory,
                    self.output_directory,
                    self.config_file,
                )
                arena_aligned = arena_alignment_task(
                    recording_row,
                    self.job_directory,
                    self.output_directory,
                    self.config_file,
                )
                ego_aligned = egocentric_alignment_task(
                    recording_row,
                    self.job_directory,
                    self.output_directory,
                    self.config_file,
                )
                # cont_feats = compute_continuous_features_task(
                #    recording_row,
                #    self.job_directory,
                #    self.output_directory,
                #    self.config_file,
                # )

                # define dependencies
                predicted_2d >> completed_2d
                [synced_cams, calibrated, completed_2d] >> triangulated
                triangulated >> gimbaled
                gimbaled >> size_normed
                size_normed >> arena_aligned
                size_normed >> ego_aligned
                # [arena_aligned, ego_aligned] >> cont_feats

            globals()[dag_id] = generated_dag
            logger.info(f"DAG {dag_id} created")
