from datetime import datetime, timedelta
import pandas as pd
import requests
from io import BytesIO
from pathlib import Path
from multicamera_airflow_pipeline.tim_240731.interface.o2 import O2Runner
from datetime import datetime

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
from airflow.decorators import task
from airflow.models.dag import DAG

sync_cameras_task = task(sync_cameras.sync_cameras)
predict_2d_task = task(predict_2d.predict_2d)
sync_cameras_to_openephys_task = task(sync_cameras_to_openephys.sync_cameras_to_openephys)
calibrate_cameras_task = task(calibrate_cameras.calibrate_cameras)
spikesorting_task = task(spikesorting.spikesorting)
triangulation_task = task(triangulation.triangulation)
run_gimbal_task = task(run_gimbal.run_gimbal)
size_normalization_task = task(size_normalization.size_normalization)
arena_alignment_task = task(arena_alignment.arena_alignment)
egocentric_alignment_task = task(egocentric_alignment.egocentric_alignment)
compute_continuous_features_task = task(compute_continuous_features.compute_continuous_features)


class AirflowDAG:
    def __init__(
        self,
        pipeline_name: str = "tim_240731",
        config_file="/n/groups/datta/tim_sainburg/projects/multicamera_airflow_pipeline/multicamera_airflow_pipeline/tim_240731/default_config.yaml",
        output_directory: str = "/n/groups/datta/kpts_pipeline/tim_240731/results",
        job_directory: str = "/n/groups/datta/kpts_pipeline/tim_240731/jobs",
        spreadsheet_url: str = "https://docs.google.com/spreadsheet/ccc?key=1jACsUmxuJ9Une59qmvzZGc1qXezKhKzD1zho2sEfcrU&output=csv&gid=0",
        default_args={
            "owner": "airflow",
            "depends_on_past": False,
            "start_date": datetime(2024, 1, 1),
            "email_on_failure": False,
            "email_on_retry": False,
            "retries": 1,
            "is_paused_upon_creation": False,
            "retry_delay": timedelta(minutes=5),
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
        self.scheduling_interval = schedule_interval
        self.default_args = default_args
        self.pipeline_name = pipeline_name

    def run(self):
        # for each recording, create a new DAG
        for idx, recording_row in self.recording_df.iterrows():
            subject_id = recording_row["Subject"]
            video_recording_id = recording_row["video_recording_id"]
            with DAG(
                dag_id=f"{self.pipeline_name}_{subject_id}_{video_recording_id}",
                default_args=self.default_args,
                catchup=False,
            ) as dag:

                # define tasks
                synced_cams = sync_cameras_task(
                    recording_row,
                    self.job_directory,
                    self.output_directory,
                    self.config_file,
                )
                predicted_2d = predict_2d_task(
                    recording_row,
                    self.job_directory,
                    self.output_directory,
                    self.config_file,
                )
                calibrated = calibrate_cameras_task(
                    recording_row,
                    self.job_directory,
                    self.output_directory,
                    self.config_file,
                )
                synced_ephys = sync_cameras_to_openephys_task(
                    recording_row,
                    self.job_directory,
                    self.output_directory,
                    self.config_file,
                )
                sorted_spikes = spikesorting_task(
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
                cont_feats = compute_continuous_features_task(
                    recording_row,
                    self.job_directory,
                    self.output_directory,
                    self.config_file,
                )

                # define dependencies
                [synced_cams, calibrated, predicted_2d] >> triangulated
                triangulated >> gimbaled
                gimbaled >> size_normed
                size_normed >> arena_aligned
                size_normed >> ego_aligned
                [arena_aligned, ego_aligned] >> cont_feats


def read_google_sheet(spreadsheet_url):
    response = requests.get(spreadsheet_url)
    recording_df = pd.read_csv(BytesIO(response.content))
    return recording_df
