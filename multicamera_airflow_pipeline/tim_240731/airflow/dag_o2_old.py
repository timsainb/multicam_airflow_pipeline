from datetime import datetime, timedelta
import pandas as pd
import requests
from io import BytesIO
from pathlib import Path
from multicamera_airflow_pipeline.tim_240731.interface.o2 import O2Runner
from datetime import datetime
from airflow.utils.dag_cycle_tester import test_cycle
import logging
import sys

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

logger = logging.getLogger(__name__)
logger.info(f"Python interpreter binary location: {sys.executable}")


from airflow.decorators import task
from airflow.models.dag import DAG
from airflow.models import DagBag
from airflow import settings
from airflow.utils.dag_cycle_tester import test_cycle
from airflow.models import DagModel
from airflow.operators.python import PythonOperator

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
await_2d_predictions_task = task(await_2d_predictions.await_2d_predictions)
predict_2d_local_task = task(predict_2d_local, pool="local_gpu_pool")


class AirflowDAG:
    def __init__(
        self,
        pipeline_name: str = "tim_240731",
        config_file: str = "/n/groups/datta/tim_sainburg/projects/multicamera_airflow_pipeline/multicamera_airflow_pipeline/tim_240731/default_config.yaml",
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
        # get all dags
        dag_bag = DagBag()
        existing_dag_ids = set(dag_bag.dag_ids)

        input_dag_ids = []

        # for each recording, create a new DAG
        for idx, recording_row in self.recording_df.iterrows():
            subject_id = recording_row["Subject"]
            video_recording_id = recording_row["video_recording_id"]
            dag_id = f"{self.pipeline_name}_{subject_id}_{video_recording_id}"
            logger.info(f"Attempting to create DAG {dag_id}")

            if dag_id in existing_dag_ids:
                logger.info(f"DAG {dag_id} already exists. Skipping.")
                continue

            input_dag_ids.append(dag_id)

            with DAG(
                dag_id=dag_id,
                default_args=self.default_args,
                catchup=False,
                schedule_interval=self.scheduling_interval,
            ) as dag:
                logger.info(f"Starting creation of DAG {dag_id}")
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

                predicted_2d_local = predict_2d_local_task(
                    recording_row,
                    self.job_directory,
                    self.output_directory,
                    self.config_file,
                )

                completed_2d = await_2d_predictions_task(
                    recording_row,
                    self.output_directory,
                    recheck_duration_s=60,
                    maximum_wait_time_s=10000,
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
                [synced_cams, calibrated, completed_2d] >> triangulated
                triangulated >> gimbaled
                gimbaled >> size_normed
                size_normed >> arena_aligned
                size_normed >> ego_aligned
                [arena_aligned, ego_aligned] >> cont_feats

                globals()[dag_id] = dag

                test_cycle(dag)
                dag_bag.bag_dag(dag, root_dag=dag)

                session = settings.Session()
                try:
                    if not session.query(DagModel).filter(DagModel.dag_id == dag_id).first():
                        session.add(DagModel(dag_id=dag_id))
                        session.commit()
                except Exception as e:
                    logger.error(f"Error committing to DagModel: {e}")
                finally:
                    session.close()

                logger.info(f"DAG created and committed: {dag_id}")

        print("\n--- DAG Registration Verification ---\n")
        print("Input DAG IDs (should be created):", input_dag_ids)
        print("Existing DAGs in dag_bag:", list(dag_bag.dag_ids))
        print("\n--- End of Verification ---\n")

        if dag_bag.dags:
            for dag_id, dag in dag_bag.dags.items():
                print(f"DAG ID: {dag_id}, DAG: {dag}")
        else:
            print("No dags found")


def read_google_sheet(spreadsheet_url):
    response = requests.get(spreadsheet_url)
    recording_df = pd.read_csv(BytesIO(response.content))
    return recording_df
