from pathlib import Path
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

from multicamera_airflow_pipeline.tim_240731.airflow.dag_o2 import AirflowDAG

# gets the dag
airflow_dag = AirflowDAG(
    spreadsheet_url="https://docs.google.com/spreadsheet/ccc?key=1jACsUmxuJ9Une59qmvzZGc1qXezKhKzD1zho2sEfcrU&output=csv&gid=0",
    config_file="/n/groups/datta/Jonah/Local_code_groups/6cam_repos/multicam_airflow_pipeline/multicamera_airflow_pipeline/jonah_241112/default_config_v2.yaml",
)

# runs the dag
airflow_dag.generate_dags()

# Path to this script
dag_file_path = Path(__file__)


def touch_self():
    """
    Touch the DAG's .py file to update its modification time.
    This will cause the Airflow scheduler to reload the DAG.
    """
    dag_file_path.touch()
    print(f"Touched file: {dag_file_path}")


# Define the DAG
dag = DAG(
    "refresh_pipeline_dag",
    default_args={
        "owner": "airflow",
        "depends_on_past": False,
        "start_date": days_ago(1),  # Update to current date
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    description="This DAG touches its own .py file to refresh the pipeline",
    schedule_interval=timedelta(minutes=10),  # Adjust the schedule as needed
    catchup=False,
    is_paused_upon_creation=False,
)

# Define the task
touch_self_task = PythonOperator(
    task_id="Refresh_DAG",
    python_callable=touch_self,
    dag=dag,
)

# Define task dependencies (if any)
touch_self_task
