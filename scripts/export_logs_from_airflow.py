from datetime import datetime
import os
import re

from airflow.configuration import conf
from airflow.models import DagRun, TaskInstance
from airflow.utils.db import create_session
from airflow.utils.state import State
from sqlalchemy import and_, func
import yaml


def get_latest_successful_dag_runs():
    with create_session() as session:
        subquery = (
            session.query(
                DagRun.dag_id, func.max(DagRun.execution_date).label("max_date")
            )
            .filter(DagRun.state == State.SUCCESS)
            .group_by(DagRun.dag_id)
            .subquery()
        )

        latest_successful_runs = (
            session.query(DagRun)
            .join(
                subquery,
                and_(
                    DagRun.dag_id == subquery.c.dag_id,
                    DagRun.execution_date == subquery.c.max_date,
                ),
            )
            .filter(DagRun.state == State.SUCCESS)
            .all()
        )
    return latest_successful_runs


def get_task_instances(dag_run):
    with create_session() as session:
        task_instances = (
            session.query(TaskInstance)
            .filter(
                TaskInstance.dag_id == dag_run.dag_id,
                TaskInstance.execution_date == dag_run.execution_date,
            )
            .all()
        )
    return task_instances


def get_task_logs(task_instance):
    try:
        log_directory = conf.get("logging", "base_log_folder")
        dag_id = task_instance.dag_id
        task_id = task_instance.task_id
        run_id = task_instance.run_id
        try_number = task_instance.try_number

        log_file_path = os.path.join(
            log_directory,
            f"dag_id={dag_id}",
            f"run_id={run_id}",
            f"task_id={task_id}",
            f"attempt={try_number}.log",
        )

        if os.path.exists(log_file_path):
            with open(log_file_path, "r") as log_file:
                log_content = log_file.read()

            # Extract job ID
            job_id_match = re.search(
                r"Job submitted successfully with job id: (\d+)", log_content
            )
            job_id = job_id_match.group(1) if job_id_match else "Not found"

            # Extract start and end dates
            date_match = re.search(
                r"start_date=(\d{8}T\d{6}), end_date=(\d{8}T\d{6})", log_content
            )
            if date_match:
                start_date = datetime.strptime(date_match.group(1), "%Y%m%dT%H%M%S")
                end_date = datetime.strptime(date_match.group(2), "%Y%m%dT%H%M%S")
                run_time = end_date - start_date
                run_time_str = str(run_time).split(".")[0]  # Format as HH:MM:SS
            else:
                start_date, end_date, run_time_str = (
                    "Not found",
                    "Not found",
                    "Not found",
                )

            # Strip date/time prefix from logs
            log_lines = log_content.split("\n")
            stripped_log_lines = [
                re.sub(r"^\[[^\]]+\]\s*", "", line) for line in log_lines
            ]
            stripped_log_content = "\n".join(stripped_log_lines)

            return {
                "log_content": stripped_log_content,
                "job_id": job_id,
                "start_date": str(start_date),
                "end_date": str(end_date),
                "run_time": run_time_str,
            }
        else:
            return {
                "log_content": f"Log file not found: {log_file_path}",
                "job_id": "Not found",
                "start_date": "Not found",
                "end_date": "Not found",
                "run_time": "Not found",
            }
    except Exception as e:
        return {
            "log_content": f"Error retrieving logs: {str(e)}",
            "job_id": "Error",
            "start_date": "Error",
            "end_date": "Error",
            "run_time": "Error",
        }


def export_logs_to_yaml():
    dag_runs = get_latest_successful_dag_runs()
    all_logs = {}

    for dag_run in dag_runs:
        dag_logs = {}
        task_instances = get_task_instances(dag_run)

        for task_instance in task_instances:
            task_log_info = get_task_logs(task_instance)
            dag_logs[task_instance.task_id] = task_log_info

        all_logs[dag_run.dag_id] = dag_logs

    with open("exported_airflow_logs.yml", "w") as yaml_file:
        yaml.dump(all_logs, yaml_file, default_flow_style=False)


if __name__ == "__main__":
    export_logs_to_yaml()
    print("Logs exported to exported_airflow_logs.yml")
