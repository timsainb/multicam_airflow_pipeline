import sys

import numpy as np
from o2_utils.slurm import evaluate_resource_usage
import yaml


def main(airflow_logs_file):
    # Read the logs
    with open(airflow_logs_file, "r") as f:
        logs = yaml.safe_load(f)

    tasks_to_analyze = [
        "arena_alignment",
        "compute_continuous_features",
        "egocentric_alignment",
        "predict_2d",
        "run_gimbal",
        "size_normalization",
        # "spikesorting",
        "triangulation",
    ]

    # See how long each task took relative to the time requested
    for task in tasks_to_analyze:
        time_fractions = []
        mem_fractions = []
        for run_name, run_info in logs.items():
            # Skip stuff we dont want
            if (
                run_name == "refresh_pipeline_dag"
                or task not in run_info
                or run_info[task]["job_id"] == "Not found"
            ):
                continue

            # Get the job info
            jobid = run_info[task]["job_id"]
            jobinfo = evaluate_resource_usage(jobid, plot=False)

            # If job isn't finished w status COMPLETED, will not be returned in the dict, so skip that
            if jobid not in jobinfo:
                continue

            # Save the info
            time_fractions.append(jobinfo[jobid]["time_fraction"])
            mem_fractions.append(jobinfo[jobid]["mem_fraction"])

        # Report the results
        print(f"Task: {task}")
        print(f"\tAnalyzed {len(time_fractions)} jobs")
        print(f"\tMedian time fraction: {np.median(time_fractions):.3f}")
        print(f"\tMinimum time fraction: {np.min(time_fractions):.3f}")
        print(f"\tMaximum time fraction: {np.max(time_fractions):.3f}")

        print(f"\tMedian mem fraction: {np.median(mem_fractions):.3f}")
        print(f"\tMinimum mem fraction: {np.min(mem_fractions):.3f}")
        print(f"\tMaximum mem fraction: {np.max(mem_fractions):.3f}")

    # TODO: Plot the results

    return


if __name__ == "__main__":
    logs_file = sys.argv[1]
    main(logs_file)
