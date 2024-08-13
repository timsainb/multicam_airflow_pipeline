from datetime import datetime, timedelta
import pandas as pd
import requests
from io import BytesIO
from pathlib import Path
from multicamera_airflow_pipeline.tim_240731.interface.o2 import O2Runner
from datetime import datetime
import textwrap
import inspect
import time
import yaml

import logging

logger = logging.getLogger(__name__)


def await_2d_predictions(
    recording_row, job_directory, output_directory, config_file, recheck_duration_s=60
):
    """This function waits for 2d predictions to complete, before returning"""

    # Find the number of videos

    # find the number of completed predictions

    # if the number of videos matches the number of completed predictions, return True

    return True
