import re
from datetime import datetime
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)

def extract_datetime_from_folder_name(folder_name):
    """
    Extracts a datetime object from a folder name based on predefined patterns.

    This function searches for datetime formats within the given folder name. 
    It supports the following formats:
    1. '%y-%m-%d-%H-%M-%S-%f' - Matches a pattern like '22-09-15-12-30-45-123456'.
    2. '%Y%m%d' - Matches a pattern like '20220915'.
    3. [add your own!]

    Parameters:
    folder_name (str): The name of the folder to search for datetime patterns.

    Returns:
    datetime or None: A datetime object if a valid datetime pattern is found, 
                        otherwise None if no pattern matches or parsing fails.

    Examples:
    >>> extract_datetime_from_folder_name("backup-22-09-15-12-30-45-123456")
    datetime.datetime(2022, 9, 15, 12, 30, 45, 123456)

    >>> extract_datetime_from_folder_name("log-20230915-report")
    datetime.datetime(2023, 9, 15, 0, 0)

    >>> extract_datetime_from_folder_name("data-2022-09-15")
    None

    """

    # Define the regex patterns for the datetime formats you expect
    patterns = [
        (r'.*(\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{6}).*', '%y-%m-%d-%H-%M-%S-%f'),
        (r'.*(\d{8}).*', '%Y%m%d')
    ]
    
    # Iterate over the patterns
    for pattern, date_format in patterns:
        match = re.search(pattern, folder_name)
        if match:
            try:
                return datetime.strptime(match.group(1), date_format)
            except ValueError:
                # If the conversion fails, continue checking other patterns
                continue
    
    # Return None if no datetime pattern was found
    logger.warning(f"No datetime pattern found in folder name: {folder_name}. Using current time.")
    return datetime.now()