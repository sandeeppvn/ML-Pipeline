'''
Author: Sandeep Pvn
Created Date: 25 March 2021
'''

import logging
import os
from datetime import datetime

# LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
# Add time and date to the log file name
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.log"
LOG_PATH = os.path.join(os.getcwd(), "logs", LOG_FILE)

os.makedirs(LOG_PATH, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_PATH, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
