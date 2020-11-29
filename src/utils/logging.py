# https://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout

import logging
import os
from datetime import datetime

try:
    os.makedirs("output/log/")
except:
    pass

date_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("output/log/" + date_time + ".log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
