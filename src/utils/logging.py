# https://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("output/exp.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
