import pandas as pd
import shutil
from src.utils.logging import logger


def cleanse(configs):
    logger.info('Running Data Cleansing "' + str(configs['cleanse'])
                + '" for ' + str(configs["data_path"]))
    data_path = configs["data_path"]
    res_path = "output/" + configs['experiment_name'] + "/data.csv"
    if configs['cleanse'] == 'none':
        shutil.copyfile(data_path, res_path)
    else:
        logger.error("Data Cleansing option " + configs['cleanse'] + " not supported")
        raise Exception("Data Cleansing option " + configs['cleanse'] + " not supported")

