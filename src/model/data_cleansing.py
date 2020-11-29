import shutil

from src.utils.logging import logger


def cleanse(configs):
    res_path = "output/" + configs['experiment_name'] + "/data.csv"
    if configs['cleanse'] == 'none':
        logger.info('Running Data Cleansing "' + str(configs['cleanse'])
                    + '" for ' + str(configs["data_path"]))
        data_path = configs["data_path"]
        shutil.copyfile(data_path, res_path)
    else:
        module = __import__(configs['cleanse'], fromlist=['blah'])
        func = getattr(module, 'cleanse')
        func(configs)
