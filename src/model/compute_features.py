import shutil

from src.utils.logging import logger


def compute_features(configs):
    data_path = "output/" + configs['experiment_name'] + "/data.csv"
    logger.info('Running Feature Compute "' + str(configs['features'])
                + '" for ' + str(data_path))
    res_path = "output/" + configs['experiment_name'] + "/features.csv"
    if configs['features'] == 'none':
        shutil.copyfile(data_path, res_path)
    else:
        logger.error("Feature Compute option " + configs['experiment_name'] + " not supported")
        raise Exception("Feature Compute option " + configs['experiment_name'] + " not supported")
