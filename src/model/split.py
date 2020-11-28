import pandas as pd
import random
from src.utils.logging import logger


def write_ids(ids, file_name):
    fid = open(file_name, "w")
    for id in ids:
        fid.write(str(id) + "\n")
    fid.close()


def split(configs):
    data_path = "output/" + configs['experiment_name'] + "/features.csv"
    logger.info('Running Data Split "' + str(configs['split'])
                + '" for ' + str(data_path))
    train_path = "output/" + configs['experiment_name'] + "/train.csv"
    val_path = train_path.replace("train.csv", "val.csv")
    test_path = train_path.replace("train.csv", "test.csv")
    data = pd.read_csv(data_path)
    random.seed(42)
    if configs['split'] == 'random':
        ids = [i for i in range(data.shape[0])]
        random.shuffle(ids)
        train_end = int(data.shape[0] * .8)
        val_end = int(data.shape[0] * .9)
        write_ids(ids[0:train_end], train_path)
        write_ids(ids[train_end:val_end], val_path)
        write_ids(ids[val_end:], test_path)
    else:
        logger.error("Data Split option " + configs['split'] + " not supported")
        raise Exception("Data Split option " + configs['split'] + " not supported")
