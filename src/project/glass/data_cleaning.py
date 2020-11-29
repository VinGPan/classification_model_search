from src.utils.logging import logger
import pandas as pd


def cleanse(configs):
    data_path = configs["data_path"]
    res_path = "output/" + configs['experiment_name'] + "/data.csv"
    logger.info('Running Data Cleansing "' + str(configs['cleanse']) + '" for ' + data_path)
    df = pd.read_csv(data_path, header=None)
    col_names = ['id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'type']
    df.columns = col_names
    df.drop(['id'], axis=1, inplace=True)
    df.to_csv(res_path, index=False)
