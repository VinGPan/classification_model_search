import pandas as pd

from src.model.utils import read_yml


def prepare_test_data(yml_name):
    configs = read_yml(yml_name)
    data_path = "output/" + configs['experiment_name'] + "/data.csv"
    test_path = "output/" + configs['experiment_name'] + "/test.csv"
    features_path = "output/" + configs['experiment_name'] + "/features.csv"

    X = pd.read_csv(data_path)
    y = X[configs['target']].values
    X = X.drop([configs['target']], axis=1)
    X = X.values

    # Make Train - Val split.
    ids = list((pd.read_csv(test_path, header=None).values)[:, 0])
    X = X[ids, :]
    y = y[ids]

    return X, y
