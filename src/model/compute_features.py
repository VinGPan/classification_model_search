import numpy as np


def compute_features(configs):
    data_path = "output/" + configs['experiment_name'] + "/data.csv"
    res_path = "output/" + configs['experiment_name'] + "/features.csv"
    if configs['features'] == 'none':
        pass
    else:
        assert False, "cleanup option " + configs['cleanup'] + " not supported"
