import pandas as pd
import shutil


def cleanup(configs):
    data_path = configs["data_path"]
    res_path = "output/" + configs['experiment_name'] + "/data.csv"
    if configs['cleanup'] == 'none':
        shutil.copyfile(data_path, res_path)
    else:
        assert False, "cleanup option " + configs['cleanup'] + " not supported"
