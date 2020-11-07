import pandas as pd
import random


def write_ids(ids, file_name):
    fid = open(file_name, "w")
    for id in ids:
        fid.write(str(id) + "\n")
    fid.close()


def split(configs):
    random.seed(42)
    data_path = "output/" + configs['experiment_name'] + "/data.csv"
    train_path = "output/" + configs['experiment_name'] + "/train.csv"
    val_path = train_path.replace("train.csv", "val.csv")
    test_path = train_path.replace("train.csv", "test.csv")
    data = pd.read_csv(data_path)
    if configs['split'] == 'random':
        ids = [i for i in range(data.shape[0])]
        random.shuffle(ids)
        train_end = int(data.shape[0] * .8)
        val_end = int(data.shape[0] * .9)
        write_ids(ids[0:train_end], train_path)
        write_ids(ids[train_end:val_end], val_path)
        write_ids(ids[val_end:], test_path)
    else:
        assert False, "cleanup option " + configs['cleanup'] + " not supported"
