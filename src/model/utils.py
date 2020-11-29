import argparse
import os

import yaml


def read_yml(yml_name):
    with open(yml_name, 'r') as ymlfile:
        configs = yaml.load(ymlfile, Loader=yaml.CLoader)
    return configs


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yml_path",
        required=True
    )
    args = parser.parse_args()
    return args.yml_path


def makedir(path):
    try:
        os.mkdir(path)
    except:
        pass
