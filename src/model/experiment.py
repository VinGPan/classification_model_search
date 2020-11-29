from src.model.build_models import build_models
from src.model.compute_features import compute_features
from src.model.data_cleansing import cleanse
from src.model.report import report
from src.model.split import split
from src.model.utils import read_args, read_yml, makedir
from src.utils.logging import logger


def run_experiment(yml_name):
    configs = read_yml(yml_name)
    makedir("output/" + configs['experiment_name'])
    cleanse(configs)
    compute_features(configs)
    split(configs)
    build_models(configs)
    return report(configs)


if __name__ == '__main__':
    exp_yml_name = read_args()
    logger.info('Running experiment ' + str(exp_yml_name))
    try:
        run_experiment(exp_yml_name)
    except Exception as e:
        logger.error(e, exc_info=True)
