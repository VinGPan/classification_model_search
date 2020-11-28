from src.model.utils import read_args, read_yml, makedir
from src.model.data_cleansing import cleanse
from src.model.compute_features import compute_features
from src.model.split import split
from src.model.build_models import build_models
from src.utils.logging import logger


def run_experiment(commnad, yml_name):
    configs = read_yml(yml_name)
    makedir("output/" + configs['experiment_name'])
    if commnad == 'cleanup':
        cleanse(configs)
    elif commnad == 'features':
        compute_features(configs)
    elif commnad == 'split':
        split(configs)
    elif commnad == 'model':
        return build_models(configs)
    elif commnad == 'all':
        cleanse(configs)
        compute_features(configs)
        split(configs)
        return build_models(configs)


if __name__ == '__main__':
    commnad, exp_yml_name = read_args()
    logger.info('Running command "' + str(commnad) + '" for ' + str(exp_yml_name))
    try:
        run_experiment(commnad, exp_yml_name)
    except Exception as e:
        logger.error(e, exc_info=True)
