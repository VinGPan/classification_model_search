from src.model.utils import read_args, read_yml, makedir
from src.model.data_cleanup import cleanup
from src.model.compute_features import compute_features
from src.model.split import split
from src.model.build_models import build_models


def run_experiment(commnad, yml_name):
    configs = read_yml(yml_name)
    makedir("output/" + configs['experiment_name'])
    if commnad == 'cleanup':
        cleanup(configs)
    elif commnad == 'features':
        compute_features(configs)
    elif commnad == 'split':
        split(configs)
    elif commnad == 'model':
        build_models(configs)
    elif commnad == 'all':
        cleanup(configs)
        compute_features(configs)
        split(configs)
        build_models(configs)


if __name__ == '__main__':
    commnad, exp_yml_name = read_args()
    run_experiment(commnad, exp_yml_name)
