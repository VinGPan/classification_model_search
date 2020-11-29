import pickle

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.utils.logging import logger


def prepare_scores_df(scores):
    col_names = ['classifier', 'preprocess', 'transform', 'accuracy', 'balanced_accuracy', 'f1_score']
    all_scores = []
    for info in scores:
        all_scores.append([info[v] for v in col_names])
    df = pd.DataFrame(np.array(all_scores), columns=col_names)
    return df


def report_training_results(configs):
    all_scores_path = "output/" + configs["experiment_name"] + "/all_scores.pkl"
    logger.info('Running Train Reports. Loading training results from ' + str(all_scores_path))
    all_scores = pickle.load(open(all_scores_path, "rb"))

    logger.info("Top 3 classifiers:")
    all_scores = sorted(all_scores, key=lambda x: x['balanced_accuracy'], reverse=True)
    prev_cls = None
    cls_count = 0
    top_scores = []
    for score in all_scores:
        if prev_cls == score['classifier']:
            continue
        res_str = 'classifier = ' + score['classifier'] + ", preproc = " + \
                  score['preprocess'] + ", transform = " + score['transform']
        res_str += (' => accuracy = ' + str(score['accuracy']) + '%, F1 = ' + str(score['f1_score']))
        logger.info(res_str)
        prev_cls = score['classifier']
        cls_count += 1
        top_scores.append(score)
        if cls_count == 3:
            break

    df = prepare_scores_df(top_scores)
    all_scores = prepare_scores_df(all_scores)
    return all_scores, df


def report_test_results(configs):
    logger.info('Running Test Reports')
    X, y = prepare_test_data(configs)
    logger.info('Predicting on test set')
    best_model_path = "output/" + configs["experiment_name"] + "/best_model.pkl"
    clf = joblib.load(best_model_path)
    pred_y = clf.predict(X)
    col_names = ['accuracy', 'balanced_accuracy', 'f1_score']
    acc = np.round(accuracy_score(y, pred_y) * 100, 1)
    b_acc = np.round(balanced_accuracy_score(y, pred_y) * 100, 1)
    f1 = np.round(f1_score(y, pred_y, average='micro'), 2)
    res_str = 'On test set => accuracy = ' + str(acc) + '%, F1 = ' + str(f1)
    logger.info(res_str)
    all_scores = [[acc, b_acc, f1]]
    df = pd.DataFrame(np.array(all_scores), columns=col_names)
    return df


def prepare_test_data(configs):
    logger.info('Reading Test Data')
    test_path = "output/" + configs['experiment_name'] + "/test.csv"
    features_path = "output/" + configs['experiment_name'] + "/features.csv"

    X = pd.read_csv(features_path)
    y = X[configs['target']].values
    X = X.drop([configs['target']], axis=1)
    X = X.values

    ids = list((pd.read_csv(test_path, header=None).values)[:, 0])
    X = X[ids, :]
    y = y[ids]

    return X, y


def report(configs):
    all_scores, df = report_training_results(configs)
    test_df = report_test_results(configs)
    return all_scores, df, test_df
