import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import confusion_matrix

from src.utils.logging import logger


def plot_confusion_matrix_(cm, ax, classes, normalize, title):
    cmap = plt.cm.Blues
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )


def plot_confusion_matrix(y_true, y_pred, classes, configs):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    logger.info("Confusion Matrix\n" + str(cm))
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cmn = np.nan_to_num(cmn)

    fig, ax = plt.subplots(1, 2, figsize=(18, 15))
    plot_confusion_matrix_(cm, ax[0], classes, False, "Confusion Matrix")
    plot_confusion_matrix_(cmn, ax[1], classes, True, "Normalized Confusion Matrix")

    fig.tight_layout()
    res = "output/" + configs["experiment_name"] + "/report/test_confusion_matrix.png"
    fig.savefig(res)


def prepare_scores_df(scores, mtype):
    if mtype == 'classification':
        col_names = ['classifier', 'preprocess', 'transform', 'accuracy', 'balanced_accuracy', 'f1_score']
    else:
        col_names = ['classifier', 'preprocess', 'transform', 'r2', 'mae', 'mse']
    all_scores = []
    for info in scores:
        all_scores.append([info[v] for v in col_names])
    df = pd.DataFrame(np.array(all_scores), columns=col_names)
    return df


def report_training_results(configs):
    mtype = configs['mtype']
    all_scores_path = "output/" + configs["experiment_name"] + "/all_scores.pkl"
    logger.info('Running Train Reports. Loading training results from ' + str(all_scores_path))
    all_scores = pickle.load(open(all_scores_path, "rb"))

    logger.info("Top 3 models:")
    if mtype == 'classification':
        all_scores = sorted(all_scores, key=lambda x: x['balanced_accuracy'], reverse=True)
    else:
        all_scores = sorted(all_scores, key=lambda x: x['r2'], reverse=True)
    prev_cls = None
    cls_count = 0
    top_scores = []
    for score in all_scores:
        if prev_cls == score['classifier']:
            continue
        res_str = 'classifier = ' + score['classifier'] + ", preproc = " + \
                  score['preprocess'] + ", transform = " + score['transform']
        if mtype == 'classification':
            res_str += (' => accuracy = ' + str(score['accuracy']) + '%, F1 = ' + str(score['f1_score']))
        else:
            res_str += (' => r2 = ' + str(score['r2']))
        logger.info(res_str)
        prev_cls = score['classifier']
        cls_count += 1
        top_scores.append(score)
        if cls_count == 3:
            break

    df = prepare_scores_df(top_scores, mtype)
    all_scores = prepare_scores_df(all_scores, mtype)
    res = "output/" + configs["experiment_name"] + "/report/cv_top_scores.csv"
    df.to_csv(res, index=False)
    res = "output/" + configs["experiment_name"] + "/report/cv_sorted_scores.csv"
    all_scores.to_csv(res, index=False)


def report_test_results_classification(configs):
    logger.info('Running Test Reports')
    ############## Make prediction on test ##############
    X, y = prepare_test_data(configs)
    logger.info('Predicting on test set')
    best_model_path = "output/" + configs["experiment_name"] + "/best_model.pkl"
    clf = joblib.load(best_model_path)
    pred_y = clf.predict(X)
    classes = list(set(np.unique(y)).union(np.unique(pred_y)))
    ####################################################

    ############## Compute scores #######################
    col_names = ['accuracy', 'balanced_accuracy', 'f1_score']
    acc = np.round(accuracy_score(y, pred_y) * 100, 1)
    b_acc = np.round(balanced_accuracy_score(y, pred_y) * 100, 1)
    f1 = np.round(f1_score(y, pred_y, average='micro'), 2)
    res_str = 'On test set => accuracy = ' + str(acc) + '%, F1 = ' + str(f1)
    all_scores = [[acc, b_acc, f1]]
    scores = pd.DataFrame(np.array(all_scores), columns=col_names)
    logger.info(res_str)
    res = "output/" + configs["experiment_name"] + "/report/test_accuracy_scores.csv"
    scores.to_csv(res, index=False)
    ####################################################

    ############## Confusion matrix #######################
    plot_confusion_matrix(y, pred_y, classes, configs)
    ####################################################

    ############## classification report #######################
    logger.info("\n" + str(classification_report(y, pred_y)))
    col_names = [' ', 'precision', 'recall', 'f1-score', 'support']
    class_report = classification_report(y, pred_y, output_dict=True)
    class_report_arr = []
    for c in classes:
        rep = class_report[str(c)]
        class_report_arr.append([c, np.round(rep['precision'], 2), np.round(rep['recall'], 2),
                                 np.round(rep['f1-score'], 2), rep['support']])
    class_report_arr.append([' ', ' ', ' ', ' ', ' '])
    for c in ['micro avg', 'macro avg', 'weighted avg']:
        rep = class_report[c]
        class_report_arr.append([c, np.round(rep['precision'], 2), np.round(rep['recall'], 2),
                                 np.round(rep['f1-score'], 2), rep['support']])
    class_report_arr = pd.DataFrame(np.array(class_report_arr), columns=col_names)
    res = "output/" + configs["experiment_name"] + "/report/test_classification_report.csv"
    class_report_arr.to_csv(res, index=False)
    ####################################################


def report_test_results_regression(configs):
    logger.info('Running Test Reports')
    ############## Make prediction on test ##############
    X, y = prepare_test_data(configs)
    logger.info('Predicting on test set')
    best_model_path = "output/" + configs["experiment_name"] + "/best_model.pkl"
    clf = joblib.load(best_model_path)
    pred_y = clf.predict(X)
    ####################################################

    ############## Compute scores #######################
    col_names = ['r2', 'mae', 'mse']
    r2 = np.round(r2_score(y, pred_y) * 100, 2)
    mse = np.round(mean_squared_error(y, pred_y) * 100, 2)
    mae = np.round(mean_absolute_error(y, pred_y), 2)
    res_str = 'On test set => r2 = ' + str(r2)
    all_scores = [[r2, mse, mae]]
    scores = pd.DataFrame(np.array(all_scores), columns=col_names)
    logger.info(res_str)
    res = "output/" + configs["experiment_name"] + "/report/test_accuracy_scores.csv"
    scores.to_csv(res, index=False)
    ####################################################


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
    try:
        os.makedirs("output/" + configs["experiment_name"] + "/report")
    except:
        pass
    report_training_results(configs)
    mtype = configs['mtype']
    if mtype == 'classification':
        report_test_results_classification(configs)
    else:
        report_test_results_regression(configs)
