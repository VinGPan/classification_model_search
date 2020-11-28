import os
import os.path
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.externals import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.utils.testing import ignore_warnings

from src.utils.logging import logger


#####################################################################
# HERE IS LIST OF VARIES LIBRARIES WE STUDIED DURING SCS_3253_024 Machine Learning COURSE  that are
# relevant to classification problem. We will tray use as many ideas as possible for this project.
#####################################################################


def add_preproc_step(preproc_str, steps):
    if preproc_str == 'min_max':
        steps.append(('preprocs', MinMaxScaler()))
    elif preproc_str == 'standard_scalar':
        steps.append(('preprocs', StandardScaler()))
    elif preproc_str == 'none':
        pass
    else:
        logger.error("unsupported preprocs option " + preproc_str)
        raise Exception("unsupported preprocs option " + preproc_str)


def add_transform_step(transforms_str, steps, transforms_info, param_grid):
    if transforms_str == 'pca':
        steps.append(('transforms', PCA()))
    elif transforms_str == 'kpca':
        steps.append(('transforms', KernelPCA(kernel='rbf')))
    elif transforms_str == 'lle':
        steps.append(('transforms', LocallyLinearEmbedding()))
    # elif transforms_str == 'mds':
    #     steps.append(('transforms', MDS())) # DOES NOT HAVE transform() function
    #     param_grid[0]["transforms__n_components"] = [3, 4, 5]
    elif transforms_str == 'isomap':
        steps.append(('transforms', Isomap()))
    # elif transforms_str == 'tsne':  # DOES NOT HAVE transform() function
    #     steps.append(('transforms', TSNE()))
    #     param_grid[0]["transforms__n_components"] = [3, 4, 5]
    elif transforms_str == 'none':
        pass
    else:
        logger.error("unsupported transforms option " + transforms_str)
        raise Exception("unsupported transforms option " + transforms_str)

    if 'params' in transforms_info:
        for trans_param in transforms_info['params']:
            vals = trans_param['vals']
            name = trans_param['name']
            for vidx in range(len(vals)):
                if vals[vidx] == 'None':
                    vals[vidx] = None
            param_grid[0]["transforms__" + name] = vals


def add_classifier_step(clf_str, steps, clf_info, param_grid, tot_classes):
    if clf_str == 'logistic':
        steps.append(('clf', LogisticRegression(multi_class='auto', random_state=0, solver='liblinear')))
    elif clf_str == 'naive_bayes':
        steps.append(('clf', GaussianNB()))
    elif clf_str == 'knn':
        steps.append(('clf', KNeighborsClassifier()))
    elif clf_str == 'random_forest':
        steps.append(('clf', RandomForestClassifier()))
    elif clf_str == 'svc':
        steps.append(('clf', SVC(class_weight='balanced', random_state=42)))
    elif clf_str == 'xgboost':
        steps.append(('clf', xgb.XGBClassifier(random_state=42, objective="multi:softmax", num_class=tot_classes)))
    elif clf_str == 'adaboost':
        steps.append(('clf', AdaBoostClassifier(random_state=42)))
    elif clf_str == 'gradboost':
        steps.append(('clf', GradientBoostingClassifier(random_state=42)))
    if 'params' in clf_info:
        for clf_param in clf_info['params']:
            vals = clf_param['vals']
            name = clf_param['name']
            for vidx in range(len(vals)):
                if vals[vidx] == 'None':
                    vals[vidx] = None
            param_grid[0]["clf__" + name] = vals


@ignore_warnings(category=ConvergenceWarning)
def build_models(configs):
    data_path = "output/" + configs['experiment_name'] + "/data.csv"
    train_path = "output/" + configs['experiment_name'] + "/train.csv"
    val_path = train_path.replace("train.csv", "val.csv")

    logger.info('Building Models for ' + str(train_path))

    X = pd.read_csv(data_path)
    y = X[configs['target']].values
    X = X.drop([configs['target']], axis=1)
    X = X.values

    # Make Train - Val split.
    ids = list((pd.read_csv(train_path, header=None).values)[:, 0])
    train_end = len(ids)
    ids.extend(list((pd.read_csv(val_path, header=None).values)[:, 0]))
    X = X[ids, :]
    y = y[ids]

    train_proportion = 0.9
    X_train = X[0:train_end, :]
    y_train = y[0:train_end]
    cv_end = int(X_train.shape[0] * train_proportion)
    X_val = X[train_end:, :]
    y_val = y[train_end:]

    split_index = [-1 if i < cv_end else 0 for i in range(X_train.shape[0])]

    # PredefinedSplit Helps in running GridSearch with single predefined split.
    pds = PredefinedSplit(test_fold=split_index)

    all_scores_path = "output/" + configs["experiment_name"] + "/all_scores.pkl"
    all_scores_done_flg = "output/" + configs["experiment_name"] + "/all_scores_flg"
    if os.path.exists(all_scores_path) and os.path.exists(all_scores_done_flg):
        # If allready model building is done just return the results.
        # This is heplful to display result in a jupyter notebook.
        logger.info('All the models have been built already. Loading results from ' + str(all_scores_path))
        all_scores = pickle.load(open(all_scores_path, "rb"))
    else:
        model_scores = {}
        all_scores = []
        tot_classes = np.unique(y).shape[0]

        # For Each classifier - For Each Data transform - For each Dimensionality reduction BUILD THE MODEL!

        for clf_info in configs["models"]["classifiers"]:
            for preproc_str in configs["models"]["preprocs"]:
                for transforms_info in configs["models"]["transforms"]:
                    transforms_str = transforms_info['name']
                    clf_str = clf_info['name']
                    res_path = "output/" + configs["experiment_name"] + "/" + clf_str + "_" + preproc_str + "_" + \
                               transforms_str + ".pkl"
                    model_str = 'classifier "' + clf_str + '" with preprocessing "' + preproc_str + \
                                '" and with transform "' + transforms_str + '"'
                    logger.info('Building ' + model_str)

                    ################# ADD IMPUTERS #################################
                    steps = [('imputer', SimpleImputer(strategy='mean'))]
                    param_grid = [{}]
                    ###################################################################

                    ################# PICK A DATA TRANSFORM ###########################
                    add_preproc_step(preproc_str, steps)
                    ###################################################################

                    ################### PICK A Dimensionality reduction method ################
                    add_transform_step(transforms_str, steps, transforms_info, param_grid)
                    ###################################################################

                    ##################### PICK A classifier #######################
                    add_classifier_step(clf_str, steps, clf_info, param_grid, tot_classes)
                    ###################################################################

                    ##################### Perform grid search #####################
                    pipeline = Pipeline(steps=steps)
                    clf = GridSearchCV(estimator=pipeline, cv=pds, param_grid=param_grid,
                                       verbose=1, scoring='balanced_accuracy')
                    if os.path.exists(res_path):
                        logger.info('Model has been built already. Loading the model ' + str(res_path))
                        clf = joblib.load(res_path)
                    else:
                        try:
                            clf.fit(X_train, y_train)
                        except Exception as e:
                            logger.info("Model building crashed for " + model_str)
                            logger.error(e, exc_info=True)
                            continue
                        joblib.dump(clf, res_path)
                    ###################################################################

                    ################### Perform Validation ################
                    logger.info("Validating the model " + model_str)
                    if clf_str not in model_scores:
                        model_scores[clf_str] = []
                    val_preds = clf.predict(X_val)
                    accuracy = np.round(accuracy_score(y_val, val_preds) * 100, 1)
                    bal_accuracy = np.round(balanced_accuracy_score(y_val, val_preds) * 100, 1)
                    f1 = np.round(f1_score(y_val, val_preds, average='weighted', labels=np.unique(val_preds)), 2)
                    model_scores[clf_str].append([res_path, accuracy, bal_accuracy, f1, clf.best_params_])
                    res_str = ' => accuracy = ' + str(accuracy) + '%, F1 = ' + str(f1)
                    logger.info(model_str + res_str)
                    all_scores.append([clf_str, preproc_str, transforms_str, accuracy, bal_accuracy, f1])
                    pickle.dump(all_scores, open(all_scores_path, "wb"))
                    ###################################################################
                # end each transforms
            # end each preprocs
        # end each classifiers
        fid = open(all_scores_done_flg, "wb")
        fid.close()
        logger.info("All the models are built.")

    # Find top three models
    logger.info("Top 3 classifiers:")
    all_scores = sorted(all_scores, key=lambda x: x[4], reverse=True)
    prev_cls = None
    cls_count = 0
    top_scores = []
    for score in all_scores:
        if prev_cls == score[0]:
            continue
        res_str = 'classifier = ' + score[0] + ", preproc = " + score[1] + ", transform = " + score[2]
        res_str += (' => accuracy = ' + str(score[3]) + '%, F1 = ' + str(score[4]))
        logger.info(res_str)
        prev_cls = score[0]
        cls_count += 1
        top_scores.append(score)
        if cls_count == 3:
            break
    col_names = ['classifier', 'preprocess', 'transform', 'accuracy', 'balanced_accuracy', 'f1_score']
    df = pd.DataFrame(np.array(top_scores), columns=col_names)
    all_scores = pd.DataFrame(np.array(all_scores), columns=col_names)
    return all_scores, df
