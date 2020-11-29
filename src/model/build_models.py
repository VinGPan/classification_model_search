import os
import os.path
import pickle
from shutil import copyfile

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
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.utils.testing import ignore_warnings

from src.model.utils import makedir
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
    ###########  classifiers  ################
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
    #########################################
    elif clf_str == 'linear':
        steps.append(('clf', LinearRegression()))
    else:
        logger.error("Classifier option " + clf_str + " not supported")
        raise Exception("Classifier option " + clf_str + " not supported")
    if 'params' in clf_info:
        for clf_param in clf_info['params']:
            vals = clf_param['vals']
            name = clf_param['name']
            for vidx in range(len(vals)):
                if vals[vidx] == 'None':
                    vals[vidx] = None
            param_grid[0]["clf__" + name] = vals


def get_crashed_list(configs):
    crashed_list_fname = "output/" + configs["experiment_name"] + "/" + "crashed.txt"
    if os.path.exists(crashed_list_fname):
        fid = open(crashed_list_fname, "r")
        crashed_list = []
        for line in fid:
            crashed_list.append(line.strip())
        fid.close()
    else:
        crashed_list = []
        fid = open(crashed_list_fname, "w")
        fid.close()
    return crashed_list, crashed_list_fname


@ignore_warnings(category=ConvergenceWarning)
def build_models(configs):
    mtype = configs['mtype']
    data_path = "output/" + configs['experiment_name'] + "/features.csv"
    train_path = "output/" + configs['experiment_name'] + "/train.csv"
    val_path = train_path.replace("train.csv", "val.csv")
    makedir("output/" + configs['experiment_name'] + "/interim")

    logger.info('Building Models for ' + str(train_path))

    # Read train and val sets
    X = pd.read_csv(data_path)
    y = X[configs['target']].values
    X = X.drop([configs['target']], axis=1)
    X = X.values

    ids = list((pd.read_csv(train_path, header=None).values)[:, 0])
    ids.extend(list((pd.read_csv(val_path, header=None).values)[:, 0]))
    X = X[ids, :]
    y = y[ids]

    all_scores_path = "output/" + configs["experiment_name"] + "/all_scores.pkl"
    all_scores_done_flg = "output/" + configs["experiment_name"] + "/all_scores_flg"
    crashed_list, crashed_list_fname = get_crashed_list(configs)
    if os.path.exists(all_scores_path) and os.path.exists(all_scores_done_flg):
        # If allready model building is done just return the results.
        # This is heplful to display result in a jupyter notebook.
        logger.info('All the models have been built already.')
    else:
        all_scores = []
        if mtype == 'classification':
            tot_classes = np.unique(y).shape[0]
        else:
            tot_classes = None

        # For Each classifier - For Each Data transform - For each Dimensionality reduction BUILD THE MODEL!

        for clf_info in configs["models"]["classifiers"]:
            for preproc_str in configs["models"]["preprocs"]:
                for transforms_info in configs["models"]["transforms"]:
                    transforms_str = transforms_info['name']
                    clf_str = clf_info['name']
                    res_path = "output/" + configs["experiment_name"] + "/interim/" + clf_str + "_" + preproc_str + \
                               "_" + transforms_str + ".pkl"
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
                    if mtype == 'classification':
                        scoring = {'balanced_accuracy': make_scorer(balanced_accuracy_score),
                                   'accuracy': make_scorer(accuracy_score),
                                   'f1': 'f1_micro'}
                        refit = 'balanced_accuracy'
                    else:
                        scoring = {'r2': make_scorer(r2_score), 'mae': make_scorer(mean_absolute_error),
                                   'mse': make_scorer(mean_squared_error)}
                        refit = 'r2'

                    clf = GridSearchCV(estimator=pipeline, cv=5, param_grid=param_grid,
                                       verbose=1, scoring=scoring, refit=refit)
                    if os.path.exists(res_path):
                        logger.info('Model has been built already. Loading the model ' + str(res_path))
                        clf = joblib.load(res_path)
                    elif model_str in crashed_list:
                        logger.info('Model fails to build. Ignoring. Please consider modifying the params.')
                        continue
                    else:
                        try:
                            clf.fit(X, y)
                        except Exception as e:
                            logger.info("Model building crashed for " + model_str)
                            logger.error(e, exc_info=True)
                            fid = open(crashed_list_fname, "a")
                            fid.write(model_str + "\n")
                            fid.close()
                            continue
                        joblib.dump(clf, res_path)
                    ###################################################################

                    ################### Record scores ################
                    cv_results = clf.cv_results_
                    score_info = {'classifier': clf_str, 'preprocess': preproc_str, 'transform': transforms_str,
                                  'res_path': res_path}
                    for scorer in scoring:
                        best_index = np.nonzero(cv_results['rank_test_%s' % scorer] == 1)[0][0]
                        best_score = cv_results['mean_test_%s' % scorer][best_index]
                        if scorer == 'balanced_accuracy':
                            score_info['balanced_accuracy'] = np.round(best_score * 100, 1)
                        elif scorer == 'accuracy':
                            score_info['accuracy'] = np.round(best_score * 100, 1)
                        elif scorer == 'f1':
                            score_info['f1_score'] = np.round(best_score, 2)
                        elif scorer == 'r2':
                            score_info['r2'] = np.round(best_score, 2)
                        elif scorer == 'mae':
                            score_info['mae'] = np.round(best_score, 2)
                        elif scorer == 'mse':
                            score_info['mse'] = np.round(best_score, 2)
                    if mtype == 'classification':
                        res_str = ' => accuracy = ' + str(score_info['accuracy']) + '%, F1 = ' + \
                                  str(score_info['f1_score'])
                    else:
                        res_str = ' => r2 = ' + str(score_info['r2'])
                    logger.info(model_str + res_str)

                    all_scores.append(score_info)
                    pickle.dump(all_scores, open(all_scores_path, "wb"))
                    ###################################################################
                # end each transforms
            # end each preprocs
        # end each classifiers
        if mtype == 'classification':
            all_scores = sorted(all_scores, key=lambda x: x['balanced_accuracy'], reverse=True)
        else:
            all_scores = sorted(all_scores, key=lambda x: x['r2'], reverse=True)

        best_model_path = "output/" + configs["experiment_name"] + "/best_model.pkl"
        copyfile(all_scores[0]['res_path'], best_model_path)

        fid = open(all_scores_done_flg, "wb")
        fid.close()
        logger.info("All the models are built.")
