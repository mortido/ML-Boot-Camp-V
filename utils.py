import os
import random
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit


def folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_model_result(model_name, train_predict, test_predict, split_target, split_predict):
    folder('models')
    folder('models/' + model_name)
    pd.DataFrame(train_predict).to_csv('models/' + model_name + '/train_predict.csv',
                                       index=False,
                                       header=False,
                                       sep=';')
    pd.DataFrame(test_predict).to_csv('models/' + model_name + '/test_predict.csv',
                                      index=False,
                                      header=False,
                                      sep=';')
    if split_target is not None:
        pd.DataFrame(split_target).to_csv('models/' + model_name + '/split_target.csv',
                                          index=False,
                                          header=False,
                                          sep=';')
        pd.DataFrame(split_predict).to_csv('models/' + model_name + '/split_predict.csv',
                                           index=False,
                                           header=False,
                                           sep=';')


def load_model(model_name):
    directory = 'models/' + model_name + '/'
    if not os.path.exists(directory):
        return 1 / 0
    train_predict = pd.read_csv(directory + 'train_predict.csv', header=None, sep=';').values.ravel()
    test_predict = pd.read_csv(directory + 'test_predict.csv', header=None, sep=';').values.ravel()
    split_target = pd.read_csv(directory + 'split_target.csv', header=None, sep=';').values.ravel()
    split_predict = pd.read_csv(directory + 'split_predict.csv', header=None, sep=';').values.ravel()

    return train_predict, test_predict, split_target, split_predict


from scipy import stats as scistats


# scistats.gmean(np.vstack((y_train,y_train/2)))

def merge_models(models, method='gmean'):
    train_target = pd.read_csv('train.csv', sep=';')['cardio'].values.ravel()
    train_predict = pd.DataFrame()
    test_predict = pd.DataFrame()
    split_predict = pd.DataFrame()
    for i, m in enumerate(models):
        tr_pr, ts_pr, split_target, sp_pr = load_model(m)
        train_predict['m' + str(i)] = tr_pr
        test_predict['m' + str(i)] = ts_pr
        split_predict['m' + str(i)] = sp_pr

        print('')
        print(m)
        print(log_loss(train_target, tr_pr),
              log_loss(split_target, sp_pr),
              log_loss(train_target,
                       train_predict.mean(axis=1) if method == 'mean' else scistats.gmean(train_predict, axis=1)),
              log_loss(split_target,
                       split_predict.mean(axis=1) if method == 'mean' else scistats.gmean(split_predict, axis=1)),
              sep='\t')
    return test_predict.mean(axis=1) if method == 'mean' else scistats.gmean(test_predict, axis=1)


def generate_interactions(data, columns, degree=3):
    result = pd.DataFrame()
    for i in range(2, degree + 1):
        for comb in combinations(columns, i):
            name = '_'.join(comb)
            result[name] = data[list(comb)].apply(lambda row: '_'.join([str(i) for i in row]), axis=1)
    return result


def get_mean_columns(x_train, y_train, x_test, columns, alpha):
    train = x_train[columns].copy()
    test = x_test[columns].copy()

    #     train.reset_index(inplace=True, drop=True)
    #     test.reset_index(inplace=True, drop=True)

    train["target"] = y_train
    glob_mean = y_train.mean()

    for c in columns:
        K = train.groupby([c]).size()
        mean_loc = train.groupby([c])["target"].mean()
        values = (mean_loc * K + glob_mean * alpha) / (K + alpha)
        values.name = c + "_target_mean"
        test = test.join(values, on=c)
        test.loc[test[values.name].isnull(), values.name] = glob_mean

    return test.drop(columns, axis=1)


def populate_mean_columns(x_train, y_train, x_test, columns, alpha, n_splits=5):
    test_extentions = get_mean_columns(x_train, y_train, x_test, columns, alpha)
    x_train = x_train.reindex(columns=np.append(x_test.columns.values, test_extentions.columns.values))
    x_test = pd.concat((x_test, test_extentions), axis=1)
    kf = StratifiedKFold(random_state=2707, n_splits=n_splits, shuffle=True)
    for train_idx, test_idx in kf.split(x_train, y_train):
        extentions = get_mean_columns(x_train.iloc[train_idx], y_train[train_idx], x_train.iloc[test_idx], columns,
                                      alpha)
        x_train.loc[x_train.index[test_idx], extentions.columns] = extentions

    return x_train, x_test


def fit_predict_model(create_callback, X_train, y_train, X_test, alpha, mean_columns=[], drop_columns=[]):
    x1, x2 = populate_mean_columns(X_train, y_train, X_test, mean_columns, alpha=alpha)
    x1.drop(drop_columns, axis=1, inplace=True)
    x2.drop(drop_columns, axis=1, inplace=True)
    model = create_callback(x1, x2)
    model.fit(x1, y_train)
    result = model.predict_proba(x2)
    return result[:, 1] if result.shape[1] > 1 else result[:, 0]


def execute_model(estimator, X_train, y_train, X_test=None, mean_columns=[], drop_columns=[], model_name="",
                  n_folds=5, n_splits=0,
                  create_callback=None, verbose=1, seed=1205, stratification_groups=None, alpha=10):
    np.random.seed(seed)
    random.seed(seed)

    if stratification_groups is None:
        stratification_groups = y_train

    if create_callback is None:
        def create_callback(tr, tst):
            return clone(estimator)

    X_train = pd.DataFrame(X_train)

    kf = StratifiedKFold(random_state=seed, n_splits=n_folds, shuffle=True)
    fold_logloss = []
    train_predict = np.zeros(X_train.shape[0])

    for train_idx, test_idx in kf.split(X_train, stratification_groups):
        train_predict[test_idx] = fit_predict_model(create_callback,
                                                    X_train.iloc[train_idx],
                                                    y_train[train_idx],
                                                    X_train.iloc[test_idx],
                                                    mean_columns=mean_columns,
                                                    drop_columns=drop_columns,
                                                    alpha=alpha)
        fold_logloss.append(log_loss(y_train[test_idx], train_predict[test_idx]))

    if verbose:
        print('')
        print(str(n_folds) + " folds logloss:")
        print(fold_logloss)
        print("mean:", np.mean(fold_logloss))
        print("std:", np.std(fold_logloss))

    split_logloss = []
    split_predict = None
    split_target = None

    if n_splits > 0:
        split_predict = np.ndarray(0)
        split_target = np.ndarray(0)
        ks = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3, train_size=None, random_state=seed)
        for i, (train_idx, test_idx) in enumerate(ks.split(X_train, stratification_groups)):
            split_target = np.append(split_target, y_train[test_idx], axis=0)
            split_predict = np.append(split_predict, fit_predict_model(create_callback,
                                                                       X_train.iloc[train_idx],
                                                                       y_train[train_idx],
                                                                       X_train.iloc[test_idx],
                                                                       mean_columns=mean_columns,
                                                                       drop_columns=drop_columns,
                                                                       alpha=alpha), axis=0)
            split_logloss.append(log_loss(y_train[test_idx], train_predict[test_idx]))
        if verbose:
            print(str(n_splits) + " Splits logloss:")
            print(split_logloss)
            print("mean:", np.mean(split_logloss))
            print("std:", np.std(split_logloss))

    if model_name:
        X_test = pd.DataFrame(X_test)
        test_predict = fit_predict_model(create_callback, X_train, y_train, X_test,
                                         mean_columns=mean_columns,
                                         drop_columns=drop_columns,
                                         alpha=alpha)

        save_model_result(model_name, train_predict, test_predict, split_target, split_predict)
        if verbose:
            print(model_name, 'results saved!')
    return np.mean(fold_logloss), np.mean(split_logloss) if n_splits > 0 else None
