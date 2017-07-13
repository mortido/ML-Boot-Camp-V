import gc
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
        print(model_name)
        return 1 / 0
    train_predict = pd.read_csv(directory + 'train_predict.csv', header=None, sep=';').values.ravel()
    test_predict = pd.read_csv(directory + 'test_predict.csv', header=None, sep=';').values.ravel()
    split_target = pd.read_csv(directory + 'split_target.csv', header=None, sep=';').values.ravel()
    split_predict = pd.read_csv(directory + 'split_predict.csv', header=None, sep=';').values.ravel()

    return train_predict, test_predict, split_target, split_predict


from scipy import stats as scistats


# scistats.gmean(np.vstack((y_train,y_train/2)))

def get_merge_score(models, method='gmean'):
    train_target = pd.read_csv('train.csv', sep=';')['cardio'].values.ravel()
    train_predict = pd.DataFrame()
    split_predict = pd.DataFrame()
    split_target = None
    for i, m in enumerate(models):
        tr_pr, _, split_target, sp_pr = load_model(m)
        train_predict['m' + str(i)] = tr_pr
        split_predict['m' + str(i)] = sp_pr

    score1 = log_loss(train_target,
                      train_predict.mean(axis=1) if method == 'mean' else scistats.gmean(train_predict, axis=1))
    score2 = log_loss(split_target,
                      split_predict.mean(axis=1) if method == 'mean' else scistats.gmean(split_predict, axis=1))

    return score1, score2


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


def merge_models2(models, method='gmean'):
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

    return test_predict.mean(axis=1) if method == 'mean' else scistats.gmean(test_predict, axis=1), train_predict.mean(
        axis=1) if method == 'mean' else scistats.gmean(train_predict, axis=1)


def score_by_folds(models, X_train, n_folds=5, seed=11241, stratification_groups=None):
    folds = [[] for _ in range(n_folds)]
    kf = StratifiedKFold(random_state=seed, n_splits=n_folds, shuffle=True)

    train_target = pd.read_csv('train.csv', sep=';')['cardio'].values.ravel()

    for m in models:
        tr_pr, _, _, _ = load_model(m)
        for i, (_, idx) in enumerate(kf.split(X_train, stratification_groups)):
            score = log_loss(train_target[idx], tr_pr[idx])
            folds[i].append((score, m))
    for f in folds:
        f.sort()
    return folds


def execute_model(estimator, X_train, y_train, X_test=None, model_name="",
                  n_folds=5, n_splits=0, create_callback=None, verbose=1, seed=11241, stratification_groups=None):
    np.random.seed(seed)
    random.seed(seed)

    if stratification_groups is None:
        stratification_groups = y_train

    if create_callback is None:
        def create_callback(xtr, xte):
            return clone(estimator)

    X_train = pd.DataFrame(X_train)
    kf = StratifiedKFold(random_state=seed, n_splits=n_folds, shuffle=True)
    fold_logloss = []
    train_predict = np.zeros(X_train.shape[0])

    for train_idx, test_idx in kf.split(X_train, stratification_groups):
        xtr = X_train.iloc[train_idx]
        xte = X_train.iloc[test_idx]
        ytr = y_train[train_idx]
        yte = y_train[test_idx]

        model = create_callback(xtr, xte)
        model.fit(xtr, ytr)
        train_predict[test_idx] = model.predict_proba(xte)[:, 1]
        fold_logloss.append(log_loss(yte, train_predict[test_idx]))

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
        for train_idx, test_idx in ks.split(X_train, stratification_groups):
            xtr = X_train.iloc[train_idx]
            xte = X_train.iloc[test_idx]
            ytr = y_train[train_idx]
            yte = y_train[test_idx]

            model = create_callback(xtr, xte)
            model.fit(xtr, ytr)
            res = model.predict_proba(xte)[:, 1]
            split_target = np.append(split_target, yte, axis=0)
            split_predict = np.append(split_predict, res, axis=0)
            split_logloss.append(log_loss(yte, res))
        if verbose:
            print(str(n_splits) + " Splits logloss:")
            print(split_logloss)
            print("mean:", np.mean(split_logloss))
            print("std:", np.std(split_logloss))

    if model_name:
        X_test = pd.DataFrame(X_test)
        model = create_callback(X_train, X_test)
        model.fit(X_train, y_train)
        test_predict = model.predict_proba(X_test)[:, 1]
        save_model_result(model_name, train_predict, test_predict, split_target, split_predict)
        if verbose:
            print(model_name, 'results saved!')
    return np.mean(fold_logloss), np.mean(split_logloss) if n_splits > 0 else None  # np.std(fold_logloss)  #


def new_features(data):
    data["BMI"] = 10000 * data["weight"] / (data["height"] * data["height"])
    data["BMI_1"] = 100 * data["weight"] / data["height"]
    data["BMI_3"] = 1000000 * data["weight"] / (data["height"] * data["height"] * data["height"])
    data["BMI_4"] = 100000000 * data["weight"] / (data["height"] * data["height"] * data["height"] * data["height"])
    data["ap_dif"] = data["ap_hi"] - data["ap_lo"]
    data["ap_dif_2"] = np.abs(data["ap_hi"] - data["ap_lo"])
    data["MAP"] = (data["ap_lo"] * 2 + data["ap_dif"]) / 3.0
    data["MAP_2"] = (data["ap_lo"] + data["ap_hi"]) / 2.0

    data["age_years"] = np.floor(data["age"] / 365.242199)
    data["age_years_2"] = np.round(data["age"] / 365.242199)

    age_bins = [0, 14000, 14980, 15700, 16420, 17140, 17890, 18625, 19355, 20090, 20820, 21555, 22280, 22990, 24000]
    age_names = list(range(1, len(age_bins)))  # [30, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64]
    data["age_group"] = pd.cut(data['age'], age_bins, labels=age_names).astype('int')
    data["age_group_MAPX"] = data["age_group"] * data["MAP"]

    age_bins = [0, 10000, 14000, 14980, 15700, 16420, 17140, 17890, 18625, 19355, 20090, 20820, 21555, 22280, 22990,
                24000]
    age_names = list(range(1, len(age_bins)))  # [30, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64]
    data["age_group_orig"] = pd.cut(data['age'], age_bins, labels=age_names).astype('int')

    bins = [0, 70, 90, 120, 140, 160, 190, 20000]
    names = list(range(len(bins) - 1))
    data["ap_hi_group"] = pd.cut(data['ap_hi'], bins, labels=names).astype('int')

    bins = [-1, 40, 60, 80, 90, 100, 2000000]
    names = list(range(len(bins) - 1))
    data["ap_lo_group"] = pd.cut(data['ap_lo'], bins, labels=names).astype('int')

    data["ap_hi_group_2"] = data['ap_hi'] // 10
    data["ap_lo_group_2"] = data['ap_lo'] // 10

    data["ap_hi_group_3"] = data['ap_hi'] // 5
    data["ap_lo_group_3"] = data['ap_lo'] // 5

    data["ap_hi_group_4"] = np.round(data['ap_hi'] / 10)
    data["ap_lo_group_4"] = np.round(data['ap_lo'] / 10)

    data["weight_group"] = pd.qcut(data['weight'], 10, labels=False).astype('int')

    data["height_group"] = pd.qcut(data['height'], 10, labels=False).astype('int')
    data["BMI_group"] = pd.qcut(data['height'], 10, labels=False).astype('int')

    data['ap_hi_1'] = 0
    data['ap_lo_1'] = 0
    data['ap_hi_2'] = 0
    data['ap_lo_2'] = 0

    idx = data['ap_hi'] % 10 != 0
    data.loc[idx, 'ap_hi_1'] = data.loc[idx, 'ap_hi']
    idx = data['ap_lo'] % 10 != 0
    data.loc[idx, 'ap_lo_1'] = data.loc[idx, 'ap_lo']
    idx = data['ap_hi'] % 10 == 0
    data.loc[idx, 'ap_hi_2'] = data.loc[idx, 'ap_hi']
    idx = data['ap_lo'] % 10 == 0
    data.loc[idx, 'ap_lo_2'] = data.loc[idx, 'ap_lo']

    ccc = ['age', 'age_group', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'BMI', 'MAP']
    for col1 in ccc:
        data[col1 + '_log'] = np.log(data[col1] + 1.1)
        for col2 in ccc:
            data['%s_mul_%s' % (col1, col2)] = data[col1] * data[col2]
            data['%s_mul_log_%s' % (col1, col2)] = data[col1] * np.log(data[col2] + 1)
            data['%s_div_log_%s' % (col1, col2)] = data[col1] / (np.log(data[col2] + 1) + 1)
            if col2 == col1:
                continue
            data['%s_div_%s' % (col1, col2)] = data[col1] / (data[col2] + 1)

    return data


def clean_data(data, light_clean=False, more_clean=False):
    data.loc[data["ap_hi"] < 0, "ap_hi"] *= -1
    data.loc[data["ap_lo"] < 0, "ap_lo"] *= -1

    # if light_clean:
    #     return data

    # data['error_group'] = 0
    # data.loc[(data['ap_lo'] < 20), 'error_group'] = 5
    # data.loc[(data['ap_hi'] < 50), 'error_group'] = 6
    # data.loc[(data['ap_lo'] > 250), 'error_group'] = 1
    # data.loc[(data['ap_lo'] > 4000), 'error_group'] = 2
    # data.loc[(data['ap_hi'] > 250), 'error_group'] = 3
    # data.loc[(data['ap_hi'] > 10000), 'error_group'] = 4
    data.loc[(data['ap_hi'] == 116) & (data['ap_lo'] == 120), 'ap_hi'] = 160
    # data.loc[(data['ap_hi'] == 120) & (data['ap_lo'] == 150), ['ap_hi', 'ap_lo']] = [120, 50]
    # data.loc[(data['ap_hi'] == 100) & (data['ap_lo'] == 150), ['ap_hi', 'ap_lo']] = [100, 50]
    # data.loc[(data['ap_hi'] == 100) & (data['ap_lo'] == 160), ['ap_hi', 'ap_lo']] = [100, 60]
    # data.loc[(data['ap_hi'] == 110) & (data['ap_lo'] == 170), ['ap_hi', 'ap_lo']] = [100, 60]
    data.loc[(data['ap_hi'] == 20) & (data['ap_lo'] == 170), ['ap_hi', 'ap_lo']] = [120, 70]

    if more_clean:
        idx = (data['ap_hi'] <= data['ap_lo']) & (data['ap_hi'] > 100) & (data['ap_lo'] > 140) & (
            data['ap_hi'] <= 250) & (data['ap_lo'] < 250)
        data.loc[idx, 'ap_lo'] %= 100

        idx = (data['ap_hi'] <= data['ap_lo']) & (data['ap_hi'] > 50) & (data['ap_lo'] > 25) & (
            data['ap_hi'] <= 250) & (data['ap_lo'] < 250)
        data.loc[idx, ['ap_hi', 'ap_lo']] = data.loc[idx, ['ap_lo', 'ap_hi']].values

    # weight/height correction
    idx = (data['height'] < 130) & (data['weight'] > 150)
    data.loc[idx, ["height", "weight"]] = data.loc[idx, ["weight", "height"]].values
    if not light_clean:
        data.loc[data['height'] < 80, "height"] += 100
        # data.loc[data['weight'] < 20, "weight"] *= 10
        # data.loc[(data['weight'] < 30) & (data['weight'] >= 20), "weight"] += 100

    # preasure correction
    data.loc[data['ap_hi'] == 11500, 'ap_hi'] = 150
    idx = data['ap_hi'] > 10000
    data.loc[idx, 'ap_hi'] = 10 * (data.loc[idx, 'ap_hi'] // 1000)
    data.loc[data['ap_lo'] >= 10000, 'ap_lo'] //= 100

    data.loc[data['ap_lo'] == 1000, 'ap_lo'] = 100
    data.loc[data['ap_lo'] == 1200, 'ap_lo'] = 120
    idx = (data['ap_lo'] == 1100) & (data['ap_hi'] < 100)
    data.loc[idx, 'ap_lo'] = data.loc[idx, 'ap_hi']
    data.loc[idx, 'ap_hi'] = 110

    data.loc[data['ap_lo'] == 1100, 'ap_lo'] = 100  # not sure...
    data.loc[data['ap_lo'] == 1110, 'ap_lo'] = 110
    data.loc[data['ap_lo'] == 1001, 'ap_lo'] = 100
    data.loc[data['ap_lo'] == 1120, 'ap_lo'] = 120
    data.loc[data['ap_lo'] == 1900, 'ap_lo'] = 90
    data.loc[data['ap_lo'] == 1130, 'ap_lo'] = 130
    data.loc[data['ap_lo'] == 1300, 'ap_lo'] = 130
    data.loc[data['ap_lo'] == 1140, 'ap_lo'] = 140
    data.loc[data['ap_lo'] == 1400, 'ap_lo'] = 140

    idx = (data['ap_lo'] > 1000) & (data['ap_hi'] == 1)
    data.loc[idx, 'ap_hi'] = 100 + data.loc[idx, 'ap_lo'] // 100
    data.loc[idx, 'ap_lo'] %= 100

    idx = (data['ap_lo'] > 250) & (data['ap_hi'] < 100)
    data.loc[idx, 'ap_hi'] = data.loc[idx, 'ap_hi'] * 10 + data.loc[idx, 'ap_lo'] // 100
    data.loc[idx, 'ap_lo'] %= 100

    data.loc[data['ap_lo'].isin([800, 8044, 802, 8000, 8099, 8079, 809,
                                 801, 810, 8200, 820, 880, 808, 8022, 8100, 8077, 8500, 850
                                 ]), 'ap_lo'] = 80

    data.loc[data['ap_lo'].isin([9100, 9800, 9011]), 'ap_lo'] = 90

    data.loc[data['ap_lo'].isin([4700, 7099, 7100]), 'ap_lo'] = 70

    data.loc[data['ap_lo'] == 5700, 'ap_lo'] = 70  # 57
    data.loc[data['ap_lo'] == 6800, 'ap_lo'] = 68
    data.loc[data['ap_lo'] == 4100, 'ap_lo'] = 100

    data.loc[data['ap_lo'] > 1000, 'ap_lo'] //= 10
    data.loc[data['ap_lo'] > 890, 'ap_lo'] = 90
    data.loc[data['ap_lo'] > 790, 'ap_lo'] = 80
    data.loc[data['ap_lo'] > 690, 'ap_lo'] = 70
    data.loc[data['ap_lo'] > 590, 'ap_lo'] = 60

    data.loc[data['ap_hi'] == 1420, 'ap_hi'] = 120
    data.loc[(data['ap_hi'] > 250) & (data['ap_hi'].astype('str').apply(lambda x: '4' in x)), 'ap_hi'] = 140

    data.loc[data['ap_hi'].isin([1202, 1205, ]), 'ap_hi'] = 120
    data.loc[data['ap_hi'].isin([1500, 1502, ]), 'ap_hi'] = 150
    data.loc[data['ap_hi'].isin([1620, 1608, ]), 'ap_hi'] = 160
    data.loc[data['ap_hi'].isin([1300, 1130, ]), 'ap_hi'] = 130
    data.loc[data['ap_hi'] == 2000, 'ap_hi'] = 200
    data.loc[data['ap_hi'] == 1110, 'ap_hi'] = 110
    data.loc[data['ap_hi'] == 701, 'ap_hi'] = 170
    data.loc[(data['ap_hi'] >= 900) & (data['ap_lo'] > 50), 'ap_hi'] = 90
    data.loc[data['ap_hi'] == 906, ['ap_hi', 'ap_lo']] = [90, 60]
    data.loc[data['ap_hi'] == 907, ['ap_hi', 'ap_lo']] = [90, 70]
    data.loc[data['ap_hi'] == 806, ['ap_hi', 'ap_lo']] = [80, 60]
    data.loc[data['ap_hi'] == 309, ['ap_hi', 'ap_lo']] = [130, 90]
    data.loc[data['ap_hi'] == 509, ['ap_hi', 'ap_lo']] = [150, 90]

    data.loc[(data['ap_hi'] == 138) & (data['ap_lo'] == 0), ['ap_hi', 'ap_lo']] = [130, 80]
    data.loc[(data['ap_hi'] == 149) & (data['ap_lo'] == 0), ['ap_hi', 'ap_lo']] = [140, 90]
    data.loc[(data['ap_hi'] == 148) & (data['ap_lo'] == 0), ['ap_hi', 'ap_lo']] = [140, 80]
    data.loc[(data['ap_hi'] == 108) & (data['ap_lo'] == 0), ['ap_hi', 'ap_lo']] = [100, 80]
    data.loc[(data['ap_hi'] == 117) & (data['ap_lo'] == 0), ['ap_hi', 'ap_lo']] = [110, 70]
    data.loc[(data['ap_hi'] == 118) & (data['ap_lo'] == 0), ['ap_hi', 'ap_lo']] = [110, 80]

    data.loc[(data['ap_hi'] <= 20) & (data['ap_hi'] > 10) & (data['ap_hi'] * 10 > data['ap_lo']), 'ap_hi'] *= 10
    data.loc[(data['ap_lo'] <= 50) & (data['ap_lo'] > 2) & (data['ap_lo'] * 10 < data['ap_hi']), 'ap_lo'] *= 10

    # data.loc[(data['ap_hi'] == 10) & (data['ap_lo'] == 80), 'ap_hi'] = 120
    # data.loc[(data['ap_hi'] == 10) & (data['ap_lo'] == 70), 'ap_hi'] = 110
    # data.loc[(data['ap_hi'] == 10) & (data['ap_lo'] == 60), 'ap_hi'] = 100
    data.loc[(data['ap_hi'] == 10) & (data['ap_lo'] == 160), ['ap_hi', 'ap_lo']] = [110, 60]
    data.loc[(data['ap_hi'] == 24) & (data['ap_lo'] == 20), ['ap_hi', 'ap_lo']] = [240, 120]
    # data.loc[(data['ap_hi'] == 70) & (data['ap_lo'] == 15), ['ap_hi', 'ap_lo']] = [150, 70]
    data.loc[data['ap_lo'] == 19, 'ap_lo'] = 90
    data.loc[data['ap_lo'] == 15, 'ap_lo'] = 50

    data.loc[(data['ap_hi'] == 180) & (data['ap_lo'] == 20), ['ap_hi', 'ap_lo']] = [180, 120]

    idx = (data['ap_hi'] < 50) & (data['ap_lo'] > 100)
    data.loc[idx, 'ap_hi'] = data.loc[idx, 'ap_lo']
    data.loc[idx, 'ap_lo'] = 0

    for ap_hi, idx in [(115, 3683),
                       (123, 7657),
                       (104, 29827),
                       (109, 41674),
                       (122, 81051),
                       (99, 82646),
                       (113, 89703),
                       (105, 5685),
                       (118, 42755),
                       (133, 43735),
                       (103, 79396)]:
        data.loc[data['id'] == idx, 'ap_hi'] = ap_hi

    for ap_lo, idx in [(73, 12550),
                       (81, 16884),
                       (83, 19258),
                       (73, 19885),
                       (72, 27069),
                       (73, 28742),
                       (79, 31965),
                       (79, 33295),
                       (79, 35356),
                       (79, 36325),
                       (87, 39577),
                       (87, 49321),
                       (83, 50210),
                       (79, 50799),
                       (79, 56048),
                       (78, 57023),
                       (78, 58088),
                       (69, 58537),
                       (79, 62937),
                       (83, 63710),
                       (89, 65470),
                       (80, 68612),
                       (83, 75007),
                       (79, 81298),
                       (80, 93224),
                       (84, 97439),
                       (79, 7465),
                       (79, 18180),
                       (79, 20962),
                       (79, 26367),
                       (51, 43735),
                       (74, 80247),
                       (73, 88937),
                       (73, 98631),
                       (79, 99499)]:
        data.loc[data['id'] == idx, 'ap_lo'] = ap_lo
    return data


class SmoothLikelihood:
    def __init__(self, columns, glob_mean_value, kf, alpha=13):
        self.columns = columns
        self.glob_mean_value = glob_mean_value
        self.alpha = alpha
        self.kf = kf
        if isinstance(columns, (list, tuple)):
            self.new_column = '_'.join(columns) + '_target_mean'
            self.columns = columns
        else:
            self.new_column = columns + '_target_mean'
            self.columns = [columns]

    def fit_transform(self, X, y):
        X = X.copy()
        X['target'] = y

        def calc(x):
            return (x['sum'] + self.glob_mean_value * self.alpha) / (x['count'] + self.alpha)

        result = np.zeros(X.shape[0])
        for itr, ite in self.kf.split(X, y):
            tr = X.iloc[itr]
            te = X.iloc[ite]

            temp = tr.groupby(self.columns)['target'].agg(["count", "sum"])
            value_dict = temp.apply(calc, axis=1).to_dict()

            result[ite] = te[self.columns].apply(lambda x: value_dict.get(tuple(x.values), self.glob_mean_value),
                                                 axis=1)

        result = pd.DataFrame(result, columns=[self.new_column])

        temp = X.groupby(self.columns)['target'].agg(["count", "sum"])
        self.value_dict = temp.apply(calc, axis=1).to_dict()
        #         X.drop('target', axis=1, inplace=True)
        return result

    def transform(self, X):
        result = pd.DataFrame()
        result[self.new_column] = X[self.columns].apply(
            lambda x: self.value_dict.get(tuple(x.values), self.glob_mean_value), axis=1)
        return result


class ColumnsFilter:
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        #         print(X.__class__)
        return self

    def transform(self, X):
        return X[self.columns]


class SmoothLikelihood2:
    def __init__(self, columns, glob_mean_value, kf, alpha=13):
        self.columns = columns
        self.glob_mean_value = glob_mean_value
        self.alpha = alpha
        self.kf = kf
        if isinstance(columns, (list, tuple)):
            self.new_column = '_'.join(columns) + '_target_mean'
            self.columns = columns
        else:
            self.new_column = columns + '_target_mean'
            self.columns = [columns]

    def fit_transform(self, X, y):
        X = X.copy()
        X['target'] = y
        self.x_train = X
        self.y_train = y

        def calc(x):
            return (x['sum'] + self.glob_mean_value * self.alpha) / (x['count'] + self.alpha)

        result = np.zeros(X.shape[0])
        for itr, ite in self.kf.split(X, y):
            tr = X.iloc[itr]
            te = X.iloc[ite]

            temp = tr.groupby(self.columns)['target'].agg(["count", "sum"])
            value_dict = temp.apply(calc, axis=1).to_dict()

            result[ite] = te[self.columns].apply(lambda x: value_dict.get(tuple(x.values), self.glob_mean_value),
                                                 axis=1)

        result = pd.DataFrame(result, columns=[self.new_column])

        temp = X.groupby(self.columns)['target'].agg(["count", "sum"])
        self.value_dict = temp.apply(calc, axis=1).to_dict()
        #         X.drop('target', axis=1, inplace=True)
        return result

    def transform(self, X):
        result = np.zeros(X.shape[0])
        n = 0

        def calc(x):
            return (x['sum'] + self.glob_mean_value * self.alpha) / (x['count'] + self.alpha)

        for itr, ite in self.kf.split(self.x_train, self.y_train):
            n += 1
            tr = self.x_train.iloc[itr]
            temp = tr.groupby(self.columns)['target'].agg(["count", "sum"])
            value_dict = temp.apply(calc, axis=1).to_dict()
            result += X[self.columns].apply(lambda x: value_dict.get(tuple(x.values), self.glob_mean_value), axis=1)

        result = pd.DataFrame(result / n, columns=[self.new_column])
        # print(X.shape, result.shape)
        return result


class SmoothLikelihood3:
    def __init__(self, columns, glob_mean_value, kf, alpha=13):
        self.columns = columns
        self.glob_mean_value = glob_mean_value
        self.alpha = alpha
        self.kf = kf
        if isinstance(columns, (list, tuple)):
            self.new_column = '_'.join(columns) + '_target_mean'
            self.columns = columns
        else:
            self.new_column = columns + '_target_mean'
            self.columns = [columns]

    def fit(self, X, y):
        X = X.copy()
        X['target'] = y

        def calc(x):
            return (x['sum'] + self.glob_mean_value * self.alpha) / (x['count'] + self.alpha)

        temp = X.groupby(self.columns)['target'].agg(["count", "sum"])
        self.value_dict = temp.apply(calc, axis=1).to_dict()
        return self

    def transform(self, X):
        result = pd.DataFrame()
        result[self.new_column] = X[self.columns].apply(
            lambda x: self.value_dict.get(tuple(x.values), self.glob_mean_value), axis=1)
        return result


class SmoothLikelihood4:
    def __init__(self, columns, glob_mean_value, kf, alpha=13, seed=0, std=0.05):
        self.columns = columns
        self.glob_mean_value = glob_mean_value
        self.alpha = alpha
        self.kf = kf
        self.std = std

        s = np.random.get_state()
        np.random.seed(seed)
        self.rnd_state = np.random.get_state()
        np.random.set_state(s)

        if isinstance(columns, (list, tuple)):
            self.new_column = '_'.join(columns) + '_target_mean'
            self.columns = columns
        else:
            self.new_column = columns + '_target_mean'
            self.columns = [columns]

    def fit_transform(self, X, y):
        X = X.copy()
        X['target'] = y

        def calc(x):
            return (x['sum'] + self.glob_mean_value * self.alpha) / (x['count'] + self.alpha)

        result = np.zeros(X.shape[0])
        for itr, ite in self.kf.split(X, y):
            tr = X.iloc[itr]
            te = X.iloc[ite]

            temp = tr.groupby(self.columns)['target'].agg(["count", "sum"])
            value_dict = temp.apply(calc, axis=1).to_dict()

            result[ite] = te[self.columns].apply(lambda x: value_dict.get(tuple(x.values), self.glob_mean_value),
                                                 axis=1)

        result = pd.DataFrame(result, columns=[self.new_column])

        s = np.random.get_state()
        np.random.set_state(self.rnd_state)

        result += np.random.normal(scale=self.std, size=result.shape)

        self.rnd_state = np.random.get_state()
        np.random.set_state(s)

        temp = X.groupby(self.columns)['target'].agg(["count", "sum"])
        self.value_dict = temp.apply(calc, axis=1).to_dict()
        #         X.drop('target', axis=1, inplace=True)
        return result

    def transform(self, X):
        result = pd.DataFrame()
        result[self.new_column] = X[self.columns].apply(
            lambda x: self.value_dict.get(tuple(x.values), self.glob_mean_value), axis=1)

        s = np.random.get_state()
        np.random.set_state(self.rnd_state)

        result += np.random.normal(scale=self.std, size=result.shape)

        self.rnd_state = np.random.get_state()
        np.random.set_state(s)

        return result
