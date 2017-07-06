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

    score1 = log_loss(train_target, train_predict.mean(axis=1) if method == 'mean' else scistats.gmean(train_predict, axis=1))
    score2 = log_loss(split_target, split_predict.mean(axis=1) if method == 'mean' else scistats.gmean(split_predict, axis=1))

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

    age_bins = [0, 10000, 14000, 14980, 15700, 16420, 17140, 17890, 18625, 19355, 20090, 20820, 21555, 22280, 22990, 24000]
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


def clean_data(data, light_clean=False):
    data.loc[data["ap_hi"] < 0, "ap_hi"] *= -1
    data.loc[data["ap_lo"] < 0, "ap_lo"] *= -1

    # if light_clean:
    #     return data

    data['error_group'] = 0
    data.loc[(data['ap_lo'] < 20), 'error_group'] = 5
    data.loc[(data['ap_hi'] < 50), 'error_group'] = 6
    data.loc[(data['ap_lo'] > 250), 'error_group'] = 1
    data.loc[(data['ap_lo'] > 4000), 'error_group'] = 2
    data.loc[(data['ap_hi'] > 250), 'error_group'] = 3
    data.loc[(data['ap_hi'] > 10000), 'error_group'] = 4

    # weight/height correction
    idx = (data['height'] < 130) & (data['weight'] > 150)
    data.loc[idx, ["height", "weight"]] = data.loc[idx, ["weight", "height"]].values
    if not light_clean:
        data.loc[data['height'] < 80, "height"] += 100
        # data.loc[data['weight'] < 20, "weight"] *= 10
        # data.loc[(data['weight'] < 30) & (data['weight'] >= 20), "weight"] += 100

    # preasure correction
    data.loc[(data["ap_hi"] < 20) & (data["ap_hi"] > 10), "ap_hi"] *= 10
    data.loc[(data["ap_lo"] < 15) & (data["ap_lo"] > 2), "ap_lo"] *= 10

    idx = data['ap_hi'] > 10000
    data.loc[idx, 'ap_hi'] = 10 * (data.loc[idx, 'ap_hi'] // 1000)
    data.loc[data['ap_lo'] >= 10000, 'ap_lo'] //= 100

    data.loc[data['ap_lo'] == 1000, 'ap_lo'] = 100
    data.loc[data['ap_lo'] == 1200, 'ap_lo'] = 120
    data.loc[data['ap_lo'] == 1001, 'ap_lo'] = 100
    data.loc[data['ap_lo'] == 1120, 'ap_lo'] = 120
    data.loc[data['ap_lo'] == 1110, 'ap_lo'] = 110

    idx = (data['ap_lo'] == 1100) & (data['ap_hi'] < 100)
    data.loc[idx, 'ap_lo'] = data.loc[idx, 'ap_hi']
    data.loc[idx, 'ap_hi'] = 110

    data.loc[data['ap_lo'] == 1100, 'ap_lo'] = 100  # not sure...

    # ...
    idx = (data['ap_hi'] - data['ap_lo'] < -10) & (data['ap_lo'] < 190) & (data['ap_hi'] > 30) & (data['ap_hi'] <= 100)
    data.loc[idx, ['ap_hi', 'ap_lo']] = data.loc[idx, ['ap_lo', 'ap_hi']].values

    data.loc[data['ap_hi'] == 20, 'ap_hi'] = 120

    data.loc[data['ap_lo'].isin([800, 8044, 802, 8000, 8099, 8079, 809,
                                 801, 810, 8200, 820, 880, 808, 8022,
                                 ]), 'ap_lo'] = 80

    data.loc[data['ap_lo'] == 1900, 'ap_lo'] = 90

    data.loc[data['ap_lo'] == 1130, 'ap_lo'] = 130
    data.loc[data['ap_lo'] == 1300, 'ap_lo'] = 130

    data.loc[data['ap_lo'] == 1140, 'ap_lo'] = 140
    data.loc[data['ap_lo'] == 1400, 'ap_lo'] = 140

    data.loc[data['ap_lo'] > 1000, 'ap_lo'] //= 10
    data.loc[data['ap_lo'] > 890, 'ap_lo'] = 90
    data.loc[data['ap_lo'] > 790, 'ap_lo'] = 80
    data.loc[data['ap_lo'] > 690, 'ap_lo'] = 70

    data.loc[data['ap_lo'] == 585, 'ap_lo'] = 85
    data.loc[data['ap_lo'] == 602, 'ap_lo'] = 60
    data.loc[data['ap_lo'] == 570, 'ap_lo'] = 70

    idx = data['ap_hi'] // 100 == 11
    data.loc[idx, 'ap_hi'] %= 1000
    data.loc[data['ap_hi'] >= 1000, 'ap_hi'] //= 10

    #     data.loc[data['ap_hi'] == 138, ['ap_hi', 'ap_lo']] = [130, 80]
    #     data.loc[data['ap_hi'] == 149, ['ap_hi', 'ap_lo']] = [140, 90]
    #     data.loc[data['ap_hi'] == 148, ['ap_hi', 'ap_lo']] = [140, 80]
    #     data.loc[data['ap_hi'] == 108, ['ap_hi', 'ap_lo']] = [100, 80]
    #     data.loc[data['ap_hi'] == 117, ['ap_hi', 'ap_lo']] = [110, 70]
    #     data.loc[data['ap_hi'] == 118, ['ap_hi', 'ap_lo']] = [110, 80]

    #     data.loc[data["ap_hi"] > 1000, "ap_hi"] //= 10
    #     idx = (data['ap_hi'] - data['ap_lo'] < -10) & (data['ap_lo'] < 250) & (data['ap_hi'] > 30)
    #     data.loc[idx, ['ap_lo']]=data.loc[idx, ['ap_lo']]%100

    manual_update = [
        # 20438	50.324435	1	160	70.0	160	7100	1	1	0.0	1.0	1.0	1
        # 29821	52.350445	1	155	81.0	160	8100	1	1	0.0	0.0	1.0	1
        # 47030	50.198494	1	156	65.0	150	9011	2	2	0.0	0.0	1.0	1
        # 59157	49.765914	1	161	60.0	150	7099	1	1	0.0	0.0	1.0	1
        # 10586	47.767283	1	160	75.0	170	4100	1	1	0.0	0.0	1.0	-5
        # 50848	61.223819	1	158	59.0	180	8100	1	2	0.0	NaN	1.0	-5
        # 63276	58.291581	1	162	69.0	160	9100	1	1	0.0	0.0	1.0	-5

        (20438, ['ap_lo'], [70]),
        (29821, ['ap_lo'], [80]),
        (47030, ['ap_lo'], [90]),
        (59157, ['ap_lo'], [70]),
        (10586, ['ap_lo'], [100]),
        (50848, ['ap_lo'], [80]),
        (63276, ['ap_lo'], [90]),

        # WORSE vvvvv
        # 9482	53.464750	1	162	69.0	130	9100	1	1	0.0	0.0	1.0	1
        # 17260	58.770705	2	169	78.0	130	9011	1	1	1.0	1.0	1.0	1
        # 22832	39.720739	2	179	70.0	120	8500	1	1	0.0	0.0	1.0	0
        # 33191	54.570842	2	170	70.0	112	5700	1	2	0.0	0.0	1.0	1
        # 62058	59.975359	2	179	62.0	130	9800	1	1	0.0	0.0	1.0	0
        # 75482	55.854894	1	164	70.0	125	6800	1	1	0.0	0.0	1.0	0
        # 90139	53.314168	1	159	61.0	110	8077	1	1	0.0	0.0	1.0	0
        # 95886	50.565366	2	165	68.0	113	5700	1	1	0.0	0.0	1.0	0
        # 26985	52.062971	1	151	74.0	125	9100	1	1	NaN	0.0	1.0	-5
        # 45450	49.623546	1	170	86.0	125	4700	2	1	0.0	0.0	1.0	-5
        # 74784	57.993155	1	165	65.0	120	8100	3	3	NaN	0.0	0.0	-5
        (9482, ['ap_lo'], [90]),
        (17260, ['ap_lo'], [90]),
        (22832, ['ap_lo'], [80]),
        (33191, ['ap_lo'], [70]),
        (62058, ['ap_lo'], [80]),
        (75482, ['ap_lo'], [80]),
        (90139, ['ap_lo'], [80]),
        (95886, ['ap_lo'], [70]),
        (26985, ['ap_lo'], [90]),
        (45450, ['ap_lo'], [70]),
        (74784, ['ap_lo'], [80]),

        # 12494	46.283368	2	163	63.0	1	2088	1	1	1.0	0.0	1.0	0
        # 60477	51.241615	1	171	80.0	1	1088	1	1	0.0	0.0	1.0	1
        # 6580	52.235455	1	176	92.0	1	1099	1	1	0.0	NaN	1.0	-5
        # 51749	50.428474	1	169	62.0	1	2088	1	1	0.0	0.0	1.0	-5

        (12494, ['ap_hi', 'ap_lo'], [120, 80]),
        (60477, ['ap_hi', 'ap_lo'], [110, 80]),
        (6580, ['ap_hi', 'ap_lo'], [110, 90]),
        (51749, ['ap_hi', 'ap_lo'], [120, 80]),
        # 2654	41.385352	1	160	60.0	902	60	1	1	0.0	0.0	1.0	0
        # 6822	39.493498	1	168	63.0	909	60	2	1	0.0	0.0	1.0	0
        # 13616	62.036961	1	155	87.0	701	110	1	1	0.0	0.0	1.0	1
        # 57646	55.638604	1	162	50.0	309	0	1	1	0.0	0.0	1.0	0
        # 58349	54.225873	1	162	67.0	401	80	1	3	0.0	0.0	1.0	1
        # 59301	57.412731	1	154	41.0	806	0	1	1	0.0	0.0	1.0	0
        # 77010	50.680356	1	164	54.0	960	60	1	1	0.0	0.0	1.0	0
        # 1079	61.796030	2	170	74.0	400	60	1	1	0.0	0.0	1.0	-5
        # 23199	49.541410	1	166	64.0	957	70	1	1	NaN	0.0	0.0	-5
        # 62837	54.516085	2	170	79.0	509	0	1	1	0.0	0.0	1.0	-5

        (2654, ['ap_hi', 'ap_lo'], [90, 60]),
        (6822, ['ap_hi', 'ap_lo'], [90, 60]),
        (13616, ['ap_hi', 'ap_lo'], [170, 110]),
        (57646, ['ap_hi', 'ap_lo'], [130, 980]),
        (58349, ['ap_hi', 'ap_lo'], [140, 80]),
        (59301, ['ap_hi', 'ap_lo'], [80, 60]),
        (77010, ['ap_hi', 'ap_lo'], [90, 60]),
        (1079, ['ap_hi', 'ap_lo'], [100, 60]),
        (23199, ['ap_hi', 'ap_lo'], [95, 70]),
        (62837, ['ap_hi', 'ap_lo'], [150, 90]),

        # 57646	55.638604	1	162	50.0	130	980	1	1	0.0	0.0	1.0	0
        (57646, ['ap_lo'], [80]),
    ]
    for idx, cols, update in manual_update:
        data.loc[data['id'] == idx, cols] = update

        #################

    #     data.loc[(data['ap_lo']==30), 'ap_lo'] = 80

    data.loc[(data['ap_hi'] == 906), ['ap_hi', 'ap_lo']] = [90, 60]
    data.loc[(data['ap_hi'] == 907), ['ap_hi', 'ap_lo']] = [90, 70]
    #     data.loc[(data['ap_hi']==806), ['ap_hi', 'ap_lo']] = [80, 60]
    #     data.loc[(data['ap_hi']==309), ['ap_hi', 'ap_lo']] = [130, 90]

    idx = (data['ap_lo'] == 0) & (data['ap_hi'] % 10 > 2)
    data.loc[idx, 'ap_lo'] = (data.loc[idx, 'ap_hi'] % 10) * 10
    data.loc[idx, 'ap_hi'] = (data.loc[idx, 'ap_hi'] // 10) * 10
    data.loc[data['ap_lo'] == 0, 'ap_lo'] = 80

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
