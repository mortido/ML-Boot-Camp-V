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

def merge_models(models, method='gmean', show_std=False, n_folds=None, n_splits=None, seed=None):
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

    if show_std:
        if not n_splits or not n_folds or not seed:
            print('Please provide n_splits, n_folds and seed params to show std')
        else:
            pass
    return test_predict.mean(axis=1) if method == 'mean' else scistats.gmean(test_predict, axis=1)


# def generate_interactions(data, columns, min_degree=2, degree=3, white_list=None):
#     result = pd.DataFrame()
#     for i in range(min_degree, degree + 1):
#         for comb in combinations(columns, i):
#             name = '_'.join(comb)
#             if white_list and name not in white_list:
#                 continue
#             result[name] = data[list(comb)].apply(lambda row: '_'.join([str(i) for i in row]), axis=1)
#     return result


def get_mean_columns(x_train, y_train, x_test, columns, alpha):
    unique_cols = set()
    for c in columns:
        if isinstance(c, (list, tuple)):
            unique_cols.update(c)
        else:
            unique_cols.add(c)
    unique_cols = list(unique_cols)

    train = x_train[unique_cols].copy()
    test = x_test[unique_cols].copy()

    #     train.reset_index(inplace=True, drop=True)
    #     test.reset_index(inplace=True, drop=True)

    train["target"] = y_train
    glob_mean = 0.5  # 0.4997  # y_train.mean()
    # print(glob_mean)
    for c in columns:
        K = train.groupby(c).size()
        mean_loc = train.groupby(c)["target"].mean()
        values = (mean_loc * K + glob_mean * alpha) / (K + alpha)

        if isinstance(c, (list, tuple)):
            values.name = '_'.join(c) + "_target_mean"
        else:
            values.name = c + "_target_mean"

        test = test.join(values, on=c)
        test.loc[test[values.name].isnull(), values.name] = glob_mean

    return test.drop(unique_cols, axis=1)


def populate_mean_columns(x_train, y_train, x_test, columns, alpha, n_splits=10):
    test_extentions = get_mean_columns(x_train, y_train, x_test, columns, alpha)
    x_train = x_train.reindex(columns=np.append(x_test.columns.values, test_extentions.columns.values))
    x_test = pd.concat((x_test, test_extentions), axis=1)
    kf = StratifiedKFold(random_state=2707, n_splits=n_splits, shuffle=True)
    for train_idx, test_idx in kf.split(x_train, y_train):
        extentions = get_mean_columns(x_train.iloc[train_idx], y_train[train_idx], x_train.iloc[test_idx], columns,
                                      alpha)
        x_train.loc[x_train.index[test_idx], extentions.columns] = extentions

    return x_train, x_test


from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold


def fit_predict_model(create_callback, X_train, y_train, X_test, alpha, mean_columns=[], drop_columns=[]):
    gc.collect()
    x1, x2 = populate_mean_columns(X_train, y_train, X_test, mean_columns, alpha=alpha)
    x1.drop(drop_columns, axis=1, inplace=True)
    x2.drop(drop_columns, axis=1, inplace=True)
    model = create_callback(x1, x2)

    # TODO: ALARM!
    # kf = StratifiedKFold(n_splits=7, random_state=12345)
    # model = CalibratedClassifierCV(model, cv=kf, method='isotonic')

    model.fit(x1, y_train)
    result = model.predict_proba(x2)
    return result[:, 1] if result.shape[1] > 1 else result[:, 0]


def execute_model(estimator, X_train, y_train, X_test=None, use_columns=None, mean_columns=[], model_name="",
                  n_folds=5, n_splits=0,
                  create_callback=None, verbose=1, seed=11241, stratification_groups=None, alpha=10):
    np.random.seed(seed)
    random.seed(seed)

    if stratification_groups is None:
        stratification_groups = y_train

    if create_callback is None:
        def create_callback(tr, tst):
            return clone(estimator)

    X_train = pd.DataFrame(X_train)

    if use_columns is None:
        use_columns = X_train.columns

    drop_columns = [c for c in X_train.columns if c not in use_columns]

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
    return np.mean(fold_logloss), np.mean(split_logloss) if n_splits > 0 else None  # np.std(fold_logloss)  #


def new_features(data):
    data["BMI"] = 10000 * data["weight"] / (data["height"] * data["height"])
    data["BMI_1"] = 100 * data["weight"] / data["height"]
    data["BMI_3"] = 1000000 * data["weight"] / (data["height"] * data["height"] * data["height"])
    data["BMI_4"] = 100000000 * data["weight"] / (data["height"] * data["height"] * data["height"] * data["height"])
    data["ap_dif"] = data["ap_hi"] - data["ap_lo"]
    data["MAP"] = (data["ap_lo"] * 2 + data["ap_dif"]) / 3.0

    data["age_years"] = np.round(data["age"] / 365)

    age_bins = [0, 14000, 14980, 15700, 16420, 17140, 17890, 18625, 19355, 20090, 20820, 21555, 22280, 22990, 24000]
    age_names = list(range(1, len(age_bins)))  # [30, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64]
    data["age_group"] = pd.cut(data['age'], age_bins, labels=age_names).astype('int')
    data["age_group_MAPX"] = data["age_group"] * data["MAP"]

    bins = [0, 70, 90, 120, 140, 160, 190, 20000]
    names = list(range(len(bins) - 1))
    data["ap_hi_group"] = pd.cut(data['ap_hi'], bins, labels=names).astype('int')

    bins = [-1, 40, 60, 80, 90, 100, 2000000]
    names = list(range(len(bins) - 1))
    data["ap_lo_group"] = pd.cut(data['ap_lo'], bins, labels=names).astype('int')

    data["weight_group"] = pd.qcut(data['weight'], 10, labels=False).astype('int')

    data["height_group"] = pd.qcut(data['height'], 10, labels=False).astype('int')
    data["BMI_group"] = pd.qcut(data['height'], 10, labels=False).astype('int')

    return data


def clean_data(data):
    data['error_group'] = 0

    # weight/height correction
    idx = (data['height'] < 130) & (data['weight'] > 150)
    data.loc[idx, ["height", "weight"]] = data.loc[idx, ["weight", "height"]].values
    #     data.loc[idx, 'error_group'] = 100-1
    #     data.loc[data['weight']<20, "weight"] *= 10
    #     data.loc[data['weight']<20, "weight"] *= 10
    #     data.loc[data['weight']<25, "weight"] += 100

    # preasure correction

    data.loc[data["ap_hi"] < 0, "ap_hi"] *= -1
    data.loc[data["ap_lo"] < 0, "ap_lo"] *= -1

    for i in range(10):
        str_i = str(i)
        data['hi_' + str_i + 's'] = data['ap_hi'].apply(lambda x: str(x).count(str_i))
        #         data[str(i)+'lo'] = data['ap_lo'].apply(lambda x: str(x).count(str(i)))
        #         data[str(i)+'hilo'] = data[str(i)+'hi']+data[str(i)+'lo']
        #         data=data.drop(str(i)+'lo', axis=1)
        for j in range(10):
            str_j = str_i + str(j)
            data['hi_' + str_j + 's'] = data['ap_hi'].apply(lambda x: str(x).count(str_j))

    data.loc[(data['ap_lo'] < 20), 'error_group'] = 5
    data.loc[(data['ap_hi'] < 50), 'error_group'] = 6
    data.loc[(data['ap_lo'] > 250), 'error_group'] = 1
    data.loc[(data['ap_lo'] > 4000), 'error_group'] = 2
    data.loc[(data['ap_hi'] > 250), 'error_group'] = 3
    data.loc[(data['ap_hi'] > 10000), 'error_group'] = 4

    data.loc[(data["ap_hi"] < 20) & (data["ap_hi"] > 10), "ap_hi"] *= 10
    data.loc[(data["ap_lo"] < 15) & (data["ap_lo"] > 2), "ap_lo"] *= 10

    idx = data['ap_hi'] > 10000
    data.loc[idx, 'ap_hi'] = 10 * (data.loc[idx, 'ap_hi'] // 1000)
    data.loc[data['ap_lo'] >= 10000, 'ap_lo'] //= 100

    #     data.loc[data['ap_lo'].isin([1100])&(data['ap_hi']>160), 'ap_lo'] = 110
    #     data.loc[data['ap_lo'].isin([1100]), 'ap_lo'] = 100
    #     data.loc[(data['ap_lo']>250)&(data['ap_lo']<4000)&(data['ap_lo']%100==0), 'ap_lo'] /= 10

    manual_update = [

        # id	age	gender	height	weight	ap_hi	ap_lo	cholesterol	gluc	smoke	alco	active	cardio	BMI
        # 12494	16905	2	163	63.0	1	2088	1	1	1.0	0.0	1.0	0	23.711845
        # 42591	18191	2	162	63.0	140	1900	1	1	1.0	0.0	1.0	1	24.005487
        # 78873	20323	1	168	68.0	130	1900	1	1	0.0	0.0	1.0	0	24.092971
        # 51749	18419	1	169	62.0	1	2088	1	1	0.0	0.0	1.0	-5	21.707923
        (12494, ['ap_hi', 'ap_lo'], [120, 80]),
        (42591, ['ap_hi', 'ap_lo'], [140, 90]),  # ?
        (78873, ['ap_hi', 'ap_lo'], [130, 100]),  # ?
        (51749, ['ap_hi', 'ap_lo'], [120, 80]),

        # 57807	20496	1	164	62.0	70	1100	1	1	0.0	0.0	0.0	0	23.051755
        # 60477	18716	1	171	80.0	1	1088	1	1	0.0	0.0	1.0	1	27.358845
        # 91198	18182	2	186	95.0	100	901	2	2	0.0	0.0	1.0	0	27.459822
        # 6580	19079	1	176	92.0	1	1099	1	1	0.0	NaN	1.0	-5	29.700413
        (57807, ['ap_hi', 'ap_lo'], [170, 100]),
        (60477, ['ap_hi', 'ap_lo'], [110, 80]),
        (91198, ['ap_hi', 'ap_lo'], [100, 90]),
        (6580, ['ap_hi', 'ap_lo'], [110, 90]),

        # 44701	22801	1	163	115.0	20	170	1	1	0.0	0.0	1.0	1	43.283526
        # 94673	22551	1	169	88.0	10	160	3	3	0.0	0.0	0.0	1	30.811246
        (44701, ['ap_hi', 'ap_lo'], [120, 70]),
        (94673, ['ap_hi', 'ap_lo'], [110, 60]),

    ]
    for idx, cols, update in manual_update:
        data.loc[data['id'] == idx, cols] = update

    return data
