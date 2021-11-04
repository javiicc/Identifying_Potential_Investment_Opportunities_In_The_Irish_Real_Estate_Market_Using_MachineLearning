import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (StandardScaler, OneHotEncoder,
                                   PolynomialFeatures)
from sklearn.compose import ColumnTransformer
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate


def split_train_test(df, test_ratio=.15):

    shuffled_indices = np.random.permutation(len(df))
    test_set_size = int(len(df) * test_ratio)

    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return df.iloc[train_indices], df.iloc[test_indices]


def split_x_y(train_set, test_set, features, target='price'):

    y_train = train_set[target].copy()
    X_train = train_set[features].copy()

    y_test = test_set[target].copy()
    X_test = test_set[features].copy()

    print('X_train:', X_train.shape, '\n' +
          'X_test:', X_test.shape, '\n' +
          'y_train:', y_train.shape, '\n' +
          'y_test:', y_test.shape, '\n')

    return X_train, X_test, y_train, y_test


def split_data(data, target='price', test_size=.15, output='X_y_train_test',
               random_state=None):
    features = list(data.columns)
    features.remove(target)

    y = data[target].copy()
    X = data[features].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state)

    train_set = X_train.copy()
    train_set[target] = y_train.copy()

    test_set = X_test.copy()
    test_set[target] = y_test.copy()

    if output == 'X_y_train_test':
        print('X_train:', X_train.shape, '\n' +
              'X_test:', X_test.shape, '\n' +
              'y_train:', y_train.shape, '\n' +
              'y_test:', y_test.shape, '\n')
        return X_train, X_test, y_train, y_test
    elif output == 'train_test':
        print(f'train_set: {train_set.shape}')
        print(f'test_set: {test_set.shape}')
        return train_set, test_set
    elif output == 'X_y':
        print(f'X: {X.shape}')
        print(f'y: {y.shape}')
        return X, y


def metrics_regression(y_test, y_pred, squared=False):
    r2_score = metrics.r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mape = metrics.mean_absolute_percentage_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = metrics.mean_squared_error(y_test, y_pred, squared=squared)

    print(f'R²: {r2_score}')
    print(f'MAE: {mae}')
    print(f'MAPE: {mape}')
    # print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')

def cross_validate_custom(estimator, scoring_dict, X_train, y_train, cv=10, return_train_score=False):
    estimator = estimator
    scoring_dict = scoring_dict

    scores = cross_validate(estimator, X=X_train, y=y_train, scoring=scoring_dict, cv=cv,
                            return_train_score=return_train_score)
    print(scores.keys())
    return scores


def scores_statistics(estimator, scoring_dict, X_train, y_train, cv=10,
                      return_train_score=False, time_info=False):
    scores = cross_validate(estimator,
                            X=X_train, y=y_train,
                            scoring=scoring_dict,
                            cv=cv,
                            return_train_score=return_train_score)

    if time_info:
        fit_time_mean = np.mean(scores['fit_time'])
        fit_time_std = np.std(scores['fit_time'])
        score_time_mean = np.mean(scores['score_time'])
        score_time_std = np.std(scores['score_time'])
        # time_list = []
        print('fit_time mean:', fit_time_mean)
        print('fit_time std:', fit_time_std)
        print('score_time mean:', score_time_mean)
        print('score_time std:', score_time_std)

    scores_resume ={}
    for key in scoring_dict:
        try:
            mean = np.mean(scores['test_' + key])
            std = np.std(scores['test_' + key])
            print(key, 'mean:', mean)
            print(key, 'std:', std, '\n')
            scores_resume[key] = (mean, std)
        except:
            continue
    return scores, scores_resume















