import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (StandardScaler, OneHotEncoder,
                                   PolynomialFeatures, PowerTransformer)
from sklearn.compose import ColumnTransformer
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_percentage_error, r2_score

from yellowbrick.regressor import ResidualsPlot


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
    """Take a dataframe and plot missing values.
    
    Parameters
    ----------
    df : 
        The dataframe to work with.
    
    Returns
    -------
    A DataFrame showing missinga values.
    """
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
    """Take a dataframe and plot missing values.
    
    Parameters
    ----------
    df : 
        The dataframe to work with.
    
    Returns
    -------
    A DataFrame showing missinga values.
    """
    r2_score = metrics.r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mape = metrics.mean_absolute_percentage_error(y_test, y_pred)
    # mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = metrics.mean_squared_error(y_test, y_pred, squared=squared)

    print(f'R²: {r2_score}')
    print(f'MAE: {mae}')
    print(f'MAPE: {mape}')
    # print(f'MSE: {mse}')
    print(f'RMSE: {rmse}\n')

def cross_validate_custom(estimator, scoring_dict, X_train, y_train, cv=10, return_train_score=False):
    estimator = estimator
    scoring_dict = scoring_dict

    scores = cross_validate(estimator, X=X_train, y=y_train, scoring=scoring_dict, cv=cv,
                            return_train_score=return_train_score)
    print(scores.keys())
    return scores


def scores_statistics(estimator, scoring_dict, X_train, y_train, cv=10,
                      return_train_score=False, time_info=False, 
                      return_est=False):
    
    scores = cross_validate(estimator,
                            X=X_train, y=y_train,
                            scoring=scoring_dict,
                            cv=cv,
                            return_train_score=return_train_score,
                            return_estimator=return_est)

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


def plot_learning_curves(model, X_train, y_train, X_test, y_test, metric):
    #   X_train, X_val, y_train, y_val =
    # X_val, y_val = X_test.copy(), y_test.copy()
    #    X_train, y_train = X_train[:250].copy(), y_train[:250].copy()

    train_errors, test_errors = [], []
    train_r2, test_r2 = [], []
    for i in range(1, len(X_train)):
        # fit model on the training dataset
        model.fit(X_train[:i], y_train[:i])

        y_train_pred = model.predict(X_train[:i])
        y_test_pred = model.predict(X_test)

        train_errors.append(mean_absolute_percentage_error(y_train[:i], y_train_pred))
        test_errors.append(mean_absolute_percentage_error(y_test, y_test_pred))

        train_r2.append(r2_score(y_train[:i], y_train_pred))
        test_r2.append(r2_score(y_test, y_test_pred))


    if metric == 'r2':
        print(f'Train R2: {train_r2[-1]}')
        print(f'Test R2: {test_r2[-1]}')

        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        plt.plot(train_r2, 'r-+', linewidth=2, label='Train', alpha=.4)
        plt.plot(test_r2, 'b-', linewidth=3, label='Test', alpha=.4)
        ax.set_ylim(0, 1)
        plt.axhline(y=train_r2[-1], color='black', linestyle='--', alpha=.8)
        plt.axhline(y=test_r2[-1], color='black', linestyle='--', alpha=.8)
        ax.legend()

    elif metric == 'mape':
        print(f'Train MAPE: {train_errors[-1]}')
        print(f'Test MAPE: {test_errors[-1]}')

        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        plt.plot(train_errors, 'r-+', linewidth=2, label='Train', alpha=.4)
        plt.plot(test_errors, 'b-', linewidth=3, label='Test', alpha=.4)
        ax.set_ylim(0, 1)
        plt.axhline(y=train_errors[-1], color='black', linestyle='--', alpha=.8)
        plt.axhline(y=test_errors[-1], color='black', linestyle='--', alpha=.8)
        ax.legend()


def compare_models(estimator, X_train, y_train,
                   scoring_dict,
                   cv, return_train_score=False,
                   ):

    scores = cross_validate(estimator,
                            X=X_train, y=y_train,
                            scoring=scoring_dict,
                            cv=cv,
                            return_train_score=return_train_score)
    scores_resume = {}
    for key in scoring_dict:
        try:
            mean = np.mean(scores['test_' + key])
            std = np.std(scores['test_' + key])
            print(key, 'mean:', mean)
            #print(key, 'std:', std, '\n')
            scores_resume[key] = (mean, std)

        except:
            continue
    print('-' * 10)
    return scores, scores_resume


from sklearn.base import BaseEstimator, TransformerMixin

class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, input_array, y=None):
        return self
    
    def transform(self, input_array, y=None):
        return input_array*1


def transformer_estimator(num_transformation, 
                          regressor, 
                          levels_list, 
                          num_feat, 
                          cat_feat, 
                          poly_degree=1):

    if num_transformation is 'power_transformer':
        num_pipe = Pipeline([
            ('power_transformer', PowerTransformer(method='yeo-johnson')), #, standardize=False
            ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)),
            ('imputer', SimpleImputer(strategy='median')),
            ])
    elif num_transformation is 'std_scaler':
        num_pipe = Pipeline([
            ('std_scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)),
            ('imputer', SimpleImputer(strategy='median')),
            ])
    elif num_transformation is 'identity':
        num_pipe = Pipeline([
            ('identity', IdentityTransformer()),
            ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)),
            ('imputer', SimpleImputer(strategy='median')),
            ])

    cat_pipe = Pipeline([
        ('one_hot_encoder', OneHotEncoder(categories=levels_list)), 
        ('imputer', SimpleImputer(strategy='constant', fill_value=None)),
        ])

    
    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_feat),
        ('cat', cat_pipe, cat_feat),
        ]) #, remainder='passthrough'

    
    pipe_estimator = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor),
        ])
    
    return pipe_estimator


def residuals(estimator, X_train, X_test, y_train, y_test):

    fig, ax =plt.subplots(2,2,figsize=(14,10))

    
    sns.regplot(x=y_train, y=estimator.fit(X_train, y_train).predict(X_train),
                scatter_kws={"color": "cornflowerblue"}, line_kws={"color": "red"}, 
                ax=ax[0,0])\
               .set_title('Actual vs Predicted, Train')
    ax[0,0].set_xlabel('Actual price')
    ax[0,0].set_ylabel('Predicted price')
    sns.regplot(x=y_test, y=estimator.fit(X_train, y_train).predict(X_test), 
                scatter_kws={"color": "cornflowerblue"}, line_kws={"color": "red"}, 
                ax=ax[0,1])\
               .set_title('Actual vs Predicted, Test')
    ax[0,1].set_xlabel('Actual price')
    ax[0,1].set_ylabel('Predicted price')
    plt.tight_layout()
    
    visualizer = ResidualsPlot(estimator, ax=ax[1,0], 
                               train_color='b', test_color='r', 
                               train_alpha=.3, test_alpha=.3,
                              )
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.show();



