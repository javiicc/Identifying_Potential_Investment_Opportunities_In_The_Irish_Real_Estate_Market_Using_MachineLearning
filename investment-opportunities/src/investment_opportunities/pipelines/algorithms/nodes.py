
from sklearn.metrics import accuracy_score
import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, ParameterGrid, cross_val_score
from sklearn.preprocessing import (StandardScaler, OneHotEncoder,
                                   PolynomialFeatures, PowerTransformer)
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor

###########################################################################
# DATA SCIENCE FUNCTIONS AND CLASSES
###########################################################################

class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        return input_array * 1


from sklearn.base import BaseEstimator, TransformerMixin


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        return input_array * 1

def transformer_estimator(num_transformation,
                          regressor,
                          levels_list,
                          num_feat,
                          cat_feat,
                          poly_degree):

    if num_transformation is 'power_transformer':
        num_pipe = Pipeline([
            ('power_transformer', PowerTransformer(method='yeo-johnson')),
            # , standardize=False
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
    ])  # , remainder='passthrough'

    pipe_estimator = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor),  #regressor
    ])

    return pipe_estimator
'''
def get_weigts(scores_dict={'poly': 74.06, 'knn': 74.57, 'dt': 71.71}):
    models_r2 = scores_dict

    tot = 0
    for key in models_r2:
        tot += models_r2[key]

    models_weigth = {}
    models_weigth_list = []
    for key in models_r2:
        weight = models_r2[key] / tot
        models_weigth[key] = weight
        models_weigth_list.append(weight)
    return models_weigth_list
'''

###########################################################################
# DATA SCIENCE NODES
###########################################################################

def get_levels(df):
    levels_type_house = df.type_house.unique()
    levels_code = df.code.unique()
    return [levels_type_house, levels_code]

def get_features_by_type(df):
    num_features = list(df.select_dtypes('number').columns)  # X_train
    num_features.remove('price')
    # num_features.remove('longitude')
    # num_features.remove('latitude')
    cat_features = list(df.select_dtypes('object').columns)
    #cat_features.remove('city_district')
    return num_features, cat_features


def get_estimator(levels_list, num_features, cat_features,
                  transformer_estimator=transformer_estimator,
                  regressor_dict={'Polynomial_Regression': LinearRegression(),
                                  'K_Nearest_Neighbors_Regressor': KNeighborsRegressor(
                                      n_neighbors=7,
                                      weights='uniform',
                                      leaf_size= 30),
                                  'Decision_Tree_Regressor': DecisionTreeRegressor(
                                      max_depth=10,
                                      min_samples_leaf=30,
                                      random_state=7), # NO DA EXACTO, CHECK IT
                                  'XGBRegressor': XGBRegressor(
                                      n_estimators= 177, #150
                                      max_depth=3,
                                      learning_rate=.1,
                                      subsample=.30),
                                  }):
    # PODRIA SER POR EL ORDEN DE LAS VARIABLES O ALGO...
    estimators_dict = {}
    for key in regressor_dict:

        if key in ['Decision_Tree_Regressor', 'XGBRegressor']:
            num_transformation = 'identity'
        else:
            num_transformation = 'power_transformer'
        if key in ['Polynomial_Regression', 'Decision_Tree_Regressor']:
            poly_degree = 3
        else:
            poly_degree = 1

        estimator_pipe = transformer_estimator(regressor=regressor_dict[key],
                                               levels_list=levels_list,
                                               num_feat=num_features,
                                               cat_feat=cat_features,
                                               poly_degree=poly_degree,
                                               num_transformation=num_transformation)
        estimators_dict[key] = estimator_pipe
    print(estimators_dict['Decision_Tree_Regressor'])

    return estimators_dict

def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                estimators_dict) -> LinearRegression:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    trained_regressors_dict = {}
    for key in estimators_dict:

        regressor = estimators_dict[key]
        regressor.fit(X_train, y_train)
        trained_regressors_dict[key] = regressor

        #print(trained_regressors_dict)
    polyr = trained_regressors_dict['Polynomial_Regression']
    knnr = trained_regressors_dict['K_Nearest_Neighbors_Regressor']
    dtr = trained_regressors_dict['Decision_Tree_Regressor']
    xgbr = trained_regressors_dict['XGBRegressor']


    return polyr, knnr, dtr, xgbr
