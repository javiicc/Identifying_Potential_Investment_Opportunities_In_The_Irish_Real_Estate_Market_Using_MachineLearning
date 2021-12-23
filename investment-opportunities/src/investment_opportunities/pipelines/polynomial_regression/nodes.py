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


import joblib

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

def transformer_estimator(#num_transformation,
                          regressor,
                          levels_list,
                          num_feat,
                          cat_feat,
                          poly_degree,
                          num_transformation='power_transformer'):

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


def get_estimators(levels_list, num_features, cat_features,
                   transformer_estimator=transformer_estimator,
                   regressor={'LinearRegression': LinearRegression()}):

    if regressor is 'LinearRegression':
        estimator_pipe = transformer_estimator(regressor=regressor,
                                                   levels_list=levels_list,
                                                   num_feat=num_features,
                                                   cat_feat=cat_features,
                                                   poly_degree=3,
                                                   num_transformation='power_transformer')
    else:
        estimator_pipe = transformer_estimator(regressor=regressor,
                                                   levels_list=levels_list,
                                                   num_feat=num_features,
                                                   cat_feat=cat_features,
                                                   poly_degree=1,
                                                   num_transformation='power_transformer')


        print(estimator_pipe)

    return estimator_pipe
