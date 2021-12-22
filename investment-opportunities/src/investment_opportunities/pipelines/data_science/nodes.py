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

import joblib
###########################################################################
# DATA SCIENCE FUNCTIONS AND CLASSES
###########################################################################
from sklearn.base import BaseEstimator, TransformerMixin


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        return input_array * 1

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




def transformer_estimator(#num_transformation,
                       #   regressor,
                          levels_list,
                          num_feat,
                          cat_feat,
                          poly_degree=1,
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
        ('regressor', LinearRegression()),  #regressor
    ])

    return pipe_estimator

def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                pipe_estimator) -> LinearRegression:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    regressor = pipe_estimator
    regressor.fit(X_train, y_train)
    return regressor

def evaluate_model(
    regressor, X_test: pd.DataFrame, y_test: pd.Series    #: LinearRegression
):
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    print('eeeeeeeeeeeeeeeeeh!!!!!!!!!!!!', score)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)
