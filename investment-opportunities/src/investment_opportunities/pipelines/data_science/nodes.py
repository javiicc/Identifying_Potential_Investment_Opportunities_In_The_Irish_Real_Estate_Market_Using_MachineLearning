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
# DATA SCIENCE FUNCTIONS
###########################################################################


###########################################################################
# DATA SCIENCE NODES
###########################################################################
def variables_to_modelize(df: pd.DataFrame) -> pd.DataFrame:
    features = [
        'price',
        'floor_area',
        #    'views',
        'latitude',
        'longitude',
        'bedroom',
        'bathroom',
        #    'sale_type',
        'type_house',
        #    'postcode',
        #    'state_district',
        #    'county',
        #    'city_district',
        #    'road',
        #    'place',
        'code',
        #    'admin1',
        #    'cities'
    ]

    data = df[features].copy()
    return data

def get_levels(df):
    levels_type_house = df.type_house.unique()
    levels_code = df.code.unique()
    return levels_type_house, levels_code

def get_features_by_type(df):
    num_features = list(df.select_dtypes('number').columns)  # X_train
    num_features.remove('price')
    # num_features.remove('longitude')
    # num_features.remove('latitude')
    cat_features = list(df.select_dtypes('object').columns)
    #cat_features.remove('city_district')
    return num_features, cat_features


def split_data(data, target='price', test_size=.15, output='X_y_train_test',
               random_state=7):
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

def transformer_estimator(levels_code, levels_type_house,
                          num_features, cat_features,
                          regressor=XGBRegressor(n_estimators= 105),
                          num_transformation='power_transformer', poly_degree=1):
    if num_transformation is 'std_scaler':
        num_pipe = Pipeline([
            ('std_scaler', StandardScaler())
            ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)),
        ])
    elif num_transformation is 'power_transformer':
        num_pipe = Pipeline([
            ('power_transformer', PowerTransformer(method='yeo-johnson')),
            # , standardize=False
            ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)),
        ])

    cat_pipe = Pipeline([
        ('one_hot_encoder', OneHotEncoder(categories=[levels_code, levels_type_house]))
        # No hace nada si ya transformadas
        # handle_unknown='ignore'
    ])
    # Las transforme antes para evitar problemas no las variables a la hora de predecir e el test_set...

    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_features),
        ('cat', cat_pipe, cat_features),
    ])  # , remainder='passthrough'

    pipe_estimator = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('imputer',
         SimpleImputer(strategy='constant',  # esto lo puedo agnadir en los otros pipes
                       fill_value=None)),
        ('regressor', regressor)
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
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)
