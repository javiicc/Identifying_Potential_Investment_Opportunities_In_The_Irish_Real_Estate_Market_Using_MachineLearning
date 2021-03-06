"""
This is a boilerplate pipeline 'algorithms'
generated using Kedro 0.17.5
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import (StandardScaler, OneHotEncoder,
                                   PolynomialFeatures, PowerTransformer)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

from typing import List, Tuple


###########################################################################
# BASIC ALGORITHM FUNCTIONS AND CLASSES
###########################################################################


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """
    This transformer returns the value multiplied by 1.
    """
    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        return input_array * 1


def transformer_estimator(num_transformation: str,
                          regressor,
                          levels_list: List[pd.Series],
                          num_feat: list,
                          cat_feat: list,
                          poly_degree: int):
    """This function will make several types of transformations on the data based on the
    arguments given before to use a given estimator as a regressor.

    Parameters
    ----------
    num_transformation :
        String to indicate the type of transformation to perform.
    regressor :
        The regressor to be used.
    levels_list :
        List of Pandas Series to indicate the different levels of type_house and
        code attributes.
    num_feat:
        Numeric features.
    cat_feat :
        Categorical features.
    poly_degree :
        The degree in case we want a polynomial transformation.

    Returns
    -------
    Estimator.
    """
    if num_transformation is 'power_transformer':
        num_pipe = Pipeline([
           # ('imputer', SimpleImputer(strategy='median')),
            ('power_transformer', PowerTransformer(method='yeo-johnson')),
            ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)),
        ])
    elif num_transformation is 'std_scaler':
        num_pipe = Pipeline([
           # ('imputer', SimpleImputer(strategy='median')),
            ('std_scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)),
        ])
    elif num_transformation is 'identity':
        num_pipe = Pipeline([
           # ('imputer', SimpleImputer(strategy='median')),
            ('identity', IdentityTransformer()),
            ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)),
        ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan,
                                  strategy='constant',
                                  fill_value='Unknown')),
        ('one_hot_encoder', OneHotEncoder(categories=levels_list)),
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_feat),
        ('cat', cat_pipe, cat_feat),
    ], remainder='passthrough')

    if regressor is None:
        pipe_estimator = Pipeline(steps=[
            ('preprocessor', preprocessor)
        ])
    else:
        pipe_estimator = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', regressor)
        ])

    return pipe_estimator


###########################################################################
# BASIC ALGORITHM NODES
###########################################################################


def get_levels(df: pd.DataFrame) -> List[list]:
    """Get a DataFrame and return the levels from the code and type_house categorical
    features.

    Parameters
    ----------
    df :
        DataFrame with the categorical features.

    Returns
    -------
    List of Pandas Series with the unique levels for each variable.
    """
    # Add `Unknown` to give it to missing values and not have an error  with levels
    levels_type_house = list(df.type_house.unique())
    levels_type_house.append('Unknown')
    levels_type_house.remove(np.nan)

    levels_place = list(df.place.unique())
    levels_place.append('Unknown')
    levels_place.remove(np.nan)

    return [levels_type_house, levels_place]


def get_features_by_type(df: pd.DataFrame) -> Tuple[list, list]:
    """Takes a DataFrame and returns a tuple of list for numeric and categorical features.

    Parameters
    ----------
    df :
        DataFrame.

    Returns
    -------
    Tuple of two list, once for numeric features and another one for categorical features.
    """
    num_features = list(df.select_dtypes('number').columns)  # X_train
    num_features.remove('price')
    # num_features.remove('longitude')
    # num_features.remove('latitude')
    cat_features = list(df.select_dtypes('object').columns)
    # cat_features.remove('city_district')
    return num_features, cat_features


def get_estimator(levels_list: List[pd.Series],
                  num_features: list,
                  cat_features: list,
                  transformer_estimator=transformer_estimator,
                  regressor_dict=None):
    """Makes a dictionary with the estimators.

    Parameters
    ----------
    levels_list :
        List of lists with variables levels.
    num_features :
        Numeric features.
    cat_features :
        Categorical features.
    transformer_estimator :
        The `transformer_estimator()` function.
    regressor_dict :
        Dictionary with regressors if it is preferred to the default one.

    Returns
    -------
    The estimators dictionary.
    """
    # PODRIA SER POR EL ORDEN DE LAS VARIABLES O ALGO...
    if regressor_dict is None:
        regressor_dict = {
            'Polynomial_Regression': LinearRegression(),
            'K_Nearest_Neighbors_Regressor': KNeighborsRegressor(
                n_neighbors=7,
                weights='uniform',
                leaf_size=30),
            'Decision_Tree_Regressor': DecisionTreeRegressor(
                max_depth=10,
                min_samples_leaf=30,
                random_state=7),
            'XGBRegressor': XGBRegressor(
                n_estimators=177,
                max_depth=3,
                learning_rate=.1,
                subsample=.25),
            'Random_Forest_Regressor': RandomForestRegressor(
                n_estimators=180,
                max_depth=10,
                min_samples_leaf=9,
                random_state=7,
                bootstrap=True,
                n_jobs=-1)
        }
    estimators_dict = {}
    for key in regressor_dict:
        # Choose transformation for numeric features
        if key in ['Decision_Tree_Regressor', 'XGBRegressor', 'Random_Forest_Regressor']:
            num_transformation = 'identity'
        else:
            num_transformation = 'power_transformer'
        # Choose optimum degree based on the analysis in the notebooks
        if key in ['Polynomial_Regression']:
            poly_degree = 4
        elif key in ['Decision_Tree_Regressor']:
            poly_degree=3
        else:
            poly_degree = 1

        estimator_pipe = transformer_estimator(regressor=regressor_dict[key],
                                               levels_list=levels_list,
                                               num_feat=num_features,
                                               cat_feat=cat_features,
                                               poly_degree=poly_degree,
                                               num_transformation=num_transformation)
        # Fill the estimator's dictionary
        estimators_dict[key] = estimator_pipe
    # print(estimators_dict['Decision_Tree_Regressor'])
    print('-'*30, 'ESTIMATORS READY!!', '-'*30)
    return estimators_dict


def train_model(X_train: pd.DataFrame,
                y_train: pd.Series,
                estimators_dict):
    """Trains the models in the estimators_dict.

    Parameters
    ----------
    X_train :
        Training features.
    y_train :
        Training target.
    estimators_dict :
        Dictionary containing the estimators to train.

    Returns
    -------
    Several trained models.
    """
    trained_regressors_dict = {}
    for key in estimators_dict:

        regressor = estimators_dict[key]
        regressor.fit(X_train, np.log(y_train))
        trained_regressors_dict[key] = regressor

    polyr = trained_regressors_dict['Polynomial_Regression']
    knnr = trained_regressors_dict['K_Nearest_Neighbors_Regressor']
    dtr = trained_regressors_dict['Decision_Tree_Regressor']
    xgbr = trained_regressors_dict['XGBRegressor']
    rfr = trained_regressors_dict['Random_Forest_Regressor']
    print('-' * 30, 'MODELS TRAINED!!', '-' * 30)

    return polyr, knnr, dtr, xgbr, rfr