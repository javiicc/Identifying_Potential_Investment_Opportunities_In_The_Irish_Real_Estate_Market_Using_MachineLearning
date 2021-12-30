import pandas as pd
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
# BASIC ALGORITHM NODES
###########################################################################


def get_levels(df: pd.DataFrame) -> List[pd.Series]:
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
    levels_type_house = df.type_house.unique()
    levels_code = df.code.unique()
    return [levels_type_house, levels_code]


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
                random_state=7),  # NO DA EXACTO, CHECK IT
            'XGBRegressor': XGBRegressor(
                n_estimators=177,  # 150
                max_depth=3,
                learning_rate=.1,
                subsample=.30),
                          }
    estimators_dict = {}
    for key in regressor_dict:
        # Choose transformation for numeric features
        if key in ['Decision_Tree_Regressor', 'XGBRegressor']:
            num_transformation = 'identity'
        else:
            num_transformation = 'power_transformer'
        # Choose optimum degree based on the analysis in the notebooks
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
        regressor.fit(X_train, y_train)
        trained_regressors_dict[key] = regressor

    polyr = trained_regressors_dict['Polynomial_Regression']
    knnr = trained_regressors_dict['K_Nearest_Neighbors_Regressor']
    dtr = trained_regressors_dict['Decision_Tree_Regressor']
    xgbr = trained_regressors_dict['XGBRegressor']
    print('-' * 30, 'MODELS TRAINED!!', '-' * 30)

    return polyr, knnr, dtr, xgbr
