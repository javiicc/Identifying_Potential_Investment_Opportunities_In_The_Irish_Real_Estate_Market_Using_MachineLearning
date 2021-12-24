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


def voting_regresor(X_train, y_train, polyr, knnr, dtr):
    #print('000000000000000000000', len(estimators_dict))
    #print('000000000000000000000', len(get_weigts()))
    #polyr = trained_regressors_dict['Polynomial_Regression']
    #knnr = trained_regressors_dict['K_Nearest_Neighbors_Regressor']
    #dtr = trained_regressors_dict['Decision_Tree_Regressor']

    voting_regressor_ba = VotingRegressor(
        estimators=[('poly', polyr),
                    ('knn', knnr),
                    ('dt', dtr)],
        weights=get_weigts()).fit(X_train, y_train)


    #voting_regressor_ba.fit(X_train, y_train)
    #print(voting_regressor_ba)
    return voting_regressor_ba