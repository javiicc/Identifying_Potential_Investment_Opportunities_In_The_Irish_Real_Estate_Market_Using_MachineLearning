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


def evaluate_models(X_test: pd.DataFrame, y_test: pd.Series,
                    polyr, knnr, dtr, voting_regressor_BA, xgbr, stackingr):
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    trained_regressors_dict={
        'Polynomial Regression': polyr,
        'K Nearest Neighbors Regression': knnr,
        'Decision Tree Regression': dtr,
        'Voting Regressor BA': voting_regressor_BA,
        'XGB Regressor': xgbr,
        'Stacking Regressor': stackingr,
    }

    for key in trained_regressors_dict:
        print('\n', '-'*60, '\n', key, '\n', '-'*60)
        y_pred = trained_regressors_dict[key].predict(X_test)

        r2_score = metrics.r2_score(y_test, y_pred)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mape = metrics.mean_absolute_percentage_error(y_test, y_pred)
        # mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)

        print(f' R²: {r2_score}')
        print(f' MAE: {mae}')
        print(f' MAPE: {mape}')
        # print(f' MSE: {mse}')
        print(f' RMSE: {rmse}\n')