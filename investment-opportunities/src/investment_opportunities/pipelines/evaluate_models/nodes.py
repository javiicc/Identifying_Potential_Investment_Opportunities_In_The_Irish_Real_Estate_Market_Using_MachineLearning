"""
This is a boilerplate pipeline 'evaluate_models'
generated using Kedro 0.17.5
"""
import pandas as pd
import numpy as np
from sklearn import metrics


###########################################################################
# EVALUATE MODELS NODE
###########################################################################


def evaluate_models(X_test: pd.DataFrame, y_test: pd.Series,
                    polyr, knnr, dtr, voting_regressor_BA, xgbr, rfr, voting_regressor_2):
    """Calculates the coefficient of determination and other metrics and prints them.

    Parameters
    ----------
    X_test:
        Testing data of independent features.
    y_test:
        Testing data for price.
    polyr
    knnr
    dtr
    voting_regressor_BA
    xgbr
    stackingr

    Returns
    -------
    Just prints the metrics.
    """
    trained_regressors_dict = {
        'Polynomial Regression': polyr,
        'K Nearest Neighbors Regression': knnr,
        'Decision Tree Regression': dtr,
        'Voting Regressor BA': voting_regressor_BA,
        'XGB Regressor': xgbr,
        'Random Forest Regressor': rfr,
        'Voting Regressor 2': voting_regressor_2,
    }

    for key in trained_regressors_dict:
        print('\n', '-'*60, '\n', key, '\n', '-'*60)
        # Antilogarithmic transformation
        y_pred = np.exp(trained_regressors_dict[key].predict(X_test))

        r2_score = metrics.r2_score(y_test, y_pred)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mape = metrics.mean_absolute_percentage_error(y_test, y_pred)
        # mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
        # r = np.corrcoef(y_test, y_pred)[0][1]

        print(f' R²: {r2_score}')
        print(f' MAE: {mae}')
        print(f' MAPE: {mape}')
        # print(f' MSE: {mse}')
        print(f' RMSE: {rmse}\n')
        # print(f'R (corr): {r}\n')
