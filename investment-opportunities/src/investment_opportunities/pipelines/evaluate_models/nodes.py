import pandas as pd
from sklearn import metrics


###########################################################################
# EVALUATE MODELS NODE
###########################################################################


def evaluate_models(X_test: pd.DataFrame, y_test: pd.Series,
                    polyr, knnr, dtr, voting_regressor_BA, xgbr, stackingr):
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
