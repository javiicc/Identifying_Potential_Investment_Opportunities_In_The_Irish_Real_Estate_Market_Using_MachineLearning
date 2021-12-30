import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor


###########################################################################
# STACKING REGRESSOR BA TRAINED NODE
###########################################################################


def get_stacking_final_model(voting_regressor_BA, xgbr, X: pd.DataFrame, y: pd.Series):
    """Takes the voting regressor and the xgboost, builds a stacking model with them and a
    linear regression and trains it with the given data.

    Parameters
    ----------
    voting_regressor_BA
    xgbr
    X
    y

    Returns
    -------
    Stacking model trained.
    """
    # Define the base models
    level0 = list()
    level0.append(('voting_regressor_BA', voting_regressor_BA))
    level0.append(('xgb', xgbr))
    # Define meta learner model
    level1 = LinearRegression()
    # Define the stacking ensemble
    final_model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
    final_model.fit(X, y)  # Training with all data
    print('-' * 30, 'FINAL MODEL TRAINED!!', '-' * 30)
    return final_model
