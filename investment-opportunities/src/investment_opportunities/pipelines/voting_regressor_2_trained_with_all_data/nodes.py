"""
This is a boilerplate pipeline 'voting_regressor_2_trained_with_all_data'
generated using Kedro 0.17.5
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import VotingRegressor


###########################################################################
# VOTING REGRESSOR 2 FUNCTIONS
###########################################################################


def get_weights(scores_dict=None):
    """Takes a dictionary with models names and performances on the test set when
    they were evaluated in the notebooks and returns a list with their respective weights.

    Parameters
    ----------
    scores_dict :
        Dictionary with names and performance metrics.

    Returns
    -------
    List with weights.
    """
    if scores_dict is None:
        scores_dict = {'vrba': 79.04, 'rfr': 78.51, 'xgb': 79.31}

    tot = 0
    for key in scores_dict:
        tot += scores_dict[key]

    models_weight_list = []
    for key in scores_dict:
        weight = scores_dict[key] / tot
        models_weight_list.append(weight)
    return models_weight_list


###########################################################################
# STACKING REGRESSOR BA TRAINED NODE
###########################################################################


def voting_regresor_2_final_model(X: pd.DataFrame, y: pd.Series,
                             voting_regressor_BA, rfr, xgbr):
    """

    Parameters
    ----------
    X
    y
    voting_regressor_BA
    rfr
    xgbr

    Returns
    -------

    """
    # Define the base models
    final_model = VotingRegressor(
        estimators=[('vrba', voting_regressor_BA),
                    ('rfr', rfr),
                    ('xgb', xgbr)],
        weights=get_weights()).fit(X, np.log(y))
    print('-' * 30, 'FINAL MODEL TRAINED!!', '-' * 30)
    return final_model
