"""
This is a boilerplate pipeline 'voting_regressor_BA'
generated using Kedro 0.17.5
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import VotingRegressor


###########################################################################
# VOTING REGRESSOR BA FUNCTIONS
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
        scores_dict = {'poly': 78.77, 'knn': 73.38, 'dt': 71.46}

    tot = 0
    for key in scores_dict:
        tot += scores_dict[key]

    models_weight_list = []
    for key in scores_dict:
        weight = scores_dict[key] / tot
        models_weight_list.append(weight)
    return models_weight_list


###########################################################################
# VOTING REGRESSOR BA NODE
###########################################################################


def voting_regresor(X_train: pd.DataFrame, y_train: pd.DataFrame, polyr, knnr, dtr):
    """Builds a voting regressor wit the models given and trains it with the data given.

    Parameters
    ----------
    X_train
    y_train
    polyr
    knnr
    dtr

    Returns
    -------
    Returns a voting regressor trained.
    """
    voting_regressor_ba = VotingRegressor(
        estimators=[('poly', polyr),
                    ('knn', knnr),
                    ('dt', dtr)],
        weights=get_weights()
    ).fit(X_train, np.log(y_train))
    print('-' * 30, 'VOTING REGRESSOR TRAINED!!', '-' * 30)
    return voting_regressor_ba
