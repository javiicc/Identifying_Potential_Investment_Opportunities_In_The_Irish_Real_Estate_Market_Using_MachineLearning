
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor

# get a stacking ensemble of models
def get_stacking_final_model(voting_regressor_BA, xgbr, X, y): #xgb_pipe_estimator
    # define the base models
    level0 = list()
    level0.append(('voting_regressor_BA', voting_regressor_BA))
 #   level0.append(('rfr', rfr))
    level0.append(('xgb', xgbr)) #xgb_pipe_estimator
    # define meta learner model
    level1 = LinearRegression()
    # define the stacking ensemble
    final_model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
    final_model.fit(X, y) # Training with all data
    return final_model