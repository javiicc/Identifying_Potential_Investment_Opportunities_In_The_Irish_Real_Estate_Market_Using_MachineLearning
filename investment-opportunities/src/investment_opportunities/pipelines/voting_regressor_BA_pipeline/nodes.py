
from sklearn.ensemble import VotingRegressor




from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error

###########################################################################
# DATA SCIENCE FUNCTIONS AND CLASSES
###########################################################################



def get_weigts(scores_dict):
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
    return models_weigth

def compare_models(models_dict, X_test, y_test):

    for key in models_dict:
        print('\n', '-' * 60, '\n', key, '\n', '-' * 60)
        y_pred = models_dict[key].predict(X_test)

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

###########################################################################
# DATA SCIENCE NODES
###########################################################################
def get_voting_regressor_BA_estimator(polyr, knnr, dtr, X_train, y_train, X_test, y_test):
    models_weigth_list = get_weigts({'poly': 74.06, 'knn': 74.57, 'dt': 71.71})

    voting_regressor_BA = VotingRegressor(
        estimators=[('poly', polyr),
                    ('knn', knnr),
                    ('dt', dtr)],
        weights=models_weigth_list)

    voting_regressor_BA.fit(X_train, y_train)
    print(voting_regressor_BA)
    models_dict = {'Polynomial Regression': polyr,
                   'K Nearest Neighbors Regressor': knnr,
                   'Decission Tree Regressor': dtr,
                   'Voting Regressor': voting_regressor_BA}


    compare_models(models_dict=models_dict, X_test=X_test, y_test=y_test)

    return voting_regressor_BA
