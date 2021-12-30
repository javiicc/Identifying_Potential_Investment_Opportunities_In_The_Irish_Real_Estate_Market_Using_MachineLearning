



def get_predictions(model, X, y):
    y_pred = model.predict(X)

    data_w_predictions = X.copy()
    data_w_predictions['actual_price'] = y
    # We convert y_pred to a list to avoid a problem with nan values
    data_w_predictions['predicted_price'] = list(y_pred)
    return data_w_predictions


def get_residuals(data_w_residuals):
    # The residual is what will tell us whether the asset is a potential opportunity
    data_w_residuals['residual'] = data_w_residuals.predicted_price - data_w_residuals.actual_price
    data_w_residuals['res_percentage'] = data_w_residuals.residual / data_w_residuals.actual_price
    return data_w_residuals


def add_features_for_frontend(data_w_residuals, model_input):
    columns_to_merge = model_input[['url', 'place']] # add m2
    data_for_frontend = data_w_residuals.merge(columns_to_merge, left_index=True, right_index=True)
    return data_for_frontend