import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (StandardScaler, OneHotEncoder,
                                   PolynomialFeatures, PowerTransformer)
from sklearn.compose import ColumnTransformer
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from yellowbrick.regressor import ResidualsPlot
from sklearn.base import BaseEstimator, TransformerMixin


def split_data(data, target='price', 
               test_size=.15, 
               output='X_y_train_test',
               random_state=None):
    """Take a DataFrame and split it depending of the output argument.

    Parameters
    ----------
    data
    target
    test_size
    output
    random_state

    Returns
    -------
    Several Dataframes.
    """
    # List with features to model
    features = list(data.columns)
    features.remove(target)
    # Separate the target from the data
    y = data[target].copy()
    X = data[features].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size,
                                                        random_state=random_state)
    # Just in case we need training set
    train_set = X_train.copy()
    train_set[target] = y_train.copy()
    # Just in case we need test test
    test_set = X_test.copy()
    test_set[target] = y_test.copy()

    # The function will return different collection of sets depending of the
    # output argument
    if output == 'X_y_train_test':
        print('X_train:', X_train.shape, '\n' +
              'X_test:', X_test.shape, '\n' +
              'y_train:', y_train.shape, '\n' +
              'y_test:', y_test.shape, '\n')
        return X_train, X_test, y_train, y_test
    elif output == 'train_test':
        print(f'train_set: {train_set.shape}')
        print(f'test_set: {test_set.shape}')
        return train_set, test_set
    elif output == 'X_y':
        print(f'X: {X.shape}')
        print(f'y: {y.shape}')
        return X, y


def metrics_regression(y_test, y_pred, squared=False):
    """Prints several metrics.

    Parameters
    ----------
    y_test
    y_pred
    squared

    Returns
    -------
    The metrics.
    """
    r2_score = metrics.r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mape = metrics.mean_absolute_percentage_error(y_test, y_pred)
    # mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = metrics.mean_squared_error(y_test, y_pred, squared=squared)
    r = np.corrcoef(y_test, y_pred)[0][1]

    print(f'R²: {r2_score}')
    print(f'MAE: {mae}')
    print(f'MAPE: {mape}')
    # print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'R (corr): {r}\n')
    return r2_score, mae, r, mape


def scores_statistics(estimator, scoring_dict, X_train, y_train, cv=10,
                      return_train_score=False, time_info=False, 
                      return_est=False):
    """Uses cross validation to evaluate the estimator and calculates the mean
    and standard deviation of the results.

    Parameters
    ----------
    estimator
    scoring_dict
    X_train
    y_train
    cv
    return_train_score
    time_info
    return_est

    Returns
    -------
    Returns two dictionaries with scores.
    """
    scores = cross_validate(estimator,
                            X=X_train, y=y_train,
                            scoring=scoring_dict,
                            cv=cv,
                            return_train_score=return_train_score,
                            return_estimator=return_est)

    if time_info:
        fit_time_mean = np.mean(scores['fit_time'])
        fit_time_std = np.std(scores['fit_time'])
        score_time_mean = np.mean(scores['score_time'])
        score_time_std = np.std(scores['score_time'])
        # time_list = []
        print('fit_time mean:', fit_time_mean)
        print('fit_time std:', fit_time_std)
        print('score_time mean:', score_time_mean)
        print('score_time std:', score_time_std)

    scores_resume = {}
    for key in scoring_dict:
        mean = np.mean(scores['test_' + key])
        std = np.std(scores['test_' + key])
        print(key, 'mean:', mean)
        print(key, 'std:', std, '\n')
        scores_resume[key] = (mean, std)

    return scores, scores_resume


def plot_learning_curves(model, X_train, y_train, X_test, y_test, metric):
    """
    I don't use this functions in the notebooks since it takes a long time.
    """
    # X_train, X_val, y_train, y_val =
    # X_val, y_val = X_test.copy(), y_test.copy()
    # X_train, y_train = X_train[:250].copy(), y_train[:250].copy()

    train_errors, test_errors = [], []
    train_r2, test_r2 = [], []
    for i in range(1, len(X_train)):
        # fit model on the training dataset
        model.fit(X_train[:i], y_train[:i])

        y_train_pred = model.predict(X_train[:i])
        y_test_pred = model.predict(X_test)

        train_errors.append(mean_absolute_percentage_error(y_train[:i], y_train_pred))
        test_errors.append(mean_absolute_percentage_error(y_test, y_test_pred))

        train_r2.append(r2_score(y_train[:i], y_train_pred))
        test_r2.append(r2_score(y_test, y_test_pred))

    if metric == 'r2':
        print(f'Train R2: {train_r2[-1]}')
        print(f'Test R2: {test_r2[-1]}')

        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        plt.plot(train_r2, 'r-+', linewidth=2, label='Train', alpha=.4)
        plt.plot(test_r2, 'b-', linewidth=3, label='Test', alpha=.4)
        ax.set_ylim(0, 1)
        plt.axhline(y=train_r2[-1], color='black', linestyle='--', alpha=.8)
        plt.axhline(y=test_r2[-1], color='black', linestyle='--', alpha=.8)
        ax.legend()

    elif metric == 'mape':
        print(f'Train MAPE: {train_errors[-1]}')
        print(f'Test MAPE: {test_errors[-1]}')

        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        plt.plot(train_errors, 'r-+', linewidth=2, label='Train', alpha=.4)
        plt.plot(test_errors, 'b-', linewidth=3, label='Test', alpha=.4)
        ax.set_ylim(0, 1)
        plt.axhline(y=train_errors[-1], color='black', linestyle='--', alpha=.8)
        plt.axhline(y=test_errors[-1], color='black', linestyle='--', alpha=.8)
        ax.legend()


def compare_models(estimator, X_train, y_train,
                   scoring_dict,
                   cv, return_train_score=False):
    """It is pretty similar to `scores_statistics()` function but this one is
    used to compare the learners of the voting regressors.

    Parameters
    ----------
    estimator
    X_train
    y_train
    scoring_dict
    cv
    return_train_score

    Returns
    -------

    """

    scores = cross_validate(estimator,
                            X=X_train, y=y_train,  
                            scoring=scoring_dict,
                            cv=cv,
                            return_train_score=return_train_score)
    scores_resume = {}
    for key in scoring_dict:
        try:
            mean = np.mean(scores['test_' + key])
            std = np.std(scores['test_' + key])
            print(key, 'mean:', mean)
            #print(key, 'std:', std, '\n')
            scores_resume[key] = (mean, std)

        except:
            continue
    print('-' * 10)
    return scores, scores_resume


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, input_array, y=None):
        return self
    
    def transform(self, input_array, y=None):
        return input_array*1


def transformer_estimator(num_transformation: str,
                          levels_list,
                          num_feat: list,
                          cat_feat: list,
                          poly_degree=1,
                          regressor=None):
    """This function will make several types of transformations on the data based on the
    arguments given before to use a given estimator as a regressor.

    Parameters
    ----------
    num_transformation :
        String to indicate the type of transformation to perform.
    regressor :
        The regressor to be used.
    levels_list :
        List of Pandas Series to indicate the different levels of type_house and
        code attributes.
    num_feat:
        Numeric features.
    cat_feat :
        Categorical features.
    poly_degree :
        The degree in case we want a polynomial transformation.

    Returns
    -------
    Estimator.
    """
    if num_transformation is 'power_transformer':
        num_pipe = Pipeline([
           # ('imputer', SimpleImputer(strategy='median')),
            ('power_transformer', PowerTransformer(method='yeo-johnson')),
            ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)),
        ])
    elif num_transformation is 'std_scaler':
        num_pipe = Pipeline([
           # ('imputer', SimpleImputer(strategy='median')),
            ('std_scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)),
        ])
    elif num_transformation is 'identity':
        num_pipe = Pipeline([
           # ('imputer', SimpleImputer(strategy='median')),
            ('identity', IdentityTransformer()),
            ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)),
        ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan,
                                  strategy='constant',
                                  fill_value='Unknown')),
        ('one_hot_encoder', OneHotEncoder(categories=levels_list)),
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_feat),
        ('cat', cat_pipe, cat_feat),
    ], remainder='passthrough')

    if regressor is None:
        pipe_estimator = Pipeline(steps=[
            ('preprocessor', preprocessor)
        ])
    else:
        pipe_estimator = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', regressor)
        ])

    return pipe_estimator


def residuals(estimator, X_train, X_test, y_train, y_test):
    """Plots several scatter plots. Two of the show predicted prices against actual ones
    in train and test sets. The other one show residuals against predicted price.

    Parameters
    ----------
    estimator
    X_train
    X_test
    y_train
    y_test
    """
    plt.style.use('seaborn')
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    sns.regplot(x=y_train, y=estimator.fit(X_train, y_train).predict(X_train),
                scatter_kws={"color": "cornflowerblue"}, line_kws={"color": "red"}, 
                ax=ax[0, 0]).set_title('Actual vs Predicted, Train')
    ax[0, 0].set_xlabel('Actual price')
    ax[0, 0].set_ylabel('Predicted price')
    sns.regplot(x=y_test, y=estimator.fit(X_train, y_train).predict(X_test), 
                scatter_kws={"color": "cornflowerblue"}, line_kws={"color": "red"}, 
                ax=ax[0, 1]).set_title('Actual vs Predicted, Test')
    ax[0, 1].set_xlabel('Actual price')
    ax[0, 1].set_ylabel('Predicted price')
    
    for location in ['left', 'bottom', 'right', 'top']:
        ax[1, 1].spines[location].set_visible(False)
    # ax[1, 3].set_xticklabels('')
    # ax[1, 3].set_yticklabels('')
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    
    plt.tight_layout()
    
    visualizer = ResidualsPlot(estimator, ax=ax[1, 0],
                               train_color='b', test_color='r', 
                               train_alpha=.3, test_alpha=.3)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.show()


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
        scores_dict = {'poly': 74.06, 'knn': 74.57, 'dt': 71.71}

    tot = 0
    for key in scores_dict:
        tot += scores_dict[key]

    models_weight_list = []
    for key in scores_dict:
        weight = scores_dict[key] / tot
        models_weight_list.append(weight)
        
    print(scores_dict)
    
    return models_weight_list


def plot_metrics(metrics_to_plot):
    """Plots metrics.

    Parameters
    ----------
    metrics_to_plot :
        Dictionary with the metrics obtained.
    """
    plt.style.use('default')
    # Make a DataFrame with the metrics
    metrics_df = pd.DataFrame.from_dict(metrics_to_plot,
                                        columns=['r2', 'mae', 'correlation', 'mape'],
                                        orient='index')

    fig, ax = plt.subplots(2, 1, figsize=(6, len(metrics_to_plot) + 1.5))

    # First graph
    metrics_df[['r2', 'correlation', 'mape']].plot(kind='barh', ax=ax[0],
                                                   color=['#2C9650',
                                                          '#024A0A',
                                                          '#CA1A1A'],
                                                   alpha=1, width=.9)
    ax[0].set_title('Coef. Determination, Coef. Correlation, MAPE',
                    weight='bold', size=10)
    ax[0].tick_params(bottom=False, left=False)
    for location in ['left', 'bottom', 'right', 'top']:
        ax[0].spines[location].set_visible(False)
    ax[0].legend(labels=['R²', 'R corr', 'MAPE'], loc=3)

    # Second graph
    metrics_df.mae.plot(kind='barh', ax=ax[1], color='#CA1A1A', alpha=1, width=.5)
    ax[1].set_title('Mean Absolute Error', weight='bold', size=10)
    ax[1].tick_params(bottom=False, left=False)
    ax[1].set_xticks([0, 60000, 100000, 140000])
    ax[1].set_xticklabels([0, 60000, 100000, 140000])
    for location in ['left', 'bottom', 'right', 'top']:
        ax[1].spines[location].set_visible(False)

    # Write values in the bars
    for ix, row in enumerate(metrics_df.iterrows()):
       # print(ix)
       # print[0])
       # print(row[1])
       # print(row[1]['r2'])
        ax[0].text(x=row[1]['r2']-.06, y=ix-.37,
                   s=f"{round(row[1]['r2'], 2)}",
                   fontsize=6.5)
        ax[0].text(x=row[1]['correlation']-.06, y=ix-.07,
                   s=f"{round(row[1]['correlation'], 2)}",
                   fontsize=6.5,
                   color='white')
        ax[0].text(x=row[1]['mape']-.06, y=ix+.22,
                   s=f"{round(row[1]['mape'], 2)}",
                   fontsize=6.5)
        ax[1].text(x=row[1]['mae']-22500, y=ix-.08,
                   s=f"{round(row[1]['mae'])}")
    plt.tight_layout()
    

def get_base_predictions(mean_prices, data_to_predict):
    """This function is the baseline model. It assigns a prediction equals to the mean
    value in its zone to each house.

    Parameters
    ----------
    mean_prices :
        DataFrame with mean prices per town (or postal district for Dublin).
    data_to_predict : The data to predict (X_train, X_test)

    Returns
    -------
    A Series with the prediction prices.
    """
    # Calculate mean price for Ireland
    mean_price = mean_prices.mean()
    # print(mean_price.values[0])
    # The prediction is the mean price in the corresponding place so we join the data
    # and mean_prices by place
    # and taking the price as prediction
    y_pred = data_to_predict.merge(mean_prices, how='left', left_on='place', right_index=True).price
    # Fill missing values with the mean price for Ireland
    y_pred.fillna(value=mean_price.values[0], inplace=True)
    
    return y_pred


def comp_met(metrics_to_plot, new_model, last_best_model):
    """Prints the comparison of the current model with the baseline model and the last
    best model.

    Parameters
    ----------
    metrics_to_plot
    new_model
    last_best_model
    """
    base_model_mae = metrics_to_plot['Baseline Model'][1]
    new_mae = metrics_to_plot[new_model][1]
    last_best_mae = metrics_to_plot[last_best_model][1]
    
    improvement_basem = base_model_mae - new_mae
    improvement_basem_perc = improvement_basem / base_model_mae
    improvement_lbestm = last_best_mae - new_mae
    improvement_lbestm_perc = improvement_lbestm / last_best_mae
    
    print(f'Improvement respect Baseline Model: {round(improvement_basem)}€ -> '
          f'{round(improvement_basem_perc*100)}%')
    print(f'Improvement respect Last Best Model: {round(improvement_lbestm)}€ -> '
          f'{round(improvement_lbestm_perc*100)}%')
