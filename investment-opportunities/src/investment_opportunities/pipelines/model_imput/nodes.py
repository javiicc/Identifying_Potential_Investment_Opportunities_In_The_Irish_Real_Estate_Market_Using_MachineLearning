import pandas as pd
from sklearn.model_selection import train_test_split


###########################################################################
# MODEL INPUT NODES
###########################################################################


def merge_tables(df: pd.DataFrame) -> pd.DataFrame:
    """This function has been created in case in there are more DataFrames in the future.

    Parameters
    ----------
    df :
        DataFrame.

    Returns
    -------
    DataFrame. Merged DataFrames in the future.
    """
    return df


def variables_to_model(df: pd.DataFrame) -> pd.DataFrame:
    """Takes a Dataframe and return with features to model.

    Parameters
    ----------
    df :
        DataFrame.

    Returns
    -------
    DataFrame with some features to model.
    """
    features = [
        'price',
        'floor_area',
        # 'views',
        'latitude',
        'longitude',
        'bedroom',
        'bathroom',
        # 'sale_type',
        'type_house',
        # 'postcode',
        # 'state_district',
        # 'county',
        # 'city_district',
        # 'road',
        # 'place',
        'code',
        # 'admin1',
        # 'cities'
    ]

    data = df[features].copy()
    return data


def split_data(data: pd.DataFrame,
               target='price',
               test_size=.15,
               output='X_y_and_X_y_train_test',
               random_state=7):
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

    y = data[target].copy()
    X = data[features].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state)

    # Just in case we need training set
    train_set = X_train.copy()
    train_set[target] = y_train.copy()

    # Just in case we need test test
    test_set = X_test.copy()
    test_set[target] = y_test.copy()

    print('-'*25)
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
    elif output == 'X_y_and_X_y_train_test':
        print('X_train:', X_train.shape, '\n' +
              'X_test:', X_test.shape, '\n' +
              'y_train:', y_train.shape, '\n' +
              'y_test:', y_test.shape, '\n')
        print(f'X: {X.shape}')
        print(f'y: {y.shape}')
        return X_train, X_test, y_train, y_test, X, y
