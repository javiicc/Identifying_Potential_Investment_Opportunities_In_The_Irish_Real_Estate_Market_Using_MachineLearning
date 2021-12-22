import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, ParameterGrid, cross_val_score

def merge_tables(df):

    return df

def variables_to_modelize(df: pd.DataFrame) -> pd.DataFrame:
    features = [
        'price',
        'floor_area',
        #    'views',
        'latitude',
        'longitude',
        'bedroom',
        'bathroom',
        #    'sale_type',
        'type_house',
        #    'postcode',
        #    'state_district',
        #    'county',
        #    'city_district',
        #    'road',
        #    'place',
        'code',
        #    'admin1',
        #    'cities'
    ]
    data = df[features].copy()
    return data

def split_data(data, target='price', test_size=.15, output='X_y_train_test',
               random_state=7):  #7   .15 random_state=random_state test_size=test_size
    features = list(data.columns)
    features.remove(target)

    y = data[target].copy()
    X = data[features].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state)

    train_set = X_train.copy()
    train_set[target] = y_train.copy()

    test_set = X_test.copy()
    test_set[target] = y_test.copy()

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
        return X_train, X_test, y_train, y_test