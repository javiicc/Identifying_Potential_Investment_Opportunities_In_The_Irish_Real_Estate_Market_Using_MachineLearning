import numpy as np
import pandas as pd

import re
from os import listdir
from os.path import isfile, join

from datetime import timedelta



###########################################################################
# OUTLIERS PREPROCESSING FUNCTIONS
###########################################################################
# Percentile based method
def pct_method(data, level, lower=True):
    """Classify outliers based on percentiles.

    Parameters
    ----------
    data :
        Column.
    level :
        Punto de corte a partir del cual se considera outlier.
    lower :
        To indicate whether there should be ...

    Returns
    -------
    .
    """
    # Upper and lower limits by percentiles
    upper = np.percentile(data, 100 - level)
    if lower:
        lower = np.percentile(data, level)
        # Returning the upper and lower limits
        return [lower, upper]
    else:
        return [upper]


# Interquartile range method
def iqr_method(data):
    """Classify outliers based on.

    Parameters
    ----------
    data :
        .

    Returns
    -------
    .
    """
    # Calculating the IQR
    perc_75 = np.percentile(data, 75)
    perc_25 = np.percentile(data, 25)
    iqr_range = perc_75 - perc_25

    # Obtaining the lower and upper bound
    iqr_upper = perc_75 + (1.5 * iqr_range)
    iqr_lower = perc_25 - (1.5 * iqr_range)

    # Returning the upper and lower limits
    return [iqr_lower, iqr_upper]


# This approach only works if the data is approximately Gaussian
def std_method(data):
    """Classify outliers based on.

    Parameters
    ----------
    data :
        .

    Returns
    -------
    .
    """
    # Creating three standard deviations away boundaries
    std = np.std(data)
    upper_3std = np.mean(data) + 3 * std
    lower_3std = np.mean(data) - 3 * std
    # Returning the upper and lower limits
    return [lower_3std, upper_3std]


def outlier_bool(df, feature, level=1, continuous=False, log=False):
    """Classify outliers based on.

    Parameters
    ----------
    data :
        .

    Returns
    -------
    .
    """
    data = df[feature]

    # Taking logs is specified
    if log is True:
        data = np.log(data + 1)

    # Obtaining the ranges
    pct_range = pct_method(data, level)
    iqr_range = iqr_method(data)
    std_range = std_method(data)

    if continuous is False:
        # Setting the lower limit fixed for discrete variables
        low_limit = np.min(data)
        # high_limit = np.max([pct_range[1],
        #                    iqr_range[1],
        #                   std_range[1]])

    elif continuous:
        if feature is 'floor_area':
            # Percentile based method is the only one that return a
            # positive value
            low_limit = pct_range[0]
        else:
            # print('no')
            low_limit = np.min([pct_range[0],
                                iqr_range[0],
                                # std_range[0]
                                ])
    high_limit = np.max([pct_range[1],
                         iqr_range[1],
                         # std_range[1]
                         ])

    print(f'Limits: {[low_limit, high_limit]}')
    # Restrict the data with the minimum and maximum
    no_outlier_bool = data.between(low_limit, high_limit)
    outlier_bool = no_outlier_bool == False
    print(f'No outliers: {no_outlier_bool.sum()}')
    print(f'Outliers: {(outlier_bool).sum()}\n')

    # Return boolean
    return outlier_bool

'''
def drop_outliers(df, feature, level=1, continuous=False, log=False, inplace=False):
    """Classify outliers based on.

    Parameters
    ----------
    data :
        .

    Returns
    -------
    .
    """
    print(f'Range before: {[df[feature].min(), df[feature].max()]}\n')

    outlier_boolean = outlier_bool(df=df, feature=feature, level=1, continuous=continuous,
                                   log=False)
    rows_before = df.shape[0]

    # Filter data to get outliers
    outliers = df[outlier_boolean == False]
    # Filter data to drop outliers
    df = df[outlier_boolean]

    rows_after = df.shape[0]

    print(f'Range after: {[df[feature].min(), df[feature].max()]}')
    print(f'Outliers dropped: {rows_before - rows_after}')

    return df, outliers
'''
'''
def common_ix(index_list):
    """Classify outliers based on.

    Parameters
    ----------
    data :
        .

    Returns
    -------
    .
    """
    data_ix = []
    for i, elem in enumerate(index_list):
        # First index list
        if i == 0:
            # initial_ix = sd_out_price.index
            initial_ix = elem
            for ix in initial_ix:
                # If ix is in the next index list then por el momento
                # cumple la condicion y por tanto se une a la lista
                if ix in index_list[i + 1]:
                    data_ix.append(ix)
            print(f'1st and 2nd index lists: {len(data_ix)} rows')

        elif i < 4:
            for ix in data_ix:
                # Check whether index from data_ix are in the next list,
                # if not -> remove it
                if ix not in index_list[i + 1]:
                    data_ix.remove(ix)
            print(
                f'{i + 2}{"rd" if i + 2 == 3 else "th"} index list: {len(data_ix)} rows')
    print('-' * 10)
    return data_ix
'''
###########################################################################
# OUTLIERS DROPPING NODES
###########################################################################


def drop_outliers(df: pd.DataFrame,
                  features=['price', 'floor_area',
                            'views', 'bedroom', 'bathroom'],
                  level=1, log=False, inplace=False): #continuous=False,
    """Classify outliers based on.

    Parameters
    ----------
    data :
        .

    Returns
    -------
    .
    """
    # List to add outliers from each feature
    outliers_list = []
    for feature in features:

        print(feature.upper())
        print(f'Range before: {[df[feature].min(), df[feature].max()]}\n')

        if feature in ['bedroom', 'bathroom']:
            outlier_boolean = outlier_bool(df=df, feature=feature, level=1, continuous=False,
                                           log=False)
        elif feature in ['price', 'floor_area', 'views']:
            outlier_boolean = outlier_bool(df=df, feature=feature, level=1,
                                           continuous=True, log=False)

        rows_before = df.shape[0]
        # Filter data to get outliers
        outliers = df[outlier_boolean]
        rows_after = df[outlier_boolean == False].shape[0]
        print(f'Range after: {[df[feature].min(), df[feature].max()]}')
        print(f'Outliers to drop: {rows_before - rows_after}')
        print('-----------')

        # Increase the list with outliers index from each feature
        outliers_list += list(outliers.index)
        #print(len(outliers_list))

        # Convert list to set to eliminate repeated index
        outliers_set = set(outliers_list)
        #print(len(outliers_set))

    before = df.shape
    print('---------------')
    print('Shape before:', before)
    # Drop outliers!
    df.drop(index=outliers_set, inplace=True)
    after = df.shape
    print('Shape after:', after)
    print('Outliers dropped:', before[0] - after[0])

    return df
'''
def drop_outliers_ix(df: pd.DataFrame, index_dict):
    """Classify outliers based on.

    Parameters
    ----------
    data :
        .

    Returns
    -------
    .
    """
    print(df.shape)
    outliers_rep = []
    for key in index_dict:
        list_ = index_dict[key]
        outliers_rep += list(list_)
        # print(len(outliers_rep))

    outliers = pd.Series(outliers_rep).unique()
    print('Outliers dropped:', len(outliers))
    print('estoy aqui')


    df_without_outliers = df.drop(index=outliers).copy()
    print(df_without_outliers.shape)

    return df
'''
