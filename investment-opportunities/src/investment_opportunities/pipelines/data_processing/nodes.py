import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re
from os import listdir
from os.path import isfile, join

from datetime import timedelta


###########################################################################
# DATA PREPROCESSING FUNCTIONS
###########################################################################

def preprocess_price(df: pd.DataFrame) -> pd.DataFrame:
    """Takes a dataframe, and does wrangling and cleaning task over the
    `price` column.

    Parameters
    ----------
    df :
        The dataframe to search.

    Returns
    -------
    The dataframe cleaned.
    """
    before = df.shape[0]
    print(f'Rows before dropping: {before}')

    # Filter to drop three kind of ads
    df = df[((df['price'] != 'Price on Application') & (
            df['price'] != 'AMV: Price on Application') &
             df['price'].notna()
             )].copy()

    index_to_drop = df.dropna(subset=['price']).loc[
        df.dropna(subset=['price'])['price'].str.contains('£')
    ].index
    # Drop ads with '£' pattern
    df.drop(index=index_to_drop, inplace=True)

    # Prices wrangling
    df['price'] = df['price'].str.split('€').str[-1] \
        .str.replace(',', '', regex=False) \
        .str.replace(')', '', regex=False) \
        .astype(float)

    after = df.shape[0]
    print(f'Rows after dropping: {after}\n' + '-' * 10)
    print(f'Difference: {after - before}')

    return df


def preprocess_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Takes a dataframe and make a column for `latitude` and another one
    for `longitude` with data from `coordinates` column. The new columns
    will be float dtype.

    Parameters
    ----------
    df :
        The dataframe to search.

    Returns
    -------
    The dataframe processed.
    """
    print(f'Shape before process: {df.shape}')

    df['latitude'] = df['coordinates'].str.split('+').str[0].astype(float)
    df['longitude'] = df['coordinates'].str.split('+').str[1].astype(float)

    df.drop(columns=['coordinates'], inplace=True)
    print(f'Shape after process: {df.shape}')
    return df


def drop_coord_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Takes a dataframe and drop coordinates outliers.

    Parameters
    ----------
    df :
        The dataframe to search.

    Returns
    -------
    The dataframe processed.
    """
    before = df.shape[0]
    print(f'Rows before dropping: {before}')

    df.drop(index=df[(df['latitude'] < 51.3) | (df['latitude'] > 55.4) | \
                     (df['longitude'] > -5.9) | (df['longitude'] < -10.6)].index,
            inplace=True)
    # Drop ads from Nothern Ireland
    df.drop(index=df[(df['latitude'] > 54.5) & (df['longitude'] > -7.9) & \
                     (df['latitude'] < 54.6)].index, inplace=True)

    after = df.shape[0]
    print(f'Rows after dropping: {after}\n' + '-' * 10)
    print(f'Difference: {after - before}')
    return df


def drop_floor_area(df: pd.DataFrame) -> pd.DataFrame:
    """Takes a dataframe and clean the `floor_area` column.

    Parameters
    ----------
    df :
        The dataframe to search.

    Returns
    -------
    The dataframe cleaned.
    """
    before = df.shape[0]
    print(f'Rows after dropping: {before}')

    # Filter missing values and those whitout the pattern `m²`
    df = df.dropna(subset=['floor_area']).loc[
        df.dropna(subset=['floor_area'])['floor_area'].str.contains('m²')
    ].copy()

    after = df.shape[0]
    print(f'Rows after dropping: {after}\n' + '-' * 10)
    print(f'Difference: {after - before}')

    return df


def floor_area_wragling(df: pd.DataFrame) -> pd.DataFrame:
    """Takes a dataframe and clean the `floor_area` column.

    Parameters
    ----------
    df :
        The dataframe to search.

    Returns
    -------
    The dataframe whit the column wrangled.
    """
    df['floor_area'] = df['floor_area'].str.split(' ').str[0]
    return df


def preprocess_floor_area(df: pd.DataFrame) -> pd.DataFrame:
    """Takes a dataframe and process it with `drop_floor_area`
    and `floor_area_wragling` functions.

    Parameters
    ----------
    df :
        The dataframe to search.

    Returns
    -------
    The dataframe processed.
    """
    df_dropped = drop_floor_area(df=df)
    df_wrangled = floor_area_wragling(df=df_dropped)
    return df_wrangled


def drop_info(df: pd.DataFrame) -> pd.DataFrame:
    """Takes a dataframe and drop rows whitout an splitted `info`
    length diferrent than four.

    Parameters
    ----------
    df :
        The dataframe to search.

    Returns
    -------
    The dataframe processed.
    """
    before = df.shape
    print(f'Shape before dropping: {before}')

    # df.dropna(subset=['info'])
    df = df[df['info'].str.split(',').apply(len) == 4]

    after = df.shape
    print(f'Shape after dropping: {after}\n' + '-' * 10)
    print(f'Dropped: {before[0] - after[0]} rows\n')

    return df


def preprocess_info(df: pd.DataFrame) -> pd.DataFrame:
    """Takes a dataframe and process it with `drop_info`
    and creates three new columns whit info from another one.
    Also, it drops the other one column.

    Parameters
    ----------
    df :
        The dataframe to search.

    Returns
    -------
    The dataframe processed.
    """
    df = drop_info(df).copy()

    before = df.shape
    print(f'Shape before dropping: {before}')

    df['bedroom'] = df['info'].str.split(',').str[0]
    df['bathroom'] = df['info'].str.split(',').str[1]
    df['plus_info'] = df['info'].str.split(',').str[3]

    df.drop(columns=['info'], inplace=True)

    after = df.shape
    print(f'Shape after dropping: {after}\n' + '-' * 10)
    print(f'Difference: {after[1] - before[1]} columns')

    return df


def preprocess_views(df: pd.DataFrame) -> pd.DataFrame:
    """Takes a dataframe and quits commas.

    Parameters
    ----------
    df :
        The dataframe to search.

    Returns
    -------
    The dataframe processed.
    """
    df['views'] = df['views'].str.replace(',', '').astype(float)
    return df


def preprocess_rooms(df: pd.DataFrame) -> pd.DataFrame:
    """Takes a dataframe and isolate the numbers of rooms.

    Parameters
    ----------
    df :
        The dataframe to search.

    Returns
    -------
    The dataframe processed.
    """
    df['bedroom'] = df['bedroom'].str.split(' ').str[0]
    df['bathroom'] = df['bathroom'].str.split(' ').str[0]
    return df

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
            # Percentile based method is the onlu oney that return a
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
    outlier = data.between(low_limit, high_limit)
    print(f'No outliers: {outlier.sum()}')
    print(f'Outliers: {(outlier == False).sum()}\n')

    # Return boolean
    return outlier


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




###########################################################################
# DATA PREPROCESSING NODES
###########################################################################

def preprocess_ads(df: pd.DataFrame) -> pd.DataFrame:

    # Drop useless columns
    df.drop(columns=['energy_performance_indicator'], inplace=True)
    df.drop(columns='item_id', inplace=True)

    df.replace('none', np.nan, inplace=True)

    # Preprocess columns
    df = preprocess_price(df.copy()).reset_index(drop=True)
    df = preprocess_coordinates(df)
    df = drop_coord_outliers(df)
    df = preprocess_floor_area(df)
    df = preprocess_info(df)
    df = preprocess_views(df)
    df = preprocess_rooms(df)

    return df

def drop_outliers(df: pd.DataFrame,
                  features=['price', 'floor_area',
                            'views', 'bedroom', 'bathroom'],
                  level=1, continuous=False, log=False, inplace=False):
    """Classify outliers based on.

    Parameters
    ----------
    data :
        .

    Returns
    -------
    .
    """
    for feature in features:

        print(f'Range before: {[df[feature].min(), df[feature].max()]}\n')

        outlier_boolean = outlier_bool(df=df, feature=feature, level=1, continuous=continuous,
                                       log=False)
        rows_before = df.shape[0]

        outliers_dict = {}

        # Filter data to get outliers
        outliers = df[outlier_boolean == False]
        # Filter data to drop outliers
        df = df[outlier_boolean]

        rows_after = df.shape[0]

        print(f'Range after: {[df[feature].min(), df[feature].max()]}')
        print(f'Outliers dropped: {rows_before - rows_after}')
        print('-----------')

        outliers_dict[feature] = outliers.index

    return outliers_dict

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







