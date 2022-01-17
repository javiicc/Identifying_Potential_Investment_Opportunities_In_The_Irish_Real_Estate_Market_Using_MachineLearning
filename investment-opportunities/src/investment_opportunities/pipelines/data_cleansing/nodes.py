"""
This is a boilerplate pipeline 'data_cleansing'
generated using Kedro 0.17.5
"""

import numpy as np
import pandas as pd

from typing import List

###########################################################################
# OUTLIERS PREPROCESSING FUNCTIONS
###########################################################################


# Percentile based method
def pct_method(data: pd.Series, level: int, lower=True) -> List[int]:
    """Classify outliers based on percentile range.

    Parameters
    ----------
    data :
        Column.
    level :
        Cut point since is considered outlier.
    lower :
        To indicate whether there should be lower limit.

    Returns
    -------
    Range.
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
def iqr_method(data: pd.Series) -> List[int]:
    """Classify outliers based on interquartile range.

    Parameters
    ----------
    data :
        Column.

    Returns
    -------
    Interquartile range.
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


def outlier_bool(df: pd.DataFrame, feature: pd.Series, level=1,
                 continuous=False, log=False) -> pd.Series:
    """Classify outliers based on percentile and interquartile ranges.

    Parameters
    ----------
    df :
        DataFrame.
    feature :
        Column.
    level :
        'p' in np.percentile method.
    continuous :
        To indicate whether the variable is continuous or not.
    log :
        To indicate whether compute natural logarithm element-wise or not.

    Returns
    -------
    A Pandas Series of booleans with True for outliers values.
    """
    data = df[feature]

    # Taking logs is specified
    if log is True:
        data = np.log(data + 1)

    # Obtaining the ranges
    pct_range = pct_method(data, level)
    iqr_range = iqr_method(data)

    if continuous is False:
        # Setting the lower limit fixed for discrete variables
        low_limit = np.min(data)
        # high_limit = np.max([pct_range[1],
        #                      iqr_range[1])

    else:
        if feature is 'floor_area':
            # Percentile based method is the only one that return a
            # positive value
            low_limit = pct_range[0]
        else:
            low_limit = np.min([pct_range[0],
                                iqr_range[0],
                                ])
    high_limit = np.max([pct_range[1],
                         iqr_range[1],
                         ])

    print(f'Limits: {[low_limit, high_limit]}')
    # Restrict the data with the minimum and maximum
    no_outlier_bool = data.between(low_limit, high_limit)
    outlier_bool = no_outlier_bool == False
    print(f'No outliers: {no_outlier_bool.sum()}')
    print(f'Outliers: {outlier_bool.sum()}\n')

    # Return boolean
    return outlier_bool


###########################################################################
# OUTLIERS DROPPING NODES
###########################################################################


def drop_outliers(df: pd.DataFrame,
                  features=('price', 'floor_area', 'views',
                            'bedroom', 'bathroom')) -> pd.DataFrame:
    """.

    Parameters
    ----------
    df :
        DataFrame with outliers.
    features :
        Column.

    Returns
    -------
    DataFrame without outliers.
    """
    # List to add outliers from each feature
    outliers_list = []
    for feature in features:

        print('-'*50, '\n' + ' '*5, feature.upper(), '\n' + '-'*50)
        print(f'Range before: {[df[feature].min(), df[feature].max()]}\n')

        # Use the outlier_bool() function to take the boolean Series
        if feature in ['bedroom', 'bathroom']:
            outlier_boolean = outlier_bool(df=df, feature=feature, level=1,
                                           continuous=False, log=False)
        elif feature in ['price', 'floor_area', 'views']:
            outlier_boolean = outlier_bool(df=df, feature=feature, level=1,
                                           continuous=True, log=False)

        rows_before = df.shape[0]
        # Filter data to get outliers
        outliers = df[outlier_boolean]
        rows_after = df[outlier_boolean == False].shape[0]
        print(f'Range after: {[df[feature].min(), df[feature].max()]}')
        print(f'Outliers to drop: {rows_before - rows_after}')

        # Increase the list with outliers index from each feature
        # outlier_list below has repeated index
        outliers_list += list(outliers.index)
        # print(len(outliers_list))

        # Convert list to set to eliminate repeated index
        outliers_set = set(outliers_list)
        # print(len(outliers_set))

    before = df.shape
    print(('-'*25 + '\n' + '-'*25) * 2)  # + '-'*25
    print('Shape before:', before)
    # Drop outliers!
    df.drop(index=outliers_set, inplace=True)
    after = df.shape
    print('Shape after:', after)
    print('TOTAL OUTLIERS DROPPED:', before[0] - after[0])

    return df
