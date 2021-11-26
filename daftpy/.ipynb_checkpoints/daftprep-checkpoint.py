import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re
from os import listdir
from os.path import isfile, join

from datetime import timedelta


###########################################################################
# DATA CLEANSING AND WRANGLING
###########################################################################

def num_perc(df, feature, pattern):
    """Takes a dataframe, a feature and a pattern and returns the number and 
    percentage of listings with that pattern y that feature.
    
    Parameters
    ----------
    df : 
        The dataframe to search.
    feature :
        The feature to explore.
    pattern :
        The pattern we want to study.
    
    Returns
    -------
    An example of a listing whit the pattern.
    """
    
    if pattern is np.nan:
        # Number of ads with pattern
        pattern_num = df[feature].isna().sum()
        print(f'Ads with "{pattern}": {pattern_num}')
        
        # % of ads with pattern
        pattern_perc = pattern_num / df.shape[0]
        print(f'Ads with "{pattern}": {round(pattern_perc * 100, 2)}%')
        
        example = df[df[feature].isna()].sample()
        
        return example
    
    if pattern == '£' or pattern == 'ac' or pattern == 'm²':
        # Number of ads with pattern
        pattern_num = df.dropna(subset=[feature]).loc[
                      df.dropna(subset=[feature])[feature].str.contains(pattern)].shape[0]
        print(f'Ads with "{pattern}": {pattern_num}')
        
        # % of ads with pattern
        pattern_perc = pattern_num / df.shape[0]
        print(f'Ads with "{pattern}": {round(pattern_perc * 100, 2)}%')
        
        example = df.dropna(subset=[feature]).loc[
                  df.dropna(subset=[feature])[feature].str.contains(pattern)].sample()
        
        return example
    
    # Number of ads with pattern
    pattern_num = df[df[feature] == pattern].shape[0]
    print(f'Ads with "{pattern}": {pattern_num}')
    
    # % of ads with pattern
    pattern_perc = pattern_num / df.shape[0]
    print(f'Ads with "{pattern}": {round(pattern_perc * 100, 2)}%')
    
    example = df[df[feature] == pattern].sample()
    
    return example


def process_price(df):
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
    df['price'] = df['price'].str.split('€').str[-1]\
                             .str.replace(',', '', regex=False)\
                             .str.replace(')', '', regex=False)\
                             .astype(float)
    
    after = df.shape[0]
    print(f'Rows after dropping: {after}\n' + '-' * 10)
    print(f'Difference: {after - before}')

    return df


def process_coordinates(df):
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

def drop_coord_outliers(df):
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
    print(f'Rows after dropping: {before}')
    
    df.drop(index=df[(df['latitude'] < 51.3) | (df['latitude'] > 55.4) | \
                     (df['longitude'] > -5.9) | (df['longitude'] < -10.6)].index, inplace=True)
    # Drop ads from Nothern Ireland
    df.drop(index=df[(df['latitude'] > 54.5) & (df['longitude'] > -7.9) & \
                     (df['latitude'] < 54.6)].index, inplace=True)
    
    after = df.shape[0]
    print(f'Rows after dropping: {after}\n' + '-' * 10)
    print(f'Difference: {after - before}')
    return df


def drop_floor_area(df):
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


def floor_area_wragling(df):
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

def process_floor_area(df):
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


def drop_info(df):
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
    print(f'Shape after dropping: {before}')
    
    #df.dropna(subset=['info'])
    df = df[df['info'].str.split(',').apply(len) == 4]
    
    after = df.shape
    print(f'Shape after dropping: {after}\n' + '-' * 10)
    print(f'Dropped: {before[0] - after[0]} rows\n')
    
    return df


def process_info(df):
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
    print(f'Shape after dropping: {before}')
    
    df['bedroom'] = df['info'].str.split(',').str[0]
    df['bathroom'] = df['info'].str.split(',').str[1]
    df['plus_info'] = df['info'].str.split(',').str[3]
    
    df.drop(columns=['info'], inplace=True)

    after = df.shape
    print(f'Shape after dropping: {after}\n' + '-' * 10)
    print(f'Difference: {after[1] - before[1]} columns')
    
    return df


def process_views(df):
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


def process_rooms(df):
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




































