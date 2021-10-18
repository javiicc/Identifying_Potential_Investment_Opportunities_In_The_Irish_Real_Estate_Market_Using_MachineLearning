import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re
from os import listdir
from os.path import isfile, join

from datetime import timedelta



def process_price(df):
    # Drop prices
    # df = df.loc[(df['price'] != 'Price on Application') & (df['price'] != 'AMV: Price on Application')]
    df = df[((df['price'] != 'Price on Application') & (
                df['price'] != 'AMV: Price on Application'))].copy()
    # Prices wrangling      £120,000 (€139,775)
    df['price'] = df['price'].str.split('€').str[-1].str.replace(',', '',
                                                                 regex=False).str.replace(
        ')', '', regex=False).astype(float)

    return df


def process_coordinates(df):
    print(df.shape)
    df['latitude'] = df['coordinates'].str.split('+').str[0].astype(float)
    df['longitude'] = df['coordinates'].str.split('+').str[1].astype(float)
    df.drop(columns=['coordinates'], inplace=True)
    print(df.shape)
    return df

def drop_coord_outliers(df):
    print(df.shape)
    df.drop(index=df[(df['latitude'] < 51.3) | (df['latitude'] > 55.4) | \
                     (df['longitude'] > -5.9) | (df['longitude'] < -10.6)].index, inplace=True)
    print(df.shape)
    return df


def drop_floor_area(df):
    index_to_drop = df[(df['floor_area'].str.contains('m²') == False) |
                       (df['floor_area'].isna())].index
    print(f'index_to_drop: {len(index_to_drop)}\n')  # esta diferenci de 8 se debe a missing values

    shape_before = df.shape[0]
    print(f'Before dropping: {df.shape}')
    df.drop(index=index_to_drop, inplace=True)
    print(f'After dropping: {df.shape}\n----------')
    print(f'Diference: {shape_before - df.shape[0]} rows')
    return df


def floor_area_wragling(df):
    df['floor_area'] = df['floor_area'].str.split(' ').str[0]
    return df

def process_floor_area(df):
    df_dropped = drop_floor_area(df)
    df_wrangled = floor_area_wragling(df_dropped)
    return df_wrangled


def drop_info(df):
    before_dropping = df.shape
    print(before_dropping)
    df.dropna(subset=['info'])
    df = df[df['info'].str.split(',').apply(len) == 4]
    print(df.shape, '\n---------')
    print(f'Dropped: {before_dropping[0] - df.shape[0]} rows\n')
    return df


def process_info(df):
    df = drop_info(df).copy()
    before_wrangling = df.shape
    print(df.shape)
    df['bedroom'] = df['info'].str.split(',').str[0]
    df['bathroom'] = df['info'].str.split(',').str[1]

    print(df.shape, '\n---------')
    print(f'Dropped: {df.shape[1] - before_wrangling[1]} columns\n')
    return df


def process_views(df):
    df['views'] = df['views'].str.replace(',', '').astype(float)
    return df


def process_rooms(df):
    df['bedroom'] = df['bedroom'].str.split(' ').str[0]
    df['bathroom'] = df['bathroom'].str.split(' ').str[0]
    return df




































