import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re
from os import listdir
from os.path import isfile, join

from datetime import timedelta

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time

import requests
import lxml.html as lh
import seaborn as sns

import pylab
from scipy.stats import kstest
import scipy.stats as stats
from scipy.stats import normaltest

import ppscore as pps
import random
from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import (RFE,
                                       SequentialFeatureSelector)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import (StandardScaler, OneHotEncoder,
                                   PolynomialFeatures, PowerTransformer)


###########################################################################
# FEATURE ENGINEERING FUNCTIONS
###########################################################################

def location_dict(df: pd.DataFrame,
                  latitude='latitude', longitude='longitude'):
    """Take a dataframe and two columns names for latitude and longitude
    and do reverse geocoding with Nominatin geolocator.

    Parameters
    ----------
    df :
        The dataframe to work with.
    latitude :
        Latitude column name.
    longitude :
        Longitude column name.

    Returns
    -------
    A dictionary with location info.
    """
    lat_series = df[latitude]
    lon_series = df[longitude]

    # Nominatin geolocation service
    geolocator = Nominatim(user_agent="ireland-geocoder")
    # Dictionary with information we want to get
    location_dict = {'country_code': [],
                     'country': [],
                     'postcode': [],
                     'state_district': [],
                     'county': [],
                     'municipality': [],
                     'city': [],
                     'town': [],
                     'city_district': [],
                     'locality': [],
                     'suburb': [],
                     'road': [],
                     'house_number': []}

    # Loop over coordinates
    for i, coordinates in enumerate(zip(lat_series, lon_series)):
        lat = coordinates[0]
        lon = coordinates[1]

        # Return an address by location point
        location = geolocator.reverse(f"{lat}, {lon}")
        # Loop over `location_dict` and try to append the `location` info
        for key in location_dict:
            try:
                location_dict[key].append(location.raw['address'][key])
            except:
                location_dict[key].append(np.nan)
        # time.sleep(1) # sleeps for 1 second

        # To check it is working
        if i in range(0, df.shape[0], 200):
            print(i)

    return location_dict


def location_dataframe(df, dictionary):
    """Take a dataframe and a dictionary with location information
    and add it to the DataFrame.

    Parameters
    ----------
    df :
        The dataframe to work with.
    dictionary :
        dictionary with location info and values with the same length
        than the DataFrame.

    Returns
    -------
    The DataFrame with location info added.
    """
    before = df.shape
    print(f'Shape before adding: {before}')

    for key in dictionary:
        df[key] = dictionary[key]

    after = df.shape
    print(f'Shape after adding: {after}\n' + '-' * 10)
    print(f'Difference: {after[1] - before[1]} columns')
    return df


def location_engineering(df, latitude='latitude', longitude='longitude'):
    """Take a dataframe and a dictionary with location information
    and add it to the DataFrame.

    Parameters
    ----------
    df :
        The dataframe to work with.
    dictionary :
        dictionary with location info and values with the same length
        than the DataFrame.

    Returns
    -------
    The DataFrame with location info added.
    """
    # Call `location_dict` function to get a dictionary with location info
    location_dictionary = location_dict(df)
    # Call `location_dataframe` function to add the `location_dict` to a df
    df = location_dataframe(df, location_dictionary)

    return df


def geonames_dict():
    """Scrape the website from the url.

    Parameters
    ----------
    df :
        The dataframe to work with.
    dictionary :
        dictionary with location info and values with the same length
        than the DataFrame.

    Returns
    -------
    The DataFrame with location info added.
    """
    url = 'http://www.geonames.org/postalcode-search.html?q=&country=IE'
    page = requests.get(url)
    doc = lh.fromstring(page.content)
    tr_elements = doc.xpath('//tr')

    # Create empty dict
    col = {}
    # For each row, store each first element (header) and an empty list
    for i, t in enumerate(tr_elements[2]):
        key = t.text_content().lower()
        # print('%d: "%s"'%(i,name))
        col[key] = []
    col['place_coordinates'] = []

    # Fill dict
    # print(tr_elements[-1].text_content())
    for tr in tr_elements[3:]:

        if len(tr) == 7:

            for key, td in zip(col, tr):
                td = td.text_content()
                # print(td)
                col[key].append(td)

        elif len(tr) == 2:

            td = tr[-1].text_content()
            # print(td)
            col['place_coordinates'].append(td)

    del col['']
    del col['country']
    del col['admin2']
    del col['admin3']

    return col


def homogenize(eircode):  # 8, 3, 7, , dublin, 6, 9, 10
    """Scrape the website from the url.

    Parameters
    ----------

    Returns
    -------

    """
    if eircode is np.nan:
        pass
    elif len(eircode) == 8:
        pass
    elif len(eircode) == 3:
        pass

    elif len(eircode) == 7:
        if re.match(r'\w{3} \w{3}', eircode):
            eircode = eircode[:3]
        else:
            routing_key = re.search(r'(\b\w{3})', eircode)[0]
            unique_identifier = re.search(r'(\w{4}\b)', eircode)[0]
            eircode = f'{routing_key} {unique_identifier}'

    elif eircode == 'DUBLIN':
        pass

    elif re.match(r'DUBLIN', eircode):
        num = eircode[-2:]
        try:
            if int(num) < 10:
                eircode = f'D0{int(num)}'
            elif int(num) < 25:
                eircode = f'D{num}'
            else:
                eircode = np.nan
        except:  # 6w
            eircode = f'D{num}'

    elif len(eircode) == 6:
        eircode = eircode[:3]

    elif (len(eircode) == 9) or (len(eircode) == 10):
        if eircode == 'CO. CLARE':
            eircode = np.nan
        elif eircode == 'CO WICKLOW':
            eircode = np.nan
        elif eircode == 'CO.ATHLONE':
            eircode = np.nan
        elif re.match(r'\b\w{3}\b \b\w{2}\b \b\w{2}\b', eircode):  # D20 HK 69
            eircode = eircode[:3]
        else:
            # print(f'"{eircode}"', 'not processed -> np.nan')
            # for i in eircode:
            #   print(i)
            # print(len(eircode))
            eircode = np.nan
            # print(eircode)

    elif len(eircode) == 1:
        eircode = 'D0' + eircode

    elif len(eircode) == 2:
        if eircode == 'D5':
            eircode = 'D05'
        else:
            eircode = 'D' + eircode



    elif re.match(r'CO WESTMEATH', eircode) or \
            re.match(r'CO. WICKLOW', eircode) or \
            re.match(r'CO. ROSCOMMON', eircode) or \
            re.match(r'CO. KILKENNY', eircode) or \
            re.match(r'0000', eircode) or \
            re.match(r'CO WICKLOW', eircode):  # re.match(r'nan', eircode)

        # print(f'"{eircode}"', 'not processed -> np.nan')
        eircode = np.nan

    return eircode


def eircode_homogenize(df):
    """Apply the `homogenize` function to the `postcode` column.

    Parameters
    ----------
    df :
        The dataframe to work with.

    Returns
    -------
    The DataFrame with location info added.
    """
    df['postcode'] = df['postcode'].str.strip().apply(homogenize)
    # df['postcode'] = df['postcode'].apply(homogenize)
    return df


def add_location(df, geonames_df):
    """Take two DataFrames and .

    Parameters
    ----------
    df :
        The dataframe to work with.

    Returns
    -------
    The DataFrame with location info added.
    """
    before = df.shape
    print(f'Shape before dropping: {before}')

    for row in df.iterrows():

        row_index = row[0]

        if row[1]['postcode'] is not np.nan:
            routing_key = row[1]['postcode'][:3]
        elif row[1]['postcode'] is np.nan:
            continue

        geonames_row = geonames_df[geonames_df['code'].str.contains(routing_key)]
        if len(geonames_row) != 0:
            for column in geonames_df:
                df.loc[row_index, column] = geonames_row[column].values[0]
        else:
            continue

    after = df.shape
    print(f'Shape after dropping: {after}\n' + '-' * 10)
    print(f'Difference: {after[1] - before[1]} columns')

    return df
###########################################################################
# FEATURE ENGINEERING NODES
###########################################################################
def location_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:

    # Feature engineering with Geopy
    df = location_engineering(df)

    # Drop ads from UK ->>>> cambiar
    df.drop(df[df.country == 'United Kingdom'].index, inplace=True)

    # Make a DataFrame with the dictionary obtained from `geonames_dict` function
    # Esto puedo hacerlo antes o ponerlo en data directamente
    #geonames_df = pd.DataFrame(geonames_dict())

    df_loc = eircode_homogenize(df)

    #df_loc = add_location(df=df_loc, geonames_df=geonames_df)  # df=df

    df_loc.drop(columns=['country_code', 'country', 'county', 'municipality',
                       'city', 'town', 'locality', 'suburb', 'road', 'house_number',
                       'admin1', 'place_coordinates'], inplace=True)
    return df_loc

def add_geonames(df: pd.DataFrame, geonames_df) -> pd.DataFrame:

    df_loc = add_location(df=df, geonames_df=geonames_df)

    return df