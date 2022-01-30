"""
This is a boilerplate pipeline 'feature_engineering_geospatial_data'
generated using Kedro 0.17.5
"""

import numpy as np
import pandas as pd
import re
from geopy.geocoders import Nominatim

from typing import Dict


###########################################################################
# FEATURE ENGINEERING FUNCTIONS
###########################################################################


def location_dict(df: pd.DataFrame,
                  latitude='latitude', longitude='longitude') -> Dict[str, list]:
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


def location_dataframe(df: pd.DataFrame, dictionary: Dict[str, list]) -> pd.DataFrame:
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


def location_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Call the `location_dict()` function to get the location dictionary and the
    `location_dataframe()` one to add the location dictionary info to the DataFrame.

    Parameters
    ----------
    df :
        The dataframe to work with.

    Returns
    -------
    The DataFrame with location info added.
    """
    # Call `location_dict` function to get a dictionary with location info
    location_dictionary = location_dict(df)
    # Call `location_dataframe` function to add the `location_dict` to a df
    df = location_dataframe(df, location_dictionary)

    return df


def homogenize(eircode: pd.Series) -> pd.Series:
    """Takes the postcode column and homogenizes it to get the routing key.

    Parameters
    ----------
    eircode :
        Column to homogenize.

    Returns
    -------
    Column processed.
    """
    if eircode is np.nan:
        pass
    # 8 should be the eircode length in all ads
    elif len(eircode) == 8:
        pass
    # 3 is the routing key
    elif len(eircode) == 3:
        pass

    elif len(eircode) == 7:
        if re.match(r'\w{3} \w{3}', eircode):
            # Only the three first digits are useful
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
            eircode = np.nan

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
    return df


def add_location(df, geonames_df):
    """Takes the first DataFrame and adds the geonames info to it.

    Parameters
    ----------
    df :
        DataFrame to add geonames info.
    geonames_df :
        DataFrame with the geonames info.

    Returns
    -------
    The df DataFrame with the geonames info.
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
    """Calls the `location_engineering()` function to get the DataFrame with
    location info and then cleans and wrangles the location data.

    Parameters
    ----------
    df :
        The DataFrame to work with.

    Returns
    -------
    Returns the DataFrame with location information cleaned.
    """
    # Feature engineering with Geopy
    df = location_engineering(df)

    # Drop ads from UK ->>>> cambiar
    df.drop(df[df.country == 'United Kingdom'].index, inplace=True)

    # Make a DataFrame with the dictionary obtained from `geonames_dict` function
    # Esto puedo hacerlo antes o ponerlo en data directamente
    # geonames_df = pd.DataFrame(geonames_dict())

    df_loc = eircode_homogenize(df)

    # df_loc = add_location(df=df_loc, geonames_df=geonames_df)  # df=df

    df_loc.drop(columns=['country_code', 'country', 'county', 'municipality',
                         'city', 'town', 'locality', 'suburb', 'road', 'house_number'],
                inplace=True)
    return df_loc


def add_geonames(df: pd.DataFrame, geonames_df) -> pd.DataFrame:
    """Simply calls the `add_location()` function and returns the DataFrame obtained.

    Parameters
    ----------
    df :
        DataFrame to add geonames info.
    geonames_df :
        DataFrame with the geonames info.

    Returns
    -------
    The df DataFrame with the geonames info.
    """
    df_loc = add_location(df=df, geonames_df=geonames_df)

    # Drop teh index column
    df_loc.drop(columns='Unnamed: 0', inplace=True)

    return df_loc
