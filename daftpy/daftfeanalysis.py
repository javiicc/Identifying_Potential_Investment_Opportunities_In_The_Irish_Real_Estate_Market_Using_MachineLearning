import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re
from os import listdir
from os.path import isfile, join

from datetime import timedelta

from scrapy.item import Field, Item
from scrapy.loader.processors import MapCompose  # , TakeFirst
# import re
from w3lib.html import remove_tags, strip_html5_whitespace
from itemloaders.processors import TakeFirst, MapCompose, Join
from scrapy.loader import ItemLoader
from scrapy.http import TextResponse


from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time

import requests
import lxml.html as lh


########################################################################
# Feature Engineering
########################################################################

def missing_values(df):
    # Check missing values in absolute and relative terms
    missing_values = pd.DataFrame({
        'Absolute': df.isna().sum(), 
        'Relative': df.isna().sum() / df.shape[0]
    })
    return missing_values


def location_dict(df, latitude='latitude', longitude='longitude'): ## this is so slow   , attempt=1, max_attempts=5
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
        #time.sleep(1) # sleeps for 1 second

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
    
    url = 'http://www.geonames.org/postalcode-search.html?q=&country=IE'
    page = requests.get(url)
    doc = lh.fromstring(page.content) 
    tr_elements = doc.xpath('//tr')
    
    #Create empty dict
    col = {}
    #For each row, store each first element (header) and an empty list
    for i, t in enumerate(tr_elements[2]):
        key = t.text_content().lower()
        #print('%d: "%s"'%(i,name))
        col[key] = [] 
    col['place_coordinates'] = []

    # Fill dict
    #print(tr_elements[-1].text_content())
    for tr in tr_elements[3:]:
        
        if len(tr) == 7:
   
            for key, td in zip(col, tr):
                td = td.text_content()
                #print(td)
                col[key].append(td)
        elif len(tr) == 2:
        
            td = tr[-1].text_content()
            #print(td)
            col['place_coordinates'].append(td)
            
    del col['']
    del col['country']
    del col['admin2']
    del col['admin3']

    return col