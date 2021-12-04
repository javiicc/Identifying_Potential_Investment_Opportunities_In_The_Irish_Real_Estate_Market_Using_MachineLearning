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
import seaborn as sns

import pylab
from scipy.stats import kstest
import scipy.stats as stats
from scipy.stats import normaltest

########################################################################
# Feature Engineering
########################################################################

def missing_values(df):
    """Take a dataframe and plot missing values.
    
    Parameters
    ----------
    df : 
        The dataframe to work with.
    
    Returns
    -------
    A DataFrame showing missinga values.
    """
    # Check missing values in absolute and relative terms
    missing_values = pd.DataFrame({
        'Absolute': df.isna().sum(), 
        'Relative': df.isna().sum() / df.shape[0]
    })
    return missing_values


def location_dict(df, latitude='latitude', longitude='longitude'): 
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


def homogenize(eircode):   # 8, 3, 7, , dublin, 6, 9, 10
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
        elif re.match(r'\b\w{3}\b \b\w{2}\b \b\w{2}\b', eircode):   #D20 HK 69
            eircode = eircode[:3]
        else:
            #print(f'"{eircode}"', 'not processed -> np.nan')
            #for i in eircode:
             #   print(i)
            #print(len(eircode))
            eircode = np.nan
            #print(eircode)
                
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
         re.match(r'CO WICKLOW', eircode): #re.match(r'nan', eircode)
                        
        #print(f'"{eircode}"', 'not processed -> np.nan')
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
    #df['postcode'] = df['postcode'].apply(homogenize)
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


########################################################################
# Data Analysis
########################################################################

def frequencies(df, variable):
    """Take a DataFrame and a variable name and return the frquencies. 
    
    Parameters
    ----------
    df : 
        The dataframe to work with.
    variable : 
        Variable name we want know the frequency.
    
    Returns
    -------
    The DataFrame with frequencies.
    """
    freq = pd.DataFrame(data=df[variable].value_counts())
    freq.rename(columns={'cities':'freq_abs'}, inplace=True)
    # Calculate relative frequencies
    freq['freq_rel'] = freq.freq_abs / df.shape[0]
    return freq

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
        #high_limit = np.max([pct_range[1],
         #                    iqr_range[1],
          #                   std_range[1]])
        
    elif continuous:
        if feature is 'floor_area':
            # Percentile based method is the onlu oney that return a 
            # positive value
            low_limit = pct_range[0]
        else:
            #print('no')
            low_limit = np.min([pct_range[0],
                                iqr_range[0],
                                #std_range[0]
                               ])
    high_limit = np.max([pct_range[1],
                         iqr_range[1],
                         #std_range[1]
                        ])
        
    print(f'Limits: {[low_limit, high_limit]}')
    # Restrict the data with the minimum and maximum
    outlier = data.between(low_limit, high_limit)
    print(f'No outliers: {outlier.sum()}')
    print(f'Outliers: {(outlier==False).sum()}\n')
    
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
    outliers = df[outlier_boolean==False]
    # Filter data to drop outliers
    df = df[outlier_boolean]
    
    rows_after = df.shape[0]
    
    print(f'Range after: {[df[feature].min(), df[feature].max()]}')
    print(f'Outliers dropped: {rows_before - rows_after}')
    
    return df, outliers

def print_limits(df, variable, level=1):
    """Classify outliers based on. 
    
    Parameters
    ----------
    data : 
        .
    
    Returns
    -------
    .
    """
    pct_range = pct_method(df[variable], level=level)
    iqr_range = iqr_method(df[variable])
    std_range = std_method(df[variable])
    
    print(f'Percentile based method: {pct_range}')
    print(f'Interquartile range method: {iqr_range}')
    #print(f'Standard deviation method: {std_range}')


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
                if ix in index_list[i+1]:
                    data_ix.append(ix)
            print(f'1st and 2nd index lists: {len(data_ix)} rows')
    
        elif i < 4:
            for ix in data_ix:
                # Check whether index from data_ix are in the next list, 
                # if not -> remove it
                if ix not in index_list[i+1]:
                    data_ix.remove(ix)
            print(f'{i + 2}{"rd" if i + 2 == 3 else "th"} index list: {len(data_ix)} rows')
    print('-' * 10)
    return data_ix

def drop_outliers_ix(df, index_list):
    """Classify outliers based on. 
    
    Parameters
    ----------
    data : 
        .
    
    Returns
    -------
    .
    """
    outliers_rep = []
    for list_ in index_list:
        outliers_rep += list(list_)
        #print(len(outliers_rep))
    
    outliers = pd.Series(outliers_rep).unique()
    print('Outliers dropped:', len(outliers))
    
    df_without_outliers = df.drop(index=outliers).copy()
    
    return df_without_outliers


def drop_all_outliers(df, index_list):
    """Classify outliers based on. 
    
    Parameters
    ----------
    data : 
        .
    
    Returns
    -------
    .
    """
    # Get ads index which are not outliers
    data_ix = outliers_ix(index_list)
    
    before = df.shape
    print(f'Shape before dropping: {before}')
    
    # Filter data to drop outliers
    sale_out = df.iloc[data_ix]
    
    after = sale_out.shape
    print(f'Shape after dropping: {after}')
    print(f'{before[0] - after[0]} rows/outliers dropped')
    
    return sale_out


def check_transformations(feature, df, df_no_out):
    """Classify outliers based on. 
    
    Parameters
    ----------
    data : 
        .
    
    Returns
    -------
    .
    """
    fig, ax = plt.subplots(6, 2, figsize=(12, 26))

    #ax[0,0] = outplots(feature='price', df=df, df_no_out=df_no_out)
    df[feature].min()
    df[feature].max()
    
    df_no_out[feature].min()
    df_no_out[feature].max()
    
    feature_name = feature.replace('_', ' ').capitalize()
    
    # No restricted (with outliers)
    sns.histplot(data=df[feature], bins=30, color='blue', ax=ax[0, 0]) 
    ### Highlihting the peak of the crisis
    ax[0,0].axvspan(df_no_out[feature].max(), df[feature].max(),
           alpha=0.3, color="crimson")
    ax[0,0].axvspan(df[feature].min(), df_no_out[feature].min(),
           alpha=0.3, color="crimson")
    ax[0,0].set_ylabel('Unrestricted')
    ax[0,0].set_xlabel(feature_name)
    
    stats.probplot(df[feature], plot=ax[0,1])
    

    # Restrictec (without outliers)
    sns.histplot(data=df_no_out[feature], bins=30, color='blue', ax=ax[1, 0]) 
    ax[1,0].set_ylabel('Unrestricted NO outliers')
    ax[1,0].set_xlabel(feature_name)
    stats.probplot(df_no_out[feature], plot=ax[1,1])
    
    # No restricted and logarithmic transformation
    sns.histplot(data=np.log(df[feature]), bins=30, color='red', ax=ax[2, 0]) 
    ax[2,0].set_ylabel('Unrestricted')
    ax[2,0].set_xlabel(f'{feature_name} logarithmic transformation')
    stats.probplot(np.log(df[feature]), plot=ax[2,1])
    ax[2,0].axvspan(np.log(df_no_out[feature].max()), np.log(df[feature].max()),
           alpha=0.3, color="crimson")
    ax[2,0].axvspan(np.log(df[feature].min()), np.log(df_no_out[feature].min()),
           alpha=0.3, color="crimson")
    
    # Restrictec and logarithmic transformation
    sns.histplot(data=np.log(df_no_out[feature]), bins=30, color='red', ax=ax[3, 0]) 
    ax[3,0].set_ylabel('Unrestricted NO outliers')
    ax[3,0].set_xlabel(f'{feature_name} logarithmic transformation')
    stats.probplot(np.log(df_no_out[feature]), plot=ax[3,1])
    # No restricted and boxcox transformation
    sns.histplot(data=stats.boxcox(df[feature])[0], bins=30, color='green', ax=ax[4, 0]) 
    ax[4,0].set_ylabel('Unrestricted')
    ax[4,0].set_xlabel(f'{feature_name} coxbox transformation')
    stats.probplot(stats.boxcox(df[feature])[0], plot=ax[4,1])
    #ax[4,0].axvspan(stats.boxcox(df_no_out[feature].max())[0], 
     #               stats.boxcox(df[feature].max())[0],
      #     alpha=0.3, color="crimson")
    
    # Restrictec and boxcox transformation
    sns.histplot(data=stats.boxcox(df_no_out[feature])[0], bins=30, color='green', ax=ax[5, 0]) 
    ax[5,0].set_ylabel('Unrestricted NO outliers')
    ax[5,0].set_xlabel(f'{feature_name} coxbox transformation')
    stats.probplot(stats.boxcox(df_no_out[feature])[0], plot=ax[5,1])
    
    
    
    
    
    fig.tight_layout()
    
    
def tchebycheff(df, num_features, k=2):
    """Classify outliers based on. 
    
    Parameters
    ----------
    data : 
        .
    
    Returns
    -------
    .
    """
    skew = df[num_features].skew()
    
    # Tchebycheff bounds
    lim_inf = df[num_features].mean() - (k * df[num_features].std())
    lim_sup = df[num_features].mean() + (k * df[num_features].std())

    print(f'k = {k} -> {(1 - (1 / (k**2))) * 100}%')
    
    return pd.DataFrame({'skewness': skew, 
                  'lim_inf': lim_inf,
                  'lim_sup': lim_sup
                 })


def outplots(df, df_no_out, feature):

    df[feature].min()
    df[feature].max()
    
    df_no_out[feature].min()
    df_no_out[feature].max()
    
    fig, ax = plt.subplots()
    sns.histplot(data=df[feature], bins=30, color='blue')
    ax.axvspan(df_no_out[feature].max(), df[feature].max(), color="crimson", alpha=0.3)
    







