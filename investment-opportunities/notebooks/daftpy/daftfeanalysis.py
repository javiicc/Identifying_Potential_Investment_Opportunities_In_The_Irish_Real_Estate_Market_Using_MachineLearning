import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re

from geopy.geocoders import Nominatim

import requests
import lxml.html as lh
import seaborn as sns

import scipy.stats as stats

from sklearn.feature_selection import (RFE, SequentialFeatureSelector)


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


def geonames_dict():
    """Scrape the website from the url. 

    Returns
    -------
    A dictionary with geonames info.
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


def homogenize(eircode: pd.Series) -> pd.Series:
    """Takes the postcode column and homogenizes it to get the routing key.
    
    Parameters
    ----------
    eircode :
        String (something similar to an eircode expected)
    
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
        elif re.match(r'\b\w{3}\b \b\w{2}\b \b\w{2}\b', eircode):   # D20 HK 69
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
         re.match(r'CO WICKLOW', eircode): # re.match(r'nan', eircode)
                        
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
    The DataFrame with the postcode column homogenized.
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


########################################################################
# Data Analysis
########################################################################


def frequencies(df, variable):
    """Take a DataFrame and a variable name and return the frequencies.
    
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
def iqr_method(data):
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

'''
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
'''


def outlier_bool(df, feature, level=1, continuous=False, log=False):
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


def drop_outliers_tmp(df, feature, level=1, continuous=False, log=False, inplace=False):
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


def drop_outliers(df: pd.DataFrame,
                  features=('price', 'floor_area', 'views',
                            'bedroom', 'bathroom')) -> pd.DataFrame:
    """Drop outliers based on the resulting boolean Series from the `outlier_bool()`
    function.

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

        print(feature.upper())
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
        print('-----------')

        # Increase the list with outliers index from each feature
        # outlier_list below has repeated index
        outliers_list += list(outliers.index)
        # print(len(outliers_list))

        # Convert list to set to eliminate repeated index
        outliers_set = set(outliers_list)
        # print(len(outliers_set))

    before = df.shape
    print('---------------')
    print('Shape before:', before)
    # Drop outliers!
    df_no_out = df.drop(index=outliers_set)
    after = df_no_out.shape
    print('Shape after:', after)
    print('Outliers dropped:', before[0] - after[0])

    return df_no_out


def print_limits(df, variable, level=1):
    """Print the lower and upper limits.

    Parameters
    ----------
    df :
        The DataFrame to search in.
    variable :
        The current variable.
    level :
        Cut point since it is considered outlier.
    """
    pct_range = pct_method(df[variable], level=level)
    iqr_range = iqr_method(df[variable])
    
    print(f'Percentile based method: {pct_range}')
    print(f'Interquartile range method: {iqr_range}')


def check_transformations(feature, df, df_no_out):
    """Plot several graphs showing the effects of dropping outliers and logarithmic and
    box-cox transformation.

    Parameters
    ----------
    feature :
        Variable to work with.
    df :
        The full DataFrame.
    df_no_out :
        DataFrame without outliers.
    """
    fig, ax = plt.subplots(6, 2, figsize=(8, 14))

    # ax[0,0] = outplots(feature='price', df=df, df_no_out=df_no_out)
    df[feature].min()
    df[feature].max()
    
    df_no_out[feature].min()
    df_no_out[feature].max()
    
    feature_name = feature.replace('_', ' ').capitalize()
    
    # No restricted (with outliers) -> 1
    sns.histplot(data=df[feature], bins=30, color='#b00b1e', ax=ax[0, 0]) 
    # Highlighting the peak of the crisis
    ax[0, 0].axvspan(df_no_out[feature].max(), df[feature].max(),
                     alpha=0.3, color="crimson")
    ax[0, 0].axvspan(df[feature].min(), df_no_out[feature].min(),
                     alpha=0.3, color="crimson")
    ax[0, 0].set_ylabel('Unrestricted')
    ax[0, 0].set_xlabel(feature_name)
    stats.probplot(df[feature], plot=ax[0, 1])

    # Restricted (without outliers) -> 2
    sns.histplot(data=df_no_out[feature], bins=30, color='#b00b1e', ax=ax[1, 0]) 
    ax[1, 0].set_ylabel('Restricted (NO outliers)')
    ax[1, 0].set_xlabel(feature_name)
    stats.probplot(df_no_out[feature], plot=ax[1, 1])

    # No restricted and logarithmic transformation -> 3
    sns.histplot(data=np.log(df[feature]), bins=30, color='#B93D14', ax=ax[2, 0]) 
    ax[2, 0].set_ylabel('Unrestricted')
    ax[2, 0].set_xlabel(f'{feature_name} logarithmic transformation')
    stats.probplot(np.log(df[feature]), plot=ax[2, 1])
    ax[2, 0].axvspan(np.log(df_no_out[feature].max()),
                    np.log(df[feature].max()),
                    alpha=0.3, color="crimson")
    ax[2, 0].axvspan(np.log(df[feature].min()),
                    np.log(df_no_out[feature].min()),
                    alpha=0.3, color="crimson")
    
    # Restricted and logarithmic transformation -> 4
    sns.histplot(data=np.log(df_no_out[feature]), bins=30, color='#B93D14', ax=ax[3, 0]) 
    ax[3, 0].set_ylabel('Restricted (NO outliers)')
    ax[3, 0].set_xlabel(f'{feature_name} logarithmic transformation')
    stats.probplot(np.log(df_no_out[feature]), plot=ax[3, 1])
    
    # No restricted and box-cox transformation -> 5
    data1, lmbda1 = stats.boxcox(df[feature])
    sns.histplot(data=data1, bins=30, color='#159819', ax=ax[4, 0]) 
    ax[4, 0].set_ylabel('Unrestricted')
    ax[4, 0].set_xlabel(f'{feature_name} Box-Cox transformation')
    stats.probplot(data1, plot=ax[4,1])
    
    # Restricted and box-cox transformation -> 6
    data2, lmbda2 = stats.boxcox(df_no_out[feature])
    sns.histplot(data=data2, bins=30, color='#159819', ax=ax[5, 0]) 
    ax[5, 0].set_ylabel('Restricted (NO outliers)')
    ax[5, 0].set_xlabel(f'{feature_name} Box-Cox transformation')
    stats.probplot(data2, plot=ax[5, 1])
    # print(lmbda1)
    # print(lmbda2)
    
    # To color the outliers with the Box-Cox transformation is necessary apply the
    # transformation to the limits and color the higher than the maximum limit and the
    # lower than the minimum one
    ax[4, 0].axvspan(((((df_no_out[feature].max() + 1) ** lmbda1) - 1) / lmbda1),
                     data1.max(),
                     alpha=0.3, color="crimson")
    ax[4, 0].axvspan(data1.min(),
                     ((((df_no_out[feature].min() - 1) ** lmbda1) - 1) / lmbda1),
                     alpha=0.3, color="crimson")
    # print(f'Min data1: {data1.min()}')
    # print(f'Min data2: {data2.min()}')
    
    # print(f'Max data1: {data1.max()}')
    # print(f'Max data2: {data2.max()}')
    fig.tight_layout()
    
    
def tchebycheff(df, num_features, k=2, transformation=None):
    """Calculate skewness, kurtosis, and Tchebycheff bounds.

    Parameters
    ----------
    df :
        The DataFrame to work with.
    num_features :
        Numeric features.
    k :
        Number of standard deviations.
    transformation :
        Type of transformation.

    Returns
    -------
    Return a DataFrame with the statistics.
    """
    if transformation is None:
        skew = df[num_features].skew()
        kurtosis = df[num_features].kurtosis()
    elif transformation == 'log':
        skew = np.log(df[num_features].dropna()).skew()
        kurtosis = df[num_features].kurtosis()
    elif transformation == 'coxbox':
        skew = stats.boxcox(df[num_features].dropna()).skew()
        kurtosis = df[num_features].kurtosis()
    
    # Tchebycheff bounds
    lim_inf = df[num_features].mean() - (k * df[num_features].std())
    lim_sup = df[num_features].mean() + (k * df[num_features].std())

    print(f'k = {k} -> {(1 - (1 / (k**2))) * 100}%')
    
    return pd.DataFrame({'skewness': skew,
                         'kurtosis': kurtosis,
                         'lim_inf': lim_inf,
                         'lim_sup': lim_sup})


def outplots(df, df_no_out, feature):

    df[feature].min()
    df[feature].max()
    
    df_no_out[feature].min()
    df_no_out[feature].max()
    
    fig, ax = plt.subplots()
    sns.histplot(data=df[feature], bins=30, color='blue')
    ax.axvspan(df_no_out[feature].max(), df[feature].max(), color="crimson", alpha=0.3)
    

def wrapper_methods(estimators_dict, method, 
                    X_train, y_train, X_test, y_test):
    """Plot two graphs showing the evolution of the score while changing the number
    of features.

    Parameters
    ----------
    estimators_dict :
        Dictionary with estimators to try.
    method :
        The feature selection method.
    X_train
    y_train
    X_test
    y_test
    """
    # scores_dict = {}
    for key in estimators_dict:
        print(key, '\n' + '-' * 10)
        estimator = estimators_dict[key]

        scores = [] 
        for i in range(1, X_train.shape[1]):  # + 1

            # Choose selector
            if method is 'rfe':
                selector = RFE(estimator=estimator, 
                               n_features_to_select=i,
                               step=1) 
            elif method is 'sfs_forward':
                selector = SequentialFeatureSelector(estimator=estimator, 
                                                     n_features_to_select=i, 
                                                     direction='forward')
            elif method is 'sfs_backward':
                selector = SequentialFeatureSelector(estimator=estimator, 
                                                     n_features_to_select=i, 
                                                     direction='backward') 
                
            # Select variables
            selector.fit(X_train, y_train)    

            # print(sfs.support_)
            # print(sfs.ranking_)
            print(X_train.columns[selector.support_].values, '\n')

            estimator.fit(X_train.loc[:, selector.support_], y_train)
            scores.append(estimator.score(X_test.loc[:, selector.support_], y_test))

        print(scores, '\n')
        # scores_dict[key] = scores
        
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.scatter(range(1, X_train.shape[1]), scores)
        ax.set_title(f'{key} Scores', weight='bold', size=15)
        ax.set_xlabel('Number of Features')
        ax.set_ylabel('R²')
        for location in ['left', 'bottom', 'right', 'top']:
            ax.spines[location].set_visible(False)
        ax.tick_params(bottom=False, left=False)
        plt.tight_layout()
