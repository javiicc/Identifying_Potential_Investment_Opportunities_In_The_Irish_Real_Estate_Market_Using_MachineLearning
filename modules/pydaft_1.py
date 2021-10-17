import numpy as np
from geopy.geocoders import Nominatim



def fix_rooms(df):
    df.loc[(df['type']=='share') & (df['type_house']!='room'), 'type_house'] = 'room'
    return df


def replace_nan(df):
    df.replace('none', np.nan, inplace=True)
    return df


def drop_prices(df):
    items_to_drop = df[df['price'].str.split().apply(len) != 3].index
    print(df.shape)
    df.drop(index=items_to_drop, inplace=True)#.reset_index(drop=True, inplace=True)
    print(df.shape)
    return df


def process_prices(df):   # attribute
     
    new_values = df['price'][df['price'].str.contains('week')]\
                 .str.replace('€', '').str.replace(',', '')\
                 .str.split().str[0].astype(float) * (52/12)
    
    df['price'] = df['price'].str.replace('€', '').str.replace(',', '')\
                .str.split().str[0].astype(float)
    
    df.loc[new_values.index.tolist(), 'price'] = new_values
    
    df['price'] = round(df['price'], 2)
    
    return df


def bef_aft(df):
    
    before = df['price']
    after = process_prices(df['price'])
    
    df = pd.DataFrame({'before': df['price'], 
                       'after': process_prices(df['price'])}).head(10)
    return df


def process_coordinates(df):
    print(df.shape)
    df['latitude'] = df['coordinates'].str.split('+').str[0].astype(float)
    df['longitude'] = df['coordinates'].str.split('+').str[1].astype(float)
    print(df.shape)
    return df


def drop_outliers(df):
    print(df.shape)
    df.drop(index=df[(df['latitude'] < 51.3) | (df['latitude'] > 55.4) | \
                     (df['longitude'] > -5.9) | (df['longitude'] < -10.6)].index,                   inplace=True)
    print(df.shape)
    return df


def address_features(df):  ## this is so slow
    
    lat_series = df['latitude']
    lon_series = df['longitude']

    geolocator = Nominatim(user_agent="my_geocoder")
    address_dict = {'country_code': [],
                    'country': [],
                    'postcode': [],
                    'state_district': [],
                    'county': [],
                    'municipality': [],
                    'city': [],
                    'town': [],
                    'city_district': [],
                    'locality': [],
                    'road': [],
                    'house_number': []}

    for i, coordinates in enumerate(zip(lat_series, lon_series)):
        lat = coordinates[0]
        lon = coordinates[1]
        location = geolocator.reverse(f"{lat}, {lon}")
        for key in address_dict:
            try:
                address_dict[key].append(location.raw['address'][key])
            except:
                address_dict[key].append(np.nan)

        if i in range(0, len(lat_series), 100):
            print(i)

    return address_dict

def add_address_features(df, address_dict):
    print(df.shape)
    for key in address_dict:
        df[key] = address_dict[key]
    print(df.shape)
    return df


def process_info(df):
    print(df.shape)
    df['room_type'] = df['info'].str.split(',').str[0]
    df['bathroom_type'] = df['info'].str.split(',').str[1]
    print(df.shape)
    return df


def available_bedrooms(df):
    print(df.shape)
    df['available_bedrooms'] = df['overview'].str.split(',').str[0].\
                                str.split(': ').str[-1].astype(int)
    print(df.shape)
    return df


def facilities_list(df):
    facilities_list = []
    for elem in df['facilities'].str.split(','):
        try:
            for facility in elem:
                if facility in facilities_list:
                    continue
                else:
                    facilities_list.append(facility)
        except: continue # continue por el nan  
    return facilities_list

def facilities_dict(facilities_list):
    facilities_dict = {}
    for facility in facilities_list:
        facilities_dict[facility] = []
    return facilities_dict
    

def fill_facilities_dict(df, facilities_dict): #facility_string
    for elem in df['facilities']:
        if elem is np.nan:      
            for key in facilities_dict:
                facilities_dict[key].append(np.nan)
        else:   
            for key in facilities_dict:
                if key in elem:
                    facilities_dict[key].append(1)
                else:
                    facilities_dict[key].append(0)
    return facilities_dict

def add_facilities(df, facilities_dict):
    print(df.shape)
    for key in facilities_dict:
        df[key] = facilities_dict[key]
    print(df.shape)
    return df



    