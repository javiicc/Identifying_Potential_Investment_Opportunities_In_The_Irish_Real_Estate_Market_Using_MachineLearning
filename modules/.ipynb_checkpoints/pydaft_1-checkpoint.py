import numpy as np
from geopy.geocoders import Nominatim

def address_features(lat_series, lon_series):  ## this is so slow

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