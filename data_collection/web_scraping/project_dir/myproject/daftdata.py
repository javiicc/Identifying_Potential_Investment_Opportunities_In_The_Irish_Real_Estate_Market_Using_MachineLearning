import sqlite3

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


########################################################################
# DATA COLLECTION
########################################################################

# Declaring Item subclass
class DaftItem(Item):
    """Class to contain common Buy and Rent item fields.
    """
    # Field objects are used to specify metadata for each field
    daft_id = Field()
    item_id = Field()
    url = Field()
    name = Field()
    price = Field()
    info = Field()
    sale_type = Field()
    floor_area = Field()
    contact = Field()  # [1]
    phone = Field()
    psr_license_number = Field()
    ber = Field()
    entered_renewed = Field()
    views = Field()
    type_house = Field()
    energy = Field()
    coordinates = Field()
    type = Field()
    scraping_date = Field()
    description = Field()
'''
    def __init__(self, daft_id=None, item_id=None, url=None, name=None,
                 price=None, info=None, sale_type=None, floor_area=None,
                 contact=None,
                 phone=None, psr_license_number=None, ber=None,
                 entered_renewed=None, views=None,
                 type_house=None, energy=None, coordinates=None,
                 type=None, scraping_date=None):

        self.daft_id = daft_id
        self.item_id = item_id
        self.url = url
        self.name = name
        self.price = price
        self.info = info
        self.sale_type = sale_type
        self.floor_area = floor_area
        self.contact = contact  # [1]
        self.phone = phone
        self.psr_license_number = psr_license_number
        self.ber = ber
        self.entered_renewed = entered_renewed
        self.views = views
        self.type_house = type_house
        self.energy = energy
        self.coordinates = coordinates
        self.type = type
        self.scraping_date = scraping_date
'''


class DaftLoader(ItemLoader):
    """Class to contain input and output processors.
    """
    # Default input processor to process all the inputs
    default_input_processor = MapCompose(strip_html5_whitespace) 

    # Input processors for some fields
    phone_in = MapCompose(lambda x: x.replace('\n', '').strip())
    views_in = MapCompose(remove_tags)
    type_house_in = MapCompose(remove_tags)
    energy_in = MapCompose(remove_tags)


def get_name(xpath: str, response: TextResponse, loader):
    """Obtain the ad's name from the response object throug a Selector and
    populate the `name` field. Populates the `name` field with `none` if 
    something goes wrong.
    
    Parameters
    ----------
    xpath : 
        Path for the Selector.
    response :
        Page contents. Response object.
    loader : 
        Item loader. Provide the mechanism for populating the item field.
    """
    try:
        name_info = response.xpath(xpath).get()
        if name_info:
            loader.add_value('name', name_info)
        else:
            loader.add_value('name', 'none')
    except:
        loader.add_value('name', 'none')

        
def get_price(xpath: str, response: TextResponse, loader):
    """Obtain the ad's price from the response object throug a Selector and
    populate the `price` field. Populates the `name` field with `none` if 
    something goes wrong.
    
    Parameters
    ----------
    xpath : 
        Path for the Selector.
    response :
        Page contents. Response object.
    loader : 
        Item loader. Provide the mechanism for populating the item field.
    """
    try:
        price_info = response.xpath(xpath).get()
        if price_info:
            loader.add_value('price', price_info)
        else:
            loader.add_value('price', 'none')
    except:
        loader.add_value('price', 'none')


def get_info(xpath: str, response: TextResponse, loader):
    """Obtain the ad's info from the response object throug a Selector and
    populate the `info` field. Populates the `info` field with `none` if 
    something goes wrong.
    
    Parameters
    ----------
    xpath : 
        Path for the Selector.
    response :
        Page contents. Response object.
    loader : 
        Item loader. Provide the mechanism for populating the item field.
    """
    try:
        info_list = response.xpath(xpath).getall()
        if info_list:
            info_string = ','.join(info_list)
            loader.add_value('info', info_string)
        else:
            loader.add_value('info', 'none')
    except:
        loader.add_value('info', 'none')


def get_ber(xpath: str, response: TextResponse, loader):
    """Obtain the ad's ber from the response object throug a Selector and
    populate the `ber` field. Populates the `ber` field with `none` if 
    something goes wrong.
    
    Parameters
    ----------
    xpath : 
        Path for the Selector.
    response :
        Page contents. Response object.
    loader : 
        Item loader. Provide the mechanism for populating the item field.
    """
    try:
        ber_info = response.xpath(xpath).get()
        if ber_info:
            loader.add_value('ber', ber_info)
        else:
            loader.add_value('ber', 'none')
    except:
        loader.add_value('ber', 'none')


def get_entered_renewed_views(xpath: str, response: TextResponse, loader):
    """Obtain the ad's entered_renewed and views info from the response object
    throug a Selector and populate the `entered_renewed` and `views` field
    Populates the `entered_renewed` and `views` field with `none` if 
    something goes wrong.
    
    Parameters
    ----------
    xpath : 
        Path for the Selector.
    response :
        Page contents. Response object.
    loader : 
        Item loader. Provide the mechanism for populating the item field.
    """
    try:
        entered_views_info = response.xpath(xpath).getall()
        if entered_views_info[0]:
            loader.add_value('entered_renewed', entered_views_info[0])
        else:
            loader.add_value('entered_renewed', 'none')
        if entered_views_info[1]:
            loader.add_value('views', entered_views_info[1])
        else:
            loader.add_value('views', 'none')
    except:
        loader.add_value('entered_renewed', 'none')
        loader.add_value('views', 'none')


def get_contact_phone(xpath_contact: str, xpath_phone: str, pattern_phone: str,
                      response: TextResponse, loader):
    """Obtain the ad's contact info from the response object throug a Selector
    and populate the `contact` and `phone` fields. Populates the `contact` and
    `phone` fields with `none` if something goes wrong.
    
    Parameters
    ----------
    xpath_contact : 
        Path for the Selector.
    xpath_phone :
        Path for the Selector.
    response :
        Page contents. Response object.
    loader : 
        Item loader. Provide the mechanism for populating the item field.
    """
    # Contact info
    try:
        contact_info = response.xpath(xpath_contact).get()
        if contact_info:
            loader.add_value('contact', contact_info)
        else:
            loader.add_value('contact', 'none')
    except:
        loader.add_value('contact', 'none')
    # Phone info
    try:
        phone_info = response.xpath(xpath_phone).re_first(pattern_phone)
        if phone_info:
            loader.add_value('phone', phone_info)
        else:
            loader.add_value('phone', 'none')
    except:
        loader.add_value('phone', 'none')


def get_psr(xpath: str, pattern: str, response: TextResponse, loader):
    """Obtain the ad's psr info from the response object throug a Selector
    and populate the `psr` field. Populates the `psr` field with `none` if
    something goes wrong.
    
    Parameters
    ----------
    xpath : 
        Path for the Selector.
    pattern :
        Regular expression.
    response :
        Page contents. Response object.
    loader : 
        Item loader. Provide the mechanism for populating the item field.
    """
    try:
        license_info = response.xpath(xpath).re_first(pattern)
        if license_info:
            loader.add_value('psr_license_number', license_info)
        else:
            loader.add_value('psr_license_number', 'none')
    except:
        loader.add_value('psr_license_number', 'none')

def get_energy(xpath: str, response: TextResponse, loader):
    """Obtain the ad's energy info from the response object throug a Selector
    and populate the `energy` field. Populates the `energy` field with `none` if
    something goes wrong.
    
    Parameters
    ----------
    xpath : 
        Path for the Selector.
    response :
        Page contents. Response object.
    loader : 
        Item loader. Provide the mechanism for populating the item field.
    """
    try:
        energy_info = response.xpath(xpath).get()
        if energy_info:
            loader.add_value('energy', energy_info)
        else:
            loader.add_value('energy', 'none')
    except:
        loader.add_value('energy', 'none')


def get_coordinates(xpath: str, pattern: str, response: TextResponse, loader):
    """Obtain the ad's coordinates from the response object throug a Selector
    and populate the `coordinates` field. Populates the `coordinates` field with `none` if
    something goes wrong.
    
    Parameters
    ----------
    xpath : 
        Path for the Selector.
    pattern :
        Regular expression.
    response :
        Page contents. Response object.
    loader : 
        Item loader. Provide the mechanism for populating the item field.
    """
    try:
        coordinates_info = response.xpath(xpath).re_first(pattern)
        if coordinates_info:
            loader.add_value('coordinates', coordinates_info)
        else:
            loader.add_value('coordinates', 'none')
    except:
        loader.add_value('coordinates', 'none')


def get_desc(xpath: str, response: TextResponse, loader):
    """Obtain the some ad's information from the response object throug a Selector
    and populate the `sale_type` and `floor_area` fields. Populates the `sale_type` and
    `floor_area` fields with `none` if something goes wrong.
    
    Parameters
    ----------
    xpath : 
        Path for the Selector.
    response :
        Page contents. Response object.
    loader : 
        Item loader. Provide the mechanism for populating the item field.
    """
    try:
        sale_type_info = response.xpath(xpath).getall()[0]
        if sale_type_info:
            loader.add_value('sale_type', sale_type_info)
        else:
            loader.add_value('sale_type', 'none')
    except:
        loader.add_value('sale_type', 'none')
    try:
        floor_area_info = response.xpath(xpath).getall()[-1]
        if floor_area_info:
            loader.add_value('floor_area', floor_area_info)
        else:
            loader.add_value('floor_area', 'none')
    except:
        loader.add_value('floor_area', 'none')


def get_overview(xpath_caracts: str, xpath_values: str, response: TextResponse,
                 loader):
    """Obtain the some ad's overview information from the response object throug a Selector
    and populate the `overview` field. Populates the `overview` field with `none` if something
    goes wrong.
    
    Parameters
    ----------
    xpath_caracts : 
        Path for the Selector.
    xpath_values : 
        Path for the Selector.
    response :
        Page contents. Response object.
    loader : 
        Item loader. Provide the mechanism for populating the item field.
    """
    try:
        caracts = response.xpath(xpath_caracts).getall()
        values = response.xpath(xpath_values).getall()
        for value in values:
            if value == ': ':
                values.remove(value)
        overview_list = []
        for caract, value in zip(caracts, values):
            overview_elem = caract + ': ' + value
            overview_list.append(overview_elem)
        overview = ','.join(overview_list)
        loader.add_value('overview', overview)
    except:
        loader.add_value('overview', 'none')


def get_facilities(xpath: str, response: TextResponse, loader):
    """Obtain the some ad's facilities information from the response object throug a Selector
    and populate the `facilities` field. Populates the `facilities` field with `none` if something
    goes wrong.
    
    Parameters
    ----------
    xpath : 
        Path for the Selector.
    response :
        Page contents. Response object.
    loader : 
        Item loader. Provide the mechanism for populating the item field.
    """
    try:
        facilities_list = response.xpath(xpath).getall()
        facilities = ','.join(facilities_list)
        loader.add_value('facilities', facilities)
    except:
        loader.add_value('facilities', 'none')


def get_description(xpath: str, response: TextResponse, loader):
    """Obtain the ad's description from the response object throug a Selector and
    populate the `description` field. Populates the `description` field with `none` if 
    something goes wrong.
    
    Parameters
    ----------
    xpath : 
        Path for the Selector.
    response :
        Page contents. Response object.
    loader : 
        Item loader. Provide the mechanism for populating the item field.
    """
    try:
        description_info = response.xpath(xpath).get()
        if description_info:
            loader.add_value('description', description_info)
        else:
            loader.add_value('description', 'none')
    except:
        loader.add_value('description', 'none')



'''
########################################################################
# DATA CLEANSING AND WRANGLING
########################################################################

def get_db(dbname: str,): ##############
    """Stablishes a connection to the database, queries it and drops 
    the advertiser'private information before return the dataframe.
    
    Parameters
    ----------
    dbname : 
        The database name to addmto `database_path`.
    query :
        The query to the database.
    
    Returns
    -------
    The data obtained from the database as a dataframe.
    """
    database_path = f'data/{dbname}'
    connection = sqlite3.connect(database_path)
    #cursor = connection.cursor()

    daft = pd.read_sql_query(query, connection)
    connection.close()
    
    daft.drop(['contact', 'phone'], axis=1, inplace=True)
    sale = daft.copy()

    return sale
'''


