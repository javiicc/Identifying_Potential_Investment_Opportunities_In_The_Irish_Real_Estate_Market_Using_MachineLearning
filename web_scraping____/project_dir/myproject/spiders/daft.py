#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 16:42:37 2021

@author: javier
"""
from abc import ABC

from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.loader import ItemLoader

import re

from ..items import DaftItemBuy, DaftItemRent

from datetime import datetime

from daftpy.daftdata import (DaftLoader, get_name, get_price, get_info, get_ber,
                             get_entered_renewed_views, get_contact_phone, get_psr,
                             get_energy, get_coordinates, 
                             get_desc, get_overview, get_facilities)
from datetime import date



class DaftSpider(CrawlSpider, ABC):
    """Class to define the Spider. It inherit from CrawlSpider.
    
    """
    
    name = 'daft_spider'

    # custom_settings = {
    #   'CLOSESPIDER_PAGECOUNT': 50,
    #  }

    # The spider won't be able to follow a link with a domain different that the bellow one
    allowed_domains = ['daft.ie']
    
    # The spider will start looking ads from these urls
    start_urls = ['https://www.daft.ie/property-for-sale/ireland/houses',
                  'https://www.daft.ie/property-for-sale/ireland/apartments',
                  'https://www.daft.ie/property-for-rent/ireland/houses',
                  'https://www.daft.ie/property-for-rent/ireland/apartments',
                  'https://www.daft.ie/sharing/ireland',
                  'https://www.daft.ie/new-homes-for-sale/ireland',
                  ]

    # The spider will follow the following rules in order to find all the required ads
    # Each `Rule` defines a certain behaviour for crawling the site
    # A link extractor is an object that extracts links from responses
    # `allow` are regular expressions that the urls must match in order to be executed
    # `callback` indicates the method from the spider object to parse the response
    # extracted with the specified link extractor
    rules = (
        # Pagination from `start_urls`
        Rule(LinkExtractor(allow=r'.+from=.+'),
             follow=True),
        # Buy ads
        Rule(LinkExtractor(allow=r'for-sale/.+/\d+'),
             follow=False, callback='parse_items_buy'),
        # Rent ads
        Rule(LinkExtractor(allow=[r'for-rent/house.+/\d+', 
                                  r'for-rent/apartment.+/\d+',
                                  r'share/.+/\d+']), # Check the list works!!!
             follow=False, callback='parse_items_rent'),
    )
    ''' NO QUITAR AUN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    rules = (
        Rule(LinkExtractor(allow=r'.+from=.+'),
             follow=True),
        Rule(LinkExtractor(allow=r'for-sale/.+/\d+'),
             follow=False, callback='parse_items_buy'),
        Rule(LinkExtractor(allow=r'for-rent/house.+/\d+'),
             follow=False, callback='parse_items_rent'),
        Rule(LinkExtractor(allow=r'for-rent/apartment.+/\d+'),
             follow=False, callback='parse_items_rent'),
        Rule(LinkExtractor(allow=r'share/.+/\d+'),
             follow=False, callback='parse_items_rent'),
    )
    '''
    
     
    def parse_items_buy(self, response):
        """The parse_items_buy method is in charge of processing the response and 
        returning scraped data.
        
        Parameters
        ----------
        response : Response object
            The response to parse.
            
        Yield
        -------
        Item loaded.
        """
        
        # Instantiate an item loader, which indicates what item fields and what process to do
        item_loader = DaftLoader(item=DaftItemBuy(), 
                                 response=response)
        
        # Xpaths:
        xpath_id = '//p[@class="DaftIDText__StyledDaftIDParagraph-vbn7aa-0 iWkymm"]/text()'
        xpath_name = '//h1[@data-testid="address"]/text()'
        xpath_price = '//div[@data-testid="title-block"]/div/p/span/text()'
        xpath_info = '//div[@data-testid="card-info"]/p/text()'
        xpath_desc = '//ul[@class="PropertyPage__InfoSection-sc-14jmnho-7 ertsXd"]/li/text()'
        xpath_contact = '//div[@class="ContactPanel__Container-sc-18zt6u1-7 fCElaB"]/div[2]/p/text()'
        xpath_phone = '//div[@class="ContactPanel__Container-sc-18zt6u1-7 fCElaB"]/div[2]/a/@href'
        xpath_license = '//div[@class="PropertyPage__MainColumn-sc-14jmnho-1 hwqROB"]/div[5][1]'
        xpath_ber = '//div[@data-testid="ber"]/img/@alt'
        xpath_entered_views = '//p[@class="Statistics__StyledLabel-sc-15tgae4-1 pBjQg"]/text()'
        xpath_energy = '//p[@data-testid="ber-epi"]/span[2]/text()'
        xpath_coordinates = '//div[@class="NewButton__ButtonContainer-yem86a-4 dFKaNf button-container"]/a'
        
        # Patterns:
        pattern_phone = r'tel:(\d+[\-\s]?\d+[\-\s]?\d+)'
        license_pattern = r'"licenceNumber":"(\d+)'
        pattern_coordinates = r'q=loc:(.+)" data-tracking="LaunchSatellite"'
        
        
        # The following code uses the selector's shortcut `response.xpath()` instead 
        # of `response.selector.xpath()`
        # `add_value` adds the given value for the given field after being passed through 
        # the field input processor from `item_loader`
        
        # Extract daft id     
        daft_id = response.xpath(xpath_id).getall()[-1]
        item_loader.add_value('daft_id', daft_id)

        # Extract an id from the ad's url with the help of the function `findall` and 
        # the package `re`
        item_id = re.findall(r'(?<=/)\d+', response.url)[-1]
        item_loader.add_value('item_id', item_id)

        # Extract ad url
        url = response.url
        item_loader.add_value('url', url)

        # Extract ad name
        get_name(xpath=xpath_name, response=response, loader=item_loader)

        # Extract ad price
        get_price(xpath=xpath_price, response=response, loader=item_loader)

        # Extract ad info
        get_info(xpath=xpath_info, response=response, loader=item_loader)

        # Extract desc
        get_desc(xpath=xpath_desc, response=response, loader=item_loader)

        # Extract contact and phone
        get_contact_phone(xpath_contact=xpath_contact, 
                          xpath_phone=xpath_phone,
                          pattern_phone=pattern_phone, 
                          response=response, 
                          loader=item_loader)

        # Extract license
        get_psr(xpath=xpath_license, 
                pattern=license_pattern, 
                response=response,
                loader=item_loader)

        # Extract ber info
        get_ber(xpath=xpath_ber, response=response, loader=item_loader)

        # Extract entered/renewed date and views
        get_entered_renewed_views(xpath=xpath_entered_views, 
                                  response=response, 
                                  loader=item_loader)

        # Extract type_house data
        if 'apartment' in response.url:
            item_loader.add_value('type_house', 'apartment')
        elif 'house' in response.url:
            item_loader.add_value('type_house', 'house')
        else:
            item_loader.add_value('type_house', 'none')

        # Extract energy data
        get_energy(xpath=xpath_energy, response=response, loader=item_loader)

        # Extract coordinates
        get_coordinates(xpath=xpath_coordinates, 
                        pattern=pattern_coordinates, 
                        response=response, 
                        loader=item_loader)

        # Add ad's type info
        item_loader.add_value('type', 'buy')

        # Add scraping date
        item_loader.add_value('scraping_date', str(date.today()))

        yield item_loader.load_item()
    
    
    
    

    def parse_items_rent(self, response):

        # Instantiate an item loader, which indicates what item fields and what process to do
        item_loader = DaftLoader(item=DaftItemRent(), 
                                 response=response)
        
        # Xpaths:
        xpath_id = '//p[@class="DaftIDText__StyledDaftIDParagraph-vbn7aa-0 iWkymm"]/text()'
        xpath_name = '//h1[@data-testid="address"]/text()'
        xpath_price = '//div[@data-testid="title-block"]/div/p/span/text()'
        xpath_info = '//div[@data-testid="card-info"]/p/text()'
        xpath_overview_caracts = '//span[@class="PropertyPage__ListLabel-sc-14jmnho-10 ' \
                                 'ssSHo"]/text() '
        xpath_overview_values = '//ul[@class="PropertyPage__InfoSection-sc-14jmnho-7 ' \
                                'ertsXd"]/li/text() '
        xpath_facilities = '//ul[' \
                           '@class="PropertyDetailsList__PropertyDetailsListContainer' \
                           '-sc-1cjwtjz-0 bnzQrB"]/li/text() '
        xpath_ber = '//div[@data-testid="ber"]/img/@alt'
        xpath_entered_views = '//p[@class="Statistics__StyledLabel-sc-15tgae4-1 ' \
                              'pBjQg"]/text() '
        xpath_contact = '//div[@class="ContactPanel__Container-sc-18zt6u1-7 fCElaB"]/div[2]/p/text()'
        xpath_phone = '//div[@class="ContactPanel__Container-sc-18zt6u1-7 fCElaB"]/div[2]/a/@href'
        xpath_license = '//div[@class="PropertyPage__MainColumn-sc-14jmnho-1 hwqROB"]/div[5][1]'
        xpath_energy = '//p[@data-testid="ber-epi"]/span[2]/text()'
        xpath_coordinates = '//div[@class="NewButton__ButtonContainer-yem86a-4 dFKaNf button-container"]/a'
        
        # Patterns:
        pattern_phone = r'tel:(\d+[\-\s]?\d+[\-\s]?\d+)'
        license_pattern = r'"licenceNumber":"(\d+)'
        pattern_coordinates = r'q=loc:(.+)" data-tracking="LaunchSatellite"'
        

        # The following code uses the selector's shortcut `response.xpath()` instead 
        # of `response.selector.xpath()`
        # `add_value` adds the given value for the given field after being passed through 
        # the field input processor from `item_loader`
        
        # Extract daft id
        daft_id = response.xpath(xpath_id).getall()[-1]
        item_loader.add_value('daft_id', daft_id)

        # Extract an id from the ad's url with the help of the function `findall` and 
        # the package `re`
        item_id = re.findall(r'(?<=/)\d+', response.url)[-1]
        item_loader.add_value('item_id', item_id)

        # Extract ad url
        url = response.url
        item_loader.add_value('url', url)

        # Extract ad name
        get_name(xpath=xpath_name, response=response,loader=item_loader)

        # Extract ad price
        get_price(xpath=xpath_price, response=response, loader=item_loader)

        # Extract ad info
        get_info(xpath=xpath_info, response=response, loader=item_loader)

        # Extract overview caracts and values
        get_overview(xpath_caracts=xpath_overview_caracts, 
                     xpath_values=xpath_overview_values,
                     response=response, 
                     loader=item_loader)

        # Extract facilities
        get_facilities(xpath=xpath_facilities, response=response, loader=item_loader)

        # EXtract ber 
        get_ber(xpath=xpath_ber, response=response, loader=item_loader)

        # Extract entered/renewed date and views
        get_entered_renewed_views(xpath=xpath_entered_views, 
                                  response=response, 
                                  loader=item_loader)

        # Extract contact and phone
        get_contact_phone(xpath_contact=xpath_contact, 
                          xpath_phone=xpath_phone, 
                          pattern_phone=pattern_phone, 
                          response=response, 
                          loader=item_loader)

        # Extract license
        get_psr(xpath=xpath_license, 
                pattern=license_pattern, 
                response=response, 
                loader=item_loader)

        # Extract type house
        if '/apartment' in response.url:
            item_loader.add_value('type_house', 'apartment')
        elif '/house' in response.url:
            item_loader.add_value('type_house', 'house')
        elif '/share/' in response.url:
            item_loader.add_value('type_house', 'room')
        else:
            item_loader.add_value('type_house', 'none')

        # Extract energy data
        get_energy(xpath=xpath_energy, response=response, loader=item_loader)

        # Extract coordinates
        get_coordinates(xpath=xpath_coordinates, 
                        pattern=pattern_coordinates, 
                        response=response, 
                        loader=item_loader)

        # Add scraping date
        # We have to convert the date object to a string to avoid issues in the
        # Item Loader
        item_loader.add_value('scraping_date', str(date.today()))

        # Add ad's type info
        if '/share/' in response.url:
            item_loader.add_value('type', 'share')
        else:
            item_loader.add_value('type', 'rent')

        yield item_loader.load_item()

        
        
  