#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 10:41:31 2021

@author: javier
"""

from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.loader import ItemLoader 
#from scrapy.loader.processors import MapCompose
#from bs4 import BeautifulSoup
#from ..items import RentIeItem
#from ..items import RentIeRoomsItem  #, table_processing
#from scrapy.loader.processors import MapCompose, TakeFirst
#from ..pipelines import get_last_id
import re
#from ..exceptions_handlers import errback_httpbin
    
#from scrapy.spidermiddlewares.httperror import HttpError
#from twisted.internet.error import DNSLookupError
#from twisted.internet.error import TimeoutError, TCPTimedOutError
from ..items import PropertyItem


class PropertySpider(CrawlSpider):
    name = 'property_spider'
    
    custom_settings = {
        'CLOSESPIDER_PAGECOUNT':50,
        }
   

    allowed_domains = ['property.ie']
    '''
    start_urls = []
    for url in open("../daft_urls.txt"):
        start_urls.append(url) #32 urls
    '''
    
    start_urls = ['https://www.property.ie/property-to-let/ireland/']

    

    rules = (
            # Paginacion
        #Rule(LinkExtractor(allow=r'/?pageSize=20&from='),    #/\d{4,} al quitar esto ceso el fallo de los 461
         #    follow=True),
        Rule(LinkExtractor(allow=r'price_international_rental-onceoff_standard/p_'),    #/\d{4,} al quitar esto ceso el fallo de los 461
             follow=True),
        #Rule(LinkExtractor(allow=r'for-rent/.+/\d+'),    #/\d{4,} al quitar esto ceso el fallo de los 461
         #    follow=True),
        # Detalles inmueble property-to-let/Bremore-Pastures-Park-Balbriggan-Co-Dublin/6160265/
        Rule(LinkExtractor(allow=r'property-to-let/.+/\d+'),  #r'/houses-to-let/.+'
             follow=False, callback='parse_items'),
        #Rule(LinkExtractor(allow=r'for-rent/apartment.+/\d+'),  #r'/houses-to-let/.+'
         #    follow=False, callback='parse_items'),
        #Rule(LinkExtractor(allow=r'share/.+/\d+'),  #r'/houses-to-let/.+'
         #    follow=False, callback='parse_items'),
        # rooms to rent
        #Rule(LinkExtractor(allow=r'rooms-to-rent/.+/\d+'),  #r'/houses-to-let/.+'
         #    follow=False, callback='parse_items_rooms')
         )

    
    #, errback='errback_httpbin'
    
    
    def parse_items(self, response):  #parse_houses
        item_loader = ItemLoader(PropertyItem(), response)
        
        item_id = re.search(r'\d{7,}', response.url)[0]   #\d{4,}' ultimo cambio
        item_loader.add_value('item_id', item_id)
        
        url = response.url
        item_loader.add_value('url', url)
        
        xpath_name = '//h1[@style="clear: left"]/text()'
        try:
            name_info = response.xpath(xpath_name).get()
            if name_info:
                item_loader.add_value('name', name_info)
            else:
                item_loader.add_value('name', 'none')
        except:
            item_loader.add_value('name', 'none')


        xpath_price = '//div[@id="searchmoreinfo_summary"]/h2/text()'
        try:
            price_info = response.xpath(xpath_price).get()
            if price_info: 
                item_loader.add_value('price', price_info)
            else:
                item_loader.add_value('price', 'none')
        except:
            item_loader.add_value('price', 'none')
                
        
        xpath_description = '//div[@id="searchmoreinfo_summary"]/text()'
        try:
            description_info = response.xpath(xpath_description).getall()
            if description_info:
                description_string = ','.join(description_info)
                item_loader.add_value('description', description_string)
            else:
                item_loader.add_value('description', 'none')
        except:
            item_loader.add_value('description', 'none')   
            
        xpath_ber = '//div[@class="ber-top left"]/p/img/@src'
        try:
            ber_info = response.xpath(xpath_ber).re_first(r'ber_(.+).png')
            if ber_info:
                item_loader.add_value('ber', ber_info)
            else:
                item_loader.add_value('ber', 'none')
        except:
            item_loader.add_value('ber', 'none')
            
            
        xpath_features = '//div[@id="searchmoreinfo_features"]/text()'
        try:
            features_info = response.xpath(xpath_features).getall()
            if features_info:
                features_string = ','.join(features_info)
                item_loader.add_value('features', features_string)
            else:
                item_loader.add_value('features', 'none')
        except:
            item_loader.add_value('features', 'none')  
            
        
        xpath_energy = '//p[@class="ber-paragraph"]/text()'
        try:
            energy_info = response.xpath(xpath_energy).getall()
            if energy_info:
                for elem in energy_info:
                    if 'Energy Performance Indicator:' in elem:
                        energy = re.findall('(?<=Energy Performance Indicator: )(.+)', elem)[0]   
                item_loader.add_value('energy', energy)
            else:
                item_loader.add_value('energy', 'none')
        except:
            item_loader.add_value('energy', 'none')
            
            
        xpath_last_updated = '//div[@id="searchmoreinfo_description"]/p/text()'
        try:
            last_updated_info = response.xpath(xpath_last_updated).getall()[-1]
            if last_updated_info: 
                item_loader.add_value('last_updated', last_updated_info)
            else:
                item_loader.add_value('last_updated', 'none')
        except:
            item_loader.add_value('last_updated', 'none')
        
        '''  
        xpath_contact = '//div[@id="searchmoreinfo_sellingagent"]/p/text()'
        try:
            contact_info = response.xpath(contact_updated).getall()[-1]
            if contact_info:
                item_loader.add_value('contact', contact_info)
            else:
                item_loader.add_value('contact', 'none')
        except:
            item_loader.add_value('contact', 'none')
            
        xpath_phone = '//div[@id="searchmoreinfo_sellingagent"]/p/text()'
        try:
            phone_info = response.xpath(contact_updated).getall()[-1]
            if phone_info:
                item_loader.add_value('contact', contact_info)
            else:
                item_loader.add_value('contact', 'none')
        except:
            item_loader.add_value('contact', 'none')
        '''    
            

  
            
            
        xpath_coordinates = '//li[@class="smi_tab_item"]/a/@href'
        pattern_coordinates = r'loc:(.+)'
        try:
            coordinates_info = response.xpath(xpath_coordinates)[-1].re_first(pattern_coordinates)
            if coordinates_info:
                item_loader.add_value('coordinates', coordinates_info)
            else:
                item_loader.add_value('coordinates', 'none')
        except:
            item_loader.add_value('coordinates', 'none')
            
        
        #item_loader.add_value('web', 'daft.ie')
        
        yield item_loader.load_item()