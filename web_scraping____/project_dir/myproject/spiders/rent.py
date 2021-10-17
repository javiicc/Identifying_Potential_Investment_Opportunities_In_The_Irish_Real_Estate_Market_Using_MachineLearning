#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 11:28:29 2021

@author: javier
"""

from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.loader import ItemLoader 
#from scrapy.loader.processors import MapCompose
#from bs4 import BeautifulSoup
from ..items import RentIeHouseItem
from ..items import RentIeRoomItem  #, table_processing
#from scrapy.loader.processors import MapCompose, TakeFirst
#from ..pipelines import get_last_id
import re
#from ..exceptions_handlers import errback_httpbin
    
from scrapy.spidermiddlewares.httperror import HttpError
from twisted.internet.error import DNSLookupError
from twisted.internet.error import TimeoutError, TCPTimedOutError

from datetime import datetime


def scraping_date():
    now = datetime.now()
    return '{}-{}-{}'.format(now.year, now.month, now.day)


class RentIeSpider(CrawlSpider):
    name = 'rent_ie_spider'
    
    #custom_settings = {
     #   'CLOSESPIDER_PAGECOUNT':50,
      #  }
   

    allowed_domains = ['rent.ie']  
    start_urls = []
    for url in open("/home/javier/Desktop/TFM/Fraud_Detection_In_The_Irish_Rental_Market/web_scraping/project_dir/urls.txt"):
        start_urls.append(url) #32 urls
    
    

    rules = (
            #Rule(LinkExtractor(allow=r'\d+/\d+'), #con los allow funciona bien el rent_type
            #    follow=True),
            # Paginacion
        Rule(LinkExtractor(allow=r'page_'),    #/\d{4,} al quitar esto ceso el fallo de los 461
             follow=True, errback='errback_httpbin'),
        # Detalles inmueble
        Rule(LinkExtractor(allow=r'houses-to-let/.+/\d+'),  #r'/houses-to-let/.+'
             follow=False, errback='errback_httpbin', callback='parse_items_houses'),
        # rooms to rent
        Rule(LinkExtractor(allow=r'rooms-to-rent/.+/\d+'),  #r'/houses-to-let/.+'
             follow=False, errback='errback_httpbin', callback='parse_items_rooms'))
    
    
    
    def errback_httpbin(self, failure):
        # log all failures
        self.logger.error(repr(failure))

        # in case you want to do something special for some errors,
        # you may need the failure's type:

        if failure.check(HttpError):
            # these exceptions come from HttpError spider middleware
            # you can get the non-200 response
            response = failure.value.response
            self.logger.error('HttpError on %s', response.url)        

        elif failure.check(DNSLookupError):
            # this is the original request
            request = failure.request
            self.logger.error('DNSLookupError on %s', request.url)

        elif failure.check(TimeoutError, TCPTimedOutError):
            request = failure.request
            self.logger.error('TimeoutError on %s', request.url)
    



# review selectors in scrapy documentation!!!!
    def parse_items_houses(self, response):
        item_loader = ItemLoader(RentIeHouseItem(), response)
        
        

        house_id = re.findall(r'(?<=/)\d+', response.url)[-1]   #\d{4,}' ultimo cambio
        item_loader.add_value('house_id', house_id)
        # devuelve error none type is not supcriptable cuando intenta extraer algo que no es un anuncio
        
        url = response.url
        item_loader.add_value('url', url)
        
        xpath_name = '//div[@class="smi_main_top"]/div[1]/h1[1]/text()'
        try:
            item_loader.add_xpath('name', xpath_name)
        except:
            item_loader.add_xpath('name', 'none')
        
        
        xpath_price = '//div[@class="smi_details_box"]/div[2]/h2[1]/text()'
        try:
            item_loader.add_xpath('price', xpath_price) 
        except:
            item_loader.add_xpath('price', 'none')


        xpath_rooms = '//div[@class="smi_details_box"]/div[2]/text()'  # improve
        try:
            item_loader.add_xpath('rooms', xpath_rooms)
        except:
            item_loader.add_xpath('rooms', 'none')
        
        
        xpath_ber = '//div[@class="ber-top left"]/img/@src'
        try:
            ber_info = response.xpath(xpath_ber).re_first(r'i/ber/(.+).png')
            if ber_info:
                item_loader.add_value('ber', ber_info)
            else:
                item_loader.add_value('ber', 'none')
        except:
            item_loader.add_value('ber', 'none')
        
        
            
        # contact
        try:
            contact_info = response.xpath('//table//text()').extract()
            contact_index = contact_info.index('Contact:') + 2
            if contact_index:
                item_loader.add_value('contact', contact_info[contact_index])
            else:
                item_loader.add_value('contact', 'none')
        except:
            item_loader.add_value('contact', 'none')    
        # phone
        try:
            contact_info = response.xpath('//table//text()').extract()
            phone_index = contact_info.index('Phone:') + 2
            if phone_index:
                item_loader.add_value('phone', contact_info[phone_index])
            else:
                item_loader.add_value('phone', 'none')
        except:
            item_loader.add_value('phone', 'none')

      
        xpath_agent = '//div[@id="smi_main_box"]/div[1]/div[2]/p/text()'
        agent_info = response.xpath(xpath_agent).getall()
        if agent_info:
            try:
                item_loader.add_value('letting_agent', agent_info[1])
            except:
                item_loader.add_value('letting_agent', 'none')
            try:
                item_loader.add_value('psr_licence_number', agent_info[3]) # arreglar, da problemas el 3
            except:
                item_loader.add_value('psr_licence_number', 'none')                
        else:
            item_loader.add_value('letting_agent', 'none')
            item_loader.add_value('psr_licence_number', 'none')
        
        
        xpath_features = '//div[@class="smi_details_box"]/div[2]/ul[1]/li[1]'
        try:
            key_features_info = response.xpath(xpath_features).re(r'<li>(.+)</li>')
            if key_features_info:
                key_features_list = []
                for elem in key_features_info:   
                    elem = elem.replace('</li><li>', ',').split(',')
                    key_features_list = key_features_list + elem
                features_string = ','.join(key_features_list)
                item_loader.add_value('key_features', features_string)
            else:
                item_loader.add_value('key_features', 'none')
        except:
            item_loader.add_value('key_features', 'none')
            
            
       
        xpath_desc = '//div[@id="smi_description"]/p'
        try:
            availability_info = response.xpath(xpath_desc).re_first(r'(?<=Available from:</strong>\n)(.+)')
            lease_info = response.xpath(xpath_desc).re_first(r'(?<=Lease:</strong>)(.+)')
            energy_info = response.xpath(xpath_desc).re_first(r'(?<=Energy Performance Indicator: )(.+)')
            entered_info = response.xpath(xpath_desc).re_first(r'(?<=This property was entered:</strong>\n)(.+)')
            
            description_fields = {'availability': availability_info,
                                  'lease': lease_info,
                                  'energy': energy_info,
                                  'entered': entered_info}
            for key in description_fields:
                if description_fields[key]:
                    item_loader.add_value(key, description_fields[key])
                else:
                    item_loader.add_value(key, 'none')
        except:
            item_loader.add_value(key, 'none')
        
        
        houses_to_let = re.search(r'houses-to-let', response.url)
        if houses_to_let:
            item_loader.add_value('type_rent', 'houses to let')
        else:
            item_loader.add_value('type_rent', 'none')
        
        
        xpath_lat = '//*[@id="button_satellite"]'
        try:
            lat = response.xpath(xpath_lat).re_first(r'latitude: (.+),')
            if lat:
                item_loader.add_value('latitude', lat)
            else:
                item_loader.add_value('latitude', 'none')
        except:
            item_loader.add_value('latitude', 'none')
            
        xpath_lon = '//*[@id="button_satellite"]'
        try:
            lon = response.xpath(xpath_lon).re_first(r'longitude: (.+),')
            if lon:
                item_loader.add_value('longitude', lon)
            else:
                item_loader.add_value('longitude', 'none')
        except:
            item_loader.add_value('longitude', 'none')
            
            
        item_loader.add_value('scraping_date', scraping_date())
        
            
        
        #item_loader.add_value('web', 'rent.ie')
            
        yield item_loader.load_item()
        
    
    def parse_items_rooms(self, response):
        item_loader = ItemLoader(RentIeRoomItem(), response)
        
        
        room_id = re.findall(r'(?<=/)\d+', response.url)[-1]  #\d{4,}' ultimo cambio
        item_loader.add_value('room_id', room_id)
        # devuelve error none type is not supcriptable cuando intenta extraer algo que no es un anuncio
        
        url = response.url
        item_loader.add_value('url', url)
        
            
        xpath_name = '//div[@class="smi_main_top"]/div[1]/h1[1]/text()'
        try:
            item_loader.add_xpath('name', xpath_name)
        except:
            item_loader.add_xpath('name', 'none')
         
            
        xpath_price = '//div[@class="smi_details_box"]/div[2]/h2[1]/text()'
        try:
            item_loader.add_xpath('price', xpath_price) 
        except:
            item_loader.add_xpath('price', 'none') 
        
        
        xpath_bed_ava = '//div[@id="smi_main_box"]/div[1]/div[2]/text()'
        try:
            bed_and_availability = response.xpath(xpath_bed_ava).getall()
        except: pass           
            #item_loader.add_value('bed', 'none')
            #item_loader.add_value('availability_time', 'none')
        bed_info = next((i for i in bed_and_availability if 'Bedroom' in i), 'none') #este es guay
        item_loader.add_value('bed', bed_info) #next(i for i in bed_and_availability if 'Bedroom' in i)
        availability_time_info = next((i for i in bed_and_availability if 'available' in i), 'none')
        item_loader.add_value('availability_time', availability_time_info)
        
  
        xpath_ber = '//div[@class="ber-top left"]/img/@src'
        try:
            ber_info = response.xpath(xpath_ber).re_first(r'i/ber/(.+).png')
            if ber_info:
                item_loader.add_value('ber', ber_info)
            else:
                item_loader.add_value('ber', 'none')
        except:
            item_loader.add_value('ber', 'none')
       
     
        # contact
        try:
            contact_info = response.xpath('//table//text()').extract()
            contact_index = contact_info.index('Contact:') + 2
            if contact_index:
                item_loader.add_value('contact', contact_info[contact_index])
            else:
                item_loader.add_value('contact', 'none')
        except:
            item_loader.add_value('contact', 'none')    
        # phone
        try:
            contact_info = response.xpath('//table//text()').extract()
            phone_index = contact_info.index('Phone:') + 2
            if phone_index:
                item_loader.add_value('phone', contact_info[phone_index])
            else:
                item_loader.add_value('phone', 'none')
        except:
            item_loader.add_value('phone', 'none')

        # este se puede mejorar!!!!!!!!
        xpath_agent = '//div[@id="smi_main_box"]/div[1]/div[2]/p/text()'
        agent_info = response.xpath(xpath_agent).getall()
        if agent_info:
            try:
                item_loader.add_value('letting_agent', agent_info[1])
            except:
                item_loader.add_value('letting_agent', 'none')
            try:
                item_loader.add_value('psr_licence_number', agent_info[3]) # arreglar, da problemas el 3
            except:
                item_loader.add_value('psr_licence_number', 'none')                
        else:
            item_loader.add_value('letting_agent', 'none')
            item_loader.add_value('psr_licence_number', 'none')
        
        
        xpath5 = '//div[@class="smi_details_box"]/div[2]/ul[1]/li[1]'
        try:
            key_features_info = response.xpath(xpath5).re(r'<li>(.+)</li>')
            if key_features_info:
                key_features_list = []
                for elem in key_features_info:   
                    elem = elem.replace('</li><li>', ',').split(',')
                    key_features_list = key_features_list + elem
                features_string = ','.join(key_features_list)
                item_loader.add_value('key_features', features_string)
            else:
                item_loader.add_value('key_features', 'none')
        except:
            item_loader.add_value('key_features', 'none')
            

        xpath_desc = '//div[@id="smi_description"]/p'
        availability_info = response.xpath(xpath_desc).re_first(r'(?<=Available from:</strong>\n)(.+)')
        lease_info = response.xpath(xpath_desc).re_first(r'(?<=Lease:</strong>)(.+)')
        energy_info = response.xpath(xpath_desc).re_first(r'(?<=Energy Performance Indicator: )(.+)')
        entered_info = response.xpath(xpath_desc).re_first(r'(?<=This property was entered:</strong>\n)(.+)')
            
        description_fields = {'availability': availability_info, 
                              'lease': lease_info, 
                              'energy': energy_info, 
                              'entered': entered_info}         
        for key in description_fields:
            if description_fields[key]:
                item_loader.add_value(key, description_fields[key])
            else:
                item_loader.add_value(key, 'none')
        
        
        rooms_to_rent = re.search(r'rooms-to-rent', response.url)
        if rooms_to_rent:
            item_loader.add_value('type_rent', 'rooms to rent')
        else:
            item_loader.add_value('type_rent', 'none')
        
        
        xpath_lat = '//*[@id="button_satellite"]'
        try:
            lat = response.xpath(xpath_lat).re_first(r'latitude: (.+),')
            if lat:
                item_loader.add_value('latitude', lat)
            else:
                item_loader.add_value('latitude', 'none')
        except:
            item_loader.add_value('latitude', 'none')
            
        xpath_lon = '//*[@id="button_satellite"]'
        try:
            lon = response.xpath(xpath_lon).re_first(r'longitude: (.+),')
            if lon:
                item_loader.add_value('longitude', lon)
            else:
                item_loader.add_value('longitude', 'none')
        except:
            item_loader.add_value('longitude', 'none')
            
        item_loader.add_value('scraping_date', scraping_date())
            
            
       # item_loader.add_value('web', 'rent.ie')
            
        yield item_loader.load_item()