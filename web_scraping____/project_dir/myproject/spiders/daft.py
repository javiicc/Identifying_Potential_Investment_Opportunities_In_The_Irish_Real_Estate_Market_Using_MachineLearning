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
# from scrapy.loader.processors import MapCompose
# from bs4 import BeautifulSoup
# from ..items import RentIeItem
# from ..items import RentIeRoomsItem  #, table_processing
# from scrapy.loader.processors import MapCompose, TakeFirst
# from ..pipelines import get_last_id
import re
# from ..exceptions_handlers import errback_httpbin

# from scrapy.spidermiddlewares.httperror import HttpError
# from twisted.internet.error import DNSLookupError
# from twisted.internet.error import TimeoutError, TCPTimedOutError
from ..items import DaftItemBuy, DaftItemRent

from datetime import datetime


def scraping_date():
    now = datetime.now()
    return '{}-{}-{}'.format(now.year, now.month, now.day)


class DaftSpider(CrawlSpider, ABC):
    name = 'daft_spider'

    # custom_settings = {
    #   'CLOSESPIDER_PAGECOUNT': 50,
    #  }

    allowed_domains = ['daft.ie']

    start_urls = ['https://www.daft.ie/property-for-sale/ireland/houses',
                  'https://www.daft.ie/property-for-sale/ireland/apartments',
                  'https://www.daft.ie/property-for-rent/ireland/houses',
                  'https://www.daft.ie/property-for-rent/ireland/apartments',
                  'https://www.daft.ie/sharing/ireland',
                  'https://www.daft.ie/new-homes-for-sale/ireland',
                  ]

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

    def parse_items_buy(self, response):

        item_loader = ItemLoader(DaftItemBuy(), response)

        xpath_id = '//p[@class="DaftIDText__StyledDaftIDParagraph-vbn7aa-0 iWkymm"]/text()'
        daft_id = response.xpath(xpath_id).getall()[-1]
        item_loader.add_value('daft_id', daft_id)

        item_id = re.findall(r'(?<=/)\d+', response.url)[-1]
        item_loader.add_value('item_id', item_id)

        url = response.url
        item_loader.add_value('url', url)

        xpath_name = '//h1[@data-testid="address"]/text()'
        try:
            name_info = response.xpath(xpath_name).get()
            if name_info:
                item_loader.add_value('name', name_info)
            else:
                item_loader.add_value('name', 'none')
        except:
            item_loader.add_value('name', 'none')

        xpath_price = '//div[@data-testid="title-block"]/div/p/span/text()'
        try:
            price_info = response.xpath(xpath_price).get()
            if price_info:
                item_loader.add_value('price', price_info)
            else:
                item_loader.add_value('price', 'none')
        except:
            item_loader.add_value('price', 'none')

        xpath_info = '//div[@data-testid="card-info"]/p/text()'
        try:
            info_list = response.xpath(xpath_info).getall()
            if info_list:
                info_string = ','.join(info_list)
                item_loader.add_value('info', info_string)
            else:
                item_loader.add_value('info', 'none')
        except:
            item_loader.add_value('info', 'none')

        xpath_desc = '//ul[@class="PropertyPage__InfoSection-sc-14jmnho-7 ertsXd"]/li/text()'
        try:
            sale_type_info = response.xpath(xpath_desc).getall()[0]
            if sale_type_info:
                item_loader.add_value('sale_type', sale_type_info)
            else:
                item_loader.add_value('sale_type', 'none')
        except:
            item_loader.add_value('sale_type', 'none')
        try:
            floor_area_info = response.xpath(xpath_desc).getall()[-1]
            if floor_area_info:
                item_loader.add_value('floor_area', floor_area_info)
            else:
                item_loader.add_value('floor_area', 'none')
        except:
            item_loader.add_value('floor_area', 'none')

        xpath_contact = '//div[@class="ContactPanel__Container-sc-18zt6u1-7 fCElaB"]/div[2]/p/text()'
        xpath_phone = '//div[@class="ContactPanel__Container-sc-18zt6u1-7 fCElaB"]/div[2]/a/@href'
        pattern_phone = r'tel:(\d+[\-\s]?\d+[\-\s]?\d+)'
        # contact
        try:
            contact_info = response.xpath(xpath_contact).get()
            if contact_info:
                item_loader.add_value('contact', contact_info)
            else:
                item_loader.add_value('contact', 'none')
        except:
            item_loader.add_value('contact', 'none')
        # phone
        try:
            phone_info = response.xpath(xpath_phone).re_first(pattern_phone)
            if phone_info:
                item_loader.add_value('phone', phone_info)
            else:
                item_loader.add_value('phone', 'none')
        except:
            item_loader.add_value('phone', 'none')

        xpath_license = '//div[@class="PropertyPage__MainColumn-sc-14jmnho-1 hwqROB"]/div[5][1]'
        license_pattern = r'"licenceNumber":"(\d+)'
        try:
            license_info = response.xpath(xpath_license).re_first(license_pattern)
            if license_info:
                item_loader.add_value('psr_license_number', license_info)
            else:
                item_loader.add_value('psr_license_number', 'none')
        except:
            item_loader.add_value('psr_license_number', 'none')

        xpath_ber = '//div[@data-testid="ber"]/img/@alt'
        try:
            ber_info = response.xpath(xpath_ber).get()
            if ber_info:
                item_loader.add_value('ber', ber_info)
            else:
                item_loader.add_value('ber', 'none')
        except:
            item_loader.add_value('ber', 'none')

        xpath_entered_views = '//p[@class="Statistics__StyledLabel-sc-15tgae4-1 pBjQg"]/text()'
        try:
            entered_views_info = response.xpath(xpath_entered_views).getall()
            if entered_views_info[0]:
                item_loader.add_value('entered_renewed', entered_views_info[0])
            else:
                item_loader.add_value('entered_renewed', 'none')
            if entered_views_info[1]:
                item_loader.add_value('views', entered_views_info[1])
            else:
                item_loader.add_value('views', 'none')
        except:
            item_loader.add_value('entered_renewed', 'none')
            item_loader.add_value('views', 'none')

        if 'apartment' in response.url:
            item_loader.add_value('type_house', 'apartment')
        elif 'house' in response.url:
            item_loader.add_value('type_house', 'house')
        else:
            item_loader.add_value('type_house', 'none')

        xpath_energy = '//p[@data-testid="ber-epi"]/span[2]/text()'
        try:
            energy_info = response.xpath(xpath_energy).get()
            if energy_info:
                item_loader.add_value('energy', energy_info)
            else:
                item_loader.add_value('energy', 'none')
        except:
            item_loader.add_value('energy', 'none')

        xpath_coordinates = '//div[@class="NewButton__ButtonContainer-yem86a-4 dFKaNf button-container"]/a'
        pattern_coordinates = r'q=loc:(.+)" data-tracking="LaunchSatellite"'
        try:
            coordinates_info = response.xpath(xpath_coordinates).re_first(pattern_coordinates)
            if coordinates_info:
                item_loader.add_value('coordinates', coordinates_info)
            else:
                item_loader.add_value('coordinates', 'none')
        except:
            item_loader.add_value('coordinates', 'none')

        item_loader.add_value('type', 'buy')

        item_loader.add_value('scraping_date', scraping_date())

        yield item_loader.load_item()

    def parse_items_rent(self, response):

        item_loader = ItemLoader(DaftItemRent(), response)

        xpath_id = '//p[@class="DaftIDText__StyledDaftIDParagraph-vbn7aa-0 iWkymm"]/text()'
        daft_id = response.xpath(xpath_id).getall()[-1]
        item_loader.add_value('daft_id', daft_id)

        item_id = re.findall(r'(?<=/)\d+', response.url)[-1]
        item_loader.add_value('item_id', item_id)

        url = response.url
        item_loader.add_value('url', url)

        xpath_name = '//h1[@data-testid="address"]/text()'
        try:
            name_info = response.xpath(xpath_name).get()
            if name_info:
                item_loader.add_value('name', name_info)
            else:
                item_loader.add_value('name', 'none')
        except:
            item_loader.add_value('name', 'none')

        xpath_price = '//div[@data-testid="title-block"]/div/p/span/text()'
        try:
            price_info = response.xpath(xpath_price).get()
            if price_info:
                item_loader.add_value('price', price_info)
            else:
                item_loader.add_value('price', 'none')
        except:
            item_loader.add_value('price', 'none')

        xpath_info = '//div[@data-testid="card-info"]/p/text()'
        try:
            info_list = response.xpath(xpath_info).getall()
            if info_list:
                info_string = ','.join(info_list)
                item_loader.add_value('info', info_string)
            else:
                item_loader.add_value('info', 'none')
        except:
            item_loader.add_value('info', 'none')

        xpath_overview_caracts = '//span[@class="PropertyPage__ListLabel-sc-14jmnho-10 ' \
                                 'ssSHo"]/text() '
        xpath_overview_values = '//ul[@class="PropertyPage__InfoSection-sc-14jmnho-7 ' \
                                'ertsXd"]/li/text() '
        try:
            caracts = response.xpath(xpath_overview_caracts).getall()
            values = response.xpath(xpath_overview_values).getall()
            for value in values:
                if value == ': ':
                    values.remove(value)
            overview_list = []
            for caract, value in zip(caracts, values):
                overview_elem = caract + ': ' + value
                overview_list.append(overview_elem)
            overview = ','.join(overview_list)
            item_loader.add_value('overview', overview)
        except:
            item_loader.add_value('overview', 'none')

        xpath_facilities = '//ul[' \
                           '@class="PropertyDetailsList__PropertyDetailsListContainer' \
                           '-sc-1cjwtjz-0 bnzQrB"]/li/text() '
        try:
            facilities_list = response.xpath(xpath_facilities).getall()
            facilities = ','.join(facilities_list)
            item_loader.add_value('facilities', facilities)
        except:
            item_loader.add_value('facilities', 'none')

        xpath_ber = '//div[@data-testid="ber"]/img/@alt'
        try:
            ber_info = response.xpath(xpath_ber).get()
            if ber_info:
                item_loader.add_value('ber', ber_info)
            else:
                item_loader.add_value('ber', 'none')
        except:
            item_loader.add_value('ber', 'none')

        xpath_entered_views = '//p[@class="Statistics__StyledLabel-sc-15tgae4-1 ' \
                              'pBjQg"]/text() '
        try:
            entered_views_info = response.xpath(xpath_entered_views).getall()
            if entered_views_info[0]:
                item_loader.add_value('entered_renewed', entered_views_info[0])
            else:
                item_loader.add_value('entered_renewed', 'none')
            if entered_views_info[1]:
                item_loader.add_value('views', entered_views_info[1])
            else:
                item_loader.add_value('views', 'none')
        except:
            item_loader.add_value('entered_renewed', 'none')
            item_loader.add_value('views', 'none')

        xpath_contact = '//div[@class="ContactPanel__Container-sc-18zt6u1-7 fCElaB"]/div[2]/p/text()'
        xpath_phone = '//div[@class="ContactPanel__Container-sc-18zt6u1-7 fCElaB"]/div[2]/a/@href'
        pattern_phone = r'tel:(\d+[\-\s]?\d+[\-\s]?\d+)'  # r'tel:(\d+[\-\s]?\d+)'
        # contact
        try:
            contact_info = response.xpath(xpath_contact).get()
            if contact_info:
                item_loader.add_value('contact', contact_info)
            else:
                item_loader.add_value('contact', 'none')
        except:
            item_loader.add_value('contact', 'none')
        # phone
        try:
            phone_info = response.xpath(xpath_phone).re_first(pattern_phone)
            if phone_info:
                item_loader.add_value('phone', phone_info)
            else:
                item_loader.add_value('phone', 'none')
        except:
            item_loader.add_value('phone', 'none')

        xpath_license = '//div[@class="PropertyPage__MainColumn-sc-14jmnho-1 hwqROB"]/div[5][1]'
        license_pattern = r'"licenceNumber":"(\d+)'
        try:
            license_info = response.xpath(xpath_license).re_first(license_pattern)
            if license_info:
                item_loader.add_value('psr_license_number', license_info)
            else:
                item_loader.add_value('psr_license_number', 'none')
        except:
            item_loader.add_value('psr_license_number', 'none')

        # for_rent = re.search(r'houses-to-let', response.url) # response.url probar sin request
        if '/apartment' in response.url:
            item_loader.add_value('type_house', 'apartment')
        elif '/house' in response.url:
            item_loader.add_value('type_house', 'house')
        elif '/share/' in response.url:
            item_loader.add_value('type_house', 'room')
        else:
            item_loader.add_value('type_house', 'none')

        xpath_energy = '//p[@data-testid="ber-epi"]/span[2]/text()'
        try:
            energy_info = response.xpath(xpath_energy).get()
            if energy_info:
                item_loader.add_value('energy', energy_info)
            else:
                item_loader.add_value('energy', 'none')
        except:
            item_loader.add_value('energy', 'none')

        xpath_coordinates = '//div[@class="NewButton__ButtonContainer-yem86a-4 dFKaNf button-container"]/a'
        pattern_coordinates = r'q=loc:(.+)" data-tracking="LaunchSatellite"'
        try:
            coordinates_info = response.xpath(xpath_coordinates).re_first(pattern_coordinates)
            if coordinates_info:
                item_loader.add_value('coordinates', coordinates_info)
            else:
                item_loader.add_value('coordinates', 'none')
        except:
            item_loader.add_value('coordinates', 'none')

        item_loader.add_value('scraping_date', scraping_date())

        if '/share/' in response.url:
            item_loader.add_value('type', 'share')
        else:
            item_loader.add_value('type', 'rent')

        yield item_loader.load_item()
