#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 01:11:09 2021

@author: javier
"""

from bs4 import BeautifulSoup as bs
import requests


def get_urls_list():
    url = 'https://www.rent.ie/'
    
    soup = bs(requests.get(url).content, 'html.parser')
              
    urls = []
    for row in soup.find('select', attrs={'id': 'county', 
                                          'name': 's[cc_id]', 
                                          'class': 'wide'}).find_all('option'):
        city = row.get_text().lower()
        urls.append(city)
    return urls


urls_path = '/home/javier/Desktop/potential-investments/Identifyin_Potential_'\
            'Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_Machine_'\
            'Learning/data_collection/web_scraping/project_dir/urls.txt'
with open(urls_path, 'w') as file:
    for city in get_urls_list():
        file.write(f'https://www.rent.ie/houses-to-let/renting_{city}/' + '\n')
        file.write(f'https://www.rent.ie/rooms-to-rent/renting_{city}/' + '\n')
