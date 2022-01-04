#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 15:58:45 2021

@author: javier
"""

from bs4 import BeautifulSoup as bs
import requests
import os.path as path


def get_ip_list():
    url = 'https://free-proxy-list.net/'
    
    soup = bs(requests.get(url).content, 'html.parser')
              
    proxies = []
    for row in soup.find('table', attrs={'class': 'table table-striped table-bordered'}
                         ).find_all('tr')[1:]:
        tds = row.find_all('td')
        try:
            ip = tds[0].text.strip()
            port = tds[1].text.strip()
            proxies.append(str(ip) + ':' + str(port))
        except IndexError:
            continue
        
    return proxies


ip_list_path = path.abspath(path.join('IP_list.py', '../proxies.txt'))

with open(ip_list_path, 'w') as file:
    for proxy in get_ip_list():
        file.write(proxy + '\n')

