# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem
from datetime import date, datetime

import sqlite3
#from .items import RentIeHouseItem, RentIeRoomItem, DaftItem, PropertyItem
from .items import DaftItemBuy, DaftItemRent


class DuplicatesPipeline:

    def __init__(self):
        #self.room_urls_seen_rent = []
        #self.house_urls_seen_rent = [] #set()
        self.item_urls_seen_daft = []
        #self.item_urls_seen_property = []

    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        if adapter['url'] in self.item_urls_seen_daft:
            raise DropItem(f"Duplicate item found: {item!r}")
        else:
            self.item_urls_seen_daft.append(adapter['url'])
            return item


class DatabasePipeline(object):
    
    def __init__(self):  
        self.create_connection()
        self.create_table()
        
    def create_connection(self):
        # Connect to a database
        today = date.today()
        data_path = '/home/javier/Desktop/TFM/Fraud_Detection_In_The_Irish_Rental_Market/data/{}.db'.format(str(today))
        #data_path = '/home/javier/Desktop/TFM/Fraud_Detection_In_The_Irish_Rental_Market/data/hola.db'
        self.conn = sqlite3.connect(data_path)
        # Create a cursor
        self.cursor = self.conn.cursor()
        
    def create_table(self):

        self.cursor.execute('''CREATE TABLE IF NOT EXISTS buy(
            daft_id TEXT,
            item_id TEXT, 
            url TEXT,
            name TEXT, 
            price TEXT, 
            info TEXT,
            sale_type TEXT,
            floor_area TEXT,
            contact TEXT, 
            phone TEXT, 
            psr TEXT,
            ber TEXT,
            entered_renewed TEXT,
            views TEXT,
            type_house TEXT,
            energy_performance_indicator TEXT,
            coordinates TEXT,
            type TEXT, 
            scraping_date TEXT
            );''')


        self.cursor.execute('''CREATE TABLE IF NOT EXISTS rent(
            daft_id TEXT,
            item_id TEXT,
            url TEXT,
            name TEXT, 
            price TEXT, 
            info TEXT,
            overview TEXT, 
            facilities TEXT, 
            ber TEXT, 
            entered_renewed TEXT,
            views TEXT,
            contact TEXT, 
            phone TEXT, 
            psr TEXT,
            type_house TEXT,
            energy_performance_indicator TEXT,
            coordinates TEXT,
            type TEXT, 
            scraping_date TEXT
            );''')



        # Datatypes:
        # NULL
        # INTEGER
        # REAL 
        # TEXT
        # BLOB -> images
        

        
    def process_item(self, item, spider):

        self.store_db(item)
        print('Pipeline: ' + item['name'][0])
        print('------------------------------------------')
        return item
         
        
    def store_db(self, item):

        if isinstance(item, DaftItemBuy):
            self.cursor.execute('''INSERT INTO buy VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);''',
                                (
                                    item['daft_id'][0],
                                    item['item_id'][0],
                                    item['url'][0],
                                    item['name'][0],
                                    item['price'][0],
                                    item['info'][0],
                                    item['sale_type'][0],
                                    item['floor_area'][0],
                                    item['contact'][0],
                                    item['phone'][0],
                                    item['psr_license_number'][0],
                                    item['ber'][0],
                                    item['entered_renewed'][0],
                                    item['views'][0],
                                    item['type_house'][0],
                                    item['energy'][0],
                                    item['coordinates'][0],
                                    item['type'][0],
                                    item['scraping_date'][0]))


        elif isinstance(item, DaftItemRent):
            self.cursor.execute('''INSERT INTO rent VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);''',
                                (
                                    item['daft_id'][0],
                                    item['item_id'][0],
                                    item['url'][0],
                                    item['name'][0],
                                    item['price'][0],
                                    item['info'][0],
                                    item['overview'][0],
                                    item['facilities'][0],
                                    item['ber'][0],
                                    item['entered_renewed'][0],
                                    item['views'][0],
                                    item['contact'][0],
                                    item['phone'][0],
                                    item['psr_license_number'][0],
                                    item['type_house'][0],
                                    item['energy'][0],
                                    item['coordinates'][0],
                                    item['type'][0],
                                    item['scraping_date'][0]))




        
        # Save (commit) the changes
        self.conn.commit()
        return item
    
    def close_spider(self, spider): 
        # We can also close the connection if we are done with it.
        # Just be sure any changes have been committed or they will be lost.
        #self.cursor.close() probarla, no hace falta creo
        self.conn.close()
