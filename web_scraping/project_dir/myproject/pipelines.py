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
    """Class to filter duplicate items.
    
    Attributes
    ----------
    item_urls_seen_daft : list
        Internal attribute to hold the world population.
    """

    def __init__(self):
        self.item_urls_seen_daft = []  #set()

    def process_item(self, item, spider):
        """Filter items.
        
        Parameters
        ----------
        item : item object
            The scraped item. Item to be filtered.
        spider : Spider object
            The spider which scraped the item
        
        Returns
        -------
        Item not duplicated.
        """
        # ItemAdapter: Wrapper class to interact with data container objects. It provides a 
        # common interface to extract and set data without having to take the object’s type into 
        # account.
        adapter = ItemAdapter(item)
        if adapter['url'] in self.item_urls_seen_daft:
            raise DropItem(f"Duplicate item found: {item!r}")
        else:
            self.item_urls_seen_daft.append(adapter['url'])
            return item


class DatabasePipeline(object):
    """Class to connect or create the database, create the tables if they are not, 
    store the database, and close the connection.
    
    Attributes
    ----------
    create_connection : 
        
    create_table :
        
    """
    def __init__(self):  
        self.create_connection()
        self.create_table()
        
    def create_connection(self):
        """Create the conection to the database and the cursor.
        """
        today = date.today()
        # Path to the database
        data_path = '/home/javier/Desktop/potential-investments/A_Study_Of_Potential_Investment_'\
                    'Opportunities_In_The_Irish_Real_Estate_Market_Using_Machine_Learning/'\
                    'data/{}.db'.format(str(today))
        # Connect to a database
        self.conn = sqlite3.connect(data_path)
        # Create a cursor
        self.cursor = self.conn.cursor()
        
    def create_table(self):
        """Create two tables in the database. One for sale houses and one for rent houses.
        """
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
            scraping_date TEXT,
            description TEXT
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
            scraping_date TEXT,
            description TEXT
            );''')
        # Datatypes:
        # NULL
        # INTEGER
        # REAL 
        # TEXT
        # BLOB -> images
        
    def process_item(self, item, spider):
        """Store the item in the database and print its name.
        
        Parameters
        ----------
        item :
        
        spider :
        
        """
        self.store_db(item)
        print('Pipeline: ' + item['name'][0])
        print('------------------------------------------')
        return item
         
    def store_db(self, item):
        """Store a given item in the database and save the changes.
        
        Parameters
        ----------
        item :
            
        Returns
        -------
        item :
        """
        if isinstance(item, DaftItemBuy):
            self.cursor.execute('''INSERT INTO buy VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);''',
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
                                    item['scraping_date'][0], 
                                    item['description'][0]
                                ))

        elif isinstance(item, DaftItemRent):
            self.cursor.execute('''INSERT INTO rent VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);''',
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
                                    item['scraping_date'][0],
                                    item['description'][0]
                                ))

        # Save (commit) the changes
        self.conn.commit()
        return item
    
    def close_spider(self, spider): 
        """Close the connection.
        
        Parameters
        ----------
        spider :
        """
        # We can also close the connection if we are done with it.
        # Just be sure any changes have been committed or they will be lost.
        # self.cursor.close() probarla, no hace falta creo
        self.conn.close()
