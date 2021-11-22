# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

from scrapy.item import Field, Item
from scrapy.loader.processors import MapCompose  # , TakeFirst
# import re
from w3lib.html import remove_tags, strip_html5_whitespace

from daftpy.daftdata import DaftItem

class DaftItemBuy(DaftItem):
    # The _init_ is inherited from parent
    pass


'''
# Declaring Item subclasses
class DaftItemBuy(Item):
    
    # Declaring fields to specify metadata for each field
    daft_id = Field()
    item_id = Field()
    url = Field()
    name = Field(
        input_processor=MapCompose(lambda x: x.strip()),
        # output_processor=TakeFirst()
    )
    price = Field(
        input_processor=MapCompose(lambda x: x.strip()),
        # output_processor=TakeFirst()
    )
    info = Field()
    sale_type = Field()
    floor_area = Field()
    contact = Field(
        input_processor=MapCompose(lambda x: x.strip()),
        # output_processor=TakeFirst()
    )  # [1]
    phone = Field(
        input_processor=MapCompose(lambda x: x.replace('\n', '').strip()),
        # output_processor=TakeFirst()
    )
    psr_license_number = Field()
    ber = Field(
        # input_processor=MapCompose(lambda x: x.replace('\n', '').strip())
    )
    entered_renewed = Field(
        # input_processor=MapCompose(remove_tags, strip_html5_whitespace)
    )
    views = Field(
        input_processor=MapCompose(remove_tags, strip_html5_whitespace)
    )
    type_house = Field(
        input_processor=MapCompose(remove_tags, strip_html5_whitespace)
    )
    energy = Field(
        input_processor=MapCompose(remove_tags, strip_html5_whitespace)
    )
    coordinates = Field(
        # input_processor=MapCompose(remove_tags, strip_html5_whitespace)
    )
    type = Field()
    scraping_date = Field() 
'''

class DaftItemRent(DaftItem):
    # The _init_ is inherited from parent
    overview = Field()
    facilities = Field()
    
    
'''
class DaftItemRent(Item):
    daft_id = Field()
    item_id = Field()
    url = Field()
    name = Field(
        input_processor=MapCompose(lambda x: x.strip()),
        # output_processor=TakeFirst()
    )
    price = Field(
        input_processor=MapCompose(lambda x: x.strip()),
        # output_processor=TakeFirst()
    )
    info = Field()
    overview = Field()
    facilities = Field()
    ber = Field(
        # input_processor=MapCompose(lambda x: x.replace('\n', '').strip())
    )
    entered_renewed = Field(
        # input_processor=MapCompose(remove_tags, strip_html5_whitespace)
    )
    views = Field(
        input_processor=MapCompose(remove_tags, strip_html5_whitespace)
    )
    contact = Field(
        input_processor=MapCompose(lambda x: x.strip()),
        # output_processor=TakeFirst()
    )  # [1]
    phone = Field(
        input_processor=MapCompose(lambda x: x.replace('\n', '').strip()),
        # output_processor=TakeFirst()
    )
    psr_license_number = Field()
    type_house = Field(
        input_processor=MapCompose(remove_tags, strip_html5_whitespace)
    )
    energy = Field(
        input_processor=MapCompose(remove_tags, strip_html5_whitespace)
    )
    coordinates = Field(
        # input_processor=MapCompose(remove_tags, strip_html5_whitespace)
    )
    type = Field()
    scraping_date = Field()
'''


'''
# Declaring Item subclasses
class DaftItemBuy(Item):
    
    # Declaring fields to specify metadata for each field
    daft_id = Field()
    item_id = Field()
    url = Field()
    name = Field(
        input_processor=MapCompose(lambda x: x.strip()),
        # output_processor=TakeFirst()
    )
    price = Field(
        input_processor=MapCompose(lambda x: x.strip()),
        # output_processor=TakeFirst()
    )
    info = Field()
    sale_type = Field()
    floor_area = Field()
    contact = Field(
        input_processor=MapCompose(lambda x: x.strip()),
        # output_processor=TakeFirst()
    )  # [1]
    phone = Field(
        input_processor=MapCompose(lambda x: x.replace('\n', '').strip()),
        # output_processor=TakeFirst()
    )
    psr_license_number = Field()
    ber = Field(
        # input_processor=MapCompose(lambda x: x.replace('\n', '').strip())
    )
    entered_renewed = Field(
        # input_processor=MapCompose(remove_tags, strip_html5_whitespace)
    )
    views = Field(
        input_processor=MapCompose(remove_tags, strip_html5_whitespace)
    )
    type_house = Field(
        input_processor=MapCompose(remove_tags, strip_html5_whitespace)
    )
    energy = Field(
        input_processor=MapCompose(remove_tags, strip_html5_whitespace)
    )
    coordinates = Field(
        # input_processor=MapCompose(remove_tags, strip_html5_whitespace)
    )
    type = Field()
    scraping_date = Field() 
'''