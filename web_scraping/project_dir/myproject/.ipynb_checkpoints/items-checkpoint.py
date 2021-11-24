# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

from scrapy.item import Field, Item
from scrapy.loader.processors import MapCompose  # , TakeFirst
# import re
from w3lib.html import remove_tags, strip_html5_whitespace

from daftpy.daftdata import DaftItem



class DaftItemBuy(DaftItem):
    """Class to contain fields in Buy ads/items.
    """
    
    # The _init_ is inherited from parent
    # All fields are inherit from DaftItem
    pass


class DaftItemRent(DaftItem):
    """Class to contain fields in Rent ads/items.
    """
    
    # The _init_ is inherited from parent
    # The rent items need two more fields
    overview = Field()
    facilities = Field()
    pass
    
    
    
    
  