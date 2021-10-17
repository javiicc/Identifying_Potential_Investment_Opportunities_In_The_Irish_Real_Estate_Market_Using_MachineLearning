# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

from scrapy.item import Field, Item
from scrapy.loader.processors import MapCompose  # , TakeFirst
# import re
from w3lib.html import remove_tags, strip_html5_whitespace


class DaftItemBuy(Item):
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


class RentIeHouseItem(Item):
    house_id = Field()
    url = Field()
    name = Field(
        input_processor=MapCompose(lambda x: x.strip()),
        # output_processor=TakeFirst()
    )
    price = Field(
        input_processor=MapCompose(lambda x: x.strip()),
        # output_processor=TakeFirst()
    )
    rooms = Field(
        input_processor=MapCompose(lambda x: x.replace('\n', '').strip())
    )
    ber = Field(
        # input_processor=MapCompose(lambda x: x.replace('\n', '').strip())
    )
    contact = Field(
        input_processor=MapCompose(lambda x: x.strip()),
        # output_processor=TakeFirst()
    )  # [1]
    phone = Field(
        input_processor=MapCompose(lambda x: x.replace('\n', '').strip()),
        # output_processor=TakeFirst()
    )  # [2],   lambda x: re.sub(r'\n', '', table_processing(x))
    letting_agent = Field(
        input_processor=MapCompose(lambda x: x.replace('\n', '').strip()),
    )
    psr_licence_number = Field(
        input_processor=MapCompose(lambda x: x.strip()),
    )
    key_features = Field(
        # input_processor=MapCompose(lambda x: x.strip()),
    )
    availability = Field(
        input_processor=MapCompose(remove_tags, strip_html5_whitespace),
    )
    lease = Field(
        input_processor=MapCompose(remove_tags, strip_html5_whitespace),
    )
    energy = Field(
        input_processor=MapCompose(remove_tags, strip_html5_whitespace)
    )
    entered = Field(
        input_processor=MapCompose(remove_tags, strip_html5_whitespace)
    )
    type_rent = Field(
        input_processor=MapCompose(remove_tags, strip_html5_whitespace)
    )
    latitude = Field(
        # input_processor=MapCompose(remove_tags, strip_html5_whitespace)
    )
    longitude = Field(
        # input_processor=MapCompose(remove_tags, strip_html5_whitespace)
    )
    scraping_date = Field()


class RentIeRoomItem(Item):
    room_id = Field()
    url = Field()
    name = Field(
        input_processor=MapCompose(lambda x: x.strip()),
        # output_processor=TakeFirst()
    )
    price = Field(
        input_processor=MapCompose(lambda x: x.strip()),
        # output_processor=TakeFirst()
    )
    bed = Field(
        input_processor=MapCompose(lambda x: x.replace('\n', '').strip())
    )
    availability_time = Field(
        input_processor=MapCompose(remove_tags, strip_html5_whitespace),
    )
    ber = Field(
        # input_processor=MapCompose(lambda x: x.replace('\n', '').strip())
    )
    contact = Field(
        input_processor=MapCompose(lambda x: x.strip()),
        # output_processor=TakeFirst()
    )  # [1]
    phone = Field(
        input_processor=MapCompose(lambda x: x.replace('\n', '').strip()),
        # output_processor=TakeFirst()
    )  # [2],   lambda x: re.sub(r'\n', '', table_processing(x))
    letting_agent = Field(
        input_processor=MapCompose(lambda x: x.replace('\n', '').strip()),
    )
    psr_licence_number = Field(
        input_processor=MapCompose(lambda x: x.strip()),
    )
    key_features = Field(
        # input_processor=MapCompose(lambda x: x.strip()),
    )
    availability = Field(
        input_processor=MapCompose(remove_tags, strip_html5_whitespace),
    )
    lease = Field(
        input_processor=MapCompose(remove_tags, strip_html5_whitespace),
    )
    energy = Field(
        input_processor=MapCompose(remove_tags, strip_html5_whitespace)
    )
    entered = Field(
        input_processor=MapCompose(remove_tags, strip_html5_whitespace)
    )
    type_rent = Field(
        input_processor=MapCompose(remove_tags, strip_html5_whitespace)
    )
    latitude = Field(
        # input_processor=MapCompose(remove_tags, strip_html5_whitespace)
    )
    longitude = Field(
        # input_processor=MapCompose(remove_tags, strip_html5_whitespace)
    )
    scraping_date = Field()


class PropertyItem(Item):  # houses and apartments
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
    description = Field(
        input_processor=MapCompose(lambda x: x.replace('\n', '').replace(' ', '')),
    )
    ber = Field(
        # input_processor=MapCompose(lambda x: x.replace('\n', '').strip())
    )
    features = Field(
        input_processor=MapCompose(lambda x: x.replace('\n', '').replace(' ', '')),
    )

    energy = Field(
        input_processor=MapCompose(remove_tags, strip_html5_whitespace)
    )
    last_updated = Field(
        input_processor=MapCompose(lambda x: x.replace('\n', '').strip())
    )
    # contact = Field(
    #   input_processor=MapCompose(lambda x: x.strip()),
    #  #output_processor=TakeFirst()
    #   ) #[1]
    # phone = Field(
    #   input_processor=MapCompose(lambda x: x.replace('\n', '').strip()),
    #  #output_processor=TakeFirst()
    # )
    coordinates = Field(
        # input_processor=MapCompose(remove_tags, strip_html5_whitespace)
    )
