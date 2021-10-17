import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re
from os import listdir
from os.path import isfile, join


def get_db(dbname, query='''SELECT * FROM buy;'''):

    database_path = f'data/{dbname}'
    connection = sqlite3.connect(database_path)
    #cursor = connection.cursor()

    daft = pd.read_sql_query(query, connection)
    connection.close()
    daft.drop(['contact', 'phone'], axis=1, inplace=True)

    sale = daft.copy()

    return sale


def db_dict():
    data_path = 'data/'
    # data_pattern = r'\d{4}-\d{2}-\d{2}.db'
    daily_db = [f for f in listdir(data_path) if isfile(join(data_path, f)) \
                and re.match(r'\d{4}-\d{2}-\d{2}.db', f)]
    # daily_db

    db_dict = {}
    for dbname in daily_db:

        db = get_db(dbname)
        key = dbname.split('.')[0]
        if key in db_dict:
            pass
        else:
            db_dict[key] = db

    return db_dict


def to_datetime(dictionary):

    for key in dictionary:
        dictionary[key]['entered_renewed'] = pd.to_datetime(dictionary[key]['entered_renewed'],
                                                            format='%d/%m/%Y')
        dictionary[key]['scraping_date'] = pd.to_datetime(dictionary[key]['scraping_date'],
                                                          format='%Y/%m/%d')


def sale_dict_daily(dictionary):
    sale_dict_daily = {}
    for key in dictionary:
        entered_renewed_day = dictionary[key]['entered_renewed'].dt.day
        scraping_day = dictionary[key]['scraping_date'].dt.day.value_counts().index[0]

        sale_dict_daily[key] = dictionary[key][entered_renewed_day == scraping_day].reset_index(drop=True)
    return sale_dict_daily




























