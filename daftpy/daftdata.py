import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re
from os import listdir
from os.path import isfile, join

from datetime import timedelta


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


def drop_renewed(old_data, new_data):
    print(f'Shape before dropping: {new_data.shape}')
    for url in new_data['url']:

        condition_1 = old_data['url'].str.contains(url).sum() != 0
        condition_2 = (new_data.loc[
                           new_data['url'].str.contains(url), ['url', 'price']].values ==
                       old_data.loc[old_data['url'].str.contains(url), ['url',
                                                                        'price']].values).all()  # axis=1

        if condition_1 and condition_2:
            index_to_drop = new_data[new_data['url'].str.contains(url)].index[0]
            new_data.drop(index=[index_to_drop], inplace=True)
    print(f'Shape after dropping: {new_data.shape}')
    print('-' * 10)
    return new_data


def concatenate_dropping_renewed(initial_key, dictionary):

    full_data = dictionary[initial_key]
    print(f'Initial shape: {full_data.shape}')
    print('-'*30)
    dictionary.pop('2021-09-25')

    for i, key in enumerate(dictionary):
        #if key == '2021-09-29': ####
         #   break
        print(f'Key: {key}')
        data_to_concat = drop_renewed(full_data, dictionary[key])
        full_data = pd.concat([data_to_concat, full_data], axis=0)
        print(f'Shape after concatenation {1}: {full_data.shape}')
        print('-'*20)
    print(f'Final shape: {full_data.shape}')
    return full_data






















