import requests
import lxml.html as lh
import pandas as pd


def geonames_dict():
    """Scrape the website from the url.

    Parameters
    ----------
    df :
        The dataframe to work with.
    dictionary :
        dictionary with location info and values with the same length
        than the DataFrame.

    Returns
    -------
    The DataFrame with location info added.
    """
    url = 'http://www.geonames.org/postalcode-search.html?q=&country=IE'
    page = requests.get(url)
    doc = lh.fromstring(page.content)
    tr_elements = doc.xpath('//tr')

    # Create empty dict
    col = {}
    # For each row, store each first element (header) and an empty list
    for i, t in enumerate(tr_elements[2]):
        key = t.text_content().lower()
        # print('%d: "%s"'%(i,name))
        col[key] = []
    col['place_coordinates'] = []

    # Fill dict
    # print(tr_elements[-1].text_content())
    for tr in tr_elements[3:]:

        if len(tr) == 7:

            for key, td in zip(col, tr):
                td = td.text_content()
                # print(td)
                col[key].append(td)

        elif len(tr) == 2:

            td = tr[-1].text_content()
            # print(td)
            col['place_coordinates'].append(td)

    del col['']
    del col['country']
    del col['admin2']
    del col['admin3']

    return col


geonames_df = pd.DataFrame(geonames_dict())

geonames_df.to_csv('/home/javier/Desktop/potential-investments/Identifyin_Potential_'
                   'Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_'
                   'Machine_Learning/investment-opportunities/data/01_raw/geonames.csv',
                   sep=',')
