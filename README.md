# Identifying Potential Investment Opportunities In The Irish Real Estate Market Using Machine Learning

This is my final project of the Master in Data Science from KSchool. It consists of a machine learning app that predicts housing prices in the Ireland’s Real Estate Market. Once the prices are predicted they are compared with the actual prices in order to find potential investment opportunities. Bellow you can see the Dash application.  

![](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_Machine_Learnin/blob/main/investment-opportunities/notebooks/imgs/thefinalgif.gif)

You can find an in-depth explanation of the project in the [Wiki](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning/wiki) section.

The project is structured in three parts and each one correspond with a folder in this repo:

- Data Collection
  - Advertisements scraped from [daft.ie](https://www.daft.ie/) and complementary data scraped from [Geonames.org](http://www.geonames.org/postalcode-search.html?q=&country=IE) about [eircodes](https://www.eircode.ie/) and cities/towns. 
  - Folder: *data_colection*
- Potential Investment Opportunities
  - This part contains most of the work
    - Cleansing and Wrangling tasks 
    - Feature Engineering Geospatial Data
    - Exploratory Data Analysis
    - Data Modeling with Machine Learning Algorithms
   - Folder: *investment-opportunities*
- Dash Application:
  - The final application which you can see above
  - Folder: *dashapp*

# Data Collection

## Sources
- [daft.ie](https://www.daft.ie/) 
- [eircodes](https://www.eircode.ie/)

## Tools
- [Scrapy](https://scrapy.org/)
- [Requests](https://docs.python-requests.org/en/latest/) and [lxml](https://lxml.de/)
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)

The data scraped with Scrapy were saved in a SQLite3 database.

# Investment Opportunities

This part was developed in notebooks and then it was relocated in a [Kedro](https://kedro.readthedocs.io/en/stable/01_introduction/01_introduction.html) project.

## Notebooks structure

## Kedro Pipeline

![Kedro Pipeline](https://raw.githubusercontent.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_Machine_Learnin/main/investment-opportunities/notebooks/imgs/kedro-viz-final.png)
