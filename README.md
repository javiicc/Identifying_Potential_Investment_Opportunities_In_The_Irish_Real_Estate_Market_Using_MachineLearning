# Identifying Potential Investment Opportunities In The Irish Real Estate Market Using Machine Learning

This is my final project of the Master in Data Science from KSchool. It consists of a machine learning app that predicts housing prices in the Ireland’s Real Estate Market. Once the prices are predicted they are compared with the actual prices in order to find potential investment opportunities. Bellow you can see the Dash application.  

![](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_Machine_Learnin/blob/main/investment-opportunities/notebooks/imgs/thefinalgif.gif)

You can find an in-depth explanation of the project in the [Wiki](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning/wiki) section.

The project is structured in three parts and each one correspond with a folder in this repo:

- **Data Collection**
  - Advertisements scraped from [daft.ie](https://www.daft.ie/) and complementary data scraped from [Geonames.org](http://www.geonames.org/postalcode-search.html?q=&country=IE) about [eircodes](https://www.eircode.ie/) and cities/towns. 
  - Folder: *data_colection*
- **Potential Investment Opportunities**
  - This part contains most of the work
    - Cleansing and Wrangling tasks 
    - Feature Engineering Geospatial Data
    - Exploratory Data Analysis
    - Data Modeling with Machine Learning Algorithms
   - Folder: *investment-opportunities*
- **Dash Application**
  - The final application which you can see above
  - Folder: *dashapp*

# Modules

# Data Collection

## Sources
- [daft.ie](https://www.daft.ie/) 
  - House advertisements. Example bellow.

![](https://raw.githubusercontent.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning/main/investment-opportunities/notebooks/imgs/ad_example.png)

- [Geonames.org](http://www.geonames.org/postalcode-search.html?q=&country=IE)
  - [*Routing Key*](https://www.eircode.ie/what-is-eircode#:~:text=The%20first%203%20characters%20of,may%20cross%20over%20county%20borders.) from [eircodes](https://www.eircode.ie/) and cities.  

![geonames_example]()

## Tools
- [Scrapy](https://scrapy.org/)
- [Requests](https://docs.python-requests.org/en/latest/) and [lxml](https://lxml.de/)
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)

The data scraped with Scrapy were saved in a SQLite3 database. The database will be provided via Google Drive.

# Investment Opportunities

This part was developed in notebooks and then it was relocated in a [Kedro](https://kedro.readthedocs.io/en/stable/01_introduction/01_introduction.html) project. Bellow you can see the structure of the project via notebooks and via Kedro. 
*You can find an in-depth explanation of the project in the [Wiki](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning/wiki).*

### Notebooks structure

![diagram]()

### Kedro Pipeline

![Kedro Pipeline](https://raw.githubusercontent.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_Machine_Learnin/main/investment-opportunities/notebooks/imgs/kedro-viz-final.png)

## Cleansing and Wrangling 

Each feature has been preprocessed in order to give it a properly format. 

**Methodology**

- Detecting the different cases or formats in which the data was and deciding what kind of task apply in it. Once the wrangling tasks were decided the next stept was building a function to do the tasks and finally to apply it.

**Example: Price column**

![Price Conclusions](https://raw.githubusercontent.com/javiicc/Identifyin_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_Machine_Learning/main/investment-opportunities/notebooks/imgs/price_conclusions.png)

```python
pd.DataFrame({'before': sale['price'], 
              'after': process_price(sale)['price']}).head(10)
```

![Price Column Before and After](https://raw.githubusercontent.com/javiicc/Identifyin_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_Machine_Learning/main/investment-opportunities/notebooks/imgs/price_column_before_and_after.png)

The outliers were handled in the *Data Analysis* notebook. However, when the prokect was relocated to a Kedro, they were handled in the [*data_cleansing*](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning/tree/main/investment-opportunities/src/investment_opportunities/pipelines/data_cleansing) pipeline which is between the [*data_processing*](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning/tree/main/investment-opportunities/src/investment_opportunities/pipelines/data_processing) pipeline (wrangling tasks) and the [*feature_engineering_geospatial_data*](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning/tree/main/investment-opportunities/src/investment_opportunities/pipelines/feature_engineering_geospatial_data) pipeline. You cal easily see the Kedro pipeline structure thanks to the [*Kedro-Viz*](https://kedro.readthedocs.io/en/stable/03_tutorial/06_visualise_pipeline.html) tool.

## Feature Engineering Geospatial Data

In order to obtain more information and being able to make a better analysis I decided to do feature engineering to get the `city/town` names and other information from the coordinates.

This thecnique is called Reverse Geocoding:
- Reverse-Geocoding is a process used to convert coordinates (latitude and longitude) to human-readable addresses.

### Tools

- [GeoPy](https://geopy.readthedocs.io/en/stable/)
- [Nominatim Geocoder](https://geopy.readthedocs.io/en/stable/#nominatim)

### Post Reverse-Geocoding

The result of the feature engineering process was the this:

![](https://raw.githubusercontent.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning/main/investment-opportunities/notebooks/imgs/missing_values4.png)

## Data Analysis

## Prices Prediction - Modeling


# Dash Application

# Conclusions

# Replicate The Project And In-Depth Explanation
