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

![geonames_example](https://raw.githubusercontent.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning/main/investment-opportunities/notebooks/imgs/geonames_example.png)

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

### Dealing With Missing Values

The column I was interested most in was the `city` one. However, it had a lot of missing values. To solve this I used the `postcode` column to find the names of the cities. I extracted the *Routing Key* from the `postcode` and matched it with the one in the data scraped from the [Geonames.org](http://www.geonames.org/postalcode-search.html?q=&country=IE). As the `Routing Key` in the data scraped from the [Geonames.org](http://www.geonames.org/postalcode-search.html?q=&country=IE) was associated with the place (city/town), it was easy make a new column named `place` containing that information. After this process I had a column named `place` with only 1,243 missing values, instead of 5,644 or 6,551. 

![](https://raw.githubusercontent.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning/main/investment-opportunities/notebooks/imgs/mv_5.png)

## Data Analysis

I did an EDA to try finding some useful insights that would help me with the modeling tasks. Bellow is the structure of my analysis and some insights I obtained.

### Ireland's Real Estate Market Analysis

![](https://raw.githubusercontent.com/javiicc/Identifyin_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_Machine_Learning/main/investment-opportunities/notebooks/imgs/number_ads_per_town.png)

As expected, Dublin is by far the place with more advertisements in Ireland, followed by Cork and Galway.

![](https://raw.githubusercontent.com/javiicc/Identifyin_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_Machine_Learning/main/investment-opportunities/notebooks/imgs/mean_price_per_city.png)

Above I put together a chart representing mean prices per city and another one representing mean m2 prices per city. 

**Insigths**
- *Places* are very importanta in order to predict prices.
- *Floor Area* could be a potentian predictor, since places with similar m2 prices have high differences in total price.

![](https://raw.githubusercontent.com/javiicc/Identifyin_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_Machine_Learning/main/investment-opportunities/notebooks/imgs/meanpricepercitybytypehouse2.png)

**Insigths**
- *Type House* could be a potential predictor as well. The difference in house prices and apartment prices are high but they m2 prices are similar.
- *Type House* could proxy a litle the variable *Floor Area*.
- Is interesting to see that some places as Cork or Galway have m2 apartment prices higher than house ones but houses have higher total prices.

### Dublin's Real Estate Market Analysis

I decided to analyse Dublin and Cork individually since they are the bigger cities. I show bellow some insights from the analysis of Dublin.

![](https://raw.githubusercontent.com/javiicc/Identifyin_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_Machine_Learning/main/investment-opportunities/notebooks/imgs/meanpriceperpostaldistrict2.png)

**Insigths**
- *Postal Districts* have a high impact in prices. They are represented in the *routing key*.
- Some postal districts have m2 prices really high but some of the smaller total prices, which is again a signal of the importance of *floor area* in prices.

![](https://raw.githubusercontent.com/javiicc/Identifyin_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_Machine_Learning/main/investment-opportunities/notebooks/imgs/meanpriceperpdandtypehouse2.png)

The insights from the above charts could be simiral to the Ireland's Market ones.

![](https://raw.githubusercontent.com/javiicc/Identifyin_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_Machine_Learning/main/investment-opportunities/notebooks/imgs/meanfloorarea2.png)


**Insigths**
- We can easily see visually the possible correlation between:
  - `type_house` and `price`
  - `floor_area` and `price`

### Distributions

The distribution shapes were heavy skewed to the right and had a lot of outliers.

![](https://raw.githubusercontent.com/javiicc/Identifyin_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_Machine_Learning/main/investment-opportunities/notebooks/imgs/distributions2.png)

To detect outliers I used the following two methods jointly. I considered a value as an outlier only when it was detected as one by the two methods.

- Percentile-Based Method
- Interquartile Range Method

After dropping the outliers the cutoff levels were the following:

![](https://raw.githubusercontent.com/javiicc/Identifyin_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_Machine_Learning/main/investment-opportunities/notebooks/imgs/cutoff_levels.png)

To check the efects of the data cleansing I did several things. One of them was to calculate some statistics before and after dropping the outliers. As it can be seen from the images bellow the process of dropping outliers transformed the feature's metrics much more similar to a Gaussian ones.

- Before:

![](https://raw.githubusercontent.com/javiicc/Identifyin_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_Machine_Learning/main/investment-opportunities/notebooks/imgs/statistics_before.png)

- After:

![](https://raw.githubusercontent.com/javiicc/Identifyin_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_Machine_Learning/main/investment-opportunities/notebooks/imgs/statistics_after.png)

I ploted the effects of the outliers elimination in charts like the bellow one. 
- Lines 1 and 2 show the distribution with outliers and without them respectively as well as the probability plots. The red area is that which contains the outliers.
- Lines 3 and 4 show the same that above ones but with a logarithmic transformation after the detection of outliers.
- Lines 5 and 6 show a Box-Cox transformation result.

![](https://raw.githubusercontent.com/javiicc/Identifyin_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_Machine_Learning/main/investment-opportunities/notebooks/imgs/check_transformations.png)

Some algorithms are built on the assumption of normality in the distribution so the graphs like the above one made me realise the potetial of some transformations could have in them.

### Realtionships and Feature Selection

I also study the relationships between variables and I used some feture selection methods in order to figure out the best variables to predict prices.

#### Scatter Plots

![](https://raw.githubusercontent.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning/main/investment-opportunities/notebooks/imgs/scatterplots.png)

Relationships:

- There is an increasing linear relationships between `price` and `floor_area`.
`bedroom` and `bathroom` seem to have a mild relationships with `price` and maybe a little stronger one with `floor_area`.
- `latitude` and `longitude` both show two rare patterns where `price` increases around two differents coordinates values. Actually these patterns have a logical explanation. The longitude with higher prices matches with Dublin and the other one that stands out from the rest matches with Cork and Galway, as both have similar coordinates. Something similar happens in the latitude plot. Dublin and Galway have similar latitudes and Cork is in the south. We can see this if we look a map as the scatter plot between `latitude` and `longitude` and compare it with the other plots.

#### Correlations - Pearson & Spearman

![](https://raw.githubusercontent.com/javiicc/Identifyin_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_Machine_Learning/main/investment-opportunities/notebooks/imgs/correlations.png)

Conclusions:

- As I concluded from the scatter plots there is a linear relationships between `price` and `floor_area`. 
- `longitude`, `bedroom`, and `bathroom` show interesting relationships with the `price`. We can note how `bedroom` and `bathroom` are also correlated with `floor_area`, which makes sense.
- There are no strong decreasing relationships between variables.

I conclude that it would be good looking for more data in order to find more variables with predictive capabilities.

#### Predictive Power Score

![](https://raw.githubusercontent.com/javiicc/Identifyin_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_Machine_Learning/main/investment-opportunities/notebooks/imgs/pps2.png)

#### Wrapper Methods

## Prices Prediction - Modeling


# Dash Application

# Conclusions

# Replicate The Project And In-Depth Explanation
