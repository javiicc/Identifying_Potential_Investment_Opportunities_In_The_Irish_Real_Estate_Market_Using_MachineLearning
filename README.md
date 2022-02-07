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

# Table of Contents
1. [Data Collection](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning/blob/main/README.md#data-collection)

  - [Sources](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#sources)
  - [Tools](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#tools)

2. [Investment Opportunities](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#investment-opportunities)

  - [Notebooks structure](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#notebooks-structure)
  - [Kedro Pipeline](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#kedro-pipeline)
  - [Cleansing and Wrangling](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#cleansing-and-wrangling)
  - [Feature Engineering Geospatial Data](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#feature-engineering-geospatial-data)
    - [Tools](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#tools-1)
    - [Post Reverse-Geocoding](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#post-reverse-geocoding)
    - [Dealing With Missing Values](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#dealing-with-missing-values)
  - [Data Analysis](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#data-analysis)
    - [Ireland's Real Estate Market Analysis](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#irelands-real-estate-market-analysis)
    - [Dublin's Real Estate Market Analysis](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#dublins-real-estate-market-analysis)
    - [Distributions](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#distributions)
    - [Realtionships and Feature Selection](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#realtionships-and-feature-selection)
      - [Scatter Plots](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#scatter-plots)
      - [Correlations - Pearson & Spearman](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#correlations---pearson--spearman)
      - [Predictive Power Score](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#predictive-power-score)
    - [Conclusions](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#conclusions)
  - [Prices Prediction - Modeling](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#prices-prediction---modeling) 
    - [Models](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#models)
    - [Conclusions and Model Selection](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#conclusions-and-model-selection)
3. [Conclusions and Future Stepts](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#conclusions-and-future-stepts)
  - [Best Models](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#best-models)
  - [Worst Models](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#worst-models)
  - [Polynomial Regression Degree 4](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#polynomial-regression-degree-4)
  - [Price Understimation](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#price-understimation)
  - [Future Steps](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#future-steps)

4. [Dash Application](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#dash-application)

5. [Replicate The Project and In-Depth Explanation](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#replicate-the-project-and-in-depth-explanation)

6. [Learned Lessons](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning#learned-lessons)

7. [Go to Wiki](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning/wiki) 

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

![diagram](https://raw.githubusercontent.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning/main/investment-opportunities/notebooks/imgs/diagram.png)

### Kedro Pipeline

![Kedro Pipeline](https://raw.githubusercontent.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning/main/investment-opportunities/notebooks/imgs/last_mg_1.png)

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

- `city_district`, `code`, `place`, and `postcode` have some predictive power to the `price`. If we pay attention we can realize that all of them are **location** variables, which makes sense. Moreover, `code` and `place` are exactly the same but with different characters as place was extracted from `code` just exchanging the Eircode for the corresponding city or town.
- `cities` has some predictive power but is the same that `place` without differenciating Dublin postal districts.
- `psr` has some predictive power and it also makes sense because the psr identify the seller. It is reasonable to think that a particular seller will tend to offer houses in the same area or city or in close cities instead of the same number of houses in each city. So I think we could understand the `psr` as something similar to another location variable.

See the whole analysis in the [Data Analysis](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning/blob/main/investment-opportunities/notebooks/Data_Analysis.ipynb) notebook!

#### Wrapper Methods

The bellow methods were used:

- Recursive Feature Elimination
  - Linear Regression Scores
  - Decision Tree Regression Scores 
- Forward Feature Selection
  - Linear Regression Scores
  - Decision Tree Regression Scores
- Backward Feature Elimination
  - Linear Regression Scores
  - Decision Tree Regression Scores

**Conclusion**

Having into account the three methods and the fact that `psr` could be similar to `place` or a location feature the five variables I choosed were the following ones:

- `floor_area`
- `latitude`
- `longitude`
- `bedroom`
- `bathroom`

### Conclusions

After the full analysis made I selected the following variables as predictors in order to predict the houses prices:

- `floor_area`
- `latitude`
- `longitude`
- `bedroom`
- `bathroom`
- `place`
- `type_house`

## Prices Prediction - Modeling

### Models

To see the **transformations** applied to the variables and the **missing values traetment** as well as an in-depth explanation go to the [Prices Prediction](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning/blob/main/investment-opportunities/notebooks/Price_Prediction.ipynb) notebook or to the [Wiki](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning/wiki/3.5.-Price-Prediction)

The graphs bellow shows the models tested and the metrics obtained for each model.

![002](https://raw.githubusercontent.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_Machine_Learnin/main/investment-opportunities/notebooks/imgs/010.png)

As you can see the model with better performance is the Voting Regressor 2. The voting regressor averages the individual predictions of the learners to form a final prediction. The learners which make up the voting regressor 2 are the following:
- Voting Regressor Basic Algorithms (it is provided with other three learners)
  - Polynomial Regression (degree 4)
  - K-Nearest Neighbors Regressor
  - Decision Tree Regressor
- Random Forest Regressor 
- XGboost

The Voting Regressor 2 is able to improve the Baseline Model by 83,000€! Which mean a 56% less MAE. It also improve the second best model (Voting Regressor Basic Algorithms) by 1,500€ which mean only a 3%. My selected model was the Voting Regressor 2, but if you are concerned about the complexity of the models, you could choose the Polynomial Regresion since its performance is almost the same.

|Improvement respect Baseline Model (MAE)|Improvemenet respect Best Last Model (MAE)|
|:---:|:---:|
|83,320€|1,732€|
|56%|3%|

### Conclusions and Model Selection

- The best models obtain *coefficient of determination* around 0.8 so those models are able to explain an 80% of the variability in prices. In *Mean Absolute Error* terms this is an error between 66,000€ and 70,000€ aproximately in test set. This is not a bad score taking into account the low number of features we are worked with and the mean prices in Ireland. Also it is improving the *Baseline Model* by around 80,000€ so I am happy with that since the mean prices are so high in Ireland.

- One surprising thing is that the Polynomial Regression has quite similar metrics than the XGBoost or the Random Forest. Despite the fact that I choosed the Voting Regressor 2 as the final model because it is the best one, it would make sense to use a much simpler model as the Polinomial Regressor if you are concerned about model's complexity since it is very good as well as simple.

- Most of models tend to understimate houses prices which actual prices are over a million euros. Despite most houses are in the range price under a million this is still worrying. Our models are able to learn the relationship between preditor variables and the response variable better when the actual price of the house is under a million. That could be due to a lack of samples of expensive houses or due to a lack of predictors that can explain better the relationship with the price when it is too much high. Some predictors that perhaps explain that expensive prices could be the house year of construction, the average anual income of residents in areas with expensive prices, etc. 

- Another important aspect is the type of error that is less dangerous. Our model will be used to find potential investment opportunities which means that if the model understimates a house price, that house would be less interesting to us as investemnt opportunity, discouraging us to invest money in that operation. That means that we would not win money but we would not loss it either. However, if the model tends to overstimate the price of a house it would be encouraging us to invest in that house so we could buy an asset which actual value is lower than its actual price. That means that we could loss money. So we can conclude that a model that tends to understimate asset prices is less dangerous to us that one that tends to overstimate them.

# Conclusions and Future Stepts

## Best Models

|Model|MAE|R²|
|:---:|:---:|:---:|
|Voting Regressor 2|66,369€|0.80|
|Stacking Model|66,729€|0.80|

- The Voting Regressor 2 is able to improve the Baseline Model by 83,000€, which mean a 56% less MAE. 
- Voting Regressor Basic Algorithms (it is provided with other three learners)
  - Polynomial Regression (degree 4)
  - K-Nearest Neighbors Regressor
  - Decision Tree Regressor
- Random Forest Regressor 
- XGboost

## Worst Models

|Model|MAE|R²|
|:---:|:---:|:---:|
|Baselime Model|149,689€|0.23|
|Decision Tree Regressor|81,283€|0.71|
|K-Nearest Neighbors|77,802€|0.73|
|Linear Regression|77,092€|0.74|

## Polynomial Regression Degree 4 

|Model|MAE|R²|
|:---:|:---:|:---:|
|Polynomial Regression|70,216€|0.78|

The Polynomial Regresion with degree 4 has a pretty good performance and is simplest than the Voting Regressor 2.

## Price Understimation

- Most of models tend to understimate houses prices which actual prices are over a million euros. The models are able to learn the relationship between preditor variables and the response variable better when the actual price of the house is under a million euros. 

- This could be solved collecting more advertisements or finding new predictors variables capable of better explaining the higher prices.

- Since we are using the model to find potential investment opportunities, having a model that tend to understimate prices is less dangerous than one that tends to overstimate them. 

## Future Steps

I like thinking about this project as a prototype for a future data product. New ideas come to my mind quite often to improve the work but they go beyong the scope of my project for now. Some of them are the following:

- Collecting more data. I am currently learning about databases, Apache Airflow, and SQL to be able to make ETL pipelines and schedule all the process. I think it would be useful to collect all new ads from [daft.ie](https://www.daft.ie/) daily and save them in a database to train the models with more data, maybe a whole month.
- Obtaining more variables. There area a lot of variables that could be used as predictor variables.
  - Distance from houses to diferent points of interest.
  - Weather data in each place.
  - Income level per capita and area or city.
  - Demographic data.
- Trying new models and maybe deep learning.
- Building models for houses with prices over a million euros to check whether that improve the metrics. 
- Improving the frontend.

# Dash Application

The Dash Application is showed at the beggining and you can see the code [here](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning/blob/main/dashapp/app.py).

Tha application has been deployed to Heroku and you can visit it here:

https://ireland-dashboard-houses.herokuapp.com/

* Unfortunately the Dash application does not display properly when the screen is too small. This is something that I need to improve in the future.

# Replicate The Project and In-Depth Explanation

To run the project follow the instructions in the [wiki](https://github.com/javiicc/Identifying_Potential_Investment_Opportunities_In_The_Irish_Real_Estate_Market_Using_MachineLearning/wiki/6.-Run-the-Project)
