# Predict the weather 📈🌦 For Airport with Prophet

💬 (Boss): *alpha苏泽, I need a weather forecast for New York state*. 

&nbsp;&nbsp;&nbsp;↪️ (Me): *What aspect of the weather would you like to know about?*

&nbsp;&nbsp;&nbsp;↪️ (Boss): *Temperature and wind metrics would be great!*

💬 (Me): *What will this analysis be used for?*

&nbsp;&nbsp;&nbsp;↪️ (Boss): *We'd like to sell this as a product to Airports.*

💬 (Me): A minute please 😎.

---


<blockquote style='
    font-size: 16px;
    color: #171716;
    border-left: 4px solid #3793EF;
    padding-left: 16px;
    line-height: 1.6;
    margin: -10px 0 0 0px;
'>
The data source: <a href='https://cloud.google.com/bigquery/public-data#sample_tables'>google cloud weather dataset</a> from Google Big Query. 
</blockquote>


```python
import altair as alt
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
```


```python
def clean_data(df):
    df['date']=pd.to_datetime(df.date)
    df['max_gust']=df['max_gust'].astype(float).replace(999.9, np.nan) 
    df['max_temp']=df['max_temp'].astype(float).replace(9999.9, np.nan) 
    df['min_temp']=df['min_temp'].astype(float).replace(9999.9, np.nan) 
    return df
```

## Weather data


```python
weather = pd.read_csv("weather-station.csv")
weather
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>name</th>
      <th>min_temp</th>
      <th>max_temp</th>
      <th>max_gust</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-08-18</td>
      <td>EAST HAMPTON AIRPORT</td>
      <td>68.1</td>
      <td>69.8</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-08-17</td>
      <td>EAST HAMPTON AIRPORT</td>
      <td>67.7</td>
      <td>75.9</td>
      <td>18.1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-08-16</td>
      <td>EAST HAMPTON AIRPORT</td>
      <td>67.5</td>
      <td>68.9</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-08-15</td>
      <td>EAST HAMPTON AIRPORT</td>
      <td>71.2</td>
      <td>74.3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-08-14</td>
      <td>EAST HAMPTON AIRPORT</td>
      <td>75.5</td>
      <td>84.2</td>
      <td>18.1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1670</th>
      <td>2016-01-09</td>
      <td>EAST HAMPTON AIRPORT</td>
      <td>43.3</td>
      <td>48.2</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>1671</th>
      <td>2016-01-08</td>
      <td>EAST HAMPTON AIRPORT</td>
      <td>34.2</td>
      <td>46.4</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>1672</th>
      <td>2016-01-07</td>
      <td>EAST HAMPTON AIRPORT</td>
      <td>38.5</td>
      <td>45.5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1673</th>
      <td>2016-01-06</td>
      <td>EAST HAMPTON AIRPORT</td>
      <td>36.3</td>
      <td>41.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1674</th>
      <td>2016-01-02</td>
      <td>EAST HAMPTON AIRPORT</td>
      <td>37.1</td>
      <td>39.2</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>1675 rows × 5 columns</p>
</div>




```python
weather=clean_data(weather)
```

## Predict weather data

Select the weather feature to predict


```python
feature = 'min_temp'
```


```python
weather_subset=weather[['date', feature]]
weather_subset.columns=['ds', 'y']

m = Prophet(daily_seasonality=True)
m.fit(weather_subset)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
```

    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1


## Plot weather forecast


```python
my_plot=plot_plotly(m, forecast)

my_plot
```




## 👉🏼  👉🏼  👉🏼 For the next please open the notebook file in order to have good visualizatioon 👈🏻  👈🏻  👈🏻 

```python
