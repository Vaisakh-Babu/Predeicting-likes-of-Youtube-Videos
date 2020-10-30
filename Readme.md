
# Predicting likes of Youtube videos

**By Vaisakh Babu**

There is no need for an introduction to Youtube. It is one of the most popular video-sharing platform. Youtubers earn money from their videos. Here we have details on videos along with some features. Our objective is to accurately predict the number of likes for each video using the set of input variables. 

> **Motivation**: Knowing what all are the important features while uploading a video, youtubers can increase their number of views/likes of the video and hence increase their earnings.

## A quick look into our dataset


```python
# Importing Libraries
import pandas as pd
import warnings
import numpy as np
warnings.simplefilter('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
```


```python
# Loading the dataset
Dataframe = pd.read_csv('train.csv')
Dataframe.head(2)
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
      <th>video_id</th>
      <th>title</th>
      <th>channel_title</th>
      <th>category_id</th>
      <th>publish_date</th>
      <th>tags</th>
      <th>views</th>
      <th>dislikes</th>
      <th>comment_count</th>
      <th>description</th>
      <th>country_code</th>
      <th>likes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>53364</td>
      <td>Alif Allah Aur Insaan Episode 34 HUM TV Drama ...</td>
      <td>HUM TV</td>
      <td>24.0</td>
      <td>2017-12-12</td>
      <td>HUM|"TV"|"Alif Allah Aur Insaan"|"Episode 34"|...</td>
      <td>351430.0</td>
      <td>298.0</td>
      <td>900.0</td>
      <td>Alif Allah Aur Insaan Episode 34 Full - 12 Dec...</td>
      <td>CA</td>
      <td>2351.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>51040</td>
      <td>It's Showtime Miss Q &amp; A: Bela gets jealous of...</td>
      <td>ABS-CBN Entertainment</td>
      <td>24.0</td>
      <td>2018-03-08</td>
      <td>ABS-CBN Entertainment|"ABS-CBN"|"ABS-CBN Onlin...</td>
      <td>461508.0</td>
      <td>74.0</td>
      <td>314.0</td>
      <td>Vice Ganda notices Bela Padilla's sudden chang...</td>
      <td>CA</td>
      <td>3264.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
Dataframe.shape
```




    (26061, 12)



**Our dataset has 26061 records and 12 columns.**

Lets see the columns in our dataset.


```python
Dataframe.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 26061 entries, 0 to 26060
    Data columns (total 12 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   video_id       26061 non-null  int64  
     1   title          26061 non-null  object 
     2   channel_title  26061 non-null  object 
     3   category_id    26061 non-null  float64
     4   publish_date   26061 non-null  object 
     5   tags           26061 non-null  object 
     6   views          26061 non-null  float64
     7   dislikes       26061 non-null  float64
     8   comment_count  26061 non-null  float64
     9   description    26061 non-null  object 
     10  country_code   26061 non-null  object 
     11  likes          26061 non-null  float64
    dtypes: float64(5), int64(1), object(6)
    memory usage: 2.4+ MB


**In our dataset first column is ID column and the last column is the target column. The remaining 10 columns are the features that we will use for our prediction.**

**In those 10 features, we can see that there are different kind of features. Numerical, categorical, textual and temporal features.**


```python
COL = Dataframe.columns.tolist()

# ID column is the first column with the name 'video_id'
ID_COL = 'video_id'

# Target column is the last column, named 'likes'
TARGET_COL = 'likes'

# Numerical columns are : 'views','dislikes','comment_count'
NUM_COL = ['views','dislikes','comment_count']

# Categorical columns are : 'channel_title','category_id','country_code'
CAT_COL = ['channel_title','category_id','country_code']

# Textual columns are : 'title','tags','description'
TEXT_COL = ['title','tags','description']

# Temporal column is : 'publish_date'
TIME_COL = 'publish_date'
```

**We don't have any null values in our dataset. Now we can check the number of unique values in the columns**


```python
Dataframe.nunique()
```




    video_id         26061
    title            26005
    channel_title     5764
    category_id         17
    publish_date       348
    tags             21462
    views            25338
    dislikes          2633
    comment_count     4993
    description      23426
    country_code         4
    likes            12134
    dtype: int64



## Hypotheses Generation

Now we know what are the columns in our dataset .We have 1 ID column, 1 target column, 3 numerical columns, 3 categorical columns, 3 textual columns and 1 temporal column. Now we will make some hypotheses

1. **Videos with more views get more likes**
2. **Videos with more comments get more likes**
3. **Videos with less dislikes get more likes**
4. **Descriptive videos get more number of likes**
5. **Channel affects the number of likes**
6. **Category of the video affects the number of likes**
7. **The country of origin affects the number of likes**
8. **People post more videos on weekends than weekdays**

## Exploratory data analysis

### Target Distribution

We have a regression problem in our hand. So, it is important to look into the target distribution. In our case it is the likes column.


```python
Dataframe[TARGET_COL].plot(kind = 'density', title = 'Likes Distribution', fontsize=8, figsize=(8, 4));
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_16_0.png)


**Right Skewed target distribution:**

We can see that the target distribution is highly skewed. In fact the distribution is **right skewed.**

We have to treat the data before building our model. Because, skewed data can cause less accurate prediction. That is, in this case we have less number of videos having a large number of likes. Therefore, our model won't be able to predict the videos that might having a large number of likes if we use this right skewed data.


```python
Dataframe[TARGET_COL].plot(kind='box', vert = False, fontsize=8, figsize=(12, 3));
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_18_0.png)


The skewness is more evident form the boxplot. We can perform statistical test and skewness calculation for providing edidence for this. We can use **Shapiro–Wilk test** for testing the normal distribution.

In Shapiro–Wilk test, the null hypothesis is, the data is normally distributed and the significance level is 5%. Therefore, a p-value less than 0.05 can reject the null hypothesis. However if the p-value is greater than 0.05, then we cannot reject the null hypothesis.  


```python
from scipy.stats import shapiro
data = Dataframe.likes
test = shapiro(data)
print('The p-value for the test is : {}'.format(test[1]))
```

    The p-value for the test is : 0.0


**The p-value is less than the significant level. Therefor we can reject the null hypothesis.**


**Checking the skewness:**


```python
Dataframe.likes.skew()
```




    26.36605568219861



When the value of the skewness is positive, the tail of the distribution is longer towards the right hand side of the curve.

**We have to perform tarnsformation to our data in order to make the target distribution to be normal.**

**The distribution after log transformation to the data is :**


```python
np.log1p(Dataframe[TARGET_COL]).plot(kind = 'density', title = 'Likes Distribution', fontsize=8, figsize=(8, 4));
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_26_0.png)



```python
np.log1p(Dataframe[TARGET_COL]).plot(kind='box', vert = False, fontsize=8, figsize=(12, 3),title='Likes');
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_27_0.png)



```python
#Applying logerithmic transfomrations to target column of the dataset
Dataframe[TARGET_COL] = np.log1p(Dataframe[TARGET_COL])
```

### Unique values in each columns

The number of unique values in the dataframe are:


```python
Dataframe.nunique()
```




    video_id         26061
    title            26005
    channel_title     5764
    category_id         17
    publish_date       348
    tags             21462
    views            25338
    dislikes          2633
    comment_count     4993
    description      23426
    country_code         4
    likes            12134
    dtype: int64



Let's check the number of nuique values in our categorical columns


```python
Dataframe[CAT_COL].nunique()
```




    channel_title    5764
    category_id        17
    country_code        4
    dtype: int64




```python
set(Dataframe.country_code)
```




    {'CA', 'GB', 'IN', 'US'}



**In the column country_code, we have 4 unique values.**

|Code| Country |
|----|---------|
| CA | Canada  |
| GB | Britain |
| IN | India   |
| US | America |

**There are 5764 channels in the datasets and the videos are belongs to 17 categories**

## Univariate Analysis

We will analyse each features individually in univariate analysis.

1. **Numerical features**


```python
Dataframe[NUM_COL].skew()
```




    views            41.679872
    dislikes         35.456527
    comment_count    33.895410
    dtype: float64



Our numerical columns are also skewed like the target variable.

**i) views**


```python
# Function for plotting boxplot and kde plot
def plot1(df,title=' '):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    _ = df.plot(kind = 'box', ax=axes[0], vert=False, title = (title+' - boxplot').upper())
    _ = df.plot.kde(ax=axes[1], title = (title+' - density plot').upper())
    fig.show()
```


```python
plot1(Dataframe[NUM_COL[0]],NUM_COL[0])
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_41_0.png)


Let's tranform the data and see the distribution.


```python
plot1(np.log1p(Dataframe[NUM_COL[0]]))
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_43_0.png)


**ii) dislikes**


```python
plot1(Dataframe[NUM_COL[1]],NUM_COL[1])
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_45_0.png)


Applying logerithmic transformation:


```python
plot1(np.log1p(Dataframe[NUM_COL[1]]),NUM_COL[1])
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_47_0.png)


**iii) comment_count**


```python
plot1(Dataframe[NUM_COL[2]],NUM_COL[2])
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_49_0.png)


Applying logerithmic transformation:


```python
plot1(np.log1p(Dataframe[NUM_COL[2]]),NUM_COL[2])
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_51_0.png)



```python
#Applying logerithmic transfomrations to the numerical colums of the dataset
for i in NUM_COL:
    Dataframe[i] = np.log1p(Dataframe[i])
```

2. **Categorical features**


```python
Dataframe[CAT_COL].nunique()
```




    channel_title    5764
    category_id        17
    country_code        4
    dtype: int64



We can see the number of videos in each categorical columns:


```python
fig, axes = plt.subplots(1, 2, figsize=(24, 8))
_ = Dataframe['country_code'].value_counts().plot(kind='pie', ax=axes[0], title='Country', autopct='%.0f', fontsize=20)
_ = Dataframe['category_id'].value_counts().plot(kind='pie', ax=axes[1], title='Category', autopct='%.0f', fontsize=20)
_ = plt.tight_layout()
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_56_0.png)



```python
d1 = Dataframe['channel_title'].value_counts()[:25]
d1[::-1].plot.barh(figsize=(8,5),title='Top 25 Channel titles');
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_57_0.png)


> From the pie chart, we can see that,
* India and Canada have highest number of videos: 40% each.
* 37% of videos are being uploaded in the category with id 24 and 14% with id 25.
* SAB TV, SET India, ESPN, Study IQ education and etvteluguindia are the channels with most number of videos.


```python
for i in Dataframe['channel_title'].value_counts()[:5].index:
    print(i, " : ", Dataframe.query('channel_title=="{}"'.format(i))['country_code'].unique())
```

    SAB TV  :  ['CA' 'IN']
    SET India  :  ['IN' 'CA']
    ESPN  :  ['US' 'GB' 'CA']
    Study IQ education  :  ['IN']
    etvteluguindia  :  ['IN']


3. **Textual features**


```python
TEXT_COL
```




    ['title', 'tags', 'description']




```python
wc = WordCloud(stopwords = set(list(STOPWORDS) + ['|']), random_state=63)

#Function to show wordcloud
def plotWord(df,title='WordCloud'):
    op = wc.generate(str(df))
    plt.figure(figsize=(10,10))
    plt.title(title.upper(), fontsize=24)
    plt.imshow(op);
```

**i) title**

> Lets see what all are the top words occuring in each textual columns


```python
plotWord(Dataframe['title'],'title')
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_65_0.png)


**ii) tag**


```python
plotWord(Dataframe['tags'],'tags')
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_67_0.png)


**iii) description**


```python
plotWord(Dataframe['description'],'description')
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_69_0.png)


## Bivariate Analysis

We will analyse features two at a time in bivariate analysis.

1. **Numerical features**

Lets see how the vairables are related to each other by correlation heatmap


```python
df = Dataframe[NUM_COL+[TARGET_COL]]
mask = np.triu(df.corr())
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True,mask=mask);
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_73_0.png)


We can see that our target coulumn(likes) is positively correlated with all the numerical columns. We can answer some of our hypotheses from this information.

1. **Videos with more views get more likes**
> Yes. Videos with more views get more likes. 
2. **Videos with more comments get more likes**
> Yes. Videos with more comments get more likes
3. **Videos with less dislikes get more likes**
> Yes. Videos with less dislikes get more likes. When number of deslikes increase, number of views increase. Hence the number of likes also increase. 

We can look at the pairplot for futher clarification.


```python
sns.pairplot(df, height=5, aspect=24/16);
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_75_0.png)


2. **Categorical features**


```python
CAT_COL
```




    ['channel_title', 'category_id', 'country_code']



**Countrywise number of videos**


```python
countrywise_data = Dataframe.groupby(['country_code', 'channel_title']).size().reset_index()
countrywise_data.columns = ['country_code', 'channel_title', 'number_of_videos']
countrywise_data = countrywise_data.sort_values(by = 'number_of_videos', ascending = False)
countrywise_data.head(2)
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
      <th>country_code</th>
      <th>channel_title</th>
      <th>number_of_videos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4937</th>
      <td>IN</td>
      <td>Study IQ education</td>
      <td>118</td>
    </tr>
    <tr>
      <th>5237</th>
      <td>IN</td>
      <td>etvteluguindia</td>
      <td>115</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, axes = plt.subplots(4,1,figsize=(10,10))
for i, c in enumerate(Dataframe['country_code'].unique()):
    data = countrywise_data[countrywise_data['country_code'] == c][:10]
    sns.barplot(data=data,x='number_of_videos',y='channel_title',ax=axes[i])
    axes[i].set_title(f'Country Code {c}')
plt.tight_layout()
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_80_0.png)


**Observation**
1. In Canada, top 3 channels are news and television programs
2. In India, top 3 channels are educaional and television/entertainment channels
3. In Britain, all the top 6 channels are talk shows.
4. In America, top 2 channels are sports channels and talk shows are in top 5.

**Mutivariate Analysis**

> We will analyse more than two variables at a time.

**Countrywise likes per channel**


```python
countrywise_data2 = Dataframe.groupby(['country_code', 'channel_title'])['likes'].max().reset_index().sort_values(by = ['likes'], ascending=False)
countrywise_data2.head(2)
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
      <th>country_code</th>
      <th>channel_title</th>
      <th>likes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3994</th>
      <td>GB</td>
      <td>ibighit</td>
      <td>15.171369</td>
    </tr>
    <tr>
      <th>3425</th>
      <td>GB</td>
      <td>LuisFonsiVEVO</td>
      <td>14.803627</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, axes = plt.subplots(4,1,figsize=(10,10))
for i, c in enumerate(Dataframe['country_code'].unique()):
    data = countrywise_data2[countrywise_data2['country_code'] == c][:10]
    sns.barplot(data=data,x='likes',y='channel_title',ax=axes[i])
    axes[i].set_title(f'Country Code {c}')
plt.tight_layout()
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_85_0.png)


**Likes distribution for each category**


```python
sns.catplot(data=Dataframe, x='category_id',y='likes', height=5, aspect=4);
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_87_0.png)


**Likes distribution for each country**


```python
sns.catplot(data=Dataframe, x='country_code',y='likes', height=6, aspect=2);
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_89_0.png)


**Mean likes per country**


```python
Dataframe.groupby('country_code')['likes'].mean().sort_values().plot(kind = 'barh');
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_91_0.png)


**Mean likes per channel**


```python
df1 = Dataframe.groupby('channel_title')['likes'].mean().sort_values(ascending = False)[:15]
df1[::-1].plot(kind = 'barh');
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_93_0.png)


**Mean likes per category**


```python
df1 = Dataframe.groupby('category_id')['likes'].mean().sort_values(ascending = False)
df1[::-1].plot(kind = 'barh');
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_95_0.png)


We can answer some of our hypotheses from this information.
5. **Channel affects the number of likes**
> Yes.
6. **Category of the video affects the number of likes**
> Yes.
7. **The country of origin affects the number of likes**
> Yes. Videos from Britain seams to have higher average number of like while in the the least.

## Datetime variable

We have one datetime column: published date.


```python
TIME_COL
```




    'publish_date'




```python
type(Dataframe['publish_date'][0])
```




    str



We need to convert the publiched column to datetime format.


```python
Dataframe['publish_date'] = pd.to_datetime(Dataframe['publish_date'])
Dataframe['publish_date']
```




    0       2017-12-12
    1       2018-03-08
    2       2018-03-26
    3       2018-02-21
    4       2018-05-10
               ...    
    26056   2018-01-16
    26057   2017-12-17
    26058   2018-03-04
    26059   2018-05-17
    26060   2018-01-16
    Name: publish_date, Length: 26061, dtype: datetime64[ns]




```python
type(Dataframe['publish_date'][0])
```




    pandas._libs.tslibs.timestamps.Timestamp



Lets see how many videos we have in each year in the dataset


```python
Dataframe['publish_date'].dt.year.value_counts()
```




    2018    18841
    2017     7132
    2015       16
    2016       16
    2011       13
    2014        9
    2013        9
    2009        8
    2012        6
    2010        4
    2008        3
    2007        3
    2006        1
    Name: publish_date, dtype: int64



**Almost every videos are from the year 2018 and 2017**

Lets consider the videos which are published in 2017 and 2018


```python
new_data = Dataframe.query('publish_date >= 2017')
```


```python
new_data.sort_values(by = 'publish_date').groupby('publish_date').size().plot(figsize=(9, 2),title = 'Number of Videos');
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_107_0.png)


We can see that, there are no videos before NOVEMBER 2017. So we will analyse the videos published after NOVEMBER 2017


```python
new_data = Dataframe.query('publish_date > "2017-11"')
```


```python
new_data.sort_values(by = 'publish_date').groupby('publish_date').size().plot(figsize=(18, 6),title = 'Number of Videos');
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_110_0.png)


There is a very sharp dip in the number of videos in the beginning of April of 2018.

**Mean Likes in Data during different months**


```python
new_data.sort_values(by = 'publish_date').groupby('publish_date')['likes'].mean().plot(figsize=(18, 6), title="Mean Likes");
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_113_0.png)


There is a very sharp increase in mean likes in the beginning of April of 2018. This is because of the sharp dip in the number of videos during this time period. 

**Now we check this countrywise**

1. Number of videos by country


```python
temp = new_data.groupby(['publish_date', 'country_code']).size().reset_index()
temp.pivot_table(index = 'publish_date', columns = 'country_code', values=0)
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
      <th>country_code</th>
      <th>CA</th>
      <th>GB</th>
      <th>IN</th>
      <th>US</th>
    </tr>
    <tr>
      <th>publish_date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-11-02</th>
      <td>1.0</td>
      <td>11.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2017-11-03</th>
      <td>1.0</td>
      <td>24.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-04</th>
      <td>NaN</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-05</th>
      <td>NaN</td>
      <td>9.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-06</th>
      <td>NaN</td>
      <td>11.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2018-06-10</th>
      <td>42.0</td>
      <td>2.0</td>
      <td>31.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2018-06-11</th>
      <td>46.0</td>
      <td>4.0</td>
      <td>45.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2018-06-12</th>
      <td>64.0</td>
      <td>5.0</td>
      <td>37.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2018-06-13</th>
      <td>51.0</td>
      <td>7.0</td>
      <td>39.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2018-06-14</th>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>225 rows × 4 columns</p>
</div>




```python
temp.pivot_table(index = 'publish_date', columns = 'country_code', values=0).plot(subplots=True, title='Number of videos by country',figsize=(18, 18),fontsize=18)
plt.tight_layout()
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_116_0.png)


2. Mean likes by country


```python
temp = new_data.groupby(['publish_date', 'country_code'])['likes'].mean().reset_index()
temp.pivot_table(index = 'publish_date', columns = 'country_code', values='likes').plot(subplots=True, title='Mean Number of likes by country',figsize=(18, 18),fontsize=18)
plt.tight_layout()
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_118_0.png)


We can check the number of videos publishing in each day of the week. This will help us to answer the hypotheses: 
8. **People post more videos on weekends than weekdays**

We will add one column to our dataset which contain the day of the week for this analysis.


```python
plt.style.use('ggplot')
```


```python
Dataframe['day'] = Dataframe['publish_date'].dt.dayofweek
temp = Dataframe['day'].value_counts().sort_index().reset_index()
temp.columns = ['Day','Number of videos']
temp.Day = ['Mon','Tue','Wed','Thur','Fri','Sat','Sun']
sns.catplot(data=temp,x='Day',y='Number of videos',aspect = 4,kind='point')
plt.title("Number of Videos Posted Per Day of Week", fontsize=20);
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_121_0.png)


We can see that most number of videos are uploaded on fridays. And our hypothesis is proven wrong. In weakends, number of videos being uploaded is the least.

We have one more hypothesis left to answer:

4. **Descriptive videos get more number of likes**

We can use correlation heatmap for answering this.


```python
Dataframe['title_len'] =Dataframe['title'].apply(lambda x:len(x))
Dataframe['description_len'] =Dataframe['description'].apply(lambda x:len(x))
Dataframe['tags_len'] =Dataframe['tags'].apply(lambda x:len(x))
plt.subplots(figsize=(5,3))
sns.heatmap(Dataframe[['title_len', 'description_len', 'tags_len', 'likes']].corr(), annot = True);

```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_123_0.png)


From the heatmap we can see that:
1. title length is negatively correlated to likes and
2. description and tag lengths have a positive correlation

So if you keep your title length short and description long, you are having higher chances of getting more likes.

## Modelling


```python
Dataframe.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 26061 entries, 0 to 26060
    Data columns (total 16 columns):
     #   Column           Non-Null Count  Dtype         
    ---  ------           --------------  -----         
     0   video_id         26061 non-null  int64         
     1   title            26061 non-null  object        
     2   channel_title    26061 non-null  object        
     3   category_id      26061 non-null  float64       
     4   publish_date     26061 non-null  datetime64[ns]
     5   tags             26061 non-null  object        
     6   views            26061 non-null  float64       
     7   dislikes         26061 non-null  float64       
     8   comment_count    26061 non-null  float64       
     9   description      26061 non-null  object        
     10  country_code     26061 non-null  object        
     11  likes            26061 non-null  float64       
     12  day              26061 non-null  int64         
     13  title_len        26061 non-null  int64         
     14  description_len  26061 non-null  int64         
     15  tags_len         26061 non-null  int64         
    dtypes: datetime64[ns](1), float64(5), int64(5), object(5)
    memory usage: 3.2+ MB



```python
Dataframe.isnull().sum()
```




    video_id           0
    title              0
    channel_title      0
    category_id        0
    publish_date       0
    tags               0
    views              0
    dislikes           0
    comment_count      0
    description        0
    country_code       0
    likes              0
    day                0
    title_len          0
    description_len    0
    tags_len           0
    dtype: int64




```python
NUM_COL = ['views', 'dislikes', 'comment_count']
CAT_COL = ['category_id', 'country_code']
TEXT_COL = ['title', 'channel_title', 'tags', 'description']
TIME_COL = ['publish_date']
```

**country_code column is in string format. We can encode it so that we can use it for modelling**


```python
Dataframe = pd.get_dummies(Dataframe, columns = CAT_COL)
Dataframe.head(2)
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
      <th>video_id</th>
      <th>title</th>
      <th>channel_title</th>
      <th>publish_date</th>
      <th>tags</th>
      <th>views</th>
      <th>dislikes</th>
      <th>comment_count</th>
      <th>description</th>
      <th>likes</th>
      <th>day</th>
      <th>title_len</th>
      <th>description_len</th>
      <th>tags_len</th>
      <th>category_id_1.0</th>
      <th>category_id_2.0</th>
      <th>category_id_10.0</th>
      <th>category_id_15.0</th>
      <th>category_id_17.0</th>
      <th>category_id_19.0</th>
      <th>category_id_20.0</th>
      <th>category_id_22.0</th>
      <th>category_id_23.0</th>
      <th>category_id_24.0</th>
      <th>category_id_25.0</th>
      <th>category_id_26.0</th>
      <th>category_id_27.0</th>
      <th>category_id_28.0</th>
      <th>category_id_29.0</th>
      <th>category_id_30.0</th>
      <th>category_id_43.0</th>
      <th>country_code_CA</th>
      <th>country_code_GB</th>
      <th>country_code_IN</th>
      <th>country_code_US</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>53364</td>
      <td>Alif Allah Aur Insaan Episode 34 HUM TV Drama ...</td>
      <td>HUM TV</td>
      <td>2017-12-12</td>
      <td>HUM|"TV"|"Alif Allah Aur Insaan"|"Episode 34"|...</td>
      <td>12.769769</td>
      <td>5.700444</td>
      <td>6.803505</td>
      <td>Alif Allah Aur Insaan Episode 34 Full - 12 Dec...</td>
      <td>7.763021</td>
      <td>1</td>
      <td>64</td>
      <td>1030</td>
      <td>187</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>51040</td>
      <td>It's Showtime Miss Q &amp; A: Bela gets jealous of...</td>
      <td>ABS-CBN Entertainment</td>
      <td>2018-03-08</td>
      <td>ABS-CBN Entertainment|"ABS-CBN"|"ABS-CBN Onlin...</td>
      <td>13.042257</td>
      <td>4.317488</td>
      <td>5.752573</td>
      <td>Vice Ganda notices Bela Padilla's sudden chang...</td>
      <td>8.091015</td>
      <td>3</td>
      <td>55</td>
      <td>599</td>
      <td>494</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Split the dataset into training and validation sets.

We will use 80-20 split with 80% of the rows belonging to training data.


```python
trn, val = train_test_split(Dataframe, test_size=0.2, random_state=63)
Dataframe.shape[0],trn.shape[0], val.shape[0]
```




    (26061, 20848, 5213)




```python
features = [c for c in Dataframe.columns if c not in [ID_COL]+[TARGET_COL]]
CAT_NUM_COL = [c for c in features if c not in TEXT_COL+TIME_COL]
CAT_NUM_COL
```




    ['views',
     'dislikes',
     'comment_count',
     'day',
     'title_len',
     'description_len',
     'tags_len',
     'category_id_1.0',
     'category_id_2.0',
     'category_id_10.0',
     'category_id_15.0',
     'category_id_17.0',
     'category_id_19.0',
     'category_id_20.0',
     'category_id_22.0',
     'category_id_23.0',
     'category_id_24.0',
     'category_id_25.0',
     'category_id_26.0',
     'category_id_27.0',
     'category_id_28.0',
     'category_id_29.0',
     'category_id_30.0',
     'category_id_43.0',
     'country_code_CA',
     'country_code_GB',
     'country_code_IN',
     'country_code_US']




```python
X_trn, X_val, y_trn, y_val = trn[features], val[features], trn[TARGET_COL], val[TARGET_COL]
# X_trn and y_trn are the dataset we will use to train our model
# We will predict the target column using X_val and check the result with the output with the actual value y_val
```

### Linear Regression
#### 1. Training using Numerical features


```python
Lin_reg = LinearRegression()
Lin_reg.fit(X_trn[NUM_COL],y_trn)
pred = Lin_reg.predict(X_val[NUM_COL])
```

**Let us take a look at our prediction and actual values**


```python
comprsn = pd.DataFrame(data={'predictions': pred, 'actual': y_val})
comprsn
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
      <th>predictions</th>
      <th>actual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12061</th>
      <td>9.129610</td>
      <td>6.655440</td>
    </tr>
    <tr>
      <th>6599</th>
      <td>4.274881</td>
      <td>5.342334</td>
    </tr>
    <tr>
      <th>1836</th>
      <td>5.689831</td>
      <td>4.779123</td>
    </tr>
    <tr>
      <th>4208</th>
      <td>8.847895</td>
      <td>7.823646</td>
    </tr>
    <tr>
      <th>20407</th>
      <td>8.814543</td>
      <td>9.152605</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>17036</th>
      <td>9.092941</td>
      <td>9.420115</td>
    </tr>
    <tr>
      <th>18025</th>
      <td>6.353081</td>
      <td>5.880533</td>
    </tr>
    <tr>
      <th>24670</th>
      <td>8.745252</td>
      <td>9.310548</td>
    </tr>
    <tr>
      <th>12125</th>
      <td>7.031010</td>
      <td>7.944847</td>
    </tr>
    <tr>
      <th>3766</th>
      <td>8.533171</td>
      <td>9.264071</td>
    </tr>
  </tbody>
</table>
<p>5213 rows × 2 columns</p>
</div>




```python
#for i in comprsn.columns:
#    comprsn[i] = comprsn[i].apply(lambda x:np.expm1(x))
#comprsn
```


```python
#plt.scatter(x=list(range(5213)), y=comprsn['predictions'],label='Predicted')
#plt.scatter(x=list(range(5213)), y=comprsn['actual'],color='red',label='Actual')
#plt.legend()
#plt.show()
```

**We need to evaluate the performance of our model. We can use various evaluation metrices such as:**

1. Mean Squared Error(MSE)
2. Root-Mean-Squared-Error(RMSE)
3. Root Mean Squared Log Error(RMSLE)
4. Mean-Absolute-Error(MAE).
5. R²
6. Adjusted R²

**We can use R² and RMSE for the evaluation. Values for R² ranges from 0 to 1. Closer the R² value to 1, bettr our model performs.**


```python
# Evaluation Metrices
def score(y_val,pred):
    print('RMSE\t: ', np.sqrt(mean_squared_error(y_val,pred)))
    print('R²\t: ',r2_score(y_val,pred))
```


```python
score(y_val,pred)
```

    RMSE	:  0.9256090513508034
    R²	:  0.7692359553327757


#### 2. Training using Numerical and Categorical features


```python
Lin_reg = LinearRegression()
Lin_reg.fit(X_trn[CAT_NUM_COL],y_trn)
pred = Lin_reg.predict(X_val[CAT_NUM_COL])
```


```python
score(y_val,pred)
```

    RMSE	:  0.8435677496099154
    R²	:  0.8083305518428854


**We can see that our results are improved. Root Mean Square Error is reduced and the R² score is increased to 80%.**

### Decition Tree Model


```python
DT = DecisionTreeRegressor(random_state=63)
DT.fit(X_trn[CAT_NUM_COL],y_trn)
```




    DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
                          max_leaf_nodes=None, min_impurity_decrease=0.0,
                          min_impurity_split=None, min_samples_leaf=1,
                          min_samples_split=2, min_weight_fraction_leaf=0.0,
                          presort=False, random_state=63, splitter='best')




```python
pred = DT.predict(X_val[CAT_NUM_COL])
```


```python
score(y_val,pred)
```

    RMSE	:  0.8976916981861263
    R²	:  0.7829462095545799


There is no improvement after using Decision tree. So we have to specify some of the hyperparameters to get a good result.


```python
DT = DecisionTreeRegressor(random_state=63, max_depth=12,min_samples_split=30, min_samples_leaf= 10, max_leaf_nodes=396)
DT.fit(X_trn[CAT_NUM_COL],y_trn)
pred = DT.predict(X_val[CAT_NUM_COL])
score(y_val,pred)
```

    RMSE	:  0.7341625868794904
    R²	:  0.8548231213805365


**Grid Search**


```python
#hyperparam_combs = {
#    'max_depth': [ 8, 10, 12, 14, 16],
#    'min_samples_split': [20, 30, 40],
#    'max_features': [0.6, 0.8, 1],
#    'max_leaf_nodes': [253, 396, 128],
#}

#clf = GridSearchCV(DecisionTreeRegressor(),
#                         hyperparam_combs,
#                         scoring='r2')

#search = clf.fit(X_trn[CAT_NUM_COL],y_trn)

#search.best_params_
```


```python
best_parm = {'max_depth': 16,
 'max_features': 0.8,
 'max_leaf_nodes': 396,
 'min_samples_split': 30}
DT = DecisionTreeRegressor(random_state=63,**best_parm)
DT.fit(X_trn[CAT_NUM_COL],y_trn)
pred = DT.predict(X_val[CAT_NUM_COL])
score(y_val,pred)
```

    RMSE	:  0.7395403850806784
    R²	:  0.8526884677440504


**Gradient Boosting**
1. LightGBM


```python
LGB = LGBMRegressor(random_state=63)
LGB.fit(X_trn[CAT_NUM_COL],y_trn)
pred = LGB.predict(X_val[CAT_NUM_COL])
score(y_val,pred)
```

    RMSE	:  0.6295551271587297
    R²	:  0.8932469020261351



```python
#hyperparam_combs = {
#    'num_leaves': [31, 127],
#    'reg_alpha': [0.1, 0.5],
#    'min_data_in_leaf': [30, 50, 100, 300, 400],
#    'lambda_l1': [0, 1, 1.5],
#    'lambda_l2': [0, 1]
#    }
#
#clf = GridSearchCV(LGBMRegressor(),
#                         hyperparam_combs,
#                         scoring='r2')
#
#search = clf.fit(X_trn[CAT_NUM_COL],y_trn)
#
#search.best_params_
```


```python
best_parm = {'lambda_l1': 0,
 'lambda_l2': 1,
 'min_data_in_leaf': 30,
 'num_leaves': 127,
 'reg_alpha': 0.1}
LGB = LGBMRegressor(random_state=63,**best_parm)
LGB.fit(X_trn[CAT_NUM_COL],y_trn)
pred = LGB.predict(X_val[CAT_NUM_COL])
score(y_val,pred)
```

    [LightGBM] [Warning] lambda_l1 is set=0, reg_alpha=0.1 will be ignored. Current value: lambda_l1=0
    [LightGBM] [Warning] lambda_l2 is set=1, reg_lambda=0.0 will be ignored. Current value: lambda_l2=1
    [LightGBM] [Warning] min_data_in_leaf is set=30, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=30
    RMSE	:  0.6028646663902235
    R²	:  0.9021067790262407



```python
comprsn = pd.DataFrame(data={'predictions': pred, 'actual': y_val})
comprsn
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
      <th>predictions</th>
      <th>actual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12061</th>
      <td>7.337127</td>
      <td>6.655440</td>
    </tr>
    <tr>
      <th>6599</th>
      <td>4.591361</td>
      <td>5.342334</td>
    </tr>
    <tr>
      <th>1836</th>
      <td>4.932083</td>
      <td>4.779123</td>
    </tr>
    <tr>
      <th>4208</th>
      <td>8.000117</td>
      <td>7.823646</td>
    </tr>
    <tr>
      <th>20407</th>
      <td>8.747728</td>
      <td>9.152605</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>17036</th>
      <td>9.585739</td>
      <td>9.420115</td>
    </tr>
    <tr>
      <th>18025</th>
      <td>5.720946</td>
      <td>5.880533</td>
    </tr>
    <tr>
      <th>24670</th>
      <td>8.688409</td>
      <td>9.310548</td>
    </tr>
    <tr>
      <th>12125</th>
      <td>7.824834</td>
      <td>7.944847</td>
    </tr>
    <tr>
      <th>3766</th>
      <td>9.158473</td>
      <td>9.264071</td>
    </tr>
  </tbody>
</table>
<p>5213 rows × 2 columns</p>
</div>




```python
x=LGB.feature_name_
y=LGB.feature_importances_
```


```python
pd.DataFrame({'x':y,'y':x}).sort_values(by = 'x',ascending=False).set_index('y')
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
      <th>x</th>
    </tr>
    <tr>
      <th>y</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>description_len</th>
      <td>2039</td>
    </tr>
    <tr>
      <th>tags_len</th>
      <td>1734</td>
    </tr>
    <tr>
      <th>views</th>
      <td>1690</td>
    </tr>
    <tr>
      <th>dislikes</th>
      <td>1652</td>
    </tr>
    <tr>
      <th>comment_count</th>
      <td>1564</td>
    </tr>
    <tr>
      <th>title_len</th>
      <td>1494</td>
    </tr>
    <tr>
      <th>day</th>
      <td>402</td>
    </tr>
    <tr>
      <th>category_id_24.0</th>
      <td>263</td>
    </tr>
    <tr>
      <th>country_code_CA</th>
      <td>262</td>
    </tr>
    <tr>
      <th>country_code_IN</th>
      <td>237</td>
    </tr>
    <tr>
      <th>category_id_25.0</th>
      <td>237</td>
    </tr>
    <tr>
      <th>category_id_10.0</th>
      <td>181</td>
    </tr>
    <tr>
      <th>category_id_17.0</th>
      <td>158</td>
    </tr>
    <tr>
      <th>category_id_22.0</th>
      <td>120</td>
    </tr>
    <tr>
      <th>category_id_23.0</th>
      <td>119</td>
    </tr>
    <tr>
      <th>category_id_26.0</th>
      <td>83</td>
    </tr>
    <tr>
      <th>country_code_US</th>
      <td>66</td>
    </tr>
    <tr>
      <th>category_id_27.0</th>
      <td>64</td>
    </tr>
    <tr>
      <th>country_code_GB</th>
      <td>57</td>
    </tr>
    <tr>
      <th>category_id_20.0</th>
      <td>52</td>
    </tr>
    <tr>
      <th>category_id_1.0</th>
      <td>48</td>
    </tr>
    <tr>
      <th>category_id_29.0</th>
      <td>20</td>
    </tr>
    <tr>
      <th>category_id_15.0</th>
      <td>17</td>
    </tr>
    <tr>
      <th>category_id_2.0</th>
      <td>17</td>
    </tr>
    <tr>
      <th>category_id_28.0</th>
      <td>13</td>
    </tr>
    <tr>
      <th>category_id_43.0</th>
      <td>11</td>
    </tr>
    <tr>
      <th>category_id_19.0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>category_id_30.0</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame({'x':y,'y':x}).sort_values(by = 'x',ascending=False).set_index('y')[::-1].plot(kind='barh',figsize=(12, 6));
```


![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_164_0.png)



```python
LGB = LGBMRegressor(n_estimators = 5000,
                        learning_rate = 0.01,
                        colsample_bytree = 0.76,
                        metric = 'None',
                        )
LGB.fit(X_trn[CAT_NUM_COL],y_trn)
pred = LGB.predict(X_val[CAT_NUM_COL])
score(y_val,pred)
x=LGB.feature_name_
y=LGB.feature_importances_
pd.DataFrame({'x':y,'y':x}).sort_values(by = 'x',ascending=False).set_index('y')[::-1].plot(kind='barh',figsize=(12, 6));
```

    RMSE	:  0.5959047429872405
    R²	:  0.9043540377169839



![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_165_1.png)



```python
CB = CatBoostRegressor(n_estimators = 3000,
                       learning_rate = 0.01,
                       rsm = 0.4
                       )
CB.fit(X_trn[CAT_NUM_COL],y_trn)
pred = CB.predict(X_val[CAT_NUM_COL])
score(y_val,pred)
x=LGB.feature_name_
y=CB.feature_importances_
pd.DataFrame({'x':y,'y':x}).sort_values(by = 'x',ascending=False).set_index('y')[::-1].plot(kind='barh',figsize=(12, 6));
```

    0:	learn: 1.9148707	total: 82.6ms	remaining: 4m 7s
    1:	learn: 1.9020447	total: 88.3ms	remaining: 2m 12s
    2:	learn: 1.8880901	total: 94.5ms	remaining: 1m 34s
    3:	learn: 1.8744430	total: 102ms	remaining: 1m 16s
    4:	learn: 1.8613186	total: 107ms	remaining: 1m 4s
    5:	learn: 1.8490073	total: 114ms	remaining: 56.9s
    6:	learn: 1.8359478	total: 119ms	remaining: 50.9s
    7:	learn: 1.8235258	total: 125ms	remaining: 46.8s
    8:	learn: 1.8105525	total: 131ms	remaining: 43.5s
    9:	learn: 1.7978627	total: 146ms	remaining: 43.7s
    10:	learn: 1.7861177	total: 153ms	remaining: 41.5s
    11:	learn: 1.7735966	total: 159ms	remaining: 39.5s
    12:	learn: 1.7628993	total: 165ms	remaining: 37.9s
    13:	learn: 1.7512042	total: 171ms	remaining: 36.4s
    14:	learn: 1.7386890	total: 177ms	remaining: 35.1s
    15:	learn: 1.7267749	total: 182ms	remaining: 33.9s
    16:	learn: 1.7155220	total: 188ms	remaining: 33.1s
    17:	learn: 1.7041148	total: 194ms	remaining: 32.2s
    18:	learn: 1.6939803	total: 198ms	remaining: 31.1s
    19:	learn: 1.6830836	total: 202ms	remaining: 30.1s
    20:	learn: 1.6719570	total: 205ms	remaining: 29.1s
    21:	learn: 1.6614309	total: 208ms	remaining: 28.1s
    22:	learn: 1.6510430	total: 210ms	remaining: 27.2s
    23:	learn: 1.6404034	total: 213ms	remaining: 26.4s
    24:	learn: 1.6303998	total: 215ms	remaining: 25.6s
    25:	learn: 1.6201363	total: 218ms	remaining: 24.9s
    26:	learn: 1.6107354	total: 220ms	remaining: 24.2s
    27:	learn: 1.6011333	total: 223ms	remaining: 23.6s
    28:	learn: 1.5912340	total: 225ms	remaining: 23s
    29:	learn: 1.5810786	total: 227ms	remaining: 22.5s
    30:	learn: 1.5713021	total: 229ms	remaining: 22s
    31:	learn: 1.5615340	total: 232ms	remaining: 21.5s
    32:	learn: 1.5524365	total: 234ms	remaining: 21s
    33:	learn: 1.5428937	total: 237ms	remaining: 20.7s
    34:	learn: 1.5339063	total: 239ms	remaining: 20.3s
    35:	learn: 1.5245465	total: 242ms	remaining: 19.9s
    36:	learn: 1.5159844	total: 244ms	remaining: 19.5s
    37:	learn: 1.5065859	total: 250ms	remaining: 19.5s
    38:	learn: 1.4976488	total: 253ms	remaining: 19.2s
    39:	learn: 1.4885187	total: 260ms	remaining: 19.3s
    40:	learn: 1.4800399	total: 263ms	remaining: 19s
    41:	learn: 1.4713564	total: 266ms	remaining: 18.7s
    42:	learn: 1.4624875	total: 269ms	remaining: 18.5s
    43:	learn: 1.4537708	total: 272ms	remaining: 18.3s
    44:	learn: 1.4450174	total: 275ms	remaining: 18s
    45:	learn: 1.4372793	total: 277ms	remaining: 17.8s
    46:	learn: 1.4292108	total: 284ms	remaining: 17.8s
    47:	learn: 1.4210064	total: 289ms	remaining: 17.8s
    48:	learn: 1.4138827	total: 295ms	remaining: 17.7s
    49:	learn: 1.4054280	total: 299ms	remaining: 17.7s
    50:	learn: 1.3978874	total: 307ms	remaining: 17.7s
    51:	learn: 1.3904062	total: 311ms	remaining: 17.6s
    52:	learn: 1.3828670	total: 314ms	remaining: 17.5s
    53:	learn: 1.3756326	total: 343ms	remaining: 18.7s
    54:	learn: 1.3685376	total: 350ms	remaining: 18.7s
    55:	learn: 1.3608767	total: 357ms	remaining: 18.7s
    56:	learn: 1.3545933	total: 364ms	remaining: 18.8s
    57:	learn: 1.3474516	total: 370ms	remaining: 18.8s
    58:	learn: 1.3401061	total: 378ms	remaining: 18.8s
    59:	learn: 1.3330072	total: 383ms	remaining: 18.8s
    60:	learn: 1.3260636	total: 391ms	remaining: 18.8s
    61:	learn: 1.3193516	total: 398ms	remaining: 18.8s
    62:	learn: 1.3122962	total: 403ms	remaining: 18.8s
    63:	learn: 1.3055152	total: 405ms	remaining: 18.6s
    64:	learn: 1.2985601	total: 407ms	remaining: 18.4s
    65:	learn: 1.2919724	total: 410ms	remaining: 18.2s
    66:	learn: 1.2851261	total: 412ms	remaining: 18s
    67:	learn: 1.2789623	total: 414ms	remaining: 17.9s
    68:	learn: 1.2731392	total: 417ms	remaining: 17.7s
    69:	learn: 1.2662765	total: 419ms	remaining: 17.6s
    70:	learn: 1.2595461	total: 422ms	remaining: 17.4s
    71:	learn: 1.2532522	total: 425ms	remaining: 17.3s
    72:	learn: 1.2472023	total: 432ms	remaining: 17.3s
    73:	learn: 1.2409245	total: 435ms	remaining: 17.2s
    74:	learn: 1.2348658	total: 437ms	remaining: 17.1s
    75:	learn: 1.2285598	total: 440ms	remaining: 16.9s
    76:	learn: 1.2224782	total: 442ms	remaining: 16.8s
    77:	learn: 1.2163077	total: 444ms	remaining: 16.6s
    78:	learn: 1.2111862	total: 447ms	remaining: 16.5s
    79:	learn: 1.2056652	total: 449ms	remaining: 16.4s
    80:	learn: 1.2002045	total: 451ms	remaining: 16.3s
    81:	learn: 1.1948410	total: 454ms	remaining: 16.1s
    82:	learn: 1.1889553	total: 462ms	remaining: 16.2s
    83:	learn: 1.1836688	total: 469ms	remaining: 16.3s
    84:	learn: 1.1784015	total: 476ms	remaining: 16.3s
    85:	learn: 1.1728685	total: 483ms	remaining: 16.4s
    86:	learn: 1.1674968	total: 488ms	remaining: 16.4s
    87:	learn: 1.1627097	total: 495ms	remaining: 16.4s
    88:	learn: 1.1572995	total: 502ms	remaining: 16.4s
    89:	learn: 1.1534254	total: 508ms	remaining: 16.4s
    90:	learn: 1.1479691	total: 514ms	remaining: 16.4s
    91:	learn: 1.1426781	total: 521ms	remaining: 16.5s
    92:	learn: 1.1377218	total: 528ms	remaining: 16.5s
    93:	learn: 1.1330924	total: 534ms	remaining: 16.5s
    94:	learn: 1.1280615	total: 541ms	remaining: 16.5s
    95:	learn: 1.1233494	total: 547ms	remaining: 16.6s
    96:	learn: 1.1195176	total: 554ms	remaining: 16.6s
    97:	learn: 1.1150288	total: 556ms	remaining: 16.5s
    98:	learn: 1.1100509	total: 561ms	remaining: 16.4s
    99:	learn: 1.1052097	total: 564ms	remaining: 16.4s
    100:	learn: 1.1003749	total: 566ms	remaining: 16.3s
    101:	learn: 1.0955229	total: 569ms	remaining: 16.2s
    102:	learn: 1.0909120	total: 571ms	remaining: 16.1s
    103:	learn: 1.0871251	total: 573ms	remaining: 16s
    104:	learn: 1.0830962	total: 577ms	remaining: 15.9s
    105:	learn: 1.0788601	total: 583ms	remaining: 15.9s
    106:	learn: 1.0753672	total: 586ms	remaining: 15.8s
    107:	learn: 1.0714343	total: 588ms	remaining: 15.7s
    108:	learn: 1.0670847	total: 590ms	remaining: 15.7s
    109:	learn: 1.0632858	total: 593ms	remaining: 15.6s
    110:	learn: 1.0591279	total: 597ms	remaining: 15.5s
    111:	learn: 1.0549723	total: 600ms	remaining: 15.5s
    112:	learn: 1.0508340	total: 602ms	remaining: 15.4s
    113:	learn: 1.0466274	total: 605ms	remaining: 15.3s
    114:	learn: 1.0425368	total: 607ms	remaining: 15.2s
    115:	learn: 1.0384475	total: 613ms	remaining: 15.2s
    116:	learn: 1.0347697	total: 618ms	remaining: 15.2s
    117:	learn: 1.0312463	total: 621ms	remaining: 15.2s
    118:	learn: 1.0277815	total: 624ms	remaining: 15.1s
    119:	learn: 1.0239670	total: 626ms	remaining: 15s
    120:	learn: 1.0204740	total: 629ms	remaining: 15s
    121:	learn: 1.0172360	total: 631ms	remaining: 14.9s
    122:	learn: 1.0135731	total: 637ms	remaining: 14.9s
    123:	learn: 1.0100101	total: 639ms	remaining: 14.8s
    124:	learn: 1.0067284	total: 645ms	remaining: 14.8s
    125:	learn: 1.0039617	total: 651ms	remaining: 14.8s
    126:	learn: 1.0003759	total: 657ms	remaining: 14.9s
    127:	learn: 0.9974951	total: 663ms	remaining: 14.9s
    128:	learn: 0.9938525	total: 671ms	remaining: 14.9s
    129:	learn: 0.9907517	total: 676ms	remaining: 14.9s
    130:	learn: 0.9876096	total: 682ms	remaining: 14.9s
    131:	learn: 0.9844753	total: 688ms	remaining: 15s
    132:	learn: 0.9813237	total: 695ms	remaining: 15s
    133:	learn: 0.9778861	total: 702ms	remaining: 15s
    134:	learn: 0.9748585	total: 709ms	remaining: 15s
    135:	learn: 0.9718489	total: 715ms	remaining: 15.1s
    136:	learn: 0.9693755	total: 724ms	remaining: 15.1s
    137:	learn: 0.9663501	total: 730ms	remaining: 15.1s
    138:	learn: 0.9634658	total: 736ms	remaining: 15.1s
    139:	learn: 0.9609360	total: 744ms	remaining: 15.2s
    140:	learn: 0.9580456	total: 751ms	remaining: 15.2s
    141:	learn: 0.9550210	total: 757ms	remaining: 15.2s
    142:	learn: 0.9522629	total: 764ms	remaining: 15.3s
    143:	learn: 0.9495248	total: 772ms	remaining: 15.3s
    144:	learn: 0.9470049	total: 785ms	remaining: 15.5s
    145:	learn: 0.9444260	total: 790ms	remaining: 15.4s
    146:	learn: 0.9418020	total: 793ms	remaining: 15.4s
    147:	learn: 0.9390674	total: 795ms	remaining: 15.3s
    148:	learn: 0.9367327	total: 797ms	remaining: 15.3s
    149:	learn: 0.9341573	total: 800ms	remaining: 15.2s
    150:	learn: 0.9318528	total: 803ms	remaining: 15.2s
    151:	learn: 0.9296161	total: 806ms	remaining: 15.1s
    152:	learn: 0.9270739	total: 808ms	remaining: 15s
    153:	learn: 0.9245969	total: 811ms	remaining: 15s
    154:	learn: 0.9221480	total: 814ms	remaining: 14.9s
    155:	learn: 0.9198107	total: 816ms	remaining: 14.9s
    156:	learn: 0.9176208	total: 818ms	remaining: 14.8s
    157:	learn: 0.9154131	total: 821ms	remaining: 14.8s
    158:	learn: 0.9130536	total: 823ms	remaining: 14.7s
    159:	learn: 0.9111037	total: 826ms	remaining: 14.7s
    160:	learn: 0.9088064	total: 828ms	remaining: 14.6s
    161:	learn: 0.9067702	total: 831ms	remaining: 14.6s
    162:	learn: 0.9045064	total: 833ms	remaining: 14.5s
    163:	learn: 0.9024028	total: 837ms	remaining: 14.5s
    164:	learn: 0.9001269	total: 840ms	remaining: 14.4s
    165:	learn: 0.8980889	total: 842ms	remaining: 14.4s
    166:	learn: 0.8960327	total: 845ms	remaining: 14.3s
    167:	learn: 0.8940142	total: 847ms	remaining: 14.3s
    168:	learn: 0.8919283	total: 849ms	remaining: 14.2s
    169:	learn: 0.8899497	total: 852ms	remaining: 14.2s
    170:	learn: 0.8879287	total: 855ms	remaining: 14.1s
    171:	learn: 0.8860696	total: 858ms	remaining: 14.1s
    172:	learn: 0.8843327	total: 860ms	remaining: 14.1s
    173:	learn: 0.8824912	total: 863ms	remaining: 14s
    174:	learn: 0.8807649	total: 866ms	remaining: 14s
    175:	learn: 0.8791622	total: 872ms	remaining: 14s
    176:	learn: 0.8774638	total: 874ms	remaining: 13.9s
    177:	learn: 0.8756883	total: 877ms	remaining: 13.9s
    178:	learn: 0.8739151	total: 880ms	remaining: 13.9s
    179:	learn: 0.8721828	total: 883ms	remaining: 13.8s
    180:	learn: 0.8704704	total: 888ms	remaining: 13.8s
    181:	learn: 0.8689805	total: 894ms	remaining: 13.8s
    182:	learn: 0.8673832	total: 902ms	remaining: 13.9s
    183:	learn: 0.8657077	total: 905ms	remaining: 13.8s
    184:	learn: 0.8639731	total: 908ms	remaining: 13.8s
    185:	learn: 0.8624691	total: 910ms	remaining: 13.8s
    186:	learn: 0.8608050	total: 913ms	remaining: 13.7s
    187:	learn: 0.8594016	total: 919ms	remaining: 13.7s
    188:	learn: 0.8577621	total: 924ms	remaining: 13.7s
    189:	learn: 0.8561541	total: 927ms	remaining: 13.7s
    190:	learn: 0.8546617	total: 929ms	remaining: 13.7s
    191:	learn: 0.8531927	total: 935ms	remaining: 13.7s
    192:	learn: 0.8517078	total: 938ms	remaining: 13.6s
    193:	learn: 0.8502697	total: 941ms	remaining: 13.6s
    194:	learn: 0.8488932	total: 944ms	remaining: 13.6s
    195:	learn: 0.8475798	total: 947ms	remaining: 13.5s
    196:	learn: 0.8460896	total: 951ms	remaining: 13.5s
    197:	learn: 0.8445176	total: 955ms	remaining: 13.5s
    198:	learn: 0.8431143	total: 958ms	remaining: 13.5s
    199:	learn: 0.8416373	total: 961ms	remaining: 13.5s
    200:	learn: 0.8401599	total: 966ms	remaining: 13.4s
    201:	learn: 0.8391495	total: 970ms	remaining: 13.4s
    202:	learn: 0.8377914	total: 973ms	remaining: 13.4s
    203:	learn: 0.8364638	total: 975ms	remaining: 13.4s
    204:	learn: 0.8352250	total: 978ms	remaining: 13.3s
    205:	learn: 0.8338885	total: 981ms	remaining: 13.3s
    206:	learn: 0.8327796	total: 984ms	remaining: 13.3s
    207:	learn: 0.8315614	total: 987ms	remaining: 13.2s
    208:	learn: 0.8302142	total: 990ms	remaining: 13.2s
    209:	learn: 0.8290928	total: 993ms	remaining: 13.2s
    210:	learn: 0.8279728	total: 1000ms	remaining: 13.2s
    211:	learn: 0.8267212	total: 1s	remaining: 13.2s
    212:	learn: 0.8256144	total: 1.01s	remaining: 13.2s
    213:	learn: 0.8243768	total: 1.01s	remaining: 13.1s
    214:	learn: 0.8232094	total: 1.01s	remaining: 13.1s
    215:	learn: 0.8219888	total: 1.02s	remaining: 13.1s
    216:	learn: 0.8209206	total: 1.02s	remaining: 13.1s
    217:	learn: 0.8197583	total: 1.02s	remaining: 13s
    218:	learn: 0.8186706	total: 1.02s	remaining: 13s
    219:	learn: 0.8174816	total: 1.03s	remaining: 13s
    220:	learn: 0.8165093	total: 1.03s	remaining: 13s
    221:	learn: 0.8154464	total: 1.04s	remaining: 13s
    222:	learn: 0.8143793	total: 1.04s	remaining: 13s
    223:	learn: 0.8134515	total: 1.04s	remaining: 12.9s
    224:	learn: 0.8124776	total: 1.05s	remaining: 12.9s
    225:	learn: 0.8114988	total: 1.05s	remaining: 12.9s
    226:	learn: 0.8105779	total: 1.05s	remaining: 12.9s
    227:	learn: 0.8094507	total: 1.06s	remaining: 12.8s
    228:	learn: 0.8084982	total: 1.06s	remaining: 12.8s
    229:	learn: 0.8075109	total: 1.06s	remaining: 12.8s
    230:	learn: 0.8064608	total: 1.07s	remaining: 12.9s
    231:	learn: 0.8055523	total: 1.08s	remaining: 12.9s
    232:	learn: 0.8045764	total: 1.08s	remaining: 12.9s
    233:	learn: 0.8036090	total: 1.09s	remaining: 12.9s
    234:	learn: 0.8026008	total: 1.09s	remaining: 12.8s
    235:	learn: 0.8017134	total: 1.09s	remaining: 12.8s
    236:	learn: 0.8008000	total: 1.1s	remaining: 12.8s
    237:	learn: 0.7999341	total: 1.1s	remaining: 12.8s
    238:	learn: 0.7990204	total: 1.1s	remaining: 12.7s
    239:	learn: 0.7981994	total: 1.1s	remaining: 12.7s
    240:	learn: 0.7974126	total: 1.11s	remaining: 12.7s
    241:	learn: 0.7965810	total: 1.11s	remaining: 12.7s
    242:	learn: 0.7957933	total: 1.11s	remaining: 12.6s
    243:	learn: 0.7947412	total: 1.12s	remaining: 12.6s
    244:	learn: 0.7939434	total: 1.12s	remaining: 12.6s
    245:	learn: 0.7931814	total: 1.13s	remaining: 12.6s
    246:	learn: 0.7923805	total: 1.13s	remaining: 12.6s
    247:	learn: 0.7915907	total: 1.13s	remaining: 12.6s
    248:	learn: 0.7907078	total: 1.14s	remaining: 12.5s
    249:	learn: 0.7899525	total: 1.14s	remaining: 12.5s
    250:	learn: 0.7892572	total: 1.15s	remaining: 12.5s
    251:	learn: 0.7884230	total: 1.15s	remaining: 12.5s
    252:	learn: 0.7877096	total: 1.15s	remaining: 12.5s
    253:	learn: 0.7870301	total: 1.16s	remaining: 12.5s
    254:	learn: 0.7862629	total: 1.16s	remaining: 12.5s
    255:	learn: 0.7856380	total: 1.16s	remaining: 12.5s
    256:	learn: 0.7850095	total: 1.17s	remaining: 12.5s
    257:	learn: 0.7843140	total: 1.17s	remaining: 12.5s
    258:	learn: 0.7836089	total: 1.18s	remaining: 12.5s
    259:	learn: 0.7830269	total: 1.18s	remaining: 12.4s
    260:	learn: 0.7823673	total: 1.18s	remaining: 12.4s
    261:	learn: 0.7816494	total: 1.19s	remaining: 12.4s
    262:	learn: 0.7810329	total: 1.19s	remaining: 12.4s
    263:	learn: 0.7803730	total: 1.19s	remaining: 12.4s
    264:	learn: 0.7797358	total: 1.2s	remaining: 12.3s
    265:	learn: 0.7790034	total: 1.2s	remaining: 12.3s
    266:	learn: 0.7783016	total: 1.2s	remaining: 12.3s
    267:	learn: 0.7776049	total: 1.21s	remaining: 12.3s
    268:	learn: 0.7769716	total: 1.21s	remaining: 12.3s
    269:	learn: 0.7763092	total: 1.21s	remaining: 12.3s
    270:	learn: 0.7755684	total: 1.22s	remaining: 12.2s
    271:	learn: 0.7749153	total: 1.22s	remaining: 12.2s
    272:	learn: 0.7743040	total: 1.22s	remaining: 12.2s
    273:	learn: 0.7736760	total: 1.22s	remaining: 12.2s
    274:	learn: 0.7731072	total: 1.23s	remaining: 12.2s
    275:	learn: 0.7724651	total: 1.23s	remaining: 12.1s
    276:	learn: 0.7718367	total: 1.23s	remaining: 12.1s
    277:	learn: 0.7713609	total: 1.24s	remaining: 12.1s
    278:	learn: 0.7707973	total: 1.24s	remaining: 12.1s
    279:	learn: 0.7702469	total: 1.24s	remaining: 12.1s
    280:	learn: 0.7696029	total: 1.25s	remaining: 12.1s
    281:	learn: 0.7690506	total: 1.25s	remaining: 12s
    282:	learn: 0.7686038	total: 1.26s	remaining: 12.1s
    283:	learn: 0.7681147	total: 1.26s	remaining: 12s
    284:	learn: 0.7676359	total: 1.26s	remaining: 12s
    285:	learn: 0.7671478	total: 1.26s	remaining: 12s
    286:	learn: 0.7665071	total: 1.27s	remaining: 12s
    287:	learn: 0.7659723	total: 1.27s	remaining: 12s
    288:	learn: 0.7654347	total: 1.3s	remaining: 12.2s
    289:	learn: 0.7648161	total: 1.31s	remaining: 12.2s
    290:	learn: 0.7642977	total: 1.32s	remaining: 12.3s
    291:	learn: 0.7637053	total: 1.32s	remaining: 12.3s
    292:	learn: 0.7632287	total: 1.33s	remaining: 12.3s
    293:	learn: 0.7626562	total: 1.34s	remaining: 12.3s
    294:	learn: 0.7621920	total: 1.35s	remaining: 12.4s
    295:	learn: 0.7617347	total: 1.36s	remaining: 12.4s
    296:	learn: 0.7613031	total: 1.36s	remaining: 12.4s
    297:	learn: 0.7607396	total: 1.37s	remaining: 12.5s
    298:	learn: 0.7602234	total: 1.38s	remaining: 12.5s
    299:	learn: 0.7596588	total: 1.39s	remaining: 12.5s
    300:	learn: 0.7591738	total: 1.39s	remaining: 12.5s
    301:	learn: 0.7587142	total: 1.4s	remaining: 12.5s
    302:	learn: 0.7582622	total: 1.4s	remaining: 12.5s
    303:	learn: 0.7577637	total: 1.4s	remaining: 12.4s
    304:	learn: 0.7573410	total: 1.41s	remaining: 12.4s
    305:	learn: 0.7568916	total: 1.41s	remaining: 12.4s
    306:	learn: 0.7565235	total: 1.41s	remaining: 12.4s
    307:	learn: 0.7560894	total: 1.41s	remaining: 12.4s
    308:	learn: 0.7557204	total: 1.42s	remaining: 12.3s
    309:	learn: 0.7553542	total: 1.42s	remaining: 12.3s
    310:	learn: 0.7549409	total: 1.42s	remaining: 12.3s
    311:	learn: 0.7544219	total: 1.43s	remaining: 12.3s
    312:	learn: 0.7540307	total: 1.43s	remaining: 12.3s
    313:	learn: 0.7536116	total: 1.44s	remaining: 12.3s
    314:	learn: 0.7530950	total: 1.44s	remaining: 12.3s
    315:	learn: 0.7526829	total: 1.44s	remaining: 12.3s
    316:	learn: 0.7523290	total: 1.45s	remaining: 12.3s
    317:	learn: 0.7519366	total: 1.46s	remaining: 12.3s
    318:	learn: 0.7515855	total: 1.46s	remaining: 12.3s
    319:	learn: 0.7512090	total: 1.46s	remaining: 12.3s
    320:	learn: 0.7507199	total: 1.47s	remaining: 12.3s
    321:	learn: 0.7503382	total: 1.48s	remaining: 12.3s
    322:	learn: 0.7499440	total: 1.49s	remaining: 12.4s
    323:	learn: 0.7495731	total: 1.51s	remaining: 12.4s
    324:	learn: 0.7491220	total: 1.52s	remaining: 12.6s
    325:	learn: 0.7487597	total: 1.53s	remaining: 12.6s
    326:	learn: 0.7484168	total: 1.55s	remaining: 12.7s
    327:	learn: 0.7480499	total: 1.57s	remaining: 12.8s
    328:	learn: 0.7476105	total: 1.58s	remaining: 12.8s
    329:	learn: 0.7472681	total: 1.59s	remaining: 12.8s
    330:	learn: 0.7468828	total: 1.59s	remaining: 12.8s
    331:	learn: 0.7464538	total: 1.59s	remaining: 12.8s
    332:	learn: 0.7460969	total: 1.6s	remaining: 12.8s
    333:	learn: 0.7457779	total: 1.6s	remaining: 12.8s
    334:	learn: 0.7453685	total: 1.6s	remaining: 12.8s
    335:	learn: 0.7449966	total: 1.61s	remaining: 12.8s
    336:	learn: 0.7446577	total: 1.62s	remaining: 12.8s
    337:	learn: 0.7443643	total: 1.63s	remaining: 12.8s
    338:	learn: 0.7440432	total: 1.63s	remaining: 12.8s
    339:	learn: 0.7437420	total: 1.64s	remaining: 12.8s
    340:	learn: 0.7434075	total: 1.65s	remaining: 12.9s
    341:	learn: 0.7430247	total: 1.66s	remaining: 12.9s
    342:	learn: 0.7425931	total: 1.66s	remaining: 12.9s
    343:	learn: 0.7423201	total: 1.67s	remaining: 12.9s
    344:	learn: 0.7420287	total: 1.68s	remaining: 12.9s
    345:	learn: 0.7417060	total: 1.69s	remaining: 12.9s
    346:	learn: 0.7413676	total: 1.7s	remaining: 13s
    347:	learn: 0.7410758	total: 1.71s	remaining: 13s
    348:	learn: 0.7407420	total: 1.71s	remaining: 13s
    349:	learn: 0.7403574	total: 1.72s	remaining: 13s
    350:	learn: 0.7399936	total: 1.72s	remaining: 13s
    351:	learn: 0.7396502	total: 1.72s	remaining: 13s
    352:	learn: 0.7393497	total: 1.73s	remaining: 13s
    353:	learn: 0.7390001	total: 1.73s	remaining: 12.9s
    354:	learn: 0.7386716	total: 1.73s	remaining: 12.9s
    355:	learn: 0.7383690	total: 1.74s	remaining: 12.9s
    356:	learn: 0.7380995	total: 1.74s	remaining: 12.9s
    357:	learn: 0.7378175	total: 1.74s	remaining: 12.9s
    358:	learn: 0.7374849	total: 1.75s	remaining: 12.9s
    359:	learn: 0.7371642	total: 1.75s	remaining: 12.9s
    360:	learn: 0.7369066	total: 1.75s	remaining: 12.8s
    361:	learn: 0.7366631	total: 1.76s	remaining: 12.8s
    362:	learn: 0.7364023	total: 1.76s	remaining: 12.8s
    363:	learn: 0.7361066	total: 1.76s	remaining: 12.8s
    364:	learn: 0.7358021	total: 1.77s	remaining: 12.8s
    365:	learn: 0.7354183	total: 1.77s	remaining: 12.7s
    366:	learn: 0.7351096	total: 1.77s	remaining: 12.7s
    367:	learn: 0.7348944	total: 1.77s	remaining: 12.7s
    368:	learn: 0.7345716	total: 1.78s	remaining: 12.7s
    369:	learn: 0.7343182	total: 1.78s	remaining: 12.7s
    370:	learn: 0.7340981	total: 1.78s	remaining: 12.6s
    371:	learn: 0.7337937	total: 1.79s	remaining: 12.6s
    372:	learn: 0.7334890	total: 1.79s	remaining: 12.6s
    373:	learn: 0.7331585	total: 1.79s	remaining: 12.6s
    374:	learn: 0.7328916	total: 1.79s	remaining: 12.6s
    375:	learn: 0.7326512	total: 1.8s	remaining: 12.6s
    376:	learn: 0.7323156	total: 1.8s	remaining: 12.5s
    377:	learn: 0.7320591	total: 1.8s	remaining: 12.5s
    378:	learn: 0.7318217	total: 1.81s	remaining: 12.5s
    379:	learn: 0.7314760	total: 1.81s	remaining: 12.5s
    380:	learn: 0.7312304	total: 1.81s	remaining: 12.5s
    381:	learn: 0.7310269	total: 1.81s	remaining: 12.4s
    382:	learn: 0.7308080	total: 1.82s	remaining: 12.4s
    383:	learn: 0.7306134	total: 1.82s	remaining: 12.4s
    384:	learn: 0.7303995	total: 1.82s	remaining: 12.4s
    385:	learn: 0.7302053	total: 1.83s	remaining: 12.4s
    386:	learn: 0.7299740	total: 1.83s	remaining: 12.4s
    387:	learn: 0.7297161	total: 1.83s	remaining: 12.3s
    388:	learn: 0.7294668	total: 1.83s	remaining: 12.3s
    389:	learn: 0.7292246	total: 1.84s	remaining: 12.3s
    390:	learn: 0.7289549	total: 1.84s	remaining: 12.3s
    391:	learn: 0.7287215	total: 1.84s	remaining: 12.3s
    392:	learn: 0.7284405	total: 1.85s	remaining: 12.2s
    393:	learn: 0.7283629	total: 1.85s	remaining: 12.2s
    394:	learn: 0.7281442	total: 1.85s	remaining: 12.2s
    395:	learn: 0.7279152	total: 1.86s	remaining: 12.2s
    396:	learn: 0.7276944	total: 1.86s	remaining: 12.2s
    397:	learn: 0.7274784	total: 1.87s	remaining: 12.2s
    398:	learn: 0.7272589	total: 1.87s	remaining: 12.2s
    399:	learn: 0.7269987	total: 1.88s	remaining: 12.2s
    400:	learn: 0.7268167	total: 1.88s	remaining: 12.2s
    401:	learn: 0.7265556	total: 1.89s	remaining: 12.2s
    402:	learn: 0.7263368	total: 1.89s	remaining: 12.2s
    403:	learn: 0.7261101	total: 1.9s	remaining: 12.2s
    404:	learn: 0.7258815	total: 1.9s	remaining: 12.2s
    405:	learn: 0.7256656	total: 1.9s	remaining: 12.2s
    406:	learn: 0.7253923	total: 1.91s	remaining: 12.1s
    407:	learn: 0.7251853	total: 1.91s	remaining: 12.1s
    408:	learn: 0.7249369	total: 1.92s	remaining: 12.2s
    409:	learn: 0.7246814	total: 1.92s	remaining: 12.1s
    410:	learn: 0.7245219	total: 1.92s	remaining: 12.1s
    411:	learn: 0.7242962	total: 1.93s	remaining: 12.1s
    412:	learn: 0.7240650	total: 1.93s	remaining: 12.1s
    413:	learn: 0.7238763	total: 1.94s	remaining: 12.1s
    414:	learn: 0.7237177	total: 1.94s	remaining: 12.1s
    415:	learn: 0.7234978	total: 1.94s	remaining: 12.1s
    416:	learn: 0.7233290	total: 1.95s	remaining: 12s
    417:	learn: 0.7231555	total: 1.95s	remaining: 12s
    418:	learn: 0.7229960	total: 1.95s	remaining: 12s
    419:	learn: 0.7227948	total: 1.95s	remaining: 12s
    420:	learn: 0.7225762	total: 1.96s	remaining: 12s
    421:	learn: 0.7224248	total: 1.96s	remaining: 12s
    422:	learn: 0.7222680	total: 1.97s	remaining: 12s
    423:	learn: 0.7219828	total: 1.97s	remaining: 12s
    424:	learn: 0.7218238	total: 1.97s	remaining: 12s
    425:	learn: 0.7216520	total: 1.98s	remaining: 12s
    426:	learn: 0.7214122	total: 1.99s	remaining: 12s
    427:	learn: 0.7211724	total: 1.99s	remaining: 12s
    428:	learn: 0.7210422	total: 1.99s	remaining: 11.9s
    429:	learn: 0.7207157	total: 2s	remaining: 11.9s
    430:	learn: 0.7205253	total: 2s	remaining: 11.9s
    431:	learn: 0.7202735	total: 2s	remaining: 11.9s
    432:	learn: 0.7200925	total: 2s	remaining: 11.9s
    433:	learn: 0.7198816	total: 2.01s	remaining: 11.9s
    434:	learn: 0.7196293	total: 2.01s	remaining: 11.9s
    435:	learn: 0.7193104	total: 2.02s	remaining: 11.9s
    436:	learn: 0.7190928	total: 2.02s	remaining: 11.8s
    437:	learn: 0.7188594	total: 2.02s	remaining: 11.8s
    438:	learn: 0.7186944	total: 2.02s	remaining: 11.8s
    439:	learn: 0.7184836	total: 2.03s	remaining: 11.8s
    440:	learn: 0.7183444	total: 2.03s	remaining: 11.8s
    441:	learn: 0.7181443	total: 2.03s	remaining: 11.8s
    442:	learn: 0.7179564	total: 2.04s	remaining: 11.8s
    443:	learn: 0.7178244	total: 2.04s	remaining: 11.7s
    444:	learn: 0.7175492	total: 2.04s	remaining: 11.7s
    445:	learn: 0.7173691	total: 2.04s	remaining: 11.7s
    446:	learn: 0.7172000	total: 2.05s	remaining: 11.7s
    447:	learn: 0.7169874	total: 2.05s	remaining: 11.7s
    448:	learn: 0.7168513	total: 2.06s	remaining: 11.7s
    449:	learn: 0.7166171	total: 2.07s	remaining: 11.7s
    450:	learn: 0.7164845	total: 2.09s	remaining: 11.8s
    451:	learn: 0.7162926	total: 2.1s	remaining: 11.8s
    452:	learn: 0.7161646	total: 2.1s	remaining: 11.8s
    453:	learn: 0.7159571	total: 2.11s	remaining: 11.8s
    454:	learn: 0.7157304	total: 2.11s	remaining: 11.8s
    455:	learn: 0.7155536	total: 2.12s	remaining: 11.8s
    456:	learn: 0.7154125	total: 2.13s	remaining: 11.8s
    457:	learn: 0.7152332	total: 2.13s	remaining: 11.8s
    458:	learn: 0.7150311	total: 2.13s	remaining: 11.8s
    459:	learn: 0.7149046	total: 2.13s	remaining: 11.8s
    460:	learn: 0.7147326	total: 2.14s	remaining: 11.8s
    461:	learn: 0.7146274	total: 2.15s	remaining: 11.8s
    462:	learn: 0.7144803	total: 2.15s	remaining: 11.8s
    463:	learn: 0.7143546	total: 2.15s	remaining: 11.8s
    464:	learn: 0.7141575	total: 2.16s	remaining: 11.8s
    465:	learn: 0.7140107	total: 2.16s	remaining: 11.7s
    466:	learn: 0.7138812	total: 2.16s	remaining: 11.7s
    467:	learn: 0.7137535	total: 2.17s	remaining: 11.7s
    468:	learn: 0.7136202	total: 2.17s	remaining: 11.7s
    469:	learn: 0.7134793	total: 2.17s	remaining: 11.7s
    470:	learn: 0.7133353	total: 2.17s	remaining: 11.7s
    471:	learn: 0.7131947	total: 2.18s	remaining: 11.7s
    472:	learn: 0.7130426	total: 2.18s	remaining: 11.7s
    473:	learn: 0.7129185	total: 2.18s	remaining: 11.6s
    474:	learn: 0.7127263	total: 2.19s	remaining: 11.6s
    475:	learn: 0.7126209	total: 2.19s	remaining: 11.6s
    476:	learn: 0.7124848	total: 2.19s	remaining: 11.6s
    477:	learn: 0.7123229	total: 2.2s	remaining: 11.6s
    478:	learn: 0.7121940	total: 2.2s	remaining: 11.6s
    479:	learn: 0.7120692	total: 2.2s	remaining: 11.6s
    480:	learn: 0.7119672	total: 2.21s	remaining: 11.6s
    481:	learn: 0.7118463	total: 2.21s	remaining: 11.5s
    482:	learn: 0.7116808	total: 2.21s	remaining: 11.5s
    483:	learn: 0.7115126	total: 2.22s	remaining: 11.5s
    484:	learn: 0.7113868	total: 2.22s	remaining: 11.5s
    485:	learn: 0.7112569	total: 2.23s	remaining: 11.5s
    486:	learn: 0.7110794	total: 2.23s	remaining: 11.5s
    487:	learn: 0.7109694	total: 2.23s	remaining: 11.5s
    488:	learn: 0.7108809	total: 2.24s	remaining: 11.5s
    489:	learn: 0.7107127	total: 2.24s	remaining: 11.5s
    490:	learn: 0.7105595	total: 2.24s	remaining: 11.5s
    491:	learn: 0.7104052	total: 2.25s	remaining: 11.4s
    492:	learn: 0.7102967	total: 2.25s	remaining: 11.4s
    493:	learn: 0.7101381	total: 2.28s	remaining: 11.6s
    494:	learn: 0.7100274	total: 2.29s	remaining: 11.6s
    495:	learn: 0.7099197	total: 2.3s	remaining: 11.6s
    496:	learn: 0.7097839	total: 2.31s	remaining: 11.6s
    497:	learn: 0.7096497	total: 2.32s	remaining: 11.6s
    498:	learn: 0.7095635	total: 2.33s	remaining: 11.7s
    499:	learn: 0.7093804	total: 2.34s	remaining: 11.7s
    500:	learn: 0.7092449	total: 2.34s	remaining: 11.7s
    501:	learn: 0.7091292	total: 2.35s	remaining: 11.7s
    502:	learn: 0.7090175	total: 2.36s	remaining: 11.7s
    503:	learn: 0.7089199	total: 2.37s	remaining: 11.7s
    504:	learn: 0.7087891	total: 2.37s	remaining: 11.7s
    505:	learn: 0.7086553	total: 2.37s	remaining: 11.7s
    506:	learn: 0.7084366	total: 2.37s	remaining: 11.7s
    507:	learn: 0.7082000	total: 2.38s	remaining: 11.7s
    508:	learn: 0.7080916	total: 2.38s	remaining: 11.7s
    509:	learn: 0.7079110	total: 2.39s	remaining: 11.7s
    510:	learn: 0.7077719	total: 2.39s	remaining: 11.6s
    511:	learn: 0.7076809	total: 2.4s	remaining: 11.7s
    512:	learn: 0.7075890	total: 2.4s	remaining: 11.7s
    513:	learn: 0.7074405	total: 2.41s	remaining: 11.6s
    514:	learn: 0.7072954	total: 2.41s	remaining: 11.6s
    515:	learn: 0.7071503	total: 2.41s	remaining: 11.6s
    516:	learn: 0.7070330	total: 2.42s	remaining: 11.6s
    517:	learn: 0.7069309	total: 2.42s	remaining: 11.6s
    518:	learn: 0.7068036	total: 2.42s	remaining: 11.6s
    519:	learn: 0.7067119	total: 2.42s	remaining: 11.6s
    520:	learn: 0.7064729	total: 2.43s	remaining: 11.6s
    521:	learn: 0.7063115	total: 2.43s	remaining: 11.5s
    522:	learn: 0.7061835	total: 2.43s	remaining: 11.5s
    523:	learn: 0.7060936	total: 2.44s	remaining: 11.5s
    524:	learn: 0.7059761	total: 2.44s	remaining: 11.5s
    525:	learn: 0.7058275	total: 2.44s	remaining: 11.5s
    526:	learn: 0.7056893	total: 2.45s	remaining: 11.5s
    527:	learn: 0.7055547	total: 2.45s	remaining: 11.5s
    528:	learn: 0.7053641	total: 2.45s	remaining: 11.5s
    529:	learn: 0.7052011	total: 2.46s	remaining: 11.4s
    530:	learn: 0.7051089	total: 2.46s	remaining: 11.4s
    531:	learn: 0.7049459	total: 2.46s	remaining: 11.4s
    532:	learn: 0.7048588	total: 2.46s	remaining: 11.4s
    533:	learn: 0.7047264	total: 2.47s	remaining: 11.4s
    534:	learn: 0.7046034	total: 2.47s	remaining: 11.4s
    535:	learn: 0.7044799	total: 2.47s	remaining: 11.4s
    536:	learn: 0.7042215	total: 2.48s	remaining: 11.4s
    537:	learn: 0.7040476	total: 2.48s	remaining: 11.4s
    538:	learn: 0.7039181	total: 2.48s	remaining: 11.3s
    539:	learn: 0.7037653	total: 2.49s	remaining: 11.3s
    540:	learn: 0.7036272	total: 2.49s	remaining: 11.3s
    541:	learn: 0.7035399	total: 2.5s	remaining: 11.3s
    542:	learn: 0.7034527	total: 2.51s	remaining: 11.3s
    543:	learn: 0.7032587	total: 2.51s	remaining: 11.4s
    544:	learn: 0.7031565	total: 2.52s	remaining: 11.3s
    545:	learn: 0.7030130	total: 2.52s	remaining: 11.3s
    546:	learn: 0.7029176	total: 2.53s	remaining: 11.3s
    547:	learn: 0.7028102	total: 2.53s	remaining: 11.3s
    548:	learn: 0.7026510	total: 2.53s	remaining: 11.3s
    549:	learn: 0.7025190	total: 2.54s	remaining: 11.3s
    550:	learn: 0.7024337	total: 2.54s	remaining: 11.3s
    551:	learn: 0.7023440	total: 2.54s	remaining: 11.3s
    552:	learn: 0.7021962	total: 2.55s	remaining: 11.3s
    553:	learn: 0.7020775	total: 2.55s	remaining: 11.3s
    554:	learn: 0.7019170	total: 2.56s	remaining: 11.3s
    555:	learn: 0.7018263	total: 2.56s	remaining: 11.2s
    556:	learn: 0.7017085	total: 2.56s	remaining: 11.2s
    557:	learn: 0.7015566	total: 2.56s	remaining: 11.2s
    558:	learn: 0.7014311	total: 2.57s	remaining: 11.2s
    559:	learn: 0.7013281	total: 2.57s	remaining: 11.2s
    560:	learn: 0.7012190	total: 2.57s	remaining: 11.2s
    561:	learn: 0.7011299	total: 2.58s	remaining: 11.2s
    562:	learn: 0.7010526	total: 2.59s	remaining: 11.2s
    563:	learn: 0.7008909	total: 2.59s	remaining: 11.2s
    564:	learn: 0.7008068	total: 2.6s	remaining: 11.2s
    565:	learn: 0.7007063	total: 2.61s	remaining: 11.2s
    566:	learn: 0.7006202	total: 2.62s	remaining: 11.2s
    567:	learn: 0.7004836	total: 2.63s	remaining: 11.2s
    568:	learn: 0.7003623	total: 2.63s	remaining: 11.3s
    569:	learn: 0.7002299	total: 2.64s	remaining: 11.3s
    570:	learn: 0.7001119	total: 2.65s	remaining: 11.3s
    571:	learn: 0.7000115	total: 2.65s	remaining: 11.3s
    572:	learn: 0.6999085	total: 2.66s	remaining: 11.3s
    573:	learn: 0.6998486	total: 2.67s	remaining: 11.3s
    574:	learn: 0.6997348	total: 2.67s	remaining: 11.3s
    575:	learn: 0.6995783	total: 2.68s	remaining: 11.3s
    576:	learn: 0.6994683	total: 2.69s	remaining: 11.3s
    577:	learn: 0.6992910	total: 2.69s	remaining: 11.3s
    578:	learn: 0.6992046	total: 2.7s	remaining: 11.3s
    579:	learn: 0.6990390	total: 2.7s	remaining: 11.3s
    580:	learn: 0.6989466	total: 2.7s	remaining: 11.3s
    581:	learn: 0.6988495	total: 2.7s	remaining: 11.2s
    582:	learn: 0.6987809	total: 2.71s	remaining: 11.2s
    583:	learn: 0.6986738	total: 2.71s	remaining: 11.2s
    584:	learn: 0.6985349	total: 2.72s	remaining: 11.2s
    585:	learn: 0.6984036	total: 2.72s	remaining: 11.2s
    586:	learn: 0.6982500	total: 2.72s	remaining: 11.2s
    587:	learn: 0.6980978	total: 2.73s	remaining: 11.2s
    588:	learn: 0.6980255	total: 2.73s	remaining: 11.2s
    589:	learn: 0.6979574	total: 2.73s	remaining: 11.2s
    590:	learn: 0.6978640	total: 2.73s	remaining: 11.2s
    591:	learn: 0.6977973	total: 2.74s	remaining: 11.1s
    592:	learn: 0.6977380	total: 2.74s	remaining: 11.1s
    593:	learn: 0.6975640	total: 2.74s	remaining: 11.1s
    594:	learn: 0.6974737	total: 2.75s	remaining: 11.1s
    595:	learn: 0.6974173	total: 2.75s	remaining: 11.1s
    596:	learn: 0.6972886	total: 2.75s	remaining: 11.1s
    597:	learn: 0.6971801	total: 2.76s	remaining: 11.1s
    598:	learn: 0.6970459	total: 2.76s	remaining: 11.1s
    599:	learn: 0.6969711	total: 2.76s	remaining: 11s
    600:	learn: 0.6968901	total: 2.77s	remaining: 11s
    601:	learn: 0.6967678	total: 2.77s	remaining: 11s
    602:	learn: 0.6965804	total: 2.77s	remaining: 11s
    603:	learn: 0.6964678	total: 2.77s	remaining: 11s
    604:	learn: 0.6964086	total: 2.78s	remaining: 11s
    605:	learn: 0.6963236	total: 2.78s	remaining: 11s
    606:	learn: 0.6961957	total: 2.78s	remaining: 11s
    607:	learn: 0.6961180	total: 2.79s	remaining: 11s
    608:	learn: 0.6959824	total: 2.79s	remaining: 11s
    609:	learn: 0.6958577	total: 2.79s	remaining: 10.9s
    610:	learn: 0.6957475	total: 2.79s	remaining: 10.9s
    611:	learn: 0.6956338	total: 2.8s	remaining: 10.9s
    612:	learn: 0.6955601	total: 2.8s	remaining: 10.9s
    613:	learn: 0.6954079	total: 2.8s	remaining: 10.9s
    614:	learn: 0.6953074	total: 2.81s	remaining: 10.9s
    615:	learn: 0.6951964	total: 2.81s	remaining: 10.9s
    616:	learn: 0.6951246	total: 2.81s	remaining: 10.9s
    617:	learn: 0.6950315	total: 2.82s	remaining: 10.9s
    618:	learn: 0.6949260	total: 2.82s	remaining: 10.8s
    619:	learn: 0.6948543	total: 2.82s	remaining: 10.8s
    620:	learn: 0.6946880	total: 2.83s	remaining: 10.8s
    621:	learn: 0.6945717	total: 2.83s	remaining: 10.8s
    622:	learn: 0.6944706	total: 2.84s	remaining: 10.8s
    623:	learn: 0.6942892	total: 2.85s	remaining: 10.9s
    624:	learn: 0.6942280	total: 2.86s	remaining: 10.9s
    625:	learn: 0.6941294	total: 2.86s	remaining: 10.9s
    626:	learn: 0.6940218	total: 2.87s	remaining: 10.9s
    627:	learn: 0.6939146	total: 2.87s	remaining: 10.8s
    628:	learn: 0.6938614	total: 2.88s	remaining: 10.9s
    629:	learn: 0.6937199	total: 2.89s	remaining: 10.9s
    630:	learn: 0.6936359	total: 2.89s	remaining: 10.9s
    631:	learn: 0.6935603	total: 2.9s	remaining: 10.9s
    632:	learn: 0.6935024	total: 2.9s	remaining: 10.9s
    633:	learn: 0.6934257	total: 2.91s	remaining: 10.8s
    634:	learn: 0.6933200	total: 2.91s	remaining: 10.8s
    635:	learn: 0.6932086	total: 2.91s	remaining: 10.8s
    636:	learn: 0.6931074	total: 2.92s	remaining: 10.8s
    637:	learn: 0.6930542	total: 2.92s	remaining: 10.8s
    638:	learn: 0.6929380	total: 2.92s	remaining: 10.8s
    639:	learn: 0.6928520	total: 2.93s	remaining: 10.8s
    640:	learn: 0.6927501	total: 2.93s	remaining: 10.8s
    641:	learn: 0.6926166	total: 2.93s	remaining: 10.8s
    642:	learn: 0.6925563	total: 2.93s	remaining: 10.8s
    643:	learn: 0.6924867	total: 2.94s	remaining: 10.7s
    644:	learn: 0.6923538	total: 2.94s	remaining: 10.7s
    645:	learn: 0.6922995	total: 2.94s	remaining: 10.7s
    646:	learn: 0.6922197	total: 2.95s	remaining: 10.7s
    647:	learn: 0.6921658	total: 2.96s	remaining: 10.7s
    648:	learn: 0.6920243	total: 2.96s	remaining: 10.7s
    649:	learn: 0.6919348	total: 2.96s	remaining: 10.7s
    650:	learn: 0.6918694	total: 2.97s	remaining: 10.7s
    651:	learn: 0.6917693	total: 2.97s	remaining: 10.7s
    652:	learn: 0.6916929	total: 2.97s	remaining: 10.7s
    653:	learn: 0.6916294	total: 2.98s	remaining: 10.7s
    654:	learn: 0.6915699	total: 2.98s	remaining: 10.7s
    655:	learn: 0.6915167	total: 2.98s	remaining: 10.7s
    656:	learn: 0.6914349	total: 2.99s	remaining: 10.7s
    657:	learn: 0.6913759	total: 2.99s	remaining: 10.6s
    658:	learn: 0.6912729	total: 2.99s	remaining: 10.6s
    659:	learn: 0.6911842	total: 3s	remaining: 10.6s
    660:	learn: 0.6911325	total: 3s	remaining: 10.6s
    661:	learn: 0.6910459	total: 3s	remaining: 10.6s
    662:	learn: 0.6909974	total: 3.01s	remaining: 10.6s
    663:	learn: 0.6909097	total: 3.01s	remaining: 10.6s
    664:	learn: 0.6908489	total: 3.01s	remaining: 10.6s
    665:	learn: 0.6907306	total: 3.02s	remaining: 10.6s
    666:	learn: 0.6906501	total: 3.02s	remaining: 10.6s
    667:	learn: 0.6905645	total: 3.02s	remaining: 10.6s
    668:	learn: 0.6904689	total: 3.02s	remaining: 10.5s
    669:	learn: 0.6903711	total: 3.03s	remaining: 10.5s
    670:	learn: 0.6902716	total: 3.03s	remaining: 10.5s
    671:	learn: 0.6902169	total: 3.03s	remaining: 10.5s
    672:	learn: 0.6901470	total: 3.04s	remaining: 10.5s
    673:	learn: 0.6900841	total: 3.04s	remaining: 10.5s
    674:	learn: 0.6900138	total: 3.04s	remaining: 10.5s
    675:	learn: 0.6899421	total: 3.05s	remaining: 10.5s
    676:	learn: 0.6898819	total: 3.05s	remaining: 10.5s
    677:	learn: 0.6898101	total: 3.05s	remaining: 10.5s
    678:	learn: 0.6897477	total: 3.06s	remaining: 10.4s
    679:	learn: 0.6896908	total: 3.06s	remaining: 10.4s
    680:	learn: 0.6895881	total: 3.06s	remaining: 10.4s
    681:	learn: 0.6894867	total: 3.06s	remaining: 10.4s
    682:	learn: 0.6893896	total: 3.07s	remaining: 10.4s
    683:	learn: 0.6893015	total: 3.08s	remaining: 10.4s
    684:	learn: 0.6892316	total: 3.08s	remaining: 10.4s
    685:	learn: 0.6890849	total: 3.09s	remaining: 10.4s
    686:	learn: 0.6890131	total: 3.1s	remaining: 10.4s
    687:	learn: 0.6889635	total: 3.11s	remaining: 10.4s
    688:	learn: 0.6889001	total: 3.11s	remaining: 10.4s
    689:	learn: 0.6888155	total: 3.12s	remaining: 10.4s
    690:	learn: 0.6887238	total: 3.12s	remaining: 10.4s
    691:	learn: 0.6886336	total: 3.12s	remaining: 10.4s
    692:	learn: 0.6885716	total: 3.13s	remaining: 10.4s
    693:	learn: 0.6884660	total: 3.13s	remaining: 10.4s
    694:	learn: 0.6883689	total: 3.13s	remaining: 10.4s
    695:	learn: 0.6882666	total: 3.14s	remaining: 10.4s
    696:	learn: 0.6881517	total: 3.14s	remaining: 10.4s
    697:	learn: 0.6880272	total: 3.14s	remaining: 10.4s
    698:	learn: 0.6879699	total: 3.15s	remaining: 10.4s
    699:	learn: 0.6878165	total: 3.15s	remaining: 10.3s
    700:	learn: 0.6877206	total: 3.15s	remaining: 10.3s
    701:	learn: 0.6875980	total: 3.15s	remaining: 10.3s
    702:	learn: 0.6875472	total: 3.16s	remaining: 10.3s
    703:	learn: 0.6874212	total: 3.16s	remaining: 10.3s
    704:	learn: 0.6872927	total: 3.16s	remaining: 10.3s
    705:	learn: 0.6872529	total: 3.17s	remaining: 10.3s
    706:	learn: 0.6871509	total: 3.17s	remaining: 10.3s
    707:	learn: 0.6870686	total: 3.17s	remaining: 10.3s
    708:	learn: 0.6869628	total: 3.17s	remaining: 10.3s
    709:	learn: 0.6868901	total: 3.18s	remaining: 10.3s
    710:	learn: 0.6868427	total: 3.18s	remaining: 10.2s
    711:	learn: 0.6867631	total: 3.19s	remaining: 10.2s
    712:	learn: 0.6866608	total: 3.19s	remaining: 10.2s
    713:	learn: 0.6866005	total: 3.19s	remaining: 10.2s
    714:	learn: 0.6865390	total: 3.19s	remaining: 10.2s
    715:	learn: 0.6864380	total: 3.2s	remaining: 10.2s
    716:	learn: 0.6863773	total: 3.2s	remaining: 10.2s
    717:	learn: 0.6862901	total: 3.2s	remaining: 10.2s
    718:	learn: 0.6862518	total: 3.21s	remaining: 10.2s
    719:	learn: 0.6861966	total: 3.21s	remaining: 10.2s
    720:	learn: 0.6861279	total: 3.21s	remaining: 10.2s
    721:	learn: 0.6860212	total: 3.23s	remaining: 10.2s
    722:	learn: 0.6859030	total: 3.25s	remaining: 10.2s
    723:	learn: 0.6858215	total: 3.25s	remaining: 10.2s
    724:	learn: 0.6857391	total: 3.26s	remaining: 10.2s
    725:	learn: 0.6856731	total: 3.27s	remaining: 10.3s
    726:	learn: 0.6856226	total: 3.28s	remaining: 10.3s
    727:	learn: 0.6855310	total: 3.29s	remaining: 10.3s
    728:	learn: 0.6854634	total: 3.3s	remaining: 10.3s
    729:	learn: 0.6854209	total: 3.31s	remaining: 10.3s
    730:	learn: 0.6853368	total: 3.32s	remaining: 10.3s
    731:	learn: 0.6852709	total: 3.32s	remaining: 10.3s
    732:	learn: 0.6852206	total: 3.33s	remaining: 10.3s
    733:	learn: 0.6851268	total: 3.33s	remaining: 10.3s
    734:	learn: 0.6850583	total: 3.33s	remaining: 10.3s
    735:	learn: 0.6849166	total: 3.33s	remaining: 10.3s
    736:	learn: 0.6847979	total: 3.34s	remaining: 10.2s
    737:	learn: 0.6847057	total: 3.34s	remaining: 10.2s
    738:	learn: 0.6846060	total: 3.35s	remaining: 10.2s
    739:	learn: 0.6845426	total: 3.35s	remaining: 10.2s
    740:	learn: 0.6844558	total: 3.35s	remaining: 10.2s
    741:	learn: 0.6844017	total: 3.36s	remaining: 10.2s
    742:	learn: 0.6843168	total: 3.36s	remaining: 10.2s
    743:	learn: 0.6842661	total: 3.37s	remaining: 10.2s
    744:	learn: 0.6841953	total: 3.38s	remaining: 10.2s
    745:	learn: 0.6841254	total: 3.38s	remaining: 10.2s
    746:	learn: 0.6840660	total: 3.38s	remaining: 10.2s
    747:	learn: 0.6839620	total: 3.39s	remaining: 10.2s
    748:	learn: 0.6838980	total: 3.39s	remaining: 10.2s
    749:	learn: 0.6838051	total: 3.4s	remaining: 10.2s
    750:	learn: 0.6837376	total: 3.4s	remaining: 10.2s
    751:	learn: 0.6836580	total: 3.4s	remaining: 10.2s
    752:	learn: 0.6835657	total: 3.41s	remaining: 10.2s
    753:	learn: 0.6834995	total: 3.41s	remaining: 10.2s
    754:	learn: 0.6834536	total: 3.41s	remaining: 10.2s
    755:	learn: 0.6833872	total: 3.42s	remaining: 10.1s
    756:	learn: 0.6832561	total: 3.42s	remaining: 10.1s
    757:	learn: 0.6832048	total: 3.42s	remaining: 10.1s
    758:	learn: 0.6831304	total: 3.43s	remaining: 10.1s
    759:	learn: 0.6830760	total: 3.43s	remaining: 10.1s
    760:	learn: 0.6829722	total: 3.43s	remaining: 10.1s
    761:	learn: 0.6829057	total: 3.44s	remaining: 10.1s
    762:	learn: 0.6828006	total: 3.44s	remaining: 10.1s
    763:	learn: 0.6827040	total: 3.44s	remaining: 10.1s
    764:	learn: 0.6826359	total: 3.45s	remaining: 10.1s
    765:	learn: 0.6825464	total: 3.45s	remaining: 10.1s
    766:	learn: 0.6824172	total: 3.45s	remaining: 10.1s
    767:	learn: 0.6822989	total: 3.46s	remaining: 10.1s
    768:	learn: 0.6822312	total: 3.46s	remaining: 10s
    769:	learn: 0.6821569	total: 3.46s	remaining: 10s
    770:	learn: 0.6820399	total: 3.47s	remaining: 10s
    771:	learn: 0.6819888	total: 3.48s	remaining: 10s
    772:	learn: 0.6818424	total: 3.49s	remaining: 10s
    773:	learn: 0.6817345	total: 3.49s	remaining: 10s
    774:	learn: 0.6816840	total: 3.49s	remaining: 10s
    775:	learn: 0.6816448	total: 3.5s	remaining: 10s
    776:	learn: 0.6815741	total: 3.5s	remaining: 10s
    777:	learn: 0.6814999	total: 3.51s	remaining: 10s
    778:	learn: 0.6814093	total: 3.52s	remaining: 10s
    779:	learn: 0.6813337	total: 3.52s	remaining: 10s
    780:	learn: 0.6812631	total: 3.52s	remaining: 10s
    781:	learn: 0.6811363	total: 3.53s	remaining: 10s
    782:	learn: 0.6810341	total: 3.53s	remaining: 9.99s
    783:	learn: 0.6809809	total: 3.53s	remaining: 9.98s
    784:	learn: 0.6808560	total: 3.54s	remaining: 9.97s
    785:	learn: 0.6807997	total: 3.54s	remaining: 9.97s
    786:	learn: 0.6807403	total: 3.54s	remaining: 9.96s
    787:	learn: 0.6806980	total: 3.54s	remaining: 9.95s
    788:	learn: 0.6805962	total: 3.55s	remaining: 9.95s
    789:	learn: 0.6805099	total: 3.56s	remaining: 9.96s
    790:	learn: 0.6804244	total: 3.57s	remaining: 9.96s
    791:	learn: 0.6803515	total: 3.58s	remaining: 9.97s
    792:	learn: 0.6802411	total: 3.58s	remaining: 9.97s
    793:	learn: 0.6801152	total: 3.59s	remaining: 9.98s
    794:	learn: 0.6800517	total: 3.6s	remaining: 9.98s
    795:	learn: 0.6799494	total: 3.61s	remaining: 9.98s
    796:	learn: 0.6798536	total: 3.61s	remaining: 9.99s
    797:	learn: 0.6797558	total: 3.62s	remaining: 10s
    798:	learn: 0.6796665	total: 3.63s	remaining: 10s
    799:	learn: 0.6795588	total: 3.64s	remaining: 10s
    800:	learn: 0.6794789	total: 3.64s	remaining: 10s
    801:	learn: 0.6793998	total: 3.65s	remaining: 9.99s
    802:	learn: 0.6793257	total: 3.65s	remaining: 9.98s
    803:	learn: 0.6792912	total: 3.66s	remaining: 9.99s
    804:	learn: 0.6792493	total: 3.66s	remaining: 9.99s
    805:	learn: 0.6791354	total: 3.69s	remaining: 10s
    806:	learn: 0.6790563	total: 3.7s	remaining: 10s
    807:	learn: 0.6789465	total: 3.71s	remaining: 10.1s
    808:	learn: 0.6788358	total: 3.72s	remaining: 10.1s
    809:	learn: 0.6787033	total: 3.73s	remaining: 10.1s
    810:	learn: 0.6786587	total: 3.75s	remaining: 10.1s
    811:	learn: 0.6785812	total: 3.77s	remaining: 10.1s
    812:	learn: 0.6785477	total: 3.78s	remaining: 10.2s
    813:	learn: 0.6785237	total: 3.79s	remaining: 10.2s
    814:	learn: 0.6784795	total: 3.8s	remaining: 10.2s
    815:	learn: 0.6784081	total: 3.82s	remaining: 10.2s
    816:	learn: 0.6783247	total: 3.84s	remaining: 10.3s
    817:	learn: 0.6782745	total: 3.85s	remaining: 10.3s
    818:	learn: 0.6782097	total: 3.86s	remaining: 10.3s
    819:	learn: 0.6781556	total: 3.87s	remaining: 10.3s
    820:	learn: 0.6780644	total: 3.9s	remaining: 10.3s
    821:	learn: 0.6779884	total: 3.91s	remaining: 10.4s
    822:	learn: 0.6779106	total: 3.92s	remaining: 10.4s
    823:	learn: 0.6778581	total: 3.93s	remaining: 10.4s
    824:	learn: 0.6777740	total: 3.94s	remaining: 10.4s
    825:	learn: 0.6776670	total: 3.94s	remaining: 10.4s
    826:	learn: 0.6775738	total: 3.94s	remaining: 10.4s
    827:	learn: 0.6774270	total: 3.95s	remaining: 10.4s
    828:	learn: 0.6773302	total: 3.95s	remaining: 10.3s
    829:	learn: 0.6772738	total: 3.95s	remaining: 10.3s
    830:	learn: 0.6772380	total: 3.95s	remaining: 10.3s
    831:	learn: 0.6771319	total: 3.96s	remaining: 10.3s
    832:	learn: 0.6770868	total: 3.96s	remaining: 10.3s
    833:	learn: 0.6770019	total: 3.96s	remaining: 10.3s
    834:	learn: 0.6769658	total: 3.96s	remaining: 10.3s
    835:	learn: 0.6768893	total: 3.97s	remaining: 10.3s
    836:	learn: 0.6768226	total: 3.97s	remaining: 10.3s
    837:	learn: 0.6767704	total: 3.97s	remaining: 10.3s
    838:	learn: 0.6767007	total: 3.98s	remaining: 10.2s
    839:	learn: 0.6766305	total: 3.98s	remaining: 10.2s
    840:	learn: 0.6764322	total: 3.98s	remaining: 10.2s
    841:	learn: 0.6763834	total: 3.98s	remaining: 10.2s
    842:	learn: 0.6763114	total: 3.99s	remaining: 10.2s
    843:	learn: 0.6762381	total: 3.99s	remaining: 10.2s
    844:	learn: 0.6761459	total: 3.99s	remaining: 10.2s
    845:	learn: 0.6760559	total: 4s	remaining: 10.2s
    846:	learn: 0.6759900	total: 4s	remaining: 10.2s
    847:	learn: 0.6758759	total: 4s	remaining: 10.2s
    848:	learn: 0.6757745	total: 4s	remaining: 10.1s
    849:	learn: 0.6756087	total: 4.01s	remaining: 10.1s
    850:	learn: 0.6755427	total: 4.01s	remaining: 10.1s
    851:	learn: 0.6754650	total: 4.01s	remaining: 10.1s
    852:	learn: 0.6754103	total: 4.01s	remaining: 10.1s
    853:	learn: 0.6753227	total: 4.02s	remaining: 10.1s
    854:	learn: 0.6752138	total: 4.02s	remaining: 10.1s
    855:	learn: 0.6751312	total: 4.02s	remaining: 10.1s
    856:	learn: 0.6750372	total: 4.03s	remaining: 10.1s
    857:	learn: 0.6749389	total: 4.03s	remaining: 10.1s
    858:	learn: 0.6748403	total: 4.03s	remaining: 10s
    859:	learn: 0.6747708	total: 4.03s	remaining: 10s
    860:	learn: 0.6747285	total: 4.04s	remaining: 10s
    861:	learn: 0.6746418	total: 4.04s	remaining: 10s
    862:	learn: 0.6745181	total: 4.04s	remaining: 10s
    863:	learn: 0.6744476	total: 4.04s	remaining: 10s
    864:	learn: 0.6743977	total: 4.05s	remaining: 9.99s
    865:	learn: 0.6743590	total: 4.05s	remaining: 9.98s
    866:	learn: 0.6742827	total: 4.05s	remaining: 9.97s
    867:	learn: 0.6742260	total: 4.06s	remaining: 9.96s
    868:	learn: 0.6741391	total: 4.06s	remaining: 9.95s
    869:	learn: 0.6740739	total: 4.06s	remaining: 9.95s
    870:	learn: 0.6740349	total: 4.07s	remaining: 9.94s
    871:	learn: 0.6739655	total: 4.07s	remaining: 9.93s
    872:	learn: 0.6739328	total: 4.07s	remaining: 9.92s
    873:	learn: 0.6738450	total: 4.07s	remaining: 9.91s
    874:	learn: 0.6737807	total: 4.08s	remaining: 9.9s
    875:	learn: 0.6736556	total: 4.08s	remaining: 9.9s
    876:	learn: 0.6736036	total: 4.09s	remaining: 9.89s
    877:	learn: 0.6735481	total: 4.09s	remaining: 9.88s
    878:	learn: 0.6734654	total: 4.09s	remaining: 9.88s
    879:	learn: 0.6733823	total: 4.1s	remaining: 9.88s
    880:	learn: 0.6733380	total: 4.11s	remaining: 9.88s
    881:	learn: 0.6732752	total: 4.12s	remaining: 9.89s
    882:	learn: 0.6732005	total: 4.12s	remaining: 9.88s
    883:	learn: 0.6731466	total: 4.12s	remaining: 9.87s
    884:	learn: 0.6730361	total: 4.13s	remaining: 9.86s
    885:	learn: 0.6729672	total: 4.13s	remaining: 9.85s
    886:	learn: 0.6729039	total: 4.13s	remaining: 9.84s
    887:	learn: 0.6728242	total: 4.13s	remaining: 9.83s
    888:	learn: 0.6727506	total: 4.14s	remaining: 9.83s
    889:	learn: 0.6726834	total: 4.14s	remaining: 9.82s
    890:	learn: 0.6726199	total: 4.14s	remaining: 9.81s
    891:	learn: 0.6725444	total: 4.15s	remaining: 9.8s
    892:	learn: 0.6724517	total: 4.15s	remaining: 9.79s
    893:	learn: 0.6723183	total: 4.15s	remaining: 9.78s
    894:	learn: 0.6722710	total: 4.16s	remaining: 9.78s
    895:	learn: 0.6722214	total: 4.16s	remaining: 9.77s
    896:	learn: 0.6721804	total: 4.16s	remaining: 9.76s
    897:	learn: 0.6720667	total: 4.17s	remaining: 9.75s
    898:	learn: 0.6719937	total: 4.17s	remaining: 9.74s
    899:	learn: 0.6719593	total: 4.17s	remaining: 9.73s
    900:	learn: 0.6718870	total: 4.17s	remaining: 9.72s
    901:	learn: 0.6718571	total: 4.18s	remaining: 9.72s
    902:	learn: 0.6717975	total: 4.18s	remaining: 9.71s
    903:	learn: 0.6717258	total: 4.18s	remaining: 9.7s
    904:	learn: 0.6716310	total: 4.18s	remaining: 9.69s
    905:	learn: 0.6715330	total: 4.21s	remaining: 9.74s
    906:	learn: 0.6714898	total: 4.22s	remaining: 9.74s
    907:	learn: 0.6714518	total: 4.23s	remaining: 9.75s
    908:	learn: 0.6714057	total: 4.24s	remaining: 9.75s
    909:	learn: 0.6712939	total: 4.25s	remaining: 9.76s
    910:	learn: 0.6712369	total: 4.26s	remaining: 9.76s
    911:	learn: 0.6711808	total: 4.26s	remaining: 9.77s
    912:	learn: 0.6711156	total: 4.27s	remaining: 9.76s
    913:	learn: 0.6710234	total: 4.28s	remaining: 9.76s
    914:	learn: 0.6709505	total: 4.28s	remaining: 9.76s
    915:	learn: 0.6708955	total: 4.29s	remaining: 9.76s
    916:	learn: 0.6707830	total: 4.3s	remaining: 9.76s
    917:	learn: 0.6707273	total: 4.3s	remaining: 9.76s
    918:	learn: 0.6706794	total: 4.31s	remaining: 9.75s
    919:	learn: 0.6706078	total: 4.32s	remaining: 9.76s
    920:	learn: 0.6705590	total: 4.32s	remaining: 9.75s
    921:	learn: 0.6704699	total: 4.32s	remaining: 9.75s
    922:	learn: 0.6704292	total: 4.33s	remaining: 9.74s
    923:	learn: 0.6703819	total: 4.33s	remaining: 9.73s
    924:	learn: 0.6703292	total: 4.33s	remaining: 9.72s
    925:	learn: 0.6702224	total: 4.34s	remaining: 9.72s
    926:	learn: 0.6701617	total: 4.34s	remaining: 9.72s
    927:	learn: 0.6700258	total: 4.35s	remaining: 9.71s
    928:	learn: 0.6699736	total: 4.35s	remaining: 9.7s
    929:	learn: 0.6699279	total: 4.35s	remaining: 9.69s
    930:	learn: 0.6698905	total: 4.36s	remaining: 9.68s
    931:	learn: 0.6698379	total: 4.36s	remaining: 9.67s
    932:	learn: 0.6697865	total: 4.36s	remaining: 9.66s
    933:	learn: 0.6696765	total: 4.36s	remaining: 9.65s
    934:	learn: 0.6695994	total: 4.37s	remaining: 9.64s
    935:	learn: 0.6695250	total: 4.37s	remaining: 9.64s
    936:	learn: 0.6694741	total: 4.37s	remaining: 9.63s
    937:	learn: 0.6694287	total: 4.38s	remaining: 9.62s
    938:	learn: 0.6693388	total: 4.38s	remaining: 9.61s
    939:	learn: 0.6692736	total: 4.38s	remaining: 9.6s
    940:	learn: 0.6692010	total: 4.38s	remaining: 9.59s
    941:	learn: 0.6691204	total: 4.39s	remaining: 9.58s
    942:	learn: 0.6690449	total: 4.39s	remaining: 9.57s
    943:	learn: 0.6690083	total: 4.39s	remaining: 9.57s
    944:	learn: 0.6689264	total: 4.39s	remaining: 9.56s
    945:	learn: 0.6688796	total: 4.4s	remaining: 9.55s
    946:	learn: 0.6688313	total: 4.4s	remaining: 9.54s
    947:	learn: 0.6687606	total: 4.4s	remaining: 9.53s
    948:	learn: 0.6686697	total: 4.41s	remaining: 9.52s
    949:	learn: 0.6685389	total: 4.41s	remaining: 9.51s
    950:	learn: 0.6684936	total: 4.41s	remaining: 9.51s
    951:	learn: 0.6684073	total: 4.41s	remaining: 9.5s
    952:	learn: 0.6683227	total: 4.42s	remaining: 9.49s
    953:	learn: 0.6682470	total: 4.42s	remaining: 9.48s
    954:	learn: 0.6681674	total: 4.42s	remaining: 9.47s
    955:	learn: 0.6681199	total: 4.43s	remaining: 9.46s
    956:	learn: 0.6680365	total: 4.43s	remaining: 9.45s
    957:	learn: 0.6679875	total: 4.43s	remaining: 9.45s
    958:	learn: 0.6679470	total: 4.43s	remaining: 9.44s
    959:	learn: 0.6678989	total: 4.44s	remaining: 9.43s
    960:	learn: 0.6678440	total: 4.44s	remaining: 9.42s
    961:	learn: 0.6677944	total: 4.44s	remaining: 9.41s
    962:	learn: 0.6677265	total: 4.45s	remaining: 9.4s
    963:	learn: 0.6676806	total: 4.45s	remaining: 9.39s
    964:	learn: 0.6675796	total: 4.45s	remaining: 9.38s
    965:	learn: 0.6675330	total: 4.45s	remaining: 9.38s
    966:	learn: 0.6674920	total: 4.46s	remaining: 9.37s
    967:	learn: 0.6674484	total: 4.46s	remaining: 9.36s
    968:	learn: 0.6674133	total: 4.46s	remaining: 9.35s
    969:	learn: 0.6673739	total: 4.46s	remaining: 9.34s
    970:	learn: 0.6673263	total: 4.47s	remaining: 9.33s
    971:	learn: 0.6672845	total: 4.47s	remaining: 9.33s
    972:	learn: 0.6672342	total: 4.47s	remaining: 9.32s
    973:	learn: 0.6671576	total: 4.47s	remaining: 9.31s
    974:	learn: 0.6670851	total: 4.48s	remaining: 9.3s
    975:	learn: 0.6670216	total: 4.48s	remaining: 9.29s
    976:	learn: 0.6669353	total: 4.48s	remaining: 9.28s
    977:	learn: 0.6668544	total: 4.49s	remaining: 9.29s
    978:	learn: 0.6668014	total: 4.5s	remaining: 9.28s
    979:	learn: 0.6667437	total: 4.5s	remaining: 9.29s
    980:	learn: 0.6666263	total: 4.51s	remaining: 9.28s
    981:	learn: 0.6665465	total: 4.51s	remaining: 9.27s
    982:	learn: 0.6664939	total: 4.51s	remaining: 9.27s
    983:	learn: 0.6664338	total: 4.52s	remaining: 9.26s
    984:	learn: 0.6663397	total: 4.52s	remaining: 9.25s
    985:	learn: 0.6662946	total: 4.52s	remaining: 9.24s
    986:	learn: 0.6662180	total: 4.53s	remaining: 9.23s
    987:	learn: 0.6661265	total: 4.53s	remaining: 9.23s
    988:	learn: 0.6660654	total: 4.53s	remaining: 9.22s
    989:	learn: 0.6660164	total: 4.54s	remaining: 9.22s
    990:	learn: 0.6659172	total: 4.55s	remaining: 9.22s
    991:	learn: 0.6658395	total: 4.55s	remaining: 9.22s
    992:	learn: 0.6657282	total: 4.56s	remaining: 9.22s
    993:	learn: 0.6656681	total: 4.57s	remaining: 9.22s
    994:	learn: 0.6656286	total: 4.58s	remaining: 9.22s
    995:	learn: 0.6655242	total: 4.58s	remaining: 9.22s
    996:	learn: 0.6654576	total: 4.59s	remaining: 9.23s
    997:	learn: 0.6653676	total: 4.6s	remaining: 9.23s
    998:	learn: 0.6653034	total: 4.6s	remaining: 9.22s
    999:	learn: 0.6652587	total: 4.61s	remaining: 9.21s
    1000:	learn: 0.6651612	total: 4.61s	remaining: 9.21s
    1001:	learn: 0.6650786	total: 4.61s	remaining: 9.2s
    1002:	learn: 0.6650414	total: 4.61s	remaining: 9.19s
    1003:	learn: 0.6649670	total: 4.62s	remaining: 9.18s
    1004:	learn: 0.6648562	total: 4.62s	remaining: 9.17s
    1005:	learn: 0.6648034	total: 4.62s	remaining: 9.16s
    1006:	learn: 0.6647378	total: 4.63s	remaining: 9.15s
    1007:	learn: 0.6646747	total: 4.63s	remaining: 9.15s
    1008:	learn: 0.6645995	total: 4.63s	remaining: 9.14s
    1009:	learn: 0.6645562	total: 4.63s	remaining: 9.13s
    1010:	learn: 0.6645065	total: 4.64s	remaining: 9.12s
    1011:	learn: 0.6644060	total: 4.64s	remaining: 9.11s
    1012:	learn: 0.6643608	total: 4.64s	remaining: 9.11s
    1013:	learn: 0.6642868	total: 4.64s	remaining: 9.1s
    1014:	learn: 0.6642080	total: 4.65s	remaining: 9.09s
    1015:	learn: 0.6641360	total: 4.65s	remaining: 9.08s
    1016:	learn: 0.6640916	total: 4.65s	remaining: 9.07s
    1017:	learn: 0.6640390	total: 4.66s	remaining: 9.06s
    1018:	learn: 0.6639868	total: 4.66s	remaining: 9.06s
    1019:	learn: 0.6639364	total: 4.66s	remaining: 9.05s
    1020:	learn: 0.6638664	total: 4.66s	remaining: 9.04s
    1021:	learn: 0.6638154	total: 4.67s	remaining: 9.03s
    1022:	learn: 0.6637154	total: 4.67s	remaining: 9.03s
    1023:	learn: 0.6636663	total: 4.67s	remaining: 9.02s
    1024:	learn: 0.6635733	total: 4.67s	remaining: 9.01s
    1025:	learn: 0.6635404	total: 4.68s	remaining: 9s
    1026:	learn: 0.6634434	total: 4.68s	remaining: 8.99s
    1027:	learn: 0.6633426	total: 4.68s	remaining: 8.98s
    1028:	learn: 0.6632486	total: 4.69s	remaining: 8.98s
    1029:	learn: 0.6631976	total: 4.7s	remaining: 8.99s
    1030:	learn: 0.6631632	total: 4.71s	remaining: 8.99s
    1031:	learn: 0.6631153	total: 4.71s	remaining: 8.98s
    1032:	learn: 0.6630738	total: 4.71s	remaining: 8.98s
    1033:	learn: 0.6629819	total: 4.72s	remaining: 8.97s
    1034:	learn: 0.6629233	total: 4.72s	remaining: 8.96s
    1035:	learn: 0.6628859	total: 4.72s	remaining: 8.95s
    1036:	learn: 0.6627831	total: 4.73s	remaining: 8.95s
    1037:	learn: 0.6627530	total: 4.73s	remaining: 8.94s
    1038:	learn: 0.6626726	total: 4.73s	remaining: 8.93s
    1039:	learn: 0.6626274	total: 4.73s	remaining: 8.92s
    1040:	learn: 0.6625531	total: 4.74s	remaining: 8.91s
    1041:	learn: 0.6624460	total: 4.74s	remaining: 8.91s
    1042:	learn: 0.6623186	total: 4.74s	remaining: 8.9s
    1043:	learn: 0.6622722	total: 4.75s	remaining: 8.89s
    1044:	learn: 0.6622124	total: 4.75s	remaining: 8.88s
    1045:	learn: 0.6621428	total: 4.75s	remaining: 8.88s
    1046:	learn: 0.6620972	total: 4.75s	remaining: 8.87s
    1047:	learn: 0.6620570	total: 4.76s	remaining: 8.86s
    1048:	learn: 0.6619852	total: 4.76s	remaining: 8.85s
    1049:	learn: 0.6619015	total: 4.76s	remaining: 8.84s
    1050:	learn: 0.6618317	total: 4.76s	remaining: 8.84s
    1051:	learn: 0.6617720	total: 4.77s	remaining: 8.83s
    1052:	learn: 0.6617166	total: 4.77s	remaining: 8.82s
    1053:	learn: 0.6616404	total: 4.77s	remaining: 8.81s
    1054:	learn: 0.6615894	total: 4.78s	remaining: 8.8s
    1055:	learn: 0.6615147	total: 4.78s	remaining: 8.8s
    1056:	learn: 0.6614259	total: 4.78s	remaining: 8.79s
    1057:	learn: 0.6613511	total: 4.78s	remaining: 8.78s
    1058:	learn: 0.6612837	total: 4.79s	remaining: 8.77s
    1059:	learn: 0.6612101	total: 4.79s	remaining: 8.77s
    1060:	learn: 0.6611551	total: 4.79s	remaining: 8.76s
    1061:	learn: 0.6610617	total: 4.79s	remaining: 8.75s
    1062:	learn: 0.6610233	total: 4.8s	remaining: 8.74s
    1063:	learn: 0.6609639	total: 4.8s	remaining: 8.73s
    1064:	learn: 0.6608678	total: 4.8s	remaining: 8.73s
    1065:	learn: 0.6607878	total: 4.8s	remaining: 8.72s
    1066:	learn: 0.6607328	total: 4.81s	remaining: 8.71s
    1067:	learn: 0.6606425	total: 4.81s	remaining: 8.7s
    1068:	learn: 0.6605807	total: 4.81s	remaining: 8.7s
    1069:	learn: 0.6605377	total: 4.82s	remaining: 8.69s
    1070:	learn: 0.6604838	total: 4.82s	remaining: 8.68s
    1071:	learn: 0.6603906	total: 4.82s	remaining: 8.67s
    1072:	learn: 0.6603154	total: 4.82s	remaining: 8.66s
    1073:	learn: 0.6602256	total: 4.83s	remaining: 8.66s
    1074:	learn: 0.6601612	total: 4.83s	remaining: 8.65s
    1075:	learn: 0.6601186	total: 4.83s	remaining: 8.64s
    1076:	learn: 0.6600563	total: 4.84s	remaining: 8.63s
    1077:	learn: 0.6600027	total: 4.84s	remaining: 8.63s
    1078:	learn: 0.6599704	total: 4.84s	remaining: 8.62s
    1079:	learn: 0.6599258	total: 4.84s	remaining: 8.61s
    1080:	learn: 0.6598647	total: 4.85s	remaining: 8.6s
    1081:	learn: 0.6598278	total: 4.85s	remaining: 8.6s
    1082:	learn: 0.6597539	total: 4.85s	remaining: 8.59s
    1083:	learn: 0.6596937	total: 4.86s	remaining: 8.58s
    1084:	learn: 0.6595822	total: 4.86s	remaining: 8.57s
    1085:	learn: 0.6594747	total: 4.86s	remaining: 8.57s
    1086:	learn: 0.6594022	total: 4.86s	remaining: 8.56s
    1087:	learn: 0.6593495	total: 4.87s	remaining: 8.55s
    1088:	learn: 0.6593015	total: 4.87s	remaining: 8.54s
    1089:	learn: 0.6592337	total: 4.87s	remaining: 8.54s
    1090:	learn: 0.6591265	total: 4.87s	remaining: 8.53s
    1091:	learn: 0.6590490	total: 4.88s	remaining: 8.52s
    1092:	learn: 0.6589686	total: 4.88s	remaining: 8.52s
    1093:	learn: 0.6588788	total: 4.88s	remaining: 8.51s
    1094:	learn: 0.6588366	total: 4.89s	remaining: 8.51s
    1095:	learn: 0.6587678	total: 4.9s	remaining: 8.51s
    1096:	learn: 0.6586905	total: 4.91s	remaining: 8.52s
    1097:	learn: 0.6585869	total: 4.91s	remaining: 8.51s
    1098:	learn: 0.6585347	total: 4.92s	remaining: 8.5s
    1099:	learn: 0.6584577	total: 4.92s	remaining: 8.49s
    1100:	learn: 0.6583982	total: 4.92s	remaining: 8.49s
    1101:	learn: 0.6583623	total: 4.92s	remaining: 8.48s
    1102:	learn: 0.6583246	total: 4.93s	remaining: 8.47s
    1103:	learn: 0.6582576	total: 4.93s	remaining: 8.47s
    1104:	learn: 0.6582078	total: 4.93s	remaining: 8.46s
    1105:	learn: 0.6581563	total: 4.93s	remaining: 8.45s
    1106:	learn: 0.6580740	total: 4.94s	remaining: 8.44s
    1107:	learn: 0.6580460	total: 4.94s	remaining: 8.44s
    1108:	learn: 0.6579666	total: 4.94s	remaining: 8.43s
    1109:	learn: 0.6578859	total: 4.95s	remaining: 8.42s
    1110:	learn: 0.6578379	total: 4.95s	remaining: 8.41s
    1111:	learn: 0.6577697	total: 4.95s	remaining: 8.41s
    1112:	learn: 0.6576946	total: 4.96s	remaining: 8.4s
    1113:	learn: 0.6576089	total: 4.96s	remaining: 8.39s
    1114:	learn: 0.6575699	total: 4.96s	remaining: 8.39s
    1115:	learn: 0.6575367	total: 4.96s	remaining: 8.38s
    1116:	learn: 0.6574844	total: 4.97s	remaining: 8.37s
    1117:	learn: 0.6574363	total: 4.97s	remaining: 8.37s
    1118:	learn: 0.6573794	total: 4.97s	remaining: 8.36s
    1119:	learn: 0.6573039	total: 4.97s	remaining: 8.35s
    1120:	learn: 0.6572321	total: 4.98s	remaining: 8.34s
    1121:	learn: 0.6571069	total: 4.98s	remaining: 8.34s
    1122:	learn: 0.6570328	total: 4.98s	remaining: 8.33s
    1123:	learn: 0.6569346	total: 4.99s	remaining: 8.32s
    1124:	learn: 0.6568864	total: 4.99s	remaining: 8.31s
    1125:	learn: 0.6568278	total: 4.99s	remaining: 8.31s
    1126:	learn: 0.6567544	total: 4.99s	remaining: 8.3s
    1127:	learn: 0.6567320	total: 5s	remaining: 8.29s
    1128:	learn: 0.6566789	total: 5s	remaining: 8.29s
    1129:	learn: 0.6565957	total: 5s	remaining: 8.28s
    1130:	learn: 0.6565176	total: 5s	remaining: 8.27s
    1131:	learn: 0.6564174	total: 5.01s	remaining: 8.27s
    1132:	learn: 0.6563261	total: 5.01s	remaining: 8.26s
    1133:	learn: 0.6562942	total: 5.01s	remaining: 8.25s
    1134:	learn: 0.6562368	total: 5.02s	remaining: 8.24s
    1135:	learn: 0.6561856	total: 5.02s	remaining: 8.24s
    1136:	learn: 0.6561528	total: 5.02s	remaining: 8.23s
    1137:	learn: 0.6561079	total: 5.03s	remaining: 8.22s
    1138:	learn: 0.6560437	total: 5.03s	remaining: 8.21s
    1139:	learn: 0.6559930	total: 5.03s	remaining: 8.21s
    1140:	learn: 0.6559417	total: 5.03s	remaining: 8.2s
    1141:	learn: 0.6558577	total: 5.04s	remaining: 8.2s
    1142:	learn: 0.6557957	total: 5.04s	remaining: 8.19s
    1143:	learn: 0.6557557	total: 5.04s	remaining: 8.18s
    1144:	learn: 0.6557115	total: 5.04s	remaining: 8.17s
    1145:	learn: 0.6556709	total: 5.05s	remaining: 8.17s
    1146:	learn: 0.6556242	total: 5.05s	remaining: 8.16s
    1147:	learn: 0.6555596	total: 5.05s	remaining: 8.15s
    1148:	learn: 0.6554958	total: 5.06s	remaining: 8.15s
    1149:	learn: 0.6554382	total: 5.06s	remaining: 8.14s
    1150:	learn: 0.6553673	total: 5.06s	remaining: 8.13s
    1151:	learn: 0.6553119	total: 5.07s	remaining: 8.13s
    1152:	learn: 0.6552709	total: 5.07s	remaining: 8.12s
    1153:	learn: 0.6551815	total: 5.07s	remaining: 8.11s
    1154:	learn: 0.6551467	total: 5.07s	remaining: 8.1s
    1155:	learn: 0.6550702	total: 5.08s	remaining: 8.1s
    1156:	learn: 0.6549894	total: 5.08s	remaining: 8.1s
    1157:	learn: 0.6549210	total: 5.09s	remaining: 8.1s
    1158:	learn: 0.6548568	total: 5.1s	remaining: 8.1s
    1159:	learn: 0.6548183	total: 5.1s	remaining: 8.09s
    1160:	learn: 0.6547674	total: 5.11s	remaining: 8.09s
    1161:	learn: 0.6547214	total: 5.11s	remaining: 8.08s
    1162:	learn: 0.6546351	total: 5.11s	remaining: 8.07s
    1163:	learn: 0.6545732	total: 5.12s	remaining: 8.07s
    1164:	learn: 0.6545038	total: 5.12s	remaining: 8.06s
    1165:	learn: 0.6544236	total: 5.12s	remaining: 8.05s
    1166:	learn: 0.6543594	total: 5.12s	remaining: 8.05s
    1167:	learn: 0.6542667	total: 5.13s	remaining: 8.04s
    1168:	learn: 0.6541771	total: 5.13s	remaining: 8.03s
    1169:	learn: 0.6541154	total: 5.13s	remaining: 8.03s
    1170:	learn: 0.6540488	total: 5.13s	remaining: 8.02s
    1171:	learn: 0.6540084	total: 5.14s	remaining: 8.01s
    1172:	learn: 0.6539138	total: 5.14s	remaining: 8.01s
    1173:	learn: 0.6538618	total: 5.14s	remaining: 8s
    1174:	learn: 0.6538253	total: 5.15s	remaining: 7.99s
    1175:	learn: 0.6537398	total: 5.15s	remaining: 7.99s
    1176:	learn: 0.6536833	total: 5.18s	remaining: 8.03s
    1177:	learn: 0.6536290	total: 5.19s	remaining: 8.03s
    1178:	learn: 0.6535512	total: 5.2s	remaining: 8.03s
    1179:	learn: 0.6534706	total: 5.21s	remaining: 8.03s
    1180:	learn: 0.6534244	total: 5.21s	remaining: 8.03s
    1181:	learn: 0.6533529	total: 5.22s	remaining: 8.03s
    1182:	learn: 0.6533141	total: 5.23s	remaining: 8.03s
    1183:	learn: 0.6532597	total: 5.24s	remaining: 8.03s
    1184:	learn: 0.6531917	total: 5.24s	remaining: 8.03s
    1185:	learn: 0.6531435	total: 5.25s	remaining: 8.03s
    1186:	learn: 0.6530944	total: 5.26s	remaining: 8.03s
    1187:	learn: 0.6530442	total: 5.26s	remaining: 8.03s
    1188:	learn: 0.6529651	total: 5.27s	remaining: 8.03s
    1189:	learn: 0.6529055	total: 5.28s	remaining: 8.03s
    1190:	learn: 0.6528597	total: 5.29s	remaining: 8.03s
    1191:	learn: 0.6528197	total: 5.29s	remaining: 8.03s
    1192:	learn: 0.6527611	total: 5.3s	remaining: 8.04s
    1193:	learn: 0.6526764	total: 5.31s	remaining: 8.03s
    1194:	learn: 0.6526167	total: 5.31s	remaining: 8.03s
    1195:	learn: 0.6525728	total: 5.32s	remaining: 8.02s
    1196:	learn: 0.6525241	total: 5.32s	remaining: 8.01s
    1197:	learn: 0.6524644	total: 5.32s	remaining: 8s
    1198:	learn: 0.6523954	total: 5.32s	remaining: 8s
    1199:	learn: 0.6523261	total: 5.33s	remaining: 7.99s
    1200:	learn: 0.6522827	total: 5.33s	remaining: 7.98s
    1201:	learn: 0.6521933	total: 5.33s	remaining: 7.98s
    1202:	learn: 0.6521408	total: 5.33s	remaining: 7.97s
    1203:	learn: 0.6520845	total: 5.34s	remaining: 7.96s
    1204:	learn: 0.6520200	total: 5.34s	remaining: 7.96s
    1205:	learn: 0.6519618	total: 5.34s	remaining: 7.95s
    1206:	learn: 0.6519077	total: 5.35s	remaining: 7.94s
    1207:	learn: 0.6518463	total: 5.35s	remaining: 7.93s
    1208:	learn: 0.6517943	total: 5.35s	remaining: 7.93s
    1209:	learn: 0.6517510	total: 5.36s	remaining: 7.92s
    1210:	learn: 0.6516854	total: 5.36s	remaining: 7.92s
    1211:	learn: 0.6516463	total: 5.36s	remaining: 7.91s
    1212:	learn: 0.6515801	total: 5.36s	remaining: 7.9s
    1213:	learn: 0.6515339	total: 5.37s	remaining: 7.89s
    1214:	learn: 0.6514859	total: 5.37s	remaining: 7.89s
    1215:	learn: 0.6514431	total: 5.37s	remaining: 7.88s
    1216:	learn: 0.6513781	total: 5.37s	remaining: 7.87s
    1217:	learn: 0.6513240	total: 5.38s	remaining: 7.87s
    1218:	learn: 0.6512825	total: 5.38s	remaining: 7.86s
    1219:	learn: 0.6512015	total: 5.38s	remaining: 7.85s
    1220:	learn: 0.6511632	total: 5.38s	remaining: 7.85s
    1221:	learn: 0.6511332	total: 5.39s	remaining: 7.84s
    1222:	learn: 0.6510773	total: 5.39s	remaining: 7.83s
    1223:	learn: 0.6510310	total: 5.39s	remaining: 7.83s
    1224:	learn: 0.6509571	total: 5.4s	remaining: 7.82s
    1225:	learn: 0.6508737	total: 5.4s	remaining: 7.81s
    1226:	learn: 0.6508083	total: 5.4s	remaining: 7.81s
    1227:	learn: 0.6507445	total: 5.4s	remaining: 7.8s
    1228:	learn: 0.6506815	total: 5.41s	remaining: 7.79s
    1229:	learn: 0.6506122	total: 5.41s	remaining: 7.79s
    1230:	learn: 0.6505335	total: 5.41s	remaining: 7.78s
    1231:	learn: 0.6504556	total: 5.42s	remaining: 7.77s
    1232:	learn: 0.6503866	total: 5.42s	remaining: 7.76s
    1233:	learn: 0.6503551	total: 5.42s	remaining: 7.76s
    1234:	learn: 0.6502664	total: 5.42s	remaining: 7.75s
    1235:	learn: 0.6502108	total: 5.43s	remaining: 7.75s
    1236:	learn: 0.6501747	total: 5.43s	remaining: 7.74s
    1237:	learn: 0.6501149	total: 5.43s	remaining: 7.73s
    1238:	learn: 0.6500711	total: 5.43s	remaining: 7.72s
    1239:	learn: 0.6500149	total: 5.44s	remaining: 7.72s
    1240:	learn: 0.6499628	total: 5.44s	remaining: 7.71s
    1241:	learn: 0.6498945	total: 5.44s	remaining: 7.71s
    1242:	learn: 0.6498287	total: 5.45s	remaining: 7.7s
    1243:	learn: 0.6497874	total: 5.45s	remaining: 7.69s
    1244:	learn: 0.6497470	total: 5.45s	remaining: 7.68s
    1245:	learn: 0.6496928	total: 5.45s	remaining: 7.68s
    1246:	learn: 0.6496287	total: 5.46s	remaining: 7.67s
    1247:	learn: 0.6495810	total: 5.46s	remaining: 7.66s
    1248:	learn: 0.6495222	total: 5.46s	remaining: 7.66s
    1249:	learn: 0.6494544	total: 5.46s	remaining: 7.65s
    1250:	learn: 0.6493982	total: 5.47s	remaining: 7.64s
    1251:	learn: 0.6493544	total: 5.47s	remaining: 7.64s
    1252:	learn: 0.6493059	total: 5.47s	remaining: 7.63s
    1253:	learn: 0.6492073	total: 5.48s	remaining: 7.63s
    1254:	learn: 0.6491382	total: 5.49s	remaining: 7.63s
    1255:	learn: 0.6490651	total: 5.5s	remaining: 7.64s
    1256:	learn: 0.6490140	total: 5.51s	remaining: 7.64s
    1257:	learn: 0.6489461	total: 5.51s	remaining: 7.64s
    1258:	learn: 0.6488901	total: 5.52s	remaining: 7.64s
    1259:	learn: 0.6488086	total: 5.53s	remaining: 7.64s
    1260:	learn: 0.6487428	total: 5.54s	remaining: 7.64s
    1261:	learn: 0.6487018	total: 5.55s	remaining: 7.64s
    1262:	learn: 0.6486318	total: 5.56s	remaining: 7.64s
    1263:	learn: 0.6486002	total: 5.57s	remaining: 7.64s
    1264:	learn: 0.6485443	total: 5.57s	remaining: 7.64s
    1265:	learn: 0.6484718	total: 5.58s	remaining: 7.64s
    1266:	learn: 0.6484185	total: 5.59s	remaining: 7.64s
    1267:	learn: 0.6483199	total: 5.59s	remaining: 7.64s
    1268:	learn: 0.6482659	total: 5.6s	remaining: 7.64s
    1269:	learn: 0.6481830	total: 5.6s	remaining: 7.63s
    1270:	learn: 0.6481163	total: 5.61s	remaining: 7.63s
    1271:	learn: 0.6480546	total: 5.61s	remaining: 7.62s
    1272:	learn: 0.6479865	total: 5.61s	remaining: 7.61s
    1273:	learn: 0.6479366	total: 5.61s	remaining: 7.61s
    1274:	learn: 0.6478526	total: 5.62s	remaining: 7.6s
    1275:	learn: 0.6477900	total: 5.62s	remaining: 7.59s
    1276:	learn: 0.6477167	total: 5.62s	remaining: 7.58s
    1277:	learn: 0.6476612	total: 5.63s	remaining: 7.58s
    1278:	learn: 0.6475767	total: 5.63s	remaining: 7.57s
    1279:	learn: 0.6475124	total: 5.63s	remaining: 7.57s
    1280:	learn: 0.6474615	total: 5.63s	remaining: 7.56s
    1281:	learn: 0.6474182	total: 5.63s	remaining: 7.55s
    1282:	learn: 0.6473612	total: 5.64s	remaining: 7.54s
    1283:	learn: 0.6473361	total: 5.64s	remaining: 7.54s
    1284:	learn: 0.6472449	total: 5.64s	remaining: 7.53s
    1285:	learn: 0.6471966	total: 5.65s	remaining: 7.53s
    1286:	learn: 0.6471207	total: 5.65s	remaining: 7.52s
    1287:	learn: 0.6470770	total: 5.65s	remaining: 7.51s
    1288:	learn: 0.6470411	total: 5.65s	remaining: 7.5s
    1289:	learn: 0.6469675	total: 5.66s	remaining: 7.5s
    1290:	learn: 0.6469106	total: 5.66s	remaining: 7.49s
    1291:	learn: 0.6468734	total: 5.66s	remaining: 7.49s
    1292:	learn: 0.6468031	total: 5.67s	remaining: 7.48s
    1293:	learn: 0.6467576	total: 5.67s	remaining: 7.47s
    1294:	learn: 0.6467084	total: 5.67s	remaining: 7.47s
    1295:	learn: 0.6466314	total: 5.67s	remaining: 7.46s
    1296:	learn: 0.6465991	total: 5.68s	remaining: 7.45s
    1297:	learn: 0.6465648	total: 5.69s	remaining: 7.46s
    1298:	learn: 0.6465060	total: 5.7s	remaining: 7.47s
    1299:	learn: 0.6464605	total: 5.71s	remaining: 7.47s
    1300:	learn: 0.6464065	total: 5.73s	remaining: 7.48s
    1301:	learn: 0.6463704	total: 5.73s	remaining: 7.47s
    1302:	learn: 0.6463190	total: 5.73s	remaining: 7.47s
    1303:	learn: 0.6462184	total: 5.74s	remaining: 7.46s
    1304:	learn: 0.6461576	total: 5.74s	remaining: 7.45s
    1305:	learn: 0.6461308	total: 5.74s	remaining: 7.45s
    1306:	learn: 0.6460900	total: 5.74s	remaining: 7.44s
    1307:	learn: 0.6460585	total: 5.75s	remaining: 7.43s
    1308:	learn: 0.6460192	total: 5.75s	remaining: 7.43s
    1309:	learn: 0.6459454	total: 5.75s	remaining: 7.42s
    1310:	learn: 0.6458971	total: 5.75s	remaining: 7.41s
    1311:	learn: 0.6458146	total: 5.76s	remaining: 7.41s
    1312:	learn: 0.6457462	total: 5.76s	remaining: 7.4s
    1313:	learn: 0.6457104	total: 5.76s	remaining: 7.39s
    1314:	learn: 0.6456450	total: 5.77s	remaining: 7.39s
    1315:	learn: 0.6456044	total: 5.77s	remaining: 7.38s
    1316:	learn: 0.6455651	total: 5.77s	remaining: 7.38s
    1317:	learn: 0.6454882	total: 5.78s	remaining: 7.37s
    1318:	learn: 0.6454385	total: 5.78s	remaining: 7.37s
    1319:	learn: 0.6453611	total: 5.78s	remaining: 7.36s
    1320:	learn: 0.6452888	total: 5.78s	remaining: 7.35s
    1321:	learn: 0.6452459	total: 5.79s	remaining: 7.34s
    1322:	learn: 0.6451886	total: 5.79s	remaining: 7.34s
    1323:	learn: 0.6451027	total: 5.79s	remaining: 7.33s
    1324:	learn: 0.6450312	total: 5.8s	remaining: 7.33s
    1325:	learn: 0.6449675	total: 5.8s	remaining: 7.32s
    1326:	learn: 0.6448903	total: 5.8s	remaining: 7.31s
    1327:	learn: 0.6448491	total: 5.8s	remaining: 7.31s
    1328:	learn: 0.6447901	total: 5.81s	remaining: 7.3s
    1329:	learn: 0.6447379	total: 5.81s	remaining: 7.29s
    1330:	learn: 0.6446649	total: 5.81s	remaining: 7.29s
    1331:	learn: 0.6446223	total: 5.82s	remaining: 7.28s
    1332:	learn: 0.6445478	total: 5.82s	remaining: 7.28s
    1333:	learn: 0.6445034	total: 5.82s	remaining: 7.27s
    1334:	learn: 0.6444171	total: 5.82s	remaining: 7.26s
    1335:	learn: 0.6443480	total: 5.83s	remaining: 7.26s
    1336:	learn: 0.6442625	total: 5.83s	remaining: 7.25s
    1337:	learn: 0.6441921	total: 5.83s	remaining: 7.24s
    1338:	learn: 0.6441688	total: 5.83s	remaining: 7.24s
    1339:	learn: 0.6441027	total: 5.84s	remaining: 7.23s
    1340:	learn: 0.6440357	total: 5.84s	remaining: 7.22s
    1341:	learn: 0.6439770	total: 5.84s	remaining: 7.22s
    1342:	learn: 0.6439204	total: 5.85s	remaining: 7.21s
    1343:	learn: 0.6438627	total: 5.85s	remaining: 7.21s
    1344:	learn: 0.6438311	total: 5.85s	remaining: 7.2s
    1345:	learn: 0.6437644	total: 5.85s	remaining: 7.19s
    1346:	learn: 0.6436973	total: 5.86s	remaining: 7.19s
    1347:	learn: 0.6436608	total: 5.86s	remaining: 7.18s
    1348:	learn: 0.6436103	total: 5.86s	remaining: 7.18s
    1349:	learn: 0.6435330	total: 5.87s	remaining: 7.17s
    1350:	learn: 0.6435072	total: 5.87s	remaining: 7.16s
    1351:	learn: 0.6434367	total: 5.87s	remaining: 7.16s
    1352:	learn: 0.6433624	total: 5.87s	remaining: 7.15s
    1353:	learn: 0.6432782	total: 5.88s	remaining: 7.14s
    1354:	learn: 0.6432264	total: 5.88s	remaining: 7.14s
    1355:	learn: 0.6431609	total: 5.88s	remaining: 7.13s
    1356:	learn: 0.6430866	total: 5.89s	remaining: 7.13s
    1357:	learn: 0.6430221	total: 5.9s	remaining: 7.14s
    1358:	learn: 0.6429637	total: 5.91s	remaining: 7.14s
    1359:	learn: 0.6428992	total: 5.91s	remaining: 7.13s
    1360:	learn: 0.6428409	total: 5.92s	remaining: 7.13s
    1361:	learn: 0.6428044	total: 5.92s	remaining: 7.12s
    1362:	learn: 0.6427699	total: 5.92s	remaining: 7.11s
    1363:	learn: 0.6427241	total: 5.92s	remaining: 7.11s
    1364:	learn: 0.6426779	total: 5.93s	remaining: 7.1s
    1365:	learn: 0.6426150	total: 5.93s	remaining: 7.09s
    1366:	learn: 0.6425761	total: 5.93s	remaining: 7.09s
    1367:	learn: 0.6425350	total: 5.94s	remaining: 7.08s
    1368:	learn: 0.6424953	total: 5.94s	remaining: 7.08s
    1369:	learn: 0.6424469	total: 5.94s	remaining: 7.07s
    1370:	learn: 0.6423870	total: 5.95s	remaining: 7.06s
    1371:	learn: 0.6423218	total: 5.95s	remaining: 7.06s
    1372:	learn: 0.6422550	total: 5.95s	remaining: 7.05s
    1373:	learn: 0.6422080	total: 5.95s	remaining: 7.04s
    1374:	learn: 0.6421375	total: 5.96s	remaining: 7.04s
    1375:	learn: 0.6420830	total: 5.96s	remaining: 7.03s
    1376:	learn: 0.6420272	total: 5.96s	remaining: 7.03s
    1377:	learn: 0.6419940	total: 5.96s	remaining: 7.02s
    1378:	learn: 0.6419458	total: 5.97s	remaining: 7.01s
    1379:	learn: 0.6419003	total: 5.97s	remaining: 7.01s
    1380:	learn: 0.6418393	total: 5.97s	remaining: 7s
    1381:	learn: 0.6417739	total: 5.97s	remaining: 7s
    1382:	learn: 0.6417440	total: 5.98s	remaining: 6.99s
    1383:	learn: 0.6416954	total: 5.98s	remaining: 6.98s
    1384:	learn: 0.6416293	total: 5.98s	remaining: 6.98s
    1385:	learn: 0.6415741	total: 5.99s	remaining: 6.97s
    1386:	learn: 0.6415409	total: 5.99s	remaining: 6.96s
    1387:	learn: 0.6414790	total: 5.99s	remaining: 6.96s
    1388:	learn: 0.6414319	total: 5.99s	remaining: 6.95s
    1389:	learn: 0.6413667	total: 6s	remaining: 6.95s
    1390:	learn: 0.6413202	total: 6s	remaining: 6.94s
    1391:	learn: 0.6412463	total: 6s	remaining: 6.93s
    1392:	learn: 0.6412024	total: 6s	remaining: 6.93s
    1393:	learn: 0.6411433	total: 6.01s	remaining: 6.92s
    1394:	learn: 0.6410874	total: 6.01s	remaining: 6.92s
    1395:	learn: 0.6410280	total: 6.01s	remaining: 6.91s
    1396:	learn: 0.6409825	total: 6.02s	remaining: 6.91s
    1397:	learn: 0.6409301	total: 6.02s	remaining: 6.9s
    1398:	learn: 0.6408740	total: 6.03s	remaining: 6.9s
    1399:	learn: 0.6408198	total: 6.03s	remaining: 6.89s
    1400:	learn: 0.6407554	total: 6.03s	remaining: 6.88s
    1401:	learn: 0.6407010	total: 6.04s	remaining: 6.88s
    1402:	learn: 0.6406148	total: 6.04s	remaining: 6.87s
    1403:	learn: 0.6405590	total: 6.04s	remaining: 6.87s
    1404:	learn: 0.6404809	total: 6.04s	remaining: 6.86s
    1405:	learn: 0.6404202	total: 6.05s	remaining: 6.86s
    1406:	learn: 0.6403756	total: 6.05s	remaining: 6.85s
    1407:	learn: 0.6403500	total: 6.05s	remaining: 6.84s
    1408:	learn: 0.6402643	total: 6.05s	remaining: 6.84s
    1409:	learn: 0.6402273	total: 6.06s	remaining: 6.83s
    1410:	learn: 0.6401749	total: 6.06s	remaining: 6.83s
    1411:	learn: 0.6401276	total: 6.06s	remaining: 6.82s
    1412:	learn: 0.6400606	total: 6.07s	remaining: 6.81s
    1413:	learn: 0.6400171	total: 6.07s	remaining: 6.81s
    1414:	learn: 0.6399562	total: 6.07s	remaining: 6.8s
    1415:	learn: 0.6398981	total: 6.08s	remaining: 6.8s
    1416:	learn: 0.6398236	total: 6.08s	remaining: 6.79s
    1417:	learn: 0.6397805	total: 6.09s	remaining: 6.79s
    1418:	learn: 0.6397173	total: 6.09s	remaining: 6.79s
    1419:	learn: 0.6396929	total: 6.1s	remaining: 6.79s
    1420:	learn: 0.6396506	total: 6.1s	remaining: 6.78s
    1421:	learn: 0.6396072	total: 6.11s	remaining: 6.78s
    1422:	learn: 0.6395650	total: 6.11s	remaining: 6.77s
    1423:	learn: 0.6395295	total: 6.11s	remaining: 6.77s
    1424:	learn: 0.6394875	total: 6.13s	remaining: 6.77s
    1425:	learn: 0.6394420	total: 6.13s	remaining: 6.77s
    1426:	learn: 0.6393512	total: 6.14s	remaining: 6.77s
    1427:	learn: 0.6392801	total: 6.15s	remaining: 6.77s
    1428:	learn: 0.6392336	total: 6.15s	remaining: 6.77s
    1429:	learn: 0.6391925	total: 6.16s	remaining: 6.76s
    1430:	learn: 0.6391470	total: 6.17s	remaining: 6.76s
    1431:	learn: 0.6390670	total: 6.18s	remaining: 6.76s
    1432:	learn: 0.6390149	total: 6.19s	remaining: 6.76s
    1433:	learn: 0.6389601	total: 6.2s	remaining: 6.76s
    1434:	learn: 0.6389044	total: 6.2s	remaining: 6.76s
    1435:	learn: 0.6388554	total: 6.21s	remaining: 6.76s
    1436:	learn: 0.6388226	total: 6.22s	remaining: 6.76s
    1437:	learn: 0.6387396	total: 6.23s	remaining: 6.76s
    1438:	learn: 0.6387147	total: 6.23s	remaining: 6.76s
    1439:	learn: 0.6386613	total: 6.23s	remaining: 6.75s
    1440:	learn: 0.6386004	total: 6.24s	remaining: 6.75s
    1441:	learn: 0.6385476	total: 6.24s	remaining: 6.74s
    1442:	learn: 0.6384874	total: 6.24s	remaining: 6.73s
    1443:	learn: 0.6384048	total: 6.24s	remaining: 6.73s
    1444:	learn: 0.6383328	total: 6.25s	remaining: 6.72s
    1445:	learn: 0.6382745	total: 6.25s	remaining: 6.72s
    1446:	learn: 0.6381927	total: 6.25s	remaining: 6.71s
    1447:	learn: 0.6381530	total: 6.26s	remaining: 6.71s
    1448:	learn: 0.6381245	total: 6.26s	remaining: 6.7s
    1449:	learn: 0.6380840	total: 6.26s	remaining: 6.7s
    1450:	learn: 0.6380524	total: 6.27s	remaining: 6.69s
    1451:	learn: 0.6380022	total: 6.27s	remaining: 6.68s
    1452:	learn: 0.6379694	total: 6.27s	remaining: 6.68s
    1453:	learn: 0.6379344	total: 6.28s	remaining: 6.67s
    1454:	learn: 0.6378722	total: 6.28s	remaining: 6.67s
    1455:	learn: 0.6378418	total: 6.29s	remaining: 6.67s
    1456:	learn: 0.6378111	total: 6.3s	remaining: 6.67s
    1457:	learn: 0.6377495	total: 6.3s	remaining: 6.66s
    1458:	learn: 0.6376859	total: 6.3s	remaining: 6.66s
    1459:	learn: 0.6376334	total: 6.31s	remaining: 6.65s
    1460:	learn: 0.6375631	total: 6.31s	remaining: 6.65s
    1461:	learn: 0.6375035	total: 6.31s	remaining: 6.64s
    1462:	learn: 0.6374501	total: 6.32s	remaining: 6.63s
    1463:	learn: 0.6373989	total: 6.32s	remaining: 6.63s
    1464:	learn: 0.6373651	total: 6.32s	remaining: 6.62s
    1465:	learn: 0.6372786	total: 6.32s	remaining: 6.62s
    1466:	learn: 0.6372379	total: 6.33s	remaining: 6.61s
    1467:	learn: 0.6371983	total: 6.33s	remaining: 6.61s
    1468:	learn: 0.6371716	total: 6.33s	remaining: 6.6s
    1469:	learn: 0.6371075	total: 6.33s	remaining: 6.59s
    1470:	learn: 0.6370828	total: 6.34s	remaining: 6.59s
    1471:	learn: 0.6370166	total: 6.34s	remaining: 6.58s
    1472:	learn: 0.6369765	total: 6.34s	remaining: 6.58s
    1473:	learn: 0.6369550	total: 6.34s	remaining: 6.57s
    1474:	learn: 0.6368901	total: 6.35s	remaining: 6.56s
    1475:	learn: 0.6368173	total: 6.35s	remaining: 6.56s
    1476:	learn: 0.6367515	total: 6.35s	remaining: 6.55s
    1477:	learn: 0.6367202	total: 6.36s	remaining: 6.55s
    1478:	learn: 0.6366875	total: 6.36s	remaining: 6.54s
    1479:	learn: 0.6366184	total: 6.36s	remaining: 6.53s
    1480:	learn: 0.6365662	total: 6.37s	remaining: 6.53s
    1481:	learn: 0.6365235	total: 6.37s	remaining: 6.52s
    1482:	learn: 0.6364686	total: 6.37s	remaining: 6.52s
    1483:	learn: 0.6363997	total: 6.37s	remaining: 6.51s
    1484:	learn: 0.6363399	total: 6.38s	remaining: 6.5s
    1485:	learn: 0.6362970	total: 6.38s	remaining: 6.5s
    1486:	learn: 0.6362325	total: 6.38s	remaining: 6.49s
    1487:	learn: 0.6361797	total: 6.38s	remaining: 6.49s
    1488:	learn: 0.6361536	total: 6.39s	remaining: 6.48s
    1489:	learn: 0.6361123	total: 6.39s	remaining: 6.48s
    1490:	learn: 0.6360618	total: 6.39s	remaining: 6.47s
    1491:	learn: 0.6360215	total: 6.4s	remaining: 6.46s
    1492:	learn: 0.6359738	total: 6.4s	remaining: 6.46s
    1493:	learn: 0.6359447	total: 6.4s	remaining: 6.45s
    1494:	learn: 0.6359003	total: 6.4s	remaining: 6.45s
    1495:	learn: 0.6358668	total: 6.41s	remaining: 6.44s
    1496:	learn: 0.6357713	total: 6.41s	remaining: 6.43s
    1497:	learn: 0.6357232	total: 6.41s	remaining: 6.43s
    1498:	learn: 0.6356822	total: 6.42s	remaining: 6.42s
    1499:	learn: 0.6356376	total: 6.42s	remaining: 6.42s
    1500:	learn: 0.6355974	total: 6.42s	remaining: 6.41s
    1501:	learn: 0.6355380	total: 6.42s	remaining: 6.41s
    1502:	learn: 0.6354773	total: 6.43s	remaining: 6.4s
    1503:	learn: 0.6354159	total: 6.43s	remaining: 6.39s
    1504:	learn: 0.6353842	total: 6.43s	remaining: 6.39s
    1505:	learn: 0.6353409	total: 6.43s	remaining: 6.38s
    1506:	learn: 0.6353018	total: 6.44s	remaining: 6.38s
    1507:	learn: 0.6352518	total: 6.44s	remaining: 6.37s
    1508:	learn: 0.6351786	total: 6.45s	remaining: 6.37s
    1509:	learn: 0.6351275	total: 6.45s	remaining: 6.37s
    1510:	learn: 0.6350824	total: 6.46s	remaining: 6.37s
    1511:	learn: 0.6350459	total: 6.47s	remaining: 6.37s
    1512:	learn: 0.6349953	total: 6.48s	remaining: 6.37s
    1513:	learn: 0.6349283	total: 6.49s	remaining: 6.37s
    1514:	learn: 0.6348644	total: 6.5s	remaining: 6.37s
    1515:	learn: 0.6348137	total: 6.51s	remaining: 6.37s
    1516:	learn: 0.6347526	total: 6.51s	remaining: 6.37s
    1517:	learn: 0.6346963	total: 6.52s	remaining: 6.37s
    1518:	learn: 0.6346624	total: 6.53s	remaining: 6.37s
    1519:	learn: 0.6345965	total: 6.54s	remaining: 6.37s
    1520:	learn: 0.6345036	total: 6.54s	remaining: 6.37s
    1521:	learn: 0.6344572	total: 6.55s	remaining: 6.36s
    1522:	learn: 0.6344228	total: 6.55s	remaining: 6.36s
    1523:	learn: 0.6343833	total: 6.56s	remaining: 6.35s
    1524:	learn: 0.6343134	total: 6.56s	remaining: 6.35s
    1525:	learn: 0.6342575	total: 6.56s	remaining: 6.34s
    1526:	learn: 0.6342130	total: 6.57s	remaining: 6.33s
    1527:	learn: 0.6341649	total: 6.57s	remaining: 6.33s
    1528:	learn: 0.6341328	total: 6.57s	remaining: 6.32s
    1529:	learn: 0.6340711	total: 6.58s	remaining: 6.32s
    1530:	learn: 0.6340397	total: 6.58s	remaining: 6.31s
    1531:	learn: 0.6339981	total: 6.58s	remaining: 6.31s
    1532:	learn: 0.6339499	total: 6.58s	remaining: 6.3s
    1533:	learn: 0.6339169	total: 6.59s	remaining: 6.29s
    1534:	learn: 0.6338552	total: 6.59s	remaining: 6.29s
    1535:	learn: 0.6337928	total: 6.59s	remaining: 6.28s
    1536:	learn: 0.6337452	total: 6.59s	remaining: 6.28s
    1537:	learn: 0.6336882	total: 6.6s	remaining: 6.27s
    1538:	learn: 0.6336604	total: 6.6s	remaining: 6.26s
    1539:	learn: 0.6335910	total: 6.6s	remaining: 6.26s
    1540:	learn: 0.6335246	total: 6.61s	remaining: 6.25s
    1541:	learn: 0.6334768	total: 6.61s	remaining: 6.25s
    1542:	learn: 0.6334538	total: 6.61s	remaining: 6.24s
    1543:	learn: 0.6334113	total: 6.61s	remaining: 6.24s
    1544:	learn: 0.6333731	total: 6.62s	remaining: 6.23s
    1545:	learn: 0.6333347	total: 6.62s	remaining: 6.22s
    1546:	learn: 0.6333049	total: 6.62s	remaining: 6.22s
    1547:	learn: 0.6332501	total: 6.62s	remaining: 6.21s
    1548:	learn: 0.6332247	total: 6.63s	remaining: 6.21s
    1549:	learn: 0.6331856	total: 6.63s	remaining: 6.2s
    1550:	learn: 0.6331484	total: 6.63s	remaining: 6.2s
    1551:	learn: 0.6330838	total: 6.63s	remaining: 6.19s
    1552:	learn: 0.6330460	total: 6.64s	remaining: 6.18s
    1553:	learn: 0.6329877	total: 6.64s	remaining: 6.18s
    1554:	learn: 0.6329572	total: 6.64s	remaining: 6.17s
    1555:	learn: 0.6328829	total: 6.65s	remaining: 6.17s
    1556:	learn: 0.6328601	total: 6.65s	remaining: 6.16s
    1557:	learn: 0.6328145	total: 6.65s	remaining: 6.16s
    1558:	learn: 0.6327752	total: 6.65s	remaining: 6.15s
    1559:	learn: 0.6327219	total: 6.66s	remaining: 6.14s
    1560:	learn: 0.6326678	total: 6.66s	remaining: 6.14s
    1561:	learn: 0.6326432	total: 6.66s	remaining: 6.13s
    1562:	learn: 0.6325929	total: 6.67s	remaining: 6.13s
    1563:	learn: 0.6325171	total: 6.68s	remaining: 6.13s
    1564:	learn: 0.6324435	total: 6.69s	remaining: 6.13s
    1565:	learn: 0.6324119	total: 6.69s	remaining: 6.13s
    1566:	learn: 0.6323645	total: 6.69s	remaining: 6.12s
    1567:	learn: 0.6323002	total: 6.7s	remaining: 6.12s
    1568:	learn: 0.6322593	total: 6.7s	remaining: 6.11s
    1569:	learn: 0.6322096	total: 6.7s	remaining: 6.11s
    1570:	learn: 0.6321674	total: 6.71s	remaining: 6.1s
    1571:	learn: 0.6320992	total: 6.71s	remaining: 6.09s
    1572:	learn: 0.6320355	total: 6.71s	remaining: 6.09s
    1573:	learn: 0.6319957	total: 6.72s	remaining: 6.08s
    1574:	learn: 0.6319377	total: 6.72s	remaining: 6.08s
    1575:	learn: 0.6319036	total: 6.72s	remaining: 6.07s
    1576:	learn: 0.6318387	total: 6.72s	remaining: 6.07s
    1577:	learn: 0.6318129	total: 6.73s	remaining: 6.06s
    1578:	learn: 0.6317633	total: 6.73s	remaining: 6.06s
    1579:	learn: 0.6317030	total: 6.74s	remaining: 6.05s
    1580:	learn: 0.6316681	total: 6.74s	remaining: 6.05s
    1581:	learn: 0.6316139	total: 6.74s	remaining: 6.04s
    1582:	learn: 0.6315548	total: 6.74s	remaining: 6.04s
    1583:	learn: 0.6315124	total: 6.75s	remaining: 6.03s
    1584:	learn: 0.6314528	total: 6.75s	remaining: 6.03s
    1585:	learn: 0.6313778	total: 6.75s	remaining: 6.02s
    1586:	learn: 0.6313326	total: 6.76s	remaining: 6.01s
    1587:	learn: 0.6312730	total: 6.76s	remaining: 6.01s
    1588:	learn: 0.6312349	total: 6.76s	remaining: 6s
    1589:	learn: 0.6311858	total: 6.76s	remaining: 6s
    1590:	learn: 0.6311408	total: 6.77s	remaining: 5.99s
    1591:	learn: 0.6310982	total: 6.77s	remaining: 5.99s
    1592:	learn: 0.6310599	total: 6.77s	remaining: 5.98s
    1593:	learn: 0.6310116	total: 6.78s	remaining: 5.98s
    1594:	learn: 0.6309642	total: 6.78s	remaining: 5.97s
    1595:	learn: 0.6309070	total: 6.78s	remaining: 5.96s
    1596:	learn: 0.6308764	total: 6.78s	remaining: 5.96s
    1597:	learn: 0.6308442	total: 6.79s	remaining: 5.95s
    1598:	learn: 0.6307926	total: 6.79s	remaining: 5.95s
    1599:	learn: 0.6307534	total: 6.79s	remaining: 5.94s
    1600:	learn: 0.6307105	total: 6.79s	remaining: 5.94s
    1601:	learn: 0.6306647	total: 6.8s	remaining: 5.93s
    1602:	learn: 0.6306140	total: 6.8s	remaining: 5.93s
    1603:	learn: 0.6305623	total: 6.8s	remaining: 5.92s
    1604:	learn: 0.6304950	total: 6.8s	remaining: 5.92s
    1605:	learn: 0.6304708	total: 6.81s	remaining: 5.91s
    1606:	learn: 0.6304266	total: 6.81s	remaining: 5.9s
    1607:	learn: 0.6303918	total: 6.81s	remaining: 5.9s
    1608:	learn: 0.6303583	total: 6.82s	remaining: 5.89s
    1609:	learn: 0.6303050	total: 6.82s	remaining: 5.89s
    1610:	learn: 0.6302558	total: 6.82s	remaining: 5.88s
    1611:	learn: 0.6301912	total: 6.83s	remaining: 5.88s
    1612:	learn: 0.6301377	total: 6.83s	remaining: 5.87s
    1613:	learn: 0.6301064	total: 6.83s	remaining: 5.87s
    1614:	learn: 0.6300780	total: 6.83s	remaining: 5.86s
    1615:	learn: 0.6300314	total: 6.84s	remaining: 5.86s
    1616:	learn: 0.6299928	total: 6.84s	remaining: 5.85s
    1617:	learn: 0.6299635	total: 6.84s	remaining: 5.85s
    1618:	learn: 0.6299125	total: 6.85s	remaining: 5.84s
    1619:	learn: 0.6298884	total: 6.85s	remaining: 5.83s
    1620:	learn: 0.6298338	total: 6.85s	remaining: 5.83s
    1621:	learn: 0.6297786	total: 6.86s	remaining: 5.82s
    1622:	learn: 0.6297435	total: 6.86s	remaining: 5.82s
    1623:	learn: 0.6297047	total: 6.86s	remaining: 5.81s
    1624:	learn: 0.6296754	total: 6.86s	remaining: 5.81s
    1625:	learn: 0.6296303	total: 6.87s	remaining: 5.81s
    1626:	learn: 0.6295917	total: 6.88s	remaining: 5.81s
    1627:	learn: 0.6295354	total: 6.89s	remaining: 5.8s
    1628:	learn: 0.6294857	total: 6.89s	remaining: 5.8s
    1629:	learn: 0.6294349	total: 6.9s	remaining: 5.8s
    1630:	learn: 0.6293924	total: 6.9s	remaining: 5.79s
    1631:	learn: 0.6293613	total: 6.9s	remaining: 5.79s
    1632:	learn: 0.6293035	total: 6.91s	remaining: 5.78s
    1633:	learn: 0.6292525	total: 6.91s	remaining: 5.78s
    1634:	learn: 0.6292175	total: 6.91s	remaining: 5.77s
    1635:	learn: 0.6291638	total: 6.92s	remaining: 5.76s
    1636:	learn: 0.6291055	total: 6.92s	remaining: 5.76s
    1637:	learn: 0.6290558	total: 6.92s	remaining: 5.75s
    1638:	learn: 0.6290016	total: 6.92s	remaining: 5.75s
    1639:	learn: 0.6289606	total: 6.93s	remaining: 5.74s
    1640:	learn: 0.6289174	total: 6.93s	remaining: 5.74s
    1641:	learn: 0.6288616	total: 6.93s	remaining: 5.73s
    1642:	learn: 0.6288059	total: 6.93s	remaining: 5.73s
    1643:	learn: 0.6287435	total: 6.94s	remaining: 5.72s
    1644:	learn: 0.6286998	total: 6.94s	remaining: 5.72s
    1645:	learn: 0.6286496	total: 6.94s	remaining: 5.71s
    1646:	learn: 0.6286039	total: 6.95s	remaining: 5.71s
    1647:	learn: 0.6285366	total: 6.95s	remaining: 5.7s
    1648:	learn: 0.6284981	total: 6.95s	remaining: 5.7s
    1649:	learn: 0.6284215	total: 6.95s	remaining: 5.69s
    1650:	learn: 0.6283964	total: 6.96s	remaining: 5.68s
    1651:	learn: 0.6283276	total: 6.96s	remaining: 5.68s
    1652:	learn: 0.6282920	total: 6.96s	remaining: 5.68s
    1653:	learn: 0.6282623	total: 6.97s	remaining: 5.67s
    1654:	learn: 0.6282310	total: 6.98s	remaining: 5.67s
    1655:	learn: 0.6281660	total: 6.98s	remaining: 5.67s
    1656:	learn: 0.6281451	total: 6.98s	remaining: 5.66s
    1657:	learn: 0.6281013	total: 6.99s	remaining: 5.65s
    1658:	learn: 0.6280308	total: 6.99s	remaining: 5.65s
    1659:	learn: 0.6280048	total: 6.99s	remaining: 5.64s
    1660:	learn: 0.6279416	total: 6.99s	remaining: 5.64s
    1661:	learn: 0.6278956	total: 7s	remaining: 5.63s
    1662:	learn: 0.6278628	total: 7s	remaining: 5.63s
    1663:	learn: 0.6278197	total: 7s	remaining: 5.62s
    1664:	learn: 0.6277869	total: 7s	remaining: 5.62s
    1665:	learn: 0.6277401	total: 7.01s	remaining: 5.61s
    1666:	learn: 0.6276773	total: 7.01s	remaining: 5.61s
    1667:	learn: 0.6276499	total: 7.01s	remaining: 5.6s
    1668:	learn: 0.6276129	total: 7.02s	remaining: 5.6s
    1669:	learn: 0.6275800	total: 7.02s	remaining: 5.59s
    1670:	learn: 0.6275083	total: 7.02s	remaining: 5.58s
    1671:	learn: 0.6274497	total: 7.03s	remaining: 5.58s
    1672:	learn: 0.6273885	total: 7.03s	remaining: 5.57s
    1673:	learn: 0.6273336	total: 7.03s	remaining: 5.57s
    1674:	learn: 0.6272648	total: 7.03s	remaining: 5.56s
    1675:	learn: 0.6272374	total: 7.04s	remaining: 5.56s
    1676:	learn: 0.6271916	total: 7.04s	remaining: 5.55s
    1677:	learn: 0.6271595	total: 7.04s	remaining: 5.55s
    1678:	learn: 0.6271098	total: 7.04s	remaining: 5.54s
    1679:	learn: 0.6270820	total: 7.05s	remaining: 5.54s
    1680:	learn: 0.6270243	total: 7.05s	remaining: 5.53s
    1681:	learn: 0.6269352	total: 7.05s	remaining: 5.53s
    1682:	learn: 0.6268864	total: 7.06s	remaining: 5.52s
    1683:	learn: 0.6268431	total: 7.06s	remaining: 5.52s
    1684:	learn: 0.6268080	total: 7.06s	remaining: 5.51s
    1685:	learn: 0.6267776	total: 7.06s	remaining: 5.51s
    1686:	learn: 0.6267506	total: 7.07s	remaining: 5.5s
    1687:	learn: 0.6267136	total: 7.1s	remaining: 5.52s
    1688:	learn: 0.6266771	total: 7.11s	remaining: 5.51s
    1689:	learn: 0.6266383	total: 7.11s	remaining: 5.51s
    1690:	learn: 0.6265868	total: 7.12s	remaining: 5.51s
    1691:	learn: 0.6265474	total: 7.13s	remaining: 5.51s
    1692:	learn: 0.6265081	total: 7.14s	remaining: 5.51s
    1693:	learn: 0.6264603	total: 7.14s	remaining: 5.51s
    1694:	learn: 0.6264199	total: 7.15s	remaining: 5.51s
    1695:	learn: 0.6263651	total: 7.16s	remaining: 5.5s
    1696:	learn: 0.6263384	total: 7.17s	remaining: 5.5s
    1697:	learn: 0.6262996	total: 7.17s	remaining: 5.5s
    1698:	learn: 0.6262313	total: 7.18s	remaining: 5.5s
    1699:	learn: 0.6261717	total: 7.19s	remaining: 5.5s
    1700:	learn: 0.6261177	total: 7.2s	remaining: 5.49s
    1701:	learn: 0.6260691	total: 7.2s	remaining: 5.49s
    1702:	learn: 0.6260192	total: 7.21s	remaining: 5.49s
    1703:	learn: 0.6259844	total: 7.21s	remaining: 5.49s
    1704:	learn: 0.6259315	total: 7.22s	remaining: 5.48s
    1705:	learn: 0.6259037	total: 7.22s	remaining: 5.48s
    1706:	learn: 0.6258590	total: 7.23s	remaining: 5.47s
    1707:	learn: 0.6258042	total: 7.23s	remaining: 5.47s
    1708:	learn: 0.6257727	total: 7.23s	remaining: 5.46s
    1709:	learn: 0.6257205	total: 7.24s	remaining: 5.46s
    1710:	learn: 0.6256588	total: 7.25s	remaining: 5.46s
    1711:	learn: 0.6256303	total: 7.25s	remaining: 5.45s
    1712:	learn: 0.6255923	total: 7.25s	remaining: 5.45s
    1713:	learn: 0.6255479	total: 7.25s	remaining: 5.44s
    1714:	learn: 0.6254981	total: 7.26s	remaining: 5.44s
    1715:	learn: 0.6254534	total: 7.26s	remaining: 5.43s
    1716:	learn: 0.6254268	total: 7.26s	remaining: 5.43s
    1717:	learn: 0.6253988	total: 7.27s	remaining: 5.43s
    1718:	learn: 0.6253511	total: 7.28s	remaining: 5.42s
    1719:	learn: 0.6252996	total: 7.29s	remaining: 5.42s
    1720:	learn: 0.6252259	total: 7.29s	remaining: 5.42s
    1721:	learn: 0.6251891	total: 7.29s	remaining: 5.41s
    1722:	learn: 0.6251580	total: 7.3s	remaining: 5.41s
    1723:	learn: 0.6251156	total: 7.3s	remaining: 5.4s
    1724:	learn: 0.6250744	total: 7.3s	remaining: 5.4s
    1725:	learn: 0.6250323	total: 7.31s	remaining: 5.39s
    1726:	learn: 0.6249932	total: 7.31s	remaining: 5.39s
    1727:	learn: 0.6249689	total: 7.31s	remaining: 5.38s
    1728:	learn: 0.6249055	total: 7.32s	remaining: 5.38s
    1729:	learn: 0.6248570	total: 7.32s	remaining: 5.38s
    1730:	learn: 0.6247852	total: 7.33s	remaining: 5.37s
    1731:	learn: 0.6247345	total: 7.33s	remaining: 5.37s
    1732:	learn: 0.6246955	total: 7.33s	remaining: 5.36s
    1733:	learn: 0.6246361	total: 7.33s	remaining: 5.36s
    1734:	learn: 0.6245837	total: 7.34s	remaining: 5.35s
    1735:	learn: 0.6245375	total: 7.34s	remaining: 5.34s
    1736:	learn: 0.6244729	total: 7.34s	remaining: 5.34s
    1737:	learn: 0.6244270	total: 7.35s	remaining: 5.33s
    1738:	learn: 0.6243839	total: 7.35s	remaining: 5.33s
    1739:	learn: 0.6243155	total: 7.35s	remaining: 5.33s
    1740:	learn: 0.6242854	total: 7.36s	remaining: 5.32s
    1741:	learn: 0.6242242	total: 7.36s	remaining: 5.32s
    1742:	learn: 0.6241945	total: 7.37s	remaining: 5.31s
    1743:	learn: 0.6241516	total: 7.37s	remaining: 5.31s
    1744:	learn: 0.6241133	total: 7.37s	remaining: 5.3s
    1745:	learn: 0.6240820	total: 7.38s	remaining: 5.3s
    1746:	learn: 0.6240295	total: 7.38s	remaining: 5.29s
    1747:	learn: 0.6239963	total: 7.39s	remaining: 5.29s
    1748:	learn: 0.6239517	total: 7.39s	remaining: 5.29s
    1749:	learn: 0.6239077	total: 7.39s	remaining: 5.28s
    1750:	learn: 0.6238715	total: 7.39s	remaining: 5.28s
    1751:	learn: 0.6238411	total: 7.4s	remaining: 5.27s
    1752:	learn: 0.6238118	total: 7.4s	remaining: 5.26s
    1753:	learn: 0.6237765	total: 7.41s	remaining: 5.26s
    1754:	learn: 0.6237405	total: 7.41s	remaining: 5.25s
    1755:	learn: 0.6237047	total: 7.41s	remaining: 5.25s
    1756:	learn: 0.6236827	total: 7.41s	remaining: 5.25s
    1757:	learn: 0.6236611	total: 7.42s	remaining: 5.24s
    1758:	learn: 0.6236223	total: 7.42s	remaining: 5.24s
    1759:	learn: 0.6235796	total: 7.42s	remaining: 5.23s
    1760:	learn: 0.6235576	total: 7.43s	remaining: 5.22s
    1761:	learn: 0.6235191	total: 7.44s	remaining: 5.23s
    1762:	learn: 0.6234975	total: 7.45s	remaining: 5.22s
    1763:	learn: 0.6234532	total: 7.45s	remaining: 5.22s
    1764:	learn: 0.6234166	total: 7.46s	remaining: 5.22s
    1765:	learn: 0.6233488	total: 7.47s	remaining: 5.22s
    1766:	learn: 0.6233111	total: 7.48s	remaining: 5.22s
    1767:	learn: 0.6232719	total: 7.49s	remaining: 5.22s
    1768:	learn: 0.6232331	total: 7.49s	remaining: 5.21s
    1769:	learn: 0.6232056	total: 7.5s	remaining: 5.21s
    1770:	learn: 0.6231614	total: 7.51s	remaining: 5.21s
    1771:	learn: 0.6231238	total: 7.52s	remaining: 5.21s
    1772:	learn: 0.6230688	total: 7.53s	remaining: 5.21s
    1773:	learn: 0.6230149	total: 7.53s	remaining: 5.21s
    1774:	learn: 0.6229808	total: 7.54s	remaining: 5.2s
    1775:	learn: 0.6229450	total: 7.54s	remaining: 5.2s
    1776:	learn: 0.6228991	total: 7.54s	remaining: 5.19s
    1777:	learn: 0.6228305	total: 7.55s	remaining: 5.19s
    1778:	learn: 0.6227931	total: 7.55s	remaining: 5.18s
    1779:	learn: 0.6227737	total: 7.55s	remaining: 5.18s
    1780:	learn: 0.6227133	total: 7.58s	remaining: 5.18s
    1781:	learn: 0.6226721	total: 7.59s	remaining: 5.18s
    1782:	learn: 0.6225893	total: 7.59s	remaining: 5.18s
    1783:	learn: 0.6225406	total: 7.59s	remaining: 5.18s
    1784:	learn: 0.6224835	total: 7.6s	remaining: 5.17s
    1785:	learn: 0.6224199	total: 7.6s	remaining: 5.17s
    1786:	learn: 0.6223587	total: 7.61s	remaining: 5.16s
    1787:	learn: 0.6223037	total: 7.61s	remaining: 5.16s
    1788:	learn: 0.6222740	total: 7.61s	remaining: 5.15s
    1789:	learn: 0.6222213	total: 7.62s	remaining: 5.15s
    1790:	learn: 0.6221850	total: 7.62s	remaining: 5.14s
    1791:	learn: 0.6221456	total: 7.62s	remaining: 5.14s
    1792:	learn: 0.6220810	total: 7.63s	remaining: 5.13s
    1793:	learn: 0.6220398	total: 7.63s	remaining: 5.13s
    1794:	learn: 0.6219994	total: 7.63s	remaining: 5.13s
    1795:	learn: 0.6219628	total: 7.64s	remaining: 5.12s
    1796:	learn: 0.6219409	total: 7.64s	remaining: 5.12s
    1797:	learn: 0.6219121	total: 7.64s	remaining: 5.11s
    1798:	learn: 0.6218877	total: 7.65s	remaining: 5.11s
    1799:	learn: 0.6218378	total: 7.65s	remaining: 5.1s
    1800:	learn: 0.6218069	total: 7.66s	remaining: 5.1s
    1801:	learn: 0.6217689	total: 7.66s	remaining: 5.09s
    1802:	learn: 0.6217468	total: 7.66s	remaining: 5.09s
    1803:	learn: 0.6217036	total: 7.67s	remaining: 5.08s
    1804:	learn: 0.6216509	total: 7.67s	remaining: 5.08s
    1805:	learn: 0.6216159	total: 7.67s	remaining: 5.07s
    1806:	learn: 0.6215865	total: 7.68s	remaining: 5.07s
    1807:	learn: 0.6215452	total: 7.68s	remaining: 5.06s
    1808:	learn: 0.6215099	total: 7.68s	remaining: 5.06s
    1809:	learn: 0.6214435	total: 7.69s	remaining: 5.06s
    1810:	learn: 0.6214094	total: 7.7s	remaining: 5.05s
    1811:	learn: 0.6213534	total: 7.71s	remaining: 5.05s
    1812:	learn: 0.6213302	total: 7.71s	remaining: 5.05s
    1813:	learn: 0.6212948	total: 7.71s	remaining: 5.04s
    1814:	learn: 0.6212433	total: 7.72s	remaining: 5.04s
    1815:	learn: 0.6211905	total: 7.72s	remaining: 5.04s
    1816:	learn: 0.6211544	total: 7.73s	remaining: 5.03s
    1817:	learn: 0.6211192	total: 7.73s	remaining: 5.03s
    1818:	learn: 0.6210694	total: 7.74s	remaining: 5.02s
    1819:	learn: 0.6210479	total: 7.74s	remaining: 5.02s
    1820:	learn: 0.6210059	total: 7.74s	remaining: 5.01s
    1821:	learn: 0.6209701	total: 7.75s	remaining: 5.01s
    1822:	learn: 0.6209235	total: 7.75s	remaining: 5s
    1823:	learn: 0.6208805	total: 7.76s	remaining: 5s
    1824:	learn: 0.6208392	total: 7.77s	remaining: 5s
    1825:	learn: 0.6207790	total: 7.8s	remaining: 5.01s
    1826:	learn: 0.6207520	total: 7.81s	remaining: 5.01s
    1827:	learn: 0.6207200	total: 7.82s	remaining: 5.01s
    1828:	learn: 0.6206660	total: 7.84s	remaining: 5.02s
    1829:	learn: 0.6206094	total: 7.85s	remaining: 5.02s
    1830:	learn: 0.6205792	total: 7.87s	remaining: 5.03s
    1831:	learn: 0.6205357	total: 7.89s	remaining: 5.03s
    1832:	learn: 0.6204792	total: 7.9s	remaining: 5.03s
    1833:	learn: 0.6204626	total: 7.91s	remaining: 5.03s
    1834:	learn: 0.6204227	total: 7.94s	remaining: 5.04s
    1835:	learn: 0.6203816	total: 7.95s	remaining: 5.04s
    1836:	learn: 0.6203501	total: 7.97s	remaining: 5.04s
    1837:	learn: 0.6203173	total: 7.97s	remaining: 5.04s
    1838:	learn: 0.6202696	total: 7.99s	remaining: 5.04s
    1839:	learn: 0.6202251	total: 7.99s	remaining: 5.04s
    1840:	learn: 0.6201794	total: 8s	remaining: 5.03s
    1841:	learn: 0.6201401	total: 8s	remaining: 5.03s
    1842:	learn: 0.6200912	total: 8s	remaining: 5.02s
    1843:	learn: 0.6200433	total: 8s	remaining: 5.02s
    1844:	learn: 0.6199987	total: 8.01s	remaining: 5.01s
    1845:	learn: 0.6199321	total: 8.01s	remaining: 5.01s
    1846:	learn: 0.6198840	total: 8.02s	remaining: 5s
    1847:	learn: 0.6198597	total: 8.02s	remaining: 5s
    1848:	learn: 0.6198310	total: 8.02s	remaining: 4.99s
    1849:	learn: 0.6197902	total: 8.03s	remaining: 4.99s
    1850:	learn: 0.6197598	total: 8.03s	remaining: 4.99s
    1851:	learn: 0.6196964	total: 8.04s	remaining: 4.98s
    1852:	learn: 0.6196608	total: 8.04s	remaining: 4.97s
    1853:	learn: 0.6196314	total: 8.04s	remaining: 4.97s
    1854:	learn: 0.6195848	total: 8.04s	remaining: 4.96s
    1855:	learn: 0.6195332	total: 8.05s	remaining: 4.96s
    1856:	learn: 0.6195075	total: 8.05s	remaining: 4.95s
    1857:	learn: 0.6194536	total: 8.05s	remaining: 4.95s
    1858:	learn: 0.6194158	total: 8.08s	remaining: 4.96s
    1859:	learn: 0.6193571	total: 8.09s	remaining: 4.96s
    1860:	learn: 0.6193170	total: 8.1s	remaining: 4.95s
    1861:	learn: 0.6192548	total: 8.1s	remaining: 4.95s
    1862:	learn: 0.6192115	total: 8.11s	remaining: 4.95s
    1863:	learn: 0.6191873	total: 8.12s	remaining: 4.95s
    1864:	learn: 0.6191410	total: 8.13s	remaining: 4.95s
    1865:	learn: 0.6191061	total: 8.13s	remaining: 4.94s
    1866:	learn: 0.6190813	total: 8.14s	remaining: 4.94s
    1867:	learn: 0.6190544	total: 8.15s	remaining: 4.94s
    1868:	learn: 0.6190258	total: 8.16s	remaining: 4.94s
    1869:	learn: 0.6189524	total: 8.17s	remaining: 4.93s
    1870:	learn: 0.6188926	total: 8.17s	remaining: 4.93s
    1871:	learn: 0.6188566	total: 8.18s	remaining: 4.93s
    1872:	learn: 0.6188289	total: 8.19s	remaining: 4.93s
    1873:	learn: 0.6187735	total: 8.2s	remaining: 4.92s
    1874:	learn: 0.6187434	total: 8.2s	remaining: 4.92s
    1875:	learn: 0.6187176	total: 8.21s	remaining: 4.92s
    1876:	learn: 0.6186991	total: 8.22s	remaining: 4.92s
    1877:	learn: 0.6186496	total: 8.22s	remaining: 4.91s
    1878:	learn: 0.6186153	total: 8.22s	remaining: 4.91s
    1879:	learn: 0.6185620	total: 8.23s	remaining: 4.9s
    1880:	learn: 0.6185365	total: 8.23s	remaining: 4.89s
    1881:	learn: 0.6185154	total: 8.23s	remaining: 4.89s
    1882:	learn: 0.6184644	total: 8.23s	remaining: 4.88s
    1883:	learn: 0.6184166	total: 8.24s	remaining: 4.88s
    1884:	learn: 0.6183662	total: 8.24s	remaining: 4.87s
    1885:	learn: 0.6183090	total: 8.24s	remaining: 4.87s
    1886:	learn: 0.6182688	total: 8.25s	remaining: 4.87s
    1887:	learn: 0.6182322	total: 8.25s	remaining: 4.86s
    1888:	learn: 0.6181964	total: 8.25s	remaining: 4.85s
    1889:	learn: 0.6181526	total: 8.26s	remaining: 4.85s
    1890:	learn: 0.6181084	total: 8.26s	remaining: 4.84s
    1891:	learn: 0.6180882	total: 8.26s	remaining: 4.84s
    1892:	learn: 0.6180495	total: 8.27s	remaining: 4.83s
    1893:	learn: 0.6180081	total: 8.27s	remaining: 4.83s
    1894:	learn: 0.6179642	total: 8.27s	remaining: 4.82s
    1895:	learn: 0.6179403	total: 8.28s	remaining: 4.82s
    1896:	learn: 0.6178858	total: 8.28s	remaining: 4.82s
    1897:	learn: 0.6178486	total: 8.29s	remaining: 4.81s
    1898:	learn: 0.6178187	total: 8.29s	remaining: 4.81s
    1899:	learn: 0.6177876	total: 8.3s	remaining: 4.8s
    1900:	learn: 0.6177543	total: 8.3s	remaining: 4.8s
    1901:	learn: 0.6177158	total: 8.31s	remaining: 4.79s
    1902:	learn: 0.6176700	total: 8.31s	remaining: 4.79s
    1903:	learn: 0.6176411	total: 8.32s	remaining: 4.79s
    1904:	learn: 0.6176137	total: 8.32s	remaining: 4.78s
    1905:	learn: 0.6175829	total: 8.33s	remaining: 4.78s
    1906:	learn: 0.6175281	total: 8.33s	remaining: 4.77s
    1907:	learn: 0.6174924	total: 8.33s	remaining: 4.77s
    1908:	learn: 0.6174603	total: 8.34s	remaining: 4.76s
    1909:	learn: 0.6174214	total: 8.34s	remaining: 4.76s
    1910:	learn: 0.6173861	total: 8.34s	remaining: 4.75s
    1911:	learn: 0.6173189	total: 8.34s	remaining: 4.75s
    1912:	learn: 0.6172656	total: 8.35s	remaining: 4.74s
    1913:	learn: 0.6172437	total: 8.35s	remaining: 4.74s
    1914:	learn: 0.6172021	total: 8.36s	remaining: 4.73s
    1915:	learn: 0.6171695	total: 8.36s	remaining: 4.73s
    1916:	learn: 0.6171340	total: 8.36s	remaining: 4.72s
    1917:	learn: 0.6171113	total: 8.36s	remaining: 4.72s
    1918:	learn: 0.6170670	total: 8.37s	remaining: 4.71s
    1919:	learn: 0.6170153	total: 8.37s	remaining: 4.71s
    1920:	learn: 0.6169890	total: 8.38s	remaining: 4.7s
    1921:	learn: 0.6169574	total: 8.38s	remaining: 4.7s
    1922:	learn: 0.6169181	total: 8.38s	remaining: 4.69s
    1923:	learn: 0.6168782	total: 8.38s	remaining: 4.69s
    1924:	learn: 0.6168307	total: 8.39s	remaining: 4.68s
    1925:	learn: 0.6167970	total: 8.39s	remaining: 4.68s
    1926:	learn: 0.6167685	total: 8.39s	remaining: 4.67s
    1927:	learn: 0.6167227	total: 8.4s	remaining: 4.67s
    1928:	learn: 0.6166727	total: 8.4s	remaining: 4.66s
    1929:	learn: 0.6166298	total: 8.41s	remaining: 4.66s
    1930:	learn: 0.6165892	total: 8.42s	remaining: 4.66s
    1931:	learn: 0.6165423	total: 8.43s	remaining: 4.66s
    1932:	learn: 0.6165041	total: 8.43s	remaining: 4.66s
    1933:	learn: 0.6164472	total: 8.44s	remaining: 4.65s
    1934:	learn: 0.6164189	total: 8.45s	remaining: 4.65s
    1935:	learn: 0.6164022	total: 8.46s	remaining: 4.65s
    1936:	learn: 0.6163741	total: 8.47s	remaining: 4.65s
    1937:	learn: 0.6163336	total: 8.47s	remaining: 4.64s
    1938:	learn: 0.6163025	total: 8.48s	remaining: 4.64s
    1939:	learn: 0.6162656	total: 8.49s	remaining: 4.64s
    1940:	learn: 0.6162406	total: 8.5s	remaining: 4.64s
    1941:	learn: 0.6162161	total: 8.51s	remaining: 4.63s
    1942:	learn: 0.6161620	total: 8.51s	remaining: 4.63s
    1943:	learn: 0.6161432	total: 8.51s	remaining: 4.62s
    1944:	learn: 0.6161116	total: 8.52s	remaining: 4.62s
    1945:	learn: 0.6160711	total: 8.52s	remaining: 4.62s
    1946:	learn: 0.6160327	total: 8.52s	remaining: 4.61s
    1947:	learn: 0.6159870	total: 8.53s	remaining: 4.6s
    1948:	learn: 0.6159485	total: 8.54s	remaining: 4.6s
    1949:	learn: 0.6159134	total: 8.55s	remaining: 4.6s
    1950:	learn: 0.6158712	total: 8.56s	remaining: 4.6s
    1951:	learn: 0.6158381	total: 8.58s	remaining: 4.61s
    1952:	learn: 0.6157882	total: 8.6s	remaining: 4.61s
    1953:	learn: 0.6157430	total: 8.61s	remaining: 4.61s
    1954:	learn: 0.6157014	total: 8.62s	remaining: 4.61s
    1955:	learn: 0.6156814	total: 8.63s	remaining: 4.61s
    1956:	learn: 0.6156251	total: 8.65s	remaining: 4.61s
    1957:	learn: 0.6155657	total: 8.67s	remaining: 4.62s
    1958:	learn: 0.6155316	total: 8.69s	remaining: 4.62s
    1959:	learn: 0.6154882	total: 8.7s	remaining: 4.62s
    1960:	learn: 0.6154426	total: 8.71s	remaining: 4.62s
    1961:	learn: 0.6153794	total: 8.73s	remaining: 4.62s
    1962:	learn: 0.6153578	total: 8.75s	remaining: 4.62s
    1963:	learn: 0.6153109	total: 8.76s	remaining: 4.62s
    1964:	learn: 0.6152709	total: 8.77s	remaining: 4.62s
    1965:	learn: 0.6152432	total: 8.77s	remaining: 4.62s
    1966:	learn: 0.6152138	total: 8.78s	remaining: 4.61s
    1967:	learn: 0.6151912	total: 8.78s	remaining: 4.61s
    1968:	learn: 0.6151524	total: 8.79s	remaining: 4.6s
    1969:	learn: 0.6151195	total: 8.79s	remaining: 4.6s
    1970:	learn: 0.6150794	total: 8.79s	remaining: 4.59s
    1971:	learn: 0.6150507	total: 8.8s	remaining: 4.58s
    1972:	learn: 0.6149976	total: 8.8s	remaining: 4.58s
    1973:	learn: 0.6149684	total: 8.8s	remaining: 4.58s
    1974:	learn: 0.6149280	total: 8.81s	remaining: 4.57s
    1975:	learn: 0.6148901	total: 8.81s	remaining: 4.57s
    1976:	learn: 0.6148434	total: 8.81s	remaining: 4.56s
    1977:	learn: 0.6148075	total: 8.82s	remaining: 4.55s
    1978:	learn: 0.6147602	total: 8.82s	remaining: 4.55s
    1979:	learn: 0.6147339	total: 8.82s	remaining: 4.54s
    1980:	learn: 0.6146995	total: 8.82s	remaining: 4.54s
    1981:	learn: 0.6146568	total: 8.83s	remaining: 4.53s
    1982:	learn: 0.6146269	total: 8.83s	remaining: 4.53s
    1983:	learn: 0.6145918	total: 8.83s	remaining: 4.52s
    1984:	learn: 0.6145572	total: 8.84s	remaining: 4.52s
    1985:	learn: 0.6145212	total: 8.84s	remaining: 4.51s
    1986:	learn: 0.6144724	total: 8.84s	remaining: 4.51s
    1987:	learn: 0.6144144	total: 8.85s	remaining: 4.5s
    1988:	learn: 0.6143566	total: 8.85s	remaining: 4.5s
    1989:	learn: 0.6143159	total: 8.86s	remaining: 4.49s
    1990:	learn: 0.6142781	total: 8.86s	remaining: 4.49s
    1991:	learn: 0.6142270	total: 8.86s	remaining: 4.48s
    1992:	learn: 0.6141828	total: 8.87s	remaining: 4.48s
    1993:	learn: 0.6141486	total: 8.87s	remaining: 4.47s
    1994:	learn: 0.6141132	total: 8.87s	remaining: 4.47s
    1995:	learn: 0.6140739	total: 8.88s	remaining: 4.46s
    1996:	learn: 0.6140482	total: 8.88s	remaining: 4.46s
    1997:	learn: 0.6140251	total: 8.88s	remaining: 4.45s
    1998:	learn: 0.6139956	total: 8.89s	remaining: 4.45s
    1999:	learn: 0.6139512	total: 8.89s	remaining: 4.45s
    2000:	learn: 0.6139278	total: 8.9s	remaining: 4.44s
    2001:	learn: 0.6138713	total: 8.91s	remaining: 4.44s
    2002:	learn: 0.6138407	total: 8.92s	remaining: 4.44s
    2003:	learn: 0.6137919	total: 8.93s	remaining: 4.44s
    2004:	learn: 0.6137648	total: 8.94s	remaining: 4.43s
    2005:	learn: 0.6137390	total: 8.95s	remaining: 4.43s
    2006:	learn: 0.6136957	total: 8.96s	remaining: 4.43s
    2007:	learn: 0.6136685	total: 8.97s	remaining: 4.43s
    2008:	learn: 0.6136192	total: 8.99s	remaining: 4.43s
    2009:	learn: 0.6135778	total: 9.05s	remaining: 4.46s
    2010:	learn: 0.6135393	total: 9.09s	remaining: 4.47s
    2011:	learn: 0.6135075	total: 9.09s	remaining: 4.47s
    2012:	learn: 0.6134680	total: 9.11s	remaining: 4.46s
    2013:	learn: 0.6134203	total: 9.12s	remaining: 4.46s
    2014:	learn: 0.6133893	total: 9.13s	remaining: 4.46s
    2015:	learn: 0.6133346	total: 9.14s	remaining: 4.46s
    2016:	learn: 0.6133125	total: 9.15s	remaining: 4.46s
    2017:	learn: 0.6132744	total: 9.16s	remaining: 4.46s
    2018:	learn: 0.6132271	total: 9.17s	remaining: 4.46s
    2019:	learn: 0.6131732	total: 9.17s	remaining: 4.45s
    2020:	learn: 0.6131321	total: 9.18s	remaining: 4.45s
    2021:	learn: 0.6130892	total: 9.18s	remaining: 4.44s
    2022:	learn: 0.6130490	total: 9.18s	remaining: 4.43s
    2023:	learn: 0.6130116	total: 9.19s	remaining: 4.43s
    2024:	learn: 0.6129786	total: 9.19s	remaining: 4.42s
    2025:	learn: 0.6129080	total: 9.2s	remaining: 4.42s
    2026:	learn: 0.6128764	total: 9.2s	remaining: 4.42s
    2027:	learn: 0.6128461	total: 9.2s	remaining: 4.41s
    2028:	learn: 0.6128051	total: 9.21s	remaining: 4.41s
    2029:	learn: 0.6127676	total: 9.21s	remaining: 4.4s
    2030:	learn: 0.6127316	total: 9.21s	remaining: 4.39s
    2031:	learn: 0.6126868	total: 9.21s	remaining: 4.39s
    2032:	learn: 0.6126375	total: 9.22s	remaining: 4.38s
    2033:	learn: 0.6125975	total: 9.22s	remaining: 4.38s
    2034:	learn: 0.6125719	total: 9.22s	remaining: 4.37s
    2035:	learn: 0.6125236	total: 9.22s	remaining: 4.37s
    2036:	learn: 0.6124814	total: 9.23s	remaining: 4.36s
    2037:	learn: 0.6124541	total: 9.23s	remaining: 4.36s
    2038:	learn: 0.6124105	total: 9.23s	remaining: 4.35s
    2039:	learn: 0.6123867	total: 9.24s	remaining: 4.35s
    2040:	learn: 0.6123373	total: 9.24s	remaining: 4.34s
    2041:	learn: 0.6123056	total: 9.24s	remaining: 4.34s
    2042:	learn: 0.6122650	total: 9.24s	remaining: 4.33s
    2043:	learn: 0.6122329	total: 9.25s	remaining: 4.33s
    2044:	learn: 0.6121790	total: 9.25s	remaining: 4.32s
    2045:	learn: 0.6121537	total: 9.25s	remaining: 4.31s
    2046:	learn: 0.6121360	total: 9.26s	remaining: 4.31s
    2047:	learn: 0.6120971	total: 9.26s	remaining: 4.3s
    2048:	learn: 0.6120787	total: 9.26s	remaining: 4.3s
    2049:	learn: 0.6120363	total: 9.27s	remaining: 4.29s
    2050:	learn: 0.6120066	total: 9.27s	remaining: 4.29s
    2051:	learn: 0.6119680	total: 9.27s	remaining: 4.28s
    2052:	learn: 0.6119220	total: 9.27s	remaining: 4.28s
    2053:	learn: 0.6118874	total: 9.28s	remaining: 4.27s
    2054:	learn: 0.6118727	total: 9.28s	remaining: 4.27s
    2055:	learn: 0.6118351	total: 9.28s	remaining: 4.26s
    2056:	learn: 0.6118168	total: 9.28s	remaining: 4.26s
    2057:	learn: 0.6117886	total: 9.29s	remaining: 4.25s
    2058:	learn: 0.6117426	total: 9.29s	remaining: 4.25s
    2059:	learn: 0.6116961	total: 9.3s	remaining: 4.24s
    2060:	learn: 0.6116481	total: 9.31s	remaining: 4.24s
    2061:	learn: 0.6115896	total: 9.31s	remaining: 4.24s
    2062:	learn: 0.6115628	total: 9.31s	remaining: 4.23s
    2063:	learn: 0.6115368	total: 9.32s	remaining: 4.23s
    2064:	learn: 0.6114981	total: 9.32s	remaining: 4.22s
    2065:	learn: 0.6114721	total: 9.33s	remaining: 4.22s
    2066:	learn: 0.6114452	total: 9.33s	remaining: 4.21s
    2067:	learn: 0.6114034	total: 9.33s	remaining: 4.21s
    2068:	learn: 0.6113768	total: 9.33s	remaining: 4.2s
    2069:	learn: 0.6113508	total: 9.34s	remaining: 4.2s
    2070:	learn: 0.6113169	total: 9.34s	remaining: 4.19s
    2071:	learn: 0.6112963	total: 9.34s	remaining: 4.18s
    2072:	learn: 0.6112696	total: 9.35s	remaining: 4.18s
    2073:	learn: 0.6112511	total: 9.35s	remaining: 4.17s
    2074:	learn: 0.6112120	total: 9.35s	remaining: 4.17s
    2075:	learn: 0.6111746	total: 9.35s	remaining: 4.16s
    2076:	learn: 0.6111441	total: 9.36s	remaining: 4.16s
    2077:	learn: 0.6111087	total: 9.36s	remaining: 4.15s
    2078:	learn: 0.6110595	total: 9.36s	remaining: 4.15s
    2079:	learn: 0.6110382	total: 9.37s	remaining: 4.14s
    2080:	learn: 0.6110049	total: 9.37s	remaining: 4.14s
    2081:	learn: 0.6109661	total: 9.37s	remaining: 4.13s
    2082:	learn: 0.6109262	total: 9.37s	remaining: 4.13s
    2083:	learn: 0.6108862	total: 9.38s	remaining: 4.12s
    2084:	learn: 0.6108568	total: 9.39s	remaining: 4.12s
    2085:	learn: 0.6108208	total: 9.4s	remaining: 4.12s
    2086:	learn: 0.6107841	total: 9.4s	remaining: 4.11s
    2087:	learn: 0.6107426	total: 9.41s	remaining: 4.11s
    2088:	learn: 0.6106775	total: 9.42s	remaining: 4.11s
    2089:	learn: 0.6106481	total: 9.43s	remaining: 4.1s
    2090:	learn: 0.6106116	total: 9.44s	remaining: 4.1s
    2091:	learn: 0.6105655	total: 9.45s	remaining: 4.1s
    2092:	learn: 0.6105388	total: 9.45s	remaining: 4.09s
    2093:	learn: 0.6104905	total: 9.47s	remaining: 4.1s
    2094:	learn: 0.6104536	total: 9.48s	remaining: 4.09s
    2095:	learn: 0.6104150	total: 9.49s	remaining: 4.09s
    2096:	learn: 0.6103799	total: 9.5s	remaining: 4.09s
    2097:	learn: 0.6103387	total: 9.5s	remaining: 4.08s
    2098:	learn: 0.6102945	total: 9.51s	remaining: 4.08s
    2099:	learn: 0.6102729	total: 9.52s	remaining: 4.08s
    2100:	learn: 0.6102399	total: 9.53s	remaining: 4.08s
    2101:	learn: 0.6102164	total: 9.53s	remaining: 4.07s
    2102:	learn: 0.6101708	total: 9.53s	remaining: 4.07s
    2103:	learn: 0.6101345	total: 9.54s	remaining: 4.06s
    2104:	learn: 0.6101124	total: 9.54s	remaining: 4.06s
    2105:	learn: 0.6100839	total: 9.54s	remaining: 4.05s
    2106:	learn: 0.6100507	total: 9.55s	remaining: 4.05s
    2107:	learn: 0.6100096	total: 9.55s	remaining: 4.04s
    2108:	learn: 0.6099781	total: 9.55s	remaining: 4.04s
    2109:	learn: 0.6099457	total: 9.56s	remaining: 4.03s
    2110:	learn: 0.6099238	total: 9.56s	remaining: 4.03s
    2111:	learn: 0.6098690	total: 9.56s	remaining: 4.02s
    2112:	learn: 0.6098104	total: 9.57s	remaining: 4.01s
    2113:	learn: 0.6097845	total: 9.57s	remaining: 4.01s
    2114:	learn: 0.6097643	total: 9.57s	remaining: 4s
    2115:	learn: 0.6097341	total: 9.57s	remaining: 4s
    2116:	learn: 0.6097131	total: 9.58s	remaining: 4s
    2117:	learn: 0.6096669	total: 9.58s	remaining: 3.99s
    2118:	learn: 0.6096152	total: 9.59s	remaining: 3.98s
    2119:	learn: 0.6095746	total: 9.59s	remaining: 3.98s
    2120:	learn: 0.6095350	total: 9.59s	remaining: 3.97s
    2121:	learn: 0.6094817	total: 9.59s	remaining: 3.97s
    2122:	learn: 0.6094459	total: 9.6s	remaining: 3.96s
    2123:	learn: 0.6094197	total: 9.6s	remaining: 3.96s
    2124:	learn: 0.6093939	total: 9.6s	remaining: 3.95s
    2125:	learn: 0.6093688	total: 9.61s	remaining: 3.95s
    2126:	learn: 0.6093264	total: 9.61s	remaining: 3.94s
    2127:	learn: 0.6092943	total: 9.61s	remaining: 3.94s
    2128:	learn: 0.6092444	total: 9.62s	remaining: 3.93s
    2129:	learn: 0.6092170	total: 9.62s	remaining: 3.93s
    2130:	learn: 0.6091899	total: 9.62s	remaining: 3.92s
    2131:	learn: 0.6091543	total: 9.63s	remaining: 3.92s
    2132:	learn: 0.6091307	total: 9.63s	remaining: 3.91s
    2133:	learn: 0.6090857	total: 9.63s	remaining: 3.91s
    2134:	learn: 0.6090527	total: 9.64s	remaining: 3.9s
    2135:	learn: 0.6090312	total: 9.64s	remaining: 3.9s
    2136:	learn: 0.6090012	total: 9.64s	remaining: 3.89s
    2137:	learn: 0.6089728	total: 9.64s	remaining: 3.89s
    2138:	learn: 0.6089476	total: 9.65s	remaining: 3.88s
    2139:	learn: 0.6089030	total: 9.65s	remaining: 3.88s
    2140:	learn: 0.6088783	total: 9.66s	remaining: 3.88s
    2141:	learn: 0.6088484	total: 9.66s	remaining: 3.87s
    2142:	learn: 0.6088244	total: 9.66s	remaining: 3.87s
    2143:	learn: 0.6088019	total: 9.67s	remaining: 3.86s
    2144:	learn: 0.6087783	total: 9.67s	remaining: 3.85s
    2145:	learn: 0.6087462	total: 9.68s	remaining: 3.85s
    2146:	learn: 0.6087239	total: 9.68s	remaining: 3.85s
    2147:	learn: 0.6086774	total: 9.68s	remaining: 3.84s
    2148:	learn: 0.6086462	total: 9.69s	remaining: 3.84s
    2149:	learn: 0.6086151	total: 9.69s	remaining: 3.83s
    2150:	learn: 0.6085768	total: 9.69s	remaining: 3.83s
    2151:	learn: 0.6085511	total: 9.7s	remaining: 3.82s
    2152:	learn: 0.6085143	total: 9.71s	remaining: 3.82s
    2153:	learn: 0.6084686	total: 9.72s	remaining: 3.82s
    2154:	learn: 0.6084374	total: 9.72s	remaining: 3.81s
    2155:	learn: 0.6084129	total: 9.73s	remaining: 3.81s
    2156:	learn: 0.6083825	total: 9.73s	remaining: 3.8s
    2157:	learn: 0.6083540	total: 9.73s	remaining: 3.8s
    2158:	learn: 0.6083196	total: 9.74s	remaining: 3.79s
    2159:	learn: 0.6082988	total: 9.74s	remaining: 3.79s
    2160:	learn: 0.6082654	total: 9.75s	remaining: 3.78s
    2161:	learn: 0.6082459	total: 9.75s	remaining: 3.78s
    2162:	learn: 0.6082087	total: 9.76s	remaining: 3.77s
    2163:	learn: 0.6081805	total: 9.76s	remaining: 3.77s
    2164:	learn: 0.6081289	total: 9.76s	remaining: 3.77s
    2165:	learn: 0.6081048	total: 9.77s	remaining: 3.76s
    2166:	learn: 0.6080665	total: 9.77s	remaining: 3.75s
    2167:	learn: 0.6080311	total: 9.77s	remaining: 3.75s
    2168:	learn: 0.6079973	total: 9.78s	remaining: 3.75s
    2169:	learn: 0.6079612	total: 9.78s	remaining: 3.74s
    2170:	learn: 0.6079413	total: 9.78s	remaining: 3.73s
    2171:	learn: 0.6079094	total: 9.79s	remaining: 3.73s
    2172:	learn: 0.6078688	total: 9.79s	remaining: 3.73s
    2173:	learn: 0.6078409	total: 9.79s	remaining: 3.72s
    2174:	learn: 0.6078220	total: 9.79s	remaining: 3.71s
    2175:	learn: 0.6078012	total: 9.8s	remaining: 3.71s
    2176:	learn: 0.6077582	total: 9.8s	remaining: 3.71s
    2177:	learn: 0.6077277	total: 9.8s	remaining: 3.7s
    2178:	learn: 0.6076993	total: 9.81s	remaining: 3.7s
    2179:	learn: 0.6076768	total: 9.81s	remaining: 3.69s
    2180:	learn: 0.6076510	total: 9.81s	remaining: 3.69s
    2181:	learn: 0.6076180	total: 9.82s	remaining: 3.68s
    2182:	learn: 0.6075965	total: 9.82s	remaining: 3.67s
    2183:	learn: 0.6075653	total: 9.83s	remaining: 3.67s
    2184:	learn: 0.6075301	total: 9.83s	remaining: 3.67s
    2185:	learn: 0.6075005	total: 9.84s	remaining: 3.66s
    2186:	learn: 0.6074664	total: 9.84s	remaining: 3.66s
    2187:	learn: 0.6074433	total: 9.84s	remaining: 3.65s
    2188:	learn: 0.6074050	total: 9.85s	remaining: 3.65s
    2189:	learn: 0.6073866	total: 9.85s	remaining: 3.64s
    2190:	learn: 0.6073367	total: 9.85s	remaining: 3.64s
    2191:	learn: 0.6073159	total: 9.86s	remaining: 3.63s
    2192:	learn: 0.6072756	total: 9.86s	remaining: 3.63s
    2193:	learn: 0.6072508	total: 9.86s	remaining: 3.62s
    2194:	learn: 0.6072105	total: 9.87s	remaining: 3.62s
    2195:	learn: 0.6071853	total: 9.87s	remaining: 3.61s
    2196:	learn: 0.6071519	total: 9.88s	remaining: 3.61s
    2197:	learn: 0.6071288	total: 9.88s	remaining: 3.61s
    2198:	learn: 0.6070908	total: 9.89s	remaining: 3.6s
    2199:	learn: 0.6070652	total: 9.9s	remaining: 3.6s
    2200:	learn: 0.6070479	total: 9.91s	remaining: 3.6s
    2201:	learn: 0.6070194	total: 9.91s	remaining: 3.59s
    2202:	learn: 0.6069704	total: 9.92s	remaining: 3.59s
    2203:	learn: 0.6069441	total: 9.93s	remaining: 3.58s
    2204:	learn: 0.6069062	total: 9.93s	remaining: 3.58s
    2205:	learn: 0.6068568	total: 9.93s	remaining: 3.57s
    2206:	learn: 0.6068141	total: 9.93s	remaining: 3.57s
    2207:	learn: 0.6067911	total: 9.94s	remaining: 3.56s
    2208:	learn: 0.6067550	total: 9.94s	remaining: 3.56s
    2209:	learn: 0.6067304	total: 9.94s	remaining: 3.55s
    2210:	learn: 0.6066976	total: 9.95s	remaining: 3.55s
    2211:	learn: 0.6066783	total: 9.95s	remaining: 3.54s
    2212:	learn: 0.6066486	total: 9.95s	remaining: 3.54s
    2213:	learn: 0.6066131	total: 9.96s	remaining: 3.53s
    2214:	learn: 0.6065921	total: 9.96s	remaining: 3.53s
    2215:	learn: 0.6065448	total: 9.96s	remaining: 3.52s
    2216:	learn: 0.6065247	total: 9.96s	remaining: 3.52s
    2217:	learn: 0.6064993	total: 9.97s	remaining: 3.51s
    2218:	learn: 0.6064627	total: 9.97s	remaining: 3.51s
    2219:	learn: 0.6064169	total: 9.97s	remaining: 3.5s
    2220:	learn: 0.6063717	total: 9.98s	remaining: 3.5s
    2221:	learn: 0.6063429	total: 9.98s	remaining: 3.49s
    2222:	learn: 0.6063187	total: 9.98s	remaining: 3.49s
    2223:	learn: 0.6062958	total: 9.98s	remaining: 3.48s
    2224:	learn: 0.6062577	total: 9.99s	remaining: 3.48s
    2225:	learn: 0.6062123	total: 9.99s	remaining: 3.47s
    2226:	learn: 0.6061941	total: 9.99s	remaining: 3.47s
    2227:	learn: 0.6061581	total: 10s	remaining: 3.47s
    2228:	learn: 0.6061325	total: 10s	remaining: 3.47s
    2229:	learn: 0.6061012	total: 10s	remaining: 3.46s
    2230:	learn: 0.6060600	total: 10s	remaining: 3.46s
    2231:	learn: 0.6060470	total: 10.1s	remaining: 3.46s
    2232:	learn: 0.6060219	total: 10.1s	remaining: 3.46s
    2233:	learn: 0.6059885	total: 10.1s	remaining: 3.45s
    2234:	learn: 0.6059577	total: 10.1s	remaining: 3.45s
    2235:	learn: 0.6059108	total: 10.1s	remaining: 3.44s
    2236:	learn: 0.6058829	total: 10.1s	remaining: 3.44s
    2237:	learn: 0.6058603	total: 10.1s	remaining: 3.44s
    2238:	learn: 0.6058180	total: 10.1s	remaining: 3.43s
    2239:	learn: 0.6057931	total: 10.1s	remaining: 3.43s
    2240:	learn: 0.6057761	total: 10.1s	remaining: 3.43s
    2241:	learn: 0.6057302	total: 10.1s	remaining: 3.42s
    2242:	learn: 0.6056956	total: 10.1s	remaining: 3.42s
    2243:	learn: 0.6056756	total: 10.1s	remaining: 3.41s
    2244:	learn: 0.6056505	total: 10.1s	remaining: 3.41s
    2245:	learn: 0.6056237	total: 10.1s	remaining: 3.4s
    2246:	learn: 0.6055785	total: 10.1s	remaining: 3.4s
    2247:	learn: 0.6055558	total: 10.1s	remaining: 3.39s
    2248:	learn: 0.6055261	total: 10.1s	remaining: 3.38s
    2249:	learn: 0.6054884	total: 10.1s	remaining: 3.38s
    2250:	learn: 0.6054504	total: 10.1s	remaining: 3.38s
    2251:	learn: 0.6054249	total: 10.1s	remaining: 3.37s
    2252:	learn: 0.6054055	total: 10.2s	remaining: 3.37s
    2253:	learn: 0.6053645	total: 10.2s	remaining: 3.36s
    2254:	learn: 0.6053367	total: 10.2s	remaining: 3.35s
    2255:	learn: 0.6052905	total: 10.2s	remaining: 3.35s
    2256:	learn: 0.6052642	total: 10.2s	remaining: 3.35s
    2257:	learn: 0.6052417	total: 10.2s	remaining: 3.34s
    2258:	learn: 0.6051761	total: 10.2s	remaining: 3.34s
    2259:	learn: 0.6051316	total: 10.2s	remaining: 3.33s
    2260:	learn: 0.6051148	total: 10.2s	remaining: 3.33s
    2261:	learn: 0.6050715	total: 10.2s	remaining: 3.32s
    2262:	learn: 0.6050496	total: 10.2s	remaining: 3.32s
    2263:	learn: 0.6049972	total: 10.2s	remaining: 3.31s
    2264:	learn: 0.6049658	total: 10.2s	remaining: 3.31s
    2265:	learn: 0.6049476	total: 10.2s	remaining: 3.3s
    2266:	learn: 0.6049295	total: 10.2s	remaining: 3.3s
    2267:	learn: 0.6049109	total: 10.2s	remaining: 3.29s
    2268:	learn: 0.6048774	total: 10.2s	remaining: 3.29s
    2269:	learn: 0.6048434	total: 10.2s	remaining: 3.28s
    2270:	learn: 0.6048009	total: 10.2s	remaining: 3.28s
    2271:	learn: 0.6047570	total: 10.2s	remaining: 3.27s
    2272:	learn: 0.6047186	total: 10.2s	remaining: 3.27s
    2273:	learn: 0.6046837	total: 10.2s	remaining: 3.26s
    2274:	learn: 0.6046550	total: 10.2s	remaining: 3.25s
    2275:	learn: 0.6046278	total: 10.2s	remaining: 3.25s
    2276:	learn: 0.6045863	total: 10.2s	remaining: 3.25s
    2277:	learn: 0.6045608	total: 10.2s	remaining: 3.24s
    2278:	learn: 0.6045308	total: 10.2s	remaining: 3.23s
    2279:	learn: 0.6045001	total: 10.2s	remaining: 3.23s
    2280:	learn: 0.6044472	total: 10.2s	remaining: 3.23s
    2281:	learn: 0.6044125	total: 10.2s	remaining: 3.22s
    2282:	learn: 0.6043883	total: 10.2s	remaining: 3.21s
    2283:	learn: 0.6043614	total: 10.2s	remaining: 3.21s
    2284:	learn: 0.6043312	total: 10.2s	remaining: 3.21s
    2285:	learn: 0.6043095	total: 10.2s	remaining: 3.2s
    2286:	learn: 0.6042672	total: 10.2s	remaining: 3.19s
    2287:	learn: 0.6042389	total: 10.3s	remaining: 3.19s
    2288:	learn: 0.6042084	total: 10.3s	remaining: 3.19s
    2289:	learn: 0.6041774	total: 10.3s	remaining: 3.18s
    2290:	learn: 0.6041350	total: 10.3s	remaining: 3.17s
    2291:	learn: 0.6040917	total: 10.3s	remaining: 3.17s
    2292:	learn: 0.6040787	total: 10.3s	remaining: 3.17s
    2293:	learn: 0.6040504	total: 10.3s	remaining: 3.16s
    2294:	learn: 0.6040010	total: 10.3s	remaining: 3.15s
    2295:	learn: 0.6039704	total: 10.3s	remaining: 3.15s
    2296:	learn: 0.6039412	total: 10.3s	remaining: 3.15s
    2297:	learn: 0.6039189	total: 10.3s	remaining: 3.14s
    2298:	learn: 0.6038922	total: 10.3s	remaining: 3.14s
    2299:	learn: 0.6038656	total: 10.3s	remaining: 3.14s
    2300:	learn: 0.6038410	total: 10.3s	remaining: 3.13s
    2301:	learn: 0.6037891	total: 10.3s	remaining: 3.13s
    2302:	learn: 0.6037669	total: 10.3s	remaining: 3.12s
    2303:	learn: 0.6037223	total: 10.3s	remaining: 3.12s
    2304:	learn: 0.6037025	total: 10.3s	remaining: 3.11s
    2305:	learn: 0.6036735	total: 10.3s	remaining: 3.11s
    2306:	learn: 0.6036299	total: 10.3s	remaining: 3.1s
    2307:	learn: 0.6036112	total: 10.3s	remaining: 3.1s
    2308:	learn: 0.6035796	total: 10.4s	remaining: 3.1s
    2309:	learn: 0.6035300	total: 10.4s	remaining: 3.09s
    2310:	learn: 0.6034928	total: 10.4s	remaining: 3.09s
    2311:	learn: 0.6034527	total: 10.4s	remaining: 3.09s
    2312:	learn: 0.6034123	total: 10.4s	remaining: 3.08s
    2313:	learn: 0.6033862	total: 10.4s	remaining: 3.08s
    2314:	learn: 0.6033589	total: 10.4s	remaining: 3.08s
    2315:	learn: 0.6033210	total: 10.4s	remaining: 3.07s
    2316:	learn: 0.6033029	total: 10.4s	remaining: 3.07s
    2317:	learn: 0.6032653	total: 10.4s	remaining: 3.06s
    2318:	learn: 0.6032502	total: 10.4s	remaining: 3.06s
    2319:	learn: 0.6032175	total: 10.4s	remaining: 3.06s
    2320:	learn: 0.6031596	total: 10.4s	remaining: 3.05s
    2321:	learn: 0.6031187	total: 10.4s	remaining: 3.05s
    2322:	learn: 0.6030882	total: 10.5s	remaining: 3.05s
    2323:	learn: 0.6030571	total: 10.5s	remaining: 3.04s
    2324:	learn: 0.6030188	total: 10.5s	remaining: 3.04s
    2325:	learn: 0.6029972	total: 10.5s	remaining: 3.03s
    2326:	learn: 0.6029696	total: 10.5s	remaining: 3.03s
    2327:	learn: 0.6029374	total: 10.5s	remaining: 3.02s
    2328:	learn: 0.6029036	total: 10.5s	remaining: 3.02s
    2329:	learn: 0.6028761	total: 10.5s	remaining: 3.01s
    2330:	learn: 0.6028450	total: 10.5s	remaining: 3.01s
    2331:	learn: 0.6028147	total: 10.5s	remaining: 3.01s
    2332:	learn: 0.6027761	total: 10.5s	remaining: 3s
    2333:	learn: 0.6027504	total: 10.5s	remaining: 3s
    2334:	learn: 0.6026933	total: 10.5s	remaining: 2.99s
    2335:	learn: 0.6026646	total: 10.5s	remaining: 2.99s
    2336:	learn: 0.6026376	total: 10.5s	remaining: 2.98s
    2337:	learn: 0.6026187	total: 10.5s	remaining: 2.98s
    2338:	learn: 0.6026045	total: 10.5s	remaining: 2.97s
    2339:	learn: 0.6025847	total: 10.5s	remaining: 2.97s
    2340:	learn: 0.6025563	total: 10.5s	remaining: 2.96s
    2341:	learn: 0.6025334	total: 10.5s	remaining: 2.96s
    2342:	learn: 0.6024770	total: 10.5s	remaining: 2.95s
    2343:	learn: 0.6024482	total: 10.5s	remaining: 2.95s
    2344:	learn: 0.6024064	total: 10.5s	remaining: 2.94s
    2345:	learn: 0.6023872	total: 10.6s	remaining: 2.94s
    2346:	learn: 0.6023607	total: 10.6s	remaining: 2.94s
    2347:	learn: 0.6023120	total: 10.6s	remaining: 2.94s
    2348:	learn: 0.6022829	total: 10.6s	remaining: 2.94s
    2349:	learn: 0.6022522	total: 10.6s	remaining: 2.93s
    2350:	learn: 0.6022258	total: 10.6s	remaining: 2.93s
    2351:	learn: 0.6022052	total: 10.6s	remaining: 2.93s
    2352:	learn: 0.6021753	total: 10.7s	remaining: 2.93s
    2353:	learn: 0.6021591	total: 10.7s	remaining: 2.93s
    2354:	learn: 0.6021374	total: 10.7s	remaining: 2.93s
    2355:	learn: 0.6020913	total: 10.7s	remaining: 2.92s
    2356:	learn: 0.6020441	total: 10.7s	remaining: 2.92s
    2357:	learn: 0.6019923	total: 10.7s	remaining: 2.92s
    2358:	learn: 0.6019605	total: 10.8s	remaining: 2.92s
    2359:	learn: 0.6019456	total: 10.8s	remaining: 2.92s
    2360:	learn: 0.6019261	total: 10.8s	remaining: 2.91s
    2361:	learn: 0.6019018	total: 10.8s	remaining: 2.91s
    2362:	learn: 0.6018745	total: 10.8s	remaining: 2.9s
    2363:	learn: 0.6018405	total: 10.8s	remaining: 2.9s
    2364:	learn: 0.6018254	total: 10.8s	remaining: 2.89s
    2365:	learn: 0.6018030	total: 10.8s	remaining: 2.89s
    2366:	learn: 0.6017819	total: 10.8s	remaining: 2.88s
    2367:	learn: 0.6017355	total: 10.8s	remaining: 2.88s
    2368:	learn: 0.6017154	total: 10.8s	remaining: 2.87s
    2369:	learn: 0.6016753	total: 10.8s	remaining: 2.87s
    2370:	learn: 0.6016490	total: 10.8s	remaining: 2.86s
    2371:	learn: 0.6016136	total: 10.8s	remaining: 2.86s
    2372:	learn: 0.6015910	total: 10.8s	remaining: 2.85s
    2373:	learn: 0.6015610	total: 10.8s	remaining: 2.85s
    2374:	learn: 0.6015259	total: 10.8s	remaining: 2.84s
    2375:	learn: 0.6015041	total: 10.8s	remaining: 2.84s
    2376:	learn: 0.6014435	total: 10.8s	remaining: 2.83s
    2377:	learn: 0.6014218	total: 10.8s	remaining: 2.83s
    2378:	learn: 0.6013882	total: 10.8s	remaining: 2.82s
    2379:	learn: 0.6013574	total: 10.8s	remaining: 2.82s
    2380:	learn: 0.6013300	total: 10.8s	remaining: 2.81s
    2381:	learn: 0.6013016	total: 10.8s	remaining: 2.81s
    2382:	learn: 0.6012612	total: 10.8s	remaining: 2.8s
    2383:	learn: 0.6012383	total: 10.8s	remaining: 2.8s
    2384:	learn: 0.6012154	total: 10.8s	remaining: 2.79s
    2385:	learn: 0.6011837	total: 10.8s	remaining: 2.79s
    2386:	learn: 0.6011583	total: 10.8s	remaining: 2.78s
    2387:	learn: 0.6011328	total: 10.8s	remaining: 2.78s
    2388:	learn: 0.6011096	total: 10.8s	remaining: 2.77s
    2389:	learn: 0.6010808	total: 10.9s	remaining: 2.77s
    2390:	learn: 0.6010460	total: 10.9s	remaining: 2.76s
    2391:	learn: 0.6009950	total: 10.9s	remaining: 2.76s
    2392:	learn: 0.6009651	total: 10.9s	remaining: 2.75s
    2393:	learn: 0.6009349	total: 10.9s	remaining: 2.75s
    2394:	learn: 0.6009103	total: 10.9s	remaining: 2.75s
    2395:	learn: 0.6008913	total: 10.9s	remaining: 2.74s
    2396:	learn: 0.6008618	total: 10.9s	remaining: 2.74s
    2397:	learn: 0.6008373	total: 10.9s	remaining: 2.73s
    2398:	learn: 0.6008132	total: 10.9s	remaining: 2.73s
    2399:	learn: 0.6007784	total: 10.9s	remaining: 2.72s
    2400:	learn: 0.6007625	total: 10.9s	remaining: 2.72s
    2401:	learn: 0.6007174	total: 10.9s	remaining: 2.71s
    2402:	learn: 0.6006843	total: 10.9s	remaining: 2.71s
    2403:	learn: 0.6006435	total: 10.9s	remaining: 2.71s
    2404:	learn: 0.6005952	total: 10.9s	remaining: 2.7s
    2405:	learn: 0.6005527	total: 10.9s	remaining: 2.7s
    2406:	learn: 0.6005309	total: 10.9s	remaining: 2.69s
    2407:	learn: 0.6005035	total: 10.9s	remaining: 2.69s
    2408:	learn: 0.6004515	total: 10.9s	remaining: 2.68s
    2409:	learn: 0.6004127	total: 10.9s	remaining: 2.68s
    2410:	learn: 0.6003859	total: 10.9s	remaining: 2.67s
    2411:	learn: 0.6003515	total: 10.9s	remaining: 2.67s
    2412:	learn: 0.6003077	total: 10.9s	remaining: 2.66s
    2413:	learn: 0.6002641	total: 10.9s	remaining: 2.66s
    2414:	learn: 0.6002399	total: 10.9s	remaining: 2.65s
    2415:	learn: 0.6002072	total: 11s	remaining: 2.65s
    2416:	learn: 0.6001826	total: 11s	remaining: 2.64s
    2417:	learn: 0.6001702	total: 11s	remaining: 2.64s
    2418:	learn: 0.6001600	total: 11s	remaining: 2.63s
    2419:	learn: 0.6001299	total: 11s	remaining: 2.63s
    2420:	learn: 0.6001014	total: 11s	remaining: 2.63s
    2421:	learn: 0.6000722	total: 11s	remaining: 2.62s
    2422:	learn: 0.6000524	total: 11s	remaining: 2.62s
    2423:	learn: 0.6000255	total: 11s	remaining: 2.62s
    2424:	learn: 0.6000044	total: 11s	remaining: 2.61s
    2425:	learn: 0.5999682	total: 11s	remaining: 2.61s
    2426:	learn: 0.5999375	total: 11s	remaining: 2.6s
    2427:	learn: 0.5999033	total: 11s	remaining: 2.6s
    2428:	learn: 0.5998741	total: 11s	remaining: 2.6s
    2429:	learn: 0.5998494	total: 11.1s	remaining: 2.59s
    2430:	learn: 0.5998225	total: 11.1s	remaining: 2.59s
    2431:	learn: 0.5998094	total: 11.1s	remaining: 2.59s
    2432:	learn: 0.5997544	total: 11.1s	remaining: 2.58s
    2433:	learn: 0.5996962	total: 11.1s	remaining: 2.58s
    2434:	learn: 0.5996661	total: 11.1s	remaining: 2.57s
    2435:	learn: 0.5996296	total: 11.1s	remaining: 2.57s
    2436:	learn: 0.5995960	total: 11.1s	remaining: 2.56s
    2437:	learn: 0.5995719	total: 11.1s	remaining: 2.56s
    2438:	learn: 0.5995447	total: 11.1s	remaining: 2.55s
    2439:	learn: 0.5995289	total: 11.1s	remaining: 2.55s
    2440:	learn: 0.5994972	total: 11.1s	remaining: 2.54s
    2441:	learn: 0.5994724	total: 11.1s	remaining: 2.54s
    2442:	learn: 0.5994483	total: 11.1s	remaining: 2.53s
    2443:	learn: 0.5994108	total: 11.1s	remaining: 2.53s
    2444:	learn: 0.5993823	total: 11.1s	remaining: 2.52s
    2445:	learn: 0.5993602	total: 11.1s	remaining: 2.52s
    2446:	learn: 0.5993326	total: 11.1s	remaining: 2.52s
    2447:	learn: 0.5993030	total: 11.1s	remaining: 2.51s
    2448:	learn: 0.5992868	total: 11.1s	remaining: 2.51s
    2449:	learn: 0.5992739	total: 11.1s	remaining: 2.5s
    2450:	learn: 0.5992465	total: 11.1s	remaining: 2.5s
    2451:	learn: 0.5992120	total: 11.2s	remaining: 2.49s
    2452:	learn: 0.5991780	total: 11.2s	remaining: 2.49s
    2453:	learn: 0.5991331	total: 11.2s	remaining: 2.48s
    2454:	learn: 0.5991153	total: 11.2s	remaining: 2.48s
    2455:	learn: 0.5990902	total: 11.2s	remaining: 2.47s
    2456:	learn: 0.5990506	total: 11.2s	remaining: 2.47s
    2457:	learn: 0.5990255	total: 11.2s	remaining: 2.46s
    2458:	learn: 0.5989986	total: 11.2s	remaining: 2.46s
    2459:	learn: 0.5989690	total: 11.2s	remaining: 2.45s
    2460:	learn: 0.5989435	total: 11.2s	remaining: 2.45s
    2461:	learn: 0.5989249	total: 11.2s	remaining: 2.44s
    2462:	learn: 0.5988955	total: 11.2s	remaining: 2.44s
    2463:	learn: 0.5988727	total: 11.2s	remaining: 2.43s
    2464:	learn: 0.5988484	total: 11.2s	remaining: 2.43s
    2465:	learn: 0.5988226	total: 11.2s	remaining: 2.42s
    2466:	learn: 0.5987932	total: 11.2s	remaining: 2.42s
    2467:	learn: 0.5987685	total: 11.2s	remaining: 2.41s
    2468:	learn: 0.5987319	total: 11.2s	remaining: 2.41s
    2469:	learn: 0.5987032	total: 11.2s	remaining: 2.4s
    2470:	learn: 0.5986826	total: 11.2s	remaining: 2.4s
    2471:	learn: 0.5986612	total: 11.2s	remaining: 2.39s
    2472:	learn: 0.5986376	total: 11.2s	remaining: 2.39s
    2473:	learn: 0.5986001	total: 11.2s	remaining: 2.38s
    2474:	learn: 0.5985694	total: 11.2s	remaining: 2.38s
    2475:	learn: 0.5985478	total: 11.2s	remaining: 2.37s
    2476:	learn: 0.5985183	total: 11.2s	remaining: 2.37s
    2477:	learn: 0.5984690	total: 11.2s	remaining: 2.36s
    2478:	learn: 0.5984274	total: 11.2s	remaining: 2.36s
    2479:	learn: 0.5984074	total: 11.2s	remaining: 2.35s
    2480:	learn: 0.5983731	total: 11.2s	remaining: 2.35s
    2481:	learn: 0.5983499	total: 11.2s	remaining: 2.34s
    2482:	learn: 0.5983037	total: 11.2s	remaining: 2.34s
    2483:	learn: 0.5982643	total: 11.2s	remaining: 2.33s
    2484:	learn: 0.5982364	total: 11.2s	remaining: 2.33s
    2485:	learn: 0.5982087	total: 11.2s	remaining: 2.33s
    2486:	learn: 0.5981635	total: 11.3s	remaining: 2.32s
    2487:	learn: 0.5981360	total: 11.3s	remaining: 2.31s
    2488:	learn: 0.5981076	total: 11.3s	remaining: 2.31s
    2489:	learn: 0.5980703	total: 11.3s	remaining: 2.31s
    2490:	learn: 0.5980449	total: 11.3s	remaining: 2.3s
    2491:	learn: 0.5980275	total: 11.3s	remaining: 2.3s
    2492:	learn: 0.5980097	total: 11.3s	remaining: 2.29s
    2493:	learn: 0.5979836	total: 11.3s	remaining: 2.29s
    2494:	learn: 0.5979595	total: 11.3s	remaining: 2.28s
    2495:	learn: 0.5979142	total: 11.3s	remaining: 2.28s
    2496:	learn: 0.5978841	total: 11.3s	remaining: 2.27s
    2497:	learn: 0.5978657	total: 11.3s	remaining: 2.27s
    2498:	learn: 0.5978219	total: 11.3s	remaining: 2.26s
    2499:	learn: 0.5978003	total: 11.3s	remaining: 2.26s
    2500:	learn: 0.5977846	total: 11.3s	remaining: 2.26s
    2501:	learn: 0.5977413	total: 11.3s	remaining: 2.25s
    2502:	learn: 0.5977157	total: 11.3s	remaining: 2.25s
    2503:	learn: 0.5976983	total: 11.3s	remaining: 2.24s
    2504:	learn: 0.5976709	total: 11.3s	remaining: 2.24s
    2505:	learn: 0.5976474	total: 11.4s	remaining: 2.24s
    2506:	learn: 0.5976017	total: 11.4s	remaining: 2.23s
    2507:	learn: 0.5975609	total: 11.4s	remaining: 2.23s
    2508:	learn: 0.5975283	total: 11.4s	remaining: 2.23s
    2509:	learn: 0.5974959	total: 11.4s	remaining: 2.22s
    2510:	learn: 0.5974620	total: 11.4s	remaining: 2.22s
    2511:	learn: 0.5974302	total: 11.4s	remaining: 2.21s
    2512:	learn: 0.5974055	total: 11.4s	remaining: 2.21s
    2513:	learn: 0.5973913	total: 11.4s	remaining: 2.2s
    2514:	learn: 0.5973477	total: 11.4s	remaining: 2.2s
    2515:	learn: 0.5973285	total: 11.4s	remaining: 2.19s
    2516:	learn: 0.5973000	total: 11.4s	remaining: 2.19s
    2517:	learn: 0.5972734	total: 11.4s	remaining: 2.18s
    2518:	learn: 0.5972542	total: 11.4s	remaining: 2.18s
    2519:	learn: 0.5972212	total: 11.4s	remaining: 2.17s
    2520:	learn: 0.5971869	total: 11.4s	remaining: 2.17s
    2521:	learn: 0.5971575	total: 11.4s	remaining: 2.17s
    2522:	learn: 0.5971414	total: 11.4s	remaining: 2.16s
    2523:	learn: 0.5970862	total: 11.4s	remaining: 2.15s
    2524:	learn: 0.5970641	total: 11.4s	remaining: 2.15s
    2525:	learn: 0.5970401	total: 11.4s	remaining: 2.15s
    2526:	learn: 0.5970147	total: 11.4s	remaining: 2.14s
    2527:	learn: 0.5970005	total: 11.4s	remaining: 2.14s
    2528:	learn: 0.5969729	total: 11.4s	remaining: 2.13s
    2529:	learn: 0.5969401	total: 11.4s	remaining: 2.13s
    2530:	learn: 0.5969033	total: 11.4s	remaining: 2.12s
    2531:	learn: 0.5968787	total: 11.5s	remaining: 2.12s
    2532:	learn: 0.5968391	total: 11.5s	remaining: 2.11s
    2533:	learn: 0.5968164	total: 11.5s	remaining: 2.11s
    2534:	learn: 0.5967862	total: 11.5s	remaining: 2.1s
    2535:	learn: 0.5967613	total: 11.5s	remaining: 2.1s
    2536:	learn: 0.5967288	total: 11.5s	remaining: 2.09s
    2537:	learn: 0.5966946	total: 11.5s	remaining: 2.09s
    2538:	learn: 0.5966509	total: 11.5s	remaining: 2.08s
    2539:	learn: 0.5966169	total: 11.5s	remaining: 2.08s
    2540:	learn: 0.5965853	total: 11.5s	remaining: 2.07s
    2541:	learn: 0.5965620	total: 11.5s	remaining: 2.07s
    2542:	learn: 0.5965275	total: 11.5s	remaining: 2.06s
    2543:	learn: 0.5965025	total: 11.5s	remaining: 2.06s
    2544:	learn: 0.5964788	total: 11.5s	remaining: 2.06s
    2545:	learn: 0.5964388	total: 11.5s	remaining: 2.05s
    2546:	learn: 0.5964067	total: 11.5s	remaining: 2.05s
    2547:	learn: 0.5963849	total: 11.5s	remaining: 2.04s
    2548:	learn: 0.5963508	total: 11.5s	remaining: 2.04s
    2549:	learn: 0.5963297	total: 11.5s	remaining: 2.03s
    2550:	learn: 0.5962901	total: 11.5s	remaining: 2.03s
    2551:	learn: 0.5962682	total: 11.5s	remaining: 2.02s
    2552:	learn: 0.5962494	total: 11.5s	remaining: 2.02s
    2553:	learn: 0.5962296	total: 11.5s	remaining: 2.01s
    2554:	learn: 0.5961918	total: 11.5s	remaining: 2.01s
    2555:	learn: 0.5961748	total: 11.5s	remaining: 2s
    2556:	learn: 0.5961574	total: 11.5s	remaining: 2s
    2557:	learn: 0.5961333	total: 11.5s	remaining: 1.99s
    2558:	learn: 0.5960979	total: 11.5s	remaining: 1.99s
    2559:	learn: 0.5960815	total: 11.5s	remaining: 1.99s
    2560:	learn: 0.5960449	total: 11.6s	remaining: 1.98s
    2561:	learn: 0.5960165	total: 11.6s	remaining: 1.98s
    2562:	learn: 0.5959842	total: 11.6s	remaining: 1.97s
    2563:	learn: 0.5959567	total: 11.6s	remaining: 1.97s
    2564:	learn: 0.5959343	total: 11.6s	remaining: 1.96s
    2565:	learn: 0.5959109	total: 11.6s	remaining: 1.96s
    2566:	learn: 0.5958865	total: 11.6s	remaining: 1.95s
    2567:	learn: 0.5958668	total: 11.6s	remaining: 1.95s
    2568:	learn: 0.5958417	total: 11.6s	remaining: 1.94s
    2569:	learn: 0.5958076	total: 11.6s	remaining: 1.94s
    2570:	learn: 0.5957790	total: 11.6s	remaining: 1.93s
    2571:	learn: 0.5957546	total: 11.6s	remaining: 1.93s
    2572:	learn: 0.5957197	total: 11.6s	remaining: 1.92s
    2573:	learn: 0.5956892	total: 11.6s	remaining: 1.92s
    2574:	learn: 0.5956614	total: 11.6s	remaining: 1.91s
    2575:	learn: 0.5956371	total: 11.6s	remaining: 1.91s
    2576:	learn: 0.5956018	total: 11.6s	remaining: 1.9s
    2577:	learn: 0.5955671	total: 11.6s	remaining: 1.9s
    2578:	learn: 0.5955378	total: 11.6s	remaining: 1.89s
    2579:	learn: 0.5955103	total: 11.6s	remaining: 1.89s
    2580:	learn: 0.5954874	total: 11.6s	remaining: 1.88s
    2581:	learn: 0.5954564	total: 11.6s	remaining: 1.88s
    2582:	learn: 0.5954340	total: 11.6s	remaining: 1.88s
    2583:	learn: 0.5954150	total: 11.6s	remaining: 1.87s
    2584:	learn: 0.5954016	total: 11.6s	remaining: 1.86s
    2585:	learn: 0.5953677	total: 11.6s	remaining: 1.86s
    2586:	learn: 0.5953439	total: 11.6s	remaining: 1.86s
    2587:	learn: 0.5953270	total: 11.6s	remaining: 1.85s
    2588:	learn: 0.5953088	total: 11.6s	remaining: 1.85s
    2589:	learn: 0.5952784	total: 11.6s	remaining: 1.84s
    2590:	learn: 0.5952620	total: 11.6s	remaining: 1.84s
    2591:	learn: 0.5952305	total: 11.6s	remaining: 1.83s
    2592:	learn: 0.5952116	total: 11.6s	remaining: 1.83s
    2593:	learn: 0.5951884	total: 11.7s	remaining: 1.82s
    2594:	learn: 0.5951749	total: 11.7s	remaining: 1.82s
    2595:	learn: 0.5951405	total: 11.7s	remaining: 1.81s
    2596:	learn: 0.5951084	total: 11.7s	remaining: 1.81s
    2597:	learn: 0.5950697	total: 11.7s	remaining: 1.8s
    2598:	learn: 0.5950499	total: 11.7s	remaining: 1.8s
    2599:	learn: 0.5950131	total: 11.7s	remaining: 1.79s
    2600:	learn: 0.5949973	total: 11.7s	remaining: 1.79s
    2601:	learn: 0.5949749	total: 11.7s	remaining: 1.79s
    2602:	learn: 0.5949402	total: 11.7s	remaining: 1.78s
    2603:	learn: 0.5949072	total: 11.7s	remaining: 1.78s
    2604:	learn: 0.5948797	total: 11.7s	remaining: 1.77s
    2605:	learn: 0.5948564	total: 11.7s	remaining: 1.77s
    2606:	learn: 0.5948188	total: 11.7s	remaining: 1.76s
    2607:	learn: 0.5947924	total: 11.7s	remaining: 1.76s
    2608:	learn: 0.5947504	total: 11.7s	remaining: 1.75s
    2609:	learn: 0.5947288	total: 11.7s	remaining: 1.75s
    2610:	learn: 0.5947030	total: 11.7s	remaining: 1.75s
    2611:	learn: 0.5946733	total: 11.7s	remaining: 1.74s
    2612:	learn: 0.5946357	total: 11.7s	remaining: 1.74s
    2613:	learn: 0.5946226	total: 11.7s	remaining: 1.73s
    2614:	learn: 0.5945872	total: 11.7s	remaining: 1.73s
    2615:	learn: 0.5945467	total: 11.7s	remaining: 1.72s
    2616:	learn: 0.5945133	total: 11.7s	remaining: 1.72s
    2617:	learn: 0.5944945	total: 11.8s	remaining: 1.71s
    2618:	learn: 0.5944627	total: 11.8s	remaining: 1.71s
    2619:	learn: 0.5944410	total: 11.8s	remaining: 1.71s
    2620:	learn: 0.5944155	total: 11.8s	remaining: 1.7s
    2621:	learn: 0.5943848	total: 11.8s	remaining: 1.7s
    2622:	learn: 0.5943760	total: 11.8s	remaining: 1.69s
    2623:	learn: 0.5943554	total: 11.8s	remaining: 1.69s
    2624:	learn: 0.5943197	total: 11.8s	remaining: 1.68s
    2625:	learn: 0.5942865	total: 11.8s	remaining: 1.68s
    2626:	learn: 0.5942545	total: 11.8s	remaining: 1.67s
    2627:	learn: 0.5942241	total: 11.8s	remaining: 1.67s
    2628:	learn: 0.5942008	total: 11.8s	remaining: 1.66s
    2629:	learn: 0.5941688	total: 11.8s	remaining: 1.66s
    2630:	learn: 0.5941533	total: 11.8s	remaining: 1.65s
    2631:	learn: 0.5941179	total: 11.8s	remaining: 1.65s
    2632:	learn: 0.5940971	total: 11.8s	remaining: 1.65s
    2633:	learn: 0.5940746	total: 11.8s	remaining: 1.64s
    2634:	learn: 0.5940487	total: 11.8s	remaining: 1.64s
    2635:	learn: 0.5940257	total: 11.8s	remaining: 1.63s
    2636:	learn: 0.5939908	total: 11.8s	remaining: 1.63s
    2637:	learn: 0.5939583	total: 11.8s	remaining: 1.62s
    2638:	learn: 0.5939275	total: 11.8s	remaining: 1.62s
    2639:	learn: 0.5938902	total: 11.8s	remaining: 1.61s
    2640:	learn: 0.5938635	total: 11.8s	remaining: 1.61s
    2641:	learn: 0.5938445	total: 11.8s	remaining: 1.6s
    2642:	learn: 0.5938192	total: 11.8s	remaining: 1.6s
    2643:	learn: 0.5937949	total: 11.9s	remaining: 1.59s
    2644:	learn: 0.5937705	total: 11.9s	remaining: 1.59s
    2645:	learn: 0.5937534	total: 11.9s	remaining: 1.59s
    2646:	learn: 0.5937400	total: 11.9s	remaining: 1.58s
    2647:	learn: 0.5936957	total: 11.9s	remaining: 1.58s
    2648:	learn: 0.5936470	total: 11.9s	remaining: 1.57s
    2649:	learn: 0.5936241	total: 11.9s	remaining: 1.57s
    2650:	learn: 0.5935846	total: 11.9s	remaining: 1.56s
    2651:	learn: 0.5935605	total: 11.9s	remaining: 1.56s
    2652:	learn: 0.5935384	total: 11.9s	remaining: 1.55s
    2653:	learn: 0.5935176	total: 11.9s	remaining: 1.55s
    2654:	learn: 0.5934990	total: 11.9s	remaining: 1.55s
    2655:	learn: 0.5934872	total: 11.9s	remaining: 1.54s
    2656:	learn: 0.5934726	total: 11.9s	remaining: 1.54s
    2657:	learn: 0.5934493	total: 11.9s	remaining: 1.53s
    2658:	learn: 0.5934198	total: 11.9s	remaining: 1.53s
    2659:	learn: 0.5933869	total: 11.9s	remaining: 1.52s
    2660:	learn: 0.5933594	total: 11.9s	remaining: 1.52s
    2661:	learn: 0.5933179	total: 12s	remaining: 1.52s
    2662:	learn: 0.5932811	total: 12s	remaining: 1.51s
    2663:	learn: 0.5932474	total: 12s	remaining: 1.51s
    2664:	learn: 0.5932161	total: 12s	remaining: 1.51s
    2665:	learn: 0.5931770	total: 12s	remaining: 1.5s
    2666:	learn: 0.5931549	total: 12s	remaining: 1.5s
    2667:	learn: 0.5931335	total: 12s	remaining: 1.5s
    2668:	learn: 0.5931078	total: 12s	remaining: 1.49s
    2669:	learn: 0.5930772	total: 12.1s	remaining: 1.49s
    2670:	learn: 0.5930500	total: 12.1s	remaining: 1.49s
    2671:	learn: 0.5930304	total: 12.1s	remaining: 1.48s
    2672:	learn: 0.5930013	total: 12.1s	remaining: 1.48s
    2673:	learn: 0.5929808	total: 12.1s	remaining: 1.48s
    2674:	learn: 0.5929690	total: 12.1s	remaining: 1.47s
    2675:	learn: 0.5929369	total: 12.1s	remaining: 1.47s
    2676:	learn: 0.5929117	total: 12.1s	remaining: 1.46s
    2677:	learn: 0.5928954	total: 12.1s	remaining: 1.46s
    2678:	learn: 0.5928677	total: 12.1s	remaining: 1.46s
    2679:	learn: 0.5928472	total: 12.2s	remaining: 1.45s
    2680:	learn: 0.5928143	total: 12.2s	remaining: 1.45s
    2681:	learn: 0.5927836	total: 12.2s	remaining: 1.44s
    2682:	learn: 0.5927433	total: 12.2s	remaining: 1.44s
    2683:	learn: 0.5927203	total: 12.2s	remaining: 1.43s
    2684:	learn: 0.5926892	total: 12.2s	remaining: 1.43s
    2685:	learn: 0.5926505	total: 12.2s	remaining: 1.43s
    2686:	learn: 0.5926180	total: 12.2s	remaining: 1.42s
    2687:	learn: 0.5925971	total: 12.2s	remaining: 1.42s
    2688:	learn: 0.5925687	total: 12.2s	remaining: 1.41s
    2689:	learn: 0.5925459	total: 12.2s	remaining: 1.41s
    2690:	learn: 0.5925169	total: 12.2s	remaining: 1.4s
    2691:	learn: 0.5924743	total: 12.2s	remaining: 1.4s
    2692:	learn: 0.5924376	total: 12.2s	remaining: 1.4s
    2693:	learn: 0.5924048	total: 12.2s	remaining: 1.39s
    2694:	learn: 0.5923899	total: 12.3s	remaining: 1.39s
    2695:	learn: 0.5923700	total: 12.3s	remaining: 1.38s
    2696:	learn: 0.5923533	total: 12.3s	remaining: 1.38s
    2697:	learn: 0.5923219	total: 12.3s	remaining: 1.37s
    2698:	learn: 0.5923070	total: 12.3s	remaining: 1.37s
    2699:	learn: 0.5922875	total: 12.3s	remaining: 1.37s
    2700:	learn: 0.5922596	total: 12.3s	remaining: 1.36s
    2701:	learn: 0.5922200	total: 12.3s	remaining: 1.36s
    2702:	learn: 0.5921870	total: 12.3s	remaining: 1.35s
    2703:	learn: 0.5921617	total: 12.3s	remaining: 1.35s
    2704:	learn: 0.5921303	total: 12.3s	remaining: 1.34s
    2705:	learn: 0.5921008	total: 12.3s	remaining: 1.34s
    2706:	learn: 0.5920808	total: 12.3s	remaining: 1.34s
    2707:	learn: 0.5920525	total: 12.4s	remaining: 1.33s
    2708:	learn: 0.5920313	total: 12.4s	remaining: 1.33s
    2709:	learn: 0.5919954	total: 12.4s	remaining: 1.32s
    2710:	learn: 0.5919792	total: 12.4s	remaining: 1.32s
    2711:	learn: 0.5919530	total: 12.4s	remaining: 1.31s
    2712:	learn: 0.5919090	total: 12.4s	remaining: 1.31s
    2713:	learn: 0.5918876	total: 12.4s	remaining: 1.31s
    2714:	learn: 0.5918587	total: 12.4s	remaining: 1.3s
    2715:	learn: 0.5918341	total: 12.4s	remaining: 1.3s
    2716:	learn: 0.5918018	total: 12.4s	remaining: 1.29s
    2717:	learn: 0.5917764	total: 12.4s	remaining: 1.29s
    2718:	learn: 0.5917428	total: 12.4s	remaining: 1.28s
    2719:	learn: 0.5917143	total: 12.4s	remaining: 1.28s
    2720:	learn: 0.5916970	total: 12.4s	remaining: 1.27s
    2721:	learn: 0.5916519	total: 12.4s	remaining: 1.27s
    2722:	learn: 0.5916295	total: 12.4s	remaining: 1.26s
    2723:	learn: 0.5916103	total: 12.4s	remaining: 1.26s
    2724:	learn: 0.5915858	total: 12.4s	remaining: 1.25s
    2725:	learn: 0.5915653	total: 12.4s	remaining: 1.25s
    2726:	learn: 0.5915350	total: 12.4s	remaining: 1.25s
    2727:	learn: 0.5915097	total: 12.4s	remaining: 1.24s
    2728:	learn: 0.5914961	total: 12.4s	remaining: 1.24s
    2729:	learn: 0.5914843	total: 12.4s	remaining: 1.23s
    2730:	learn: 0.5914365	total: 12.5s	remaining: 1.23s
    2731:	learn: 0.5913995	total: 12.5s	remaining: 1.22s
    2732:	learn: 0.5913706	total: 12.5s	remaining: 1.22s
    2733:	learn: 0.5913435	total: 12.5s	remaining: 1.21s
    2734:	learn: 0.5913124	total: 12.5s	remaining: 1.21s
    2735:	learn: 0.5912892	total: 12.5s	remaining: 1.2s
    2736:	learn: 0.5912606	total: 12.5s	remaining: 1.2s
    2737:	learn: 0.5912383	total: 12.5s	remaining: 1.19s
    2738:	learn: 0.5912185	total: 12.5s	remaining: 1.19s
    2739:	learn: 0.5911819	total: 12.5s	remaining: 1.18s
    2740:	learn: 0.5911546	total: 12.5s	remaining: 1.18s
    2741:	learn: 0.5911365	total: 12.5s	remaining: 1.18s
    2742:	learn: 0.5911219	total: 12.5s	remaining: 1.17s
    2743:	learn: 0.5910927	total: 12.5s	remaining: 1.17s
    2744:	learn: 0.5910540	total: 12.5s	remaining: 1.16s
    2745:	learn: 0.5910304	total: 12.5s	remaining: 1.16s
    2746:	learn: 0.5910076	total: 12.5s	remaining: 1.15s
    2747:	learn: 0.5909856	total: 12.5s	remaining: 1.15s
    2748:	learn: 0.5909509	total: 12.5s	remaining: 1.14s
    2749:	learn: 0.5909326	total: 12.5s	remaining: 1.14s
    2750:	learn: 0.5909040	total: 12.5s	remaining: 1.13s
    2751:	learn: 0.5908612	total: 12.5s	remaining: 1.13s
    2752:	learn: 0.5908467	total: 12.5s	remaining: 1.13s
    2753:	learn: 0.5908204	total: 12.5s	remaining: 1.12s
    2754:	learn: 0.5907852	total: 12.5s	remaining: 1.11s
    2755:	learn: 0.5907728	total: 12.5s	remaining: 1.11s
    2756:	learn: 0.5907430	total: 12.6s	remaining: 1.11s
    2757:	learn: 0.5907157	total: 12.6s	remaining: 1.1s
    2758:	learn: 0.5906802	total: 12.6s	remaining: 1.1s
    2759:	learn: 0.5906585	total: 12.6s	remaining: 1.09s
    2760:	learn: 0.5906382	total: 12.6s	remaining: 1.09s
    2761:	learn: 0.5906043	total: 12.6s	remaining: 1.08s
    2762:	learn: 0.5905900	total: 12.6s	remaining: 1.08s
    2763:	learn: 0.5905739	total: 12.6s	remaining: 1.07s
    2764:	learn: 0.5905473	total: 12.6s	remaining: 1.07s
    2765:	learn: 0.5905060	total: 12.6s	remaining: 1.06s
    2766:	learn: 0.5904850	total: 12.6s	remaining: 1.06s
    2767:	learn: 0.5904617	total: 12.6s	remaining: 1.05s
    2768:	learn: 0.5904252	total: 12.6s	remaining: 1.05s
    2769:	learn: 0.5903954	total: 12.6s	remaining: 1.04s
    2770:	learn: 0.5903679	total: 12.6s	remaining: 1.04s
    2771:	learn: 0.5903449	total: 12.6s	remaining: 1.03s
    2772:	learn: 0.5903064	total: 12.6s	remaining: 1.03s
    2773:	learn: 0.5902878	total: 12.6s	remaining: 1.03s
    2774:	learn: 0.5902513	total: 12.6s	remaining: 1.02s
    2775:	learn: 0.5902298	total: 12.6s	remaining: 1.02s
    2776:	learn: 0.5902152	total: 12.6s	remaining: 1.01s
    2777:	learn: 0.5901959	total: 12.6s	remaining: 1.01s
    2778:	learn: 0.5901798	total: 12.6s	remaining: 1s
    2779:	learn: 0.5901570	total: 12.6s	remaining: 999ms
    2780:	learn: 0.5901255	total: 12.6s	remaining: 995ms
    2781:	learn: 0.5900937	total: 12.6s	remaining: 991ms
    2782:	learn: 0.5900617	total: 12.6s	remaining: 986ms
    2783:	learn: 0.5900512	total: 12.7s	remaining: 982ms
    2784:	learn: 0.5900178	total: 12.7s	remaining: 978ms
    2785:	learn: 0.5899848	total: 12.7s	remaining: 973ms
    2786:	learn: 0.5899520	total: 12.7s	remaining: 969ms
    2787:	learn: 0.5899255	total: 12.7s	remaining: 965ms
    2788:	learn: 0.5898934	total: 12.7s	remaining: 961ms
    2789:	learn: 0.5898664	total: 12.7s	remaining: 956ms
    2790:	learn: 0.5898365	total: 12.7s	remaining: 952ms
    2791:	learn: 0.5898028	total: 12.7s	remaining: 948ms
    2792:	learn: 0.5897836	total: 12.7s	remaining: 944ms
    2793:	learn: 0.5897667	total: 12.7s	remaining: 939ms
    2794:	learn: 0.5897516	total: 12.8s	remaining: 935ms
    2795:	learn: 0.5897343	total: 12.8s	remaining: 931ms
    2796:	learn: 0.5897110	total: 12.8s	remaining: 926ms
    2797:	learn: 0.5896858	total: 12.8s	remaining: 922ms
    2798:	learn: 0.5896589	total: 12.8s	remaining: 917ms
    2799:	learn: 0.5896412	total: 12.8s	remaining: 913ms
    2800:	learn: 0.5896222	total: 12.8s	remaining: 908ms
    2801:	learn: 0.5895981	total: 12.8s	remaining: 904ms
    2802:	learn: 0.5895717	total: 12.8s	remaining: 900ms
    2803:	learn: 0.5895578	total: 12.8s	remaining: 896ms
    2804:	learn: 0.5895272	total: 12.8s	remaining: 892ms
    2805:	learn: 0.5894999	total: 12.8s	remaining: 887ms
    2806:	learn: 0.5894714	total: 12.8s	remaining: 883ms
    2807:	learn: 0.5894287	total: 12.8s	remaining: 878ms
    2808:	learn: 0.5894076	total: 12.8s	remaining: 873ms
    2809:	learn: 0.5893867	total: 12.8s	remaining: 869ms
    2810:	learn: 0.5893651	total: 12.8s	remaining: 864ms
    2811:	learn: 0.5893398	total: 12.9s	remaining: 859ms
    2812:	learn: 0.5892974	total: 12.9s	remaining: 855ms
    2813:	learn: 0.5892854	total: 12.9s	remaining: 850ms
    2814:	learn: 0.5892671	total: 12.9s	remaining: 845ms
    2815:	learn: 0.5892557	total: 12.9s	remaining: 841ms
    2816:	learn: 0.5892287	total: 12.9s	remaining: 836ms
    2817:	learn: 0.5892061	total: 12.9s	remaining: 832ms
    2818:	learn: 0.5891755	total: 12.9s	remaining: 827ms
    2819:	learn: 0.5891577	total: 12.9s	remaining: 822ms
    2820:	learn: 0.5891352	total: 12.9s	remaining: 818ms
    2821:	learn: 0.5890899	total: 12.9s	remaining: 814ms
    2822:	learn: 0.5890660	total: 12.9s	remaining: 810ms
    2823:	learn: 0.5890296	total: 12.9s	remaining: 806ms
    2824:	learn: 0.5889981	total: 12.9s	remaining: 802ms
    2825:	learn: 0.5889750	total: 13s	remaining: 797ms
    2826:	learn: 0.5889370	total: 13s	remaining: 793ms
    2827:	learn: 0.5889160	total: 13s	remaining: 789ms
    2828:	learn: 0.5888760	total: 13s	remaining: 784ms
    2829:	learn: 0.5888402	total: 13s	remaining: 780ms
    2830:	learn: 0.5888188	total: 13s	remaining: 776ms
    2831:	learn: 0.5887934	total: 13s	remaining: 771ms
    2832:	learn: 0.5887651	total: 13s	remaining: 767ms
    2833:	learn: 0.5887320	total: 13s	remaining: 762ms
    2834:	learn: 0.5887052	total: 13s	remaining: 758ms
    2835:	learn: 0.5886908	total: 13s	remaining: 753ms
    2836:	learn: 0.5886593	total: 13s	remaining: 748ms
    2837:	learn: 0.5886272	total: 13s	remaining: 744ms
    2838:	learn: 0.5885965	total: 13s	remaining: 739ms
    2839:	learn: 0.5885658	total: 13s	remaining: 734ms
    2840:	learn: 0.5885334	total: 13s	remaining: 730ms
    2841:	learn: 0.5884974	total: 13s	remaining: 725ms
    2842:	learn: 0.5884643	total: 13s	remaining: 720ms
    2843:	learn: 0.5884329	total: 13s	remaining: 716ms
    2844:	learn: 0.5884061	total: 13s	remaining: 711ms
    2845:	learn: 0.5883735	total: 13.1s	remaining: 706ms
    2846:	learn: 0.5883433	total: 13.1s	remaining: 702ms
    2847:	learn: 0.5883209	total: 13.1s	remaining: 697ms
    2848:	learn: 0.5882959	total: 13.1s	remaining: 692ms
    2849:	learn: 0.5882557	total: 13.1s	remaining: 687ms
    2850:	learn: 0.5882378	total: 13.1s	remaining: 683ms
    2851:	learn: 0.5882190	total: 13.1s	remaining: 678ms
    2852:	learn: 0.5881981	total: 13.1s	remaining: 673ms
    2853:	learn: 0.5881711	total: 13.1s	remaining: 669ms
    2854:	learn: 0.5881362	total: 13.1s	remaining: 664ms
    2855:	learn: 0.5881100	total: 13.1s	remaining: 660ms
    2856:	learn: 0.5880798	total: 13.1s	remaining: 655ms
    2857:	learn: 0.5880573	total: 13.1s	remaining: 650ms
    2858:	learn: 0.5880274	total: 13.1s	remaining: 646ms
    2859:	learn: 0.5879710	total: 13.1s	remaining: 641ms
    2860:	learn: 0.5879470	total: 13.1s	remaining: 637ms
    2861:	learn: 0.5879019	total: 13.1s	remaining: 632ms
    2862:	learn: 0.5878643	total: 13.1s	remaining: 628ms
    2863:	learn: 0.5878409	total: 13.1s	remaining: 623ms
    2864:	learn: 0.5878217	total: 13.1s	remaining: 618ms
    2865:	learn: 0.5878026	total: 13.1s	remaining: 614ms
    2866:	learn: 0.5877733	total: 13.1s	remaining: 609ms
    2867:	learn: 0.5877565	total: 13.1s	remaining: 604ms
    2868:	learn: 0.5877411	total: 13.1s	remaining: 600ms
    2869:	learn: 0.5877049	total: 13.1s	remaining: 595ms
    2870:	learn: 0.5876815	total: 13.1s	remaining: 590ms
    2871:	learn: 0.5876693	total: 13.1s	remaining: 586ms
    2872:	learn: 0.5876572	total: 13.1s	remaining: 581ms
    2873:	learn: 0.5876203	total: 13.1s	remaining: 577ms
    2874:	learn: 0.5875934	total: 13.2s	remaining: 572ms
    2875:	learn: 0.5875700	total: 13.2s	remaining: 567ms
    2876:	learn: 0.5875535	total: 13.2s	remaining: 563ms
    2877:	learn: 0.5875327	total: 13.2s	remaining: 558ms
    2878:	learn: 0.5875086	total: 13.2s	remaining: 553ms
    2879:	learn: 0.5874765	total: 13.2s	remaining: 549ms
    2880:	learn: 0.5874452	total: 13.2s	remaining: 544ms
    2881:	learn: 0.5874223	total: 13.2s	remaining: 539ms
    2882:	learn: 0.5873987	total: 13.2s	remaining: 535ms
    2883:	learn: 0.5873804	total: 13.2s	remaining: 530ms
    2884:	learn: 0.5873558	total: 13.2s	remaining: 526ms
    2885:	learn: 0.5873260	total: 13.2s	remaining: 521ms
    2886:	learn: 0.5873091	total: 13.2s	remaining: 516ms
    2887:	learn: 0.5872843	total: 13.2s	remaining: 512ms
    2888:	learn: 0.5872718	total: 13.2s	remaining: 507ms
    2889:	learn: 0.5872530	total: 13.2s	remaining: 502ms
    2890:	learn: 0.5872154	total: 13.2s	remaining: 498ms
    2891:	learn: 0.5871888	total: 13.2s	remaining: 493ms
    2892:	learn: 0.5871730	total: 13.2s	remaining: 489ms
    2893:	learn: 0.5871440	total: 13.2s	remaining: 484ms
    2894:	learn: 0.5871196	total: 13.2s	remaining: 479ms
    2895:	learn: 0.5870910	total: 13.2s	remaining: 475ms
    2896:	learn: 0.5870641	total: 13.2s	remaining: 470ms
    2897:	learn: 0.5870429	total: 13.2s	remaining: 466ms
    2898:	learn: 0.5870244	total: 13.2s	remaining: 461ms
    2899:	learn: 0.5869949	total: 13.3s	remaining: 457ms
    2900:	learn: 0.5869708	total: 13.3s	remaining: 452ms
    2901:	learn: 0.5869580	total: 13.3s	remaining: 448ms
    2902:	learn: 0.5869375	total: 13.3s	remaining: 443ms
    2903:	learn: 0.5869142	total: 13.3s	remaining: 439ms
    2904:	learn: 0.5868974	total: 13.3s	remaining: 435ms
    2905:	learn: 0.5868707	total: 13.3s	remaining: 430ms
    2906:	learn: 0.5868508	total: 13.3s	remaining: 426ms
    2907:	learn: 0.5868322	total: 13.3s	remaining: 421ms
    2908:	learn: 0.5868046	total: 13.3s	remaining: 417ms
    2909:	learn: 0.5867880	total: 13.3s	remaining: 412ms
    2910:	learn: 0.5867527	total: 13.3s	remaining: 408ms
    2911:	learn: 0.5867280	total: 13.3s	remaining: 403ms
    2912:	learn: 0.5867077	total: 13.3s	remaining: 399ms
    2913:	learn: 0.5866853	total: 13.4s	remaining: 394ms
    2914:	learn: 0.5866552	total: 13.4s	remaining: 390ms
    2915:	learn: 0.5866222	total: 13.4s	remaining: 386ms
    2916:	learn: 0.5865900	total: 13.4s	remaining: 381ms
    2917:	learn: 0.5865653	total: 13.4s	remaining: 376ms
    2918:	learn: 0.5865333	total: 13.4s	remaining: 372ms
    2919:	learn: 0.5865124	total: 13.4s	remaining: 367ms
    2920:	learn: 0.5864941	total: 13.4s	remaining: 363ms
    2921:	learn: 0.5864700	total: 13.4s	remaining: 358ms
    2922:	learn: 0.5864491	total: 13.4s	remaining: 354ms
    2923:	learn: 0.5864345	total: 13.4s	remaining: 349ms
    2924:	learn: 0.5863846	total: 13.4s	remaining: 345ms
    2925:	learn: 0.5863668	total: 13.5s	remaining: 340ms
    2926:	learn: 0.5863281	total: 13.5s	remaining: 336ms
    2927:	learn: 0.5862992	total: 13.5s	remaining: 331ms
    2928:	learn: 0.5862860	total: 13.5s	remaining: 327ms
    2929:	learn: 0.5862503	total: 13.5s	remaining: 322ms
    2930:	learn: 0.5862216	total: 13.5s	remaining: 318ms
    2931:	learn: 0.5861884	total: 13.5s	remaining: 313ms
    2932:	learn: 0.5861412	total: 13.5s	remaining: 309ms
    2933:	learn: 0.5861173	total: 13.5s	remaining: 304ms
    2934:	learn: 0.5861080	total: 13.5s	remaining: 300ms
    2935:	learn: 0.5860849	total: 13.5s	remaining: 295ms
    2936:	learn: 0.5860667	total: 13.6s	remaining: 291ms
    2937:	learn: 0.5860422	total: 13.6s	remaining: 286ms
    2938:	learn: 0.5860177	total: 13.6s	remaining: 282ms
    2939:	learn: 0.5859893	total: 13.6s	remaining: 277ms
    2940:	learn: 0.5859508	total: 13.6s	remaining: 272ms
    2941:	learn: 0.5859155	total: 13.6s	remaining: 268ms
    2942:	learn: 0.5858779	total: 13.6s	remaining: 263ms
    2943:	learn: 0.5858583	total: 13.6s	remaining: 259ms
    2944:	learn: 0.5858358	total: 13.6s	remaining: 254ms
    2945:	learn: 0.5858049	total: 13.6s	remaining: 250ms
    2946:	learn: 0.5857837	total: 13.6s	remaining: 245ms
    2947:	learn: 0.5857553	total: 13.6s	remaining: 240ms
    2948:	learn: 0.5857347	total: 13.6s	remaining: 236ms
    2949:	learn: 0.5857165	total: 13.7s	remaining: 231ms
    2950:	learn: 0.5856936	total: 13.7s	remaining: 227ms
    2951:	learn: 0.5856553	total: 13.7s	remaining: 222ms
    2952:	learn: 0.5856293	total: 13.7s	remaining: 218ms
    2953:	learn: 0.5856135	total: 13.7s	remaining: 213ms
    2954:	learn: 0.5855843	total: 13.7s	remaining: 208ms
    2955:	learn: 0.5855635	total: 13.7s	remaining: 204ms
    2956:	learn: 0.5855465	total: 13.7s	remaining: 199ms
    2957:	learn: 0.5855180	total: 13.7s	remaining: 195ms
    2958:	learn: 0.5854911	total: 13.7s	remaining: 190ms
    2959:	learn: 0.5854545	total: 13.7s	remaining: 186ms
    2960:	learn: 0.5854336	total: 13.7s	remaining: 181ms
    2961:	learn: 0.5854163	total: 13.7s	remaining: 176ms
    2962:	learn: 0.5853807	total: 13.8s	remaining: 172ms
    2963:	learn: 0.5853540	total: 13.8s	remaining: 167ms
    2964:	learn: 0.5853312	total: 13.8s	remaining: 163ms
    2965:	learn: 0.5853072	total: 13.8s	remaining: 158ms
    2966:	learn: 0.5852707	total: 13.8s	remaining: 154ms
    2967:	learn: 0.5852510	total: 13.8s	remaining: 149ms
    2968:	learn: 0.5852207	total: 13.8s	remaining: 144ms
    2969:	learn: 0.5851876	total: 13.8s	remaining: 140ms
    2970:	learn: 0.5851714	total: 13.8s	remaining: 135ms
    2971:	learn: 0.5851500	total: 13.8s	remaining: 130ms
    2972:	learn: 0.5851119	total: 13.9s	remaining: 126ms
    2973:	learn: 0.5850743	total: 13.9s	remaining: 121ms
    2974:	learn: 0.5850518	total: 13.9s	remaining: 116ms
    2975:	learn: 0.5850195	total: 13.9s	remaining: 112ms
    2976:	learn: 0.5850065	total: 13.9s	remaining: 107ms
    2977:	learn: 0.5849629	total: 13.9s	remaining: 103ms
    2978:	learn: 0.5849352	total: 13.9s	remaining: 98.3ms
    2979:	learn: 0.5849167	total: 14s	remaining: 93.6ms
    2980:	learn: 0.5849008	total: 14s	remaining: 89ms
    2981:	learn: 0.5848697	total: 14s	remaining: 84.3ms
    2982:	learn: 0.5848334	total: 14s	remaining: 79.7ms
    2983:	learn: 0.5847959	total: 14s	remaining: 75ms
    2984:	learn: 0.5847734	total: 14s	remaining: 70.3ms
    2985:	learn: 0.5847465	total: 14s	remaining: 65.7ms
    2986:	learn: 0.5847310	total: 14s	remaining: 61ms
    2987:	learn: 0.5847023	total: 14s	remaining: 56.3ms
    2988:	learn: 0.5846720	total: 14s	remaining: 51.7ms
    2989:	learn: 0.5846419	total: 14s	remaining: 47ms
    2990:	learn: 0.5846166	total: 14.1s	remaining: 42.3ms
    2991:	learn: 0.5845998	total: 14.1s	remaining: 37.6ms
    2992:	learn: 0.5845715	total: 14.1s	remaining: 32.9ms
    2993:	learn: 0.5845475	total: 14.1s	remaining: 28.2ms
    2994:	learn: 0.5845254	total: 14.1s	remaining: 23.5ms
    2995:	learn: 0.5844954	total: 14.1s	remaining: 18.8ms
    2996:	learn: 0.5844792	total: 14.1s	remaining: 14.1ms
    2997:	learn: 0.5844518	total: 14.1s	remaining: 9.41ms
    2998:	learn: 0.5844285	total: 14.1s	remaining: 4.71ms
    2999:	learn: 0.5844064	total: 14.1s	remaining: 0us
    RMSE	:  0.6238810536473509
    R²	:  0.8951625254607929



![png](Predicting%20likes%20of%20Youtube%20videos_files/Predicting%20likes%20of%20Youtube%20videos_166_1.png)



```python
XGB = XGBRegressor(random_state=63)
XGB.fit(X_trn[CAT_NUM_COL],y_trn)
pred = XGB.predict(X_val[CAT_NUM_COL])
score(y_val,pred)
```

    RMSE	:  0.6310015574551118
    R²	:  0.8927557987331236



```python
xgb_params = {'learning_rate': 0.01, 
              'max_depth': 12,
              'subsample': 0.9,        
              'colsample_bytree': 0.9,
              'n_estimators':600, 
              'gamma':1,         
              'min_child_weight':4}   
XGB = XGBRegressor(**xgb_params, seed = 10)
XGB.fit(X_trn[CAT_NUM_COL],y_trn)
pred = XGB.predict(X_val[CAT_NUM_COL])
score(y_val,pred)
```

    RMSE	:  0.5959782440362891
    R²	:  0.9043304416231267



```python

```
