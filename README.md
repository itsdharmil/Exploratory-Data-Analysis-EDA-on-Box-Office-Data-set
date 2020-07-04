# Movie-Recommendation-Netflix

### Exploratory Data Analysis (EDA) on Box-Office Data-set
Data Analysis or sometimes referred to as exploratory data analysis (EDA) is one of the core components of data science. It is also the part on which data scientists, data engineers and data analysts spend their majority of the time which makes it extremely important in the field of data science. This repository demonstartes some common exploratory data analysis methods and techniques using python.  Good luck with your EDA on this dataset.

### Data Overview 
This dataset is taken from [kaggle](https://www.kaggle.com/ "kaggle"). In this dataset, you are provided with 7398 movies and a variety of metadata obtained from The Movie Database ([TMDB](https://www.themoviedb.org/ "TMDB")). Movies are labeled with id. Data points include cast, crew, plot keywords, budget, posters, release dates, languages, production companies, and countries.
Get the data from : [Data-set](https://www.kaggle.com/c/tmdb-box-office-prediction/data "DATASET")

Data files :
1. train.csv
2. test.csv

###  Installation of packages
- Install **eli5** using pip command: `install eli5`
- Install **datetime** using pip command: `from datetime import datetime`
- Install **pandas** using pip command: `import pandas as pd`
- Install **numpy** using pip command: `import numpy as np`
- Install **matplotlib** using pip command: `import matplotlib`
- Install **matplotlib.pyplot** using pip command: `import matplotlib.pyplot as plt`
- Install **seaborn** using pip command: `import seaborn as sns`
- Install **os** using pip command: `import os`
- Install **scipy** using pip command: `from scipy import sparse`
- Install **scipy.sparse** using pip command: `from scipy.sparse import csr_matrix`
- Install **sklearn.decomposition** using pip command: `from sklearn.decomposition import TruncatedSVD`
- Install **sklearn.metrics.pairwise** using pip command: `from sklearn.metrics.pairwise import cosine_similarity`
- Install **random** using pip command: `import random`

###  Data Loading and Exploration
 We import are datset into dataframes using **pandas**

`train = pd.read_csv('train.csv')`
`test = pd.read_csv('test.csv')`

### Visualizing the Target Distribution

In this visualization, we are plotting a histogram of the distribution of **Revenue**, however, it is skew, hence others readings are not visible. To overcome this we will use logarithmic values of revenue.We will use 'log1p' because there might values equal to 0 and we don't want any infinity or nones as an error



```Python
fig, ax = plt.subplots(figsize = (16, 6))
plt.subplot(1, 2, 1)
plt.hist(train['revenue']);
plt.title('Distribution of revenue');
plt.subplot(1, 2, 2)
plt.hist(np.log1p(train['revenue']));
plt.title('Distribution of log of revenue');
```

![DistributionOfRevenue](/plots/DistributionOfRevenue.png "Analysis 1")
It is better to use log for further analysis for the representation of revenue, therefore we will add a 'log_revenue' column in our data frame
`train['log_revenue'] = np.log1p(train['revenue'])`

We shall follow the same for** Budget**

```Python
fig, ax = plt.subplots(figsize = (16, 6))
plt.subplot(1, 2, 1)
plt.hist(train['budget']);
plt.title('Distribution of budget');
plt.subplot(1, 2, 2)
plt.hist(np.log1p(train['budget']));
plt.title('Distribution of log of budget');
```
![DistributionOfBudget](/plots/DistributionOfBudget.png "Analysis 1")

We shall also add a 'log_budget' column for future refernce
`train['log_budget'] = np.log1p(train['budget'])`
### Relationship between Film Revenue and Budget
By common intuition, Film Revenue should be high if the budget is high, so let us confirm our hypothesis, and analyze for any trends or correlation


```Python
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.scatter(train['budget'], train['revenue'])
plt.title('Revenue vs budget');
plt.subplot(1, 2, 2)
plt.scatter(np.log1p(train['budget']), train['log_revenue'])
plt.title('Log Revenue vs log budget');
```
![RevenueAndBudget](/plots/RevenueAndBudget.png "Analysis 2")

### Does having an Official Homepage Affect Revenue?

We will take look at Homepages of the first 10 most popular movies and count its occurrences.
`train['homepage'].value_counts().head(10)`

```
http://www.transformersmovie.com/                                                  4
http://www.thehobbit.com/                                                          2
http://www.lordoftherings.net/                                                     2
http://www.neverbackdownthemovie.com/                                              1
http://disney.go.com/disneypictures/the-odd-life-of-timothy-green/                 1
http://www.magpictures.com/profile.aspx?id=9833910c-fd4a-4fcb-a734-b3d252473a03    1
http://michaelclayton.warnerbros.com/                                              1
http://www.ceremonyfilm.com/                                                       1
http://www.howtotrainyourdragon.com/                                               1
http://www.iceagemovies.com/films/ice-age                                          1
```
From the analysis we can conclude, we can see that some movies have a unique homepage and some not. This feature can not be of much use so let us create a binary feature which indicates presence or absence of homepage for a particular movie and plot that against revenue and identify any trends.



```Python
train['has_homepage'] = 0 #'0' Stands for no homepage
train.loc[train['homepage'].isnull() == False, 'has_homepage'] = 1 # '1' stands for homepage
test['has_homepage'] = 0
test.loc[test['homepage'].isnull() == False, 'has_homepage'] = 1
sns.catplot(x='has_homepage', y='revenue', data=train);
plt.title('Revenue for film without and with homepage');
```

![Homepage](/plots/Homepage.png "Analysis 3")

On analysis, we can conclude that movies that have a webpage, have more revenue compared to those which did not.


### Distribution of Languages in Film

Let us take a look at the distribution languages of the film and how they affect the revenue.

```Python
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
sns.boxplot(x='original_language', y='revenue', data=train.loc[train['original_language'].isin(train['original_language'].value_counts().head(10).index)]);
plt.title('Mean revenue per language');
plt.subplot(1, 2, 2)
sns.boxplot(x='original_language', y='log_revenue', data=train.loc[train['original_language'].isin(train['original_language'].value_counts().head(10).index)]);
plt.title('Mean log revenue per language');
```

![Language](/plots/Language.png "Analysis 4")

On analysis, from the left graph, we can conclude that English overshadows other languages. and from the right graph, we can conclude that other languages also generate higher-value revenue. However, English is the highest in terms of total revenue.

### Frequent Words in Film Titles and Discriptions
Let us create a word cloud where words with high frequency will be larger compare to fewer frequent words.

**Words in title :**
```Python
plt.figure(figsize = (12, 12))
text = ' '.join(train['original_title'].values)
wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=1000).generate(text)
plt.imshow(wordcloud)
plt.title('Top words in titles')
plt.axis("off")
plt.show()
```
![WordsInTitle](/plots/WordsInTitle.png "Analysis 5")

**Words in Description**

```Python
plt.figure(figsize = (12, 12))
text = ' '.join(train['overview'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=1000).generate(text)
plt.imshow(wordcloud)
plt.title('Top words in overview')
plt.axis("off")
plt.show()
```
![WordsInDes](/plots/WordsInDes.png "Analysis 5")

### Do Film Descriptions Impact Revenue?
Lets us check if Description has an impact on revenue and if so which words impact the most. We will do linear regression, however since words are in string format, we have to use 'TFID" vector

```Python
vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            analyzer='word',
            token_pattern=r'\w{1,}',
            ngram_range=(1, 2),
            min_df=5)

overview_text = vectorizer.fit_transform(train['overview'].fillna(''))
linreg = LinearRegression()
linreg.fit(overview_text, train['log_revenue'])
eli5.show_weights(linreg, vec=vectorizer, top=20, feature_filter=lambda x: x != '<BIAS>')
```

```
+13.074	to
+10.131	bombing
+9.981	the
+9.777	complications
… 3858 more positive …
… 3315 more negative …
-9.281	politicians
-9.391	18
-9.481	violence
-9.628	escape and
-9.716	life they
-10.021	ones
-10.111	sally
-10.291	attracted to
-10.321	who also
-10.421	casino
-10.614	receiving
-10.759	kept
-12.139	and be
-12.939	campaign
-13.858	mike
-15.273	woman from
```

### Data Wrangling
##### Preprocessing Features
Let us check is there are any null entry in release date

`test.loc[test['release_date'].isnull() == True].head()`

We get following result

```828	3829	0	NaN	tt0210130	en	Jails, Hospitals & Hip-Hop	Jails, Hospitals &amp; Hip-Hop is a cinematic```

So we will update correct infomation

`test.loc[test['release_date'].isnull() == True, 'release_date'] = '05/01/00'`

##### Analyzing Movie Release Dates

Let us first find out few relase date formats

`test.loc[test['release_date'].isnull()==False,'release_date'].head()`

```
0    7/14/07
1    5/19/58
2    5/23/97
3     9/4/10
4    2/11/05
```
So in our dataset, Relase date is in format "DD/MM/YY" format. Since are data is from both 20th century and 21st century, it would be beneficial if we convert it into "DD/MM/YYYY" format.

```python
def fix_date(x):
    year = x.split('/')[2]
    if int(year)<=19:
        return x[:-2] + '20' + year
    else:
        return x[:-2] + '19' + year
train['release_date'] = train['release_date'].apply(lambda x: fix_date(x))
test['release_date'] = test['release_date'].apply(lambda x: fix_date(x))
```

##### Creating Features Based on Release Date

Let us also convert our release date into ['year','weekday','month','weekofyear','day','quarter'] for better understanding

```python
def  process_date(df):
    date_parts = ['year','weekday','month','weekofyear','day','quarter']
    for part in date_parts:
        part_col = 'release_date' + '_' + part
        df[part_col] = getattr(df['release_date'].dt,part).astype(int)
    return df
train = process_date(train)
test= process_date(test)
```

### Using Plotly to Visualize the Number of Films Per Year

Let us get insights in which year no films released was highest

![NoFlimsPerYear](/plots/NoFlimsPerYear.png "Analysis 7")

We can see that no of films has increased significantly in the 21st century as compared to 20th.

### Number of Films and Revenue Per Year
Let us find some trends between number of films and total revenue per year and number of films and mean revenue per year.

![FilmCountAndRev](/plots/FilmCountAndRev.png "Analysis 8")

![FilmCountAndAvgRev](/plots/FilmCountAndAvgRev.png "Analysis 8")

### Do Release Days Impact Revenue?
Let us find a relation between Released day and it's impact on the revenue.

![DaysImpactRev](/plots/DaysImpactRev.png "Analysis 9")

### Relationship between Runtime and Revenue
Lets us find the distribution of runtime and its relatioon with revenue
![DistOfFlimsInHrs](/plots/DistOfFlimsInHrs.png "Analysis 10")
![RuntimeVsRev](/plots/RuntimeVsRev.png "Analysis 10")






