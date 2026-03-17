import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator  # Fixed: lowercase module, STOPWORDS (plural), commas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor

# Load the Instagram dataset with proper encoding
try:
    df = pd.read_csv("Instagram data.csv", encoding='latin-1')
except UnicodeDecodeError:
    try:
        df = pd.read_csv("Instagram data.csv", encoding='cp1252')
    except UnicodeDecodeError:
        df = pd.read_csv("Instagram data.csv", encoding='iso-8859-1')

# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())

# Display first few rows
print("\nFirst 5 rows:")
print(df.head())

# Display basic statistics
print("\nDataset Info:")
print(df.info())

# Display summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display data types
print("\nData Types:")
print(df.dtypes)

plt.figure(figsize=(10, 8))
plt.style.use('fivethirtyeight')
plt.title("distribution of impressions from home")
sns.distplot(df['From Home'])
plt.show()


plt.figure(figsize=(10, 8))
plt.style.use('fivethirtyeight')
plt.title("distribution of impressions from hashtags")
sns.distplot(df['From Hashtags'])
plt.show()

home=df['From Home'].sum()
hashtags=df['From Hashtags'].sum()
explore=df['From Explore'].sum()
other=df['From Other'].sum()

laberls = ['From Home', 'From Hashtags', 'From Explore', 'From Other']
values=[home, hashtags, explore, other]

fig=px.pie(df,values=values,names=laberls,title='Impressions on instagram post from various sources',hole=0.5)
fig.show()

text=" ".join(i for i in df.Caption)
STOPWORDS=set(STOPWORDS)
wordcloud=WordCloud(stopwords=STOPWORDS,background_color='white',width=1600,height=800).generate(text)
plt.style.use('classic')
plt.figure(figsize=(10,8))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.show()

#figure=px.scatter(data_frame=df,x='Likes',y='impressions',size="Likes",title='Likes vs Impressions',trendline='ols')
#figure.show()


numeric_df = df.select_dtypes(include=['int64','float64'])
correlation = numeric_df.corr()
print(correlation["Impressions"].sort_values(ascending=False))


conversion_rate = (df["Follows"].sum()/df["Profile Visits"].sum())*100
print(conversion_rate)


x = np.array(df[['Likes','Saves','Comments','Shares','Profile Visits','Follows']])
y = np.array(df["Impressions"])
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.2,random_state = 42)

model = PassiveAggressiveRegressor()
model.fit(xtrain,ytrain)
model.score(xtest,ytest)