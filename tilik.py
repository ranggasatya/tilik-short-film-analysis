# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 17:42:55 2020

@author: Rangga Satya Prakoso
"""

import pandas as pd
import GetOldTweets3 as got
import re
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
nltk.download('stopwords')
from nltk.corpus import stopwords
import itertools
import collections
from wordcloud import WordCloud
import matplotlib

# Create function that pulls tweets based on a general search query
# and turns to csv file

# Parameters:
# (text query you want to search)
# (max number of most recent tweets to pull from)
def text_query_to_csv(text_query, count):
    # Creation of query object
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(text_query).setSince(since_date).setUntil(until_date).setMaxTweets(count)
    
    # Creation of list that contains all tweets
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)

    # Creating list of chosen tweet data
    text_tweets = [[tweet.id, tweet.permalink, tweet.date, tweet.username,
                    tweet.text, tweet.retweets, tweet.favorites,
                    tweet.mentions, tweet.hashtags] for tweet in tweets]

    # Creation of dataframe from tweets
    tweets_df = pd.DataFrame(text_tweets, columns = ['Id', 'Permalink',
                                                     'Datetime', 'User',
                                                     'Text', 'RT', 'Fav',
                                                     'Mentions', 'Hashtags'])

    # Converting tweets dataframe to csv file
    tweets_df.to_csv('{}-{}k-tweets.csv'.format(text_query, int(count/1000)),
                     sep=',')

# Input search query to scrape tweets and name csv file
# Max recent tweets pulls x amount of most recent tweets from that user
text_query = 'tilik'
since_date = '2020-08-17'
until_date = '2020-08-20'
count = 10000

# Calling function to query X amount of relevant tweets and create a CSV file
text_query_to_csv(text_query, count)

#=============================================================================
#============================= CREATE DATAFRAME ==============================
#=============================================================================

# Create dataframe from csv file containing all the data set
tilik = pd.read_csv('tilik-10k-tweets.csv',
                        parse_dates=[2], dayfirst=True, 
                        encoding= 'unicode_escape')

# Create a function to clean the tweets
def cleanTxt(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) #removing @mentions
    text = re.sub(r'#', '', text) #removing # symbol
    text = re.sub(r'RT[\s]+', '', text) #removing RT
    text = re.sub(r'https?:\/\/\S+', '', text) #removing hyperlink
    
    return text

# Add two columns with year and month information
tilik['Datetime'] = pd.to_datetime(tilik['Datetime'])
tilik['year'] = tilik['Datetime'].dt.year
tilik['month'] = tilik['Datetime'].dt.month
tilik['day'] = tilik['Datetime'].dt.day
tilik['hour'] = tilik['Datetime'].dt.hour

# Tweets across time
matplotlib.rcParams['font.size'] = 10.0
tilik.groupby(['day', 'hour'])['Text'].count().groupby(['day', 'hour']).mean().plot.bar(figsize=(12, 6))
plt.xlabel('Date, Hour')
plt.ylabel('Count of Tweets')

#=============================================================================
#============================ MOST FREQUENT WORDS ============================
#=============================================================================

# Create a list of lists containing lowercase words for each tweet
words_in_tweet_id = [tweet.lower().split() for tweet in tilik['Text']]

# Stop words
stopWords_id = set(stopwords.words('indonesian'))

# Remove stop words from each tweet list of words
tweets_nsw_id = [[word for word in tweet_words if not word in stopWords_id]
                 for tweet_words in words_in_tweet_id]

# Collection words
collection_words = ['ya', 'tilik', 'yg', '#tilik', '"tilik"', 'tilik,', '-', 'tilik.']

# Remove collection words from each tweet list of words
tweets_nsw_nc_id = [[word for word in tweet_words if not word in
                     collection_words]
                 for tweet_words in tweets_nsw_id]

# List of all words across tweets
all_words_id = list(itertools.chain(*tweets_nsw_nc_id))

# Create counter
words_count_id = collections.Counter(all_words_id)

words_count_id.most_common(10)

# Create data frame
words_df_id = pd.DataFrame(words_count_id.most_common(10),columns=['words','count'])

# Visualisation
fig, ax = plt.subplots(figsize=(8, 8))

# Plot horizontal bar graph
words_df_id.sort_values(by='count').plot.barh(x='words',
    y='count',
    ax=ax)

rects = ax.patches
y_labels = words_df_id['count']

# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label. Change to your liking.
    space = 2
    # Vertical alignment for positive values
    ha = 'left'

    # If value of bar is negative: Place label left of bar
    if x_value < 0:
        # Invert space to place label to the left
        space *= -1
        # Horizontally align label at right
        ha = 'right'

    # Use X value as label and format number with one decimal place
    label = "{:.0f}".format(x_value)

    # Create annotation
    plt.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha)                      # Horizontally align label differently for
                                    # positive and negative values.
    
plt.show()

#=============================================================================
#=============================== HASHTAGS COUNT ==============================
#=============================================================================

tilik['Hashtags'] = tilik['Hashtags'].fillna('DUMMYHASHTAG')

# Clean the tweets
tilik['Hashtags'] = tilik['Hashtags'].apply(cleanTxt)

# Create a list of lists containing lowercase words for each tweet
hashtags_in_tweet_id = [tweet.lower().split() for tweet in tilik['Hashtags']]

# Dummy hashtag
dummy_hashtag = ['dummyhashtag']

# Remove dummy hashtag from list of hashtags
hashtags_list = [[word for word in tweet_words if not word in dummy_hashtag]
                 for tweet_words in hashtags_in_tweet_id]

# List of all words across tweets
all_hashtags = list(itertools.chain(*hashtags_list))

# Create counter
hashtags_count_id = collections.Counter(all_hashtags)

hashtags_count_id.most_common(5)

# Create data frame
hashtag_df_id = pd.DataFrame(hashtags_count_id.most_common(5),columns=['hashtags', 'count'])

##########################
### HASHTAGS COUNT CHART
##########################
fig, ax = plt.subplots(figsize=(8, 8))

# Plot horizontal bar graph
hashtag_df_id.sort_values(by='count').plot.barh(x='hashtags',
    y='count',
    ax=ax)

rects = ax.patches
y_labels = hashtag_df_id['count']

# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label. Change to your liking.
    space = 2
    # Vertical alignment for positive values
    ha = 'left'

    # If value of bar is negative: Place label left of bar
    if x_value < 0:
        # Invert space to place label to the left
        space *= -1
        # Horizontally align label at right
        ha = 'right'

    # Use X value as label and format number with one decimal place
    label = "{:.0f}".format(x_value)

    # Create annotation
    plt.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha)                      # Horizontally align label differently for
                                    # positive and negative values.
    
plt.show()

#=============================================================================
#=============================== MENTIONS COUNT ==============================
#=============================================================================

# Create a function to remove @
def cleanMention(text):
    text = re.sub(r'@', '', text) #removing @mentions

    return text

tilik['Mentions'] = tilik['Mentions'].fillna('DUMMYUSER')

# Clean the tweets
tilik['Mentions'] = tilik['Mentions'].apply(cleanMention)

# Create a list of lists containing lowercase words for each tweet
mentions_in_tweet_id = [tweet.lower().split() for tweet in tilik['Mentions']]

# Dummy hashtag
dummy_user = ['dummyuser','null']

# Remove dummy hashtag from list of hashtags
mentions_list = [[word for word in tweet_words if not word in dummy_user]
                 for tweet_words in mentions_in_tweet_id]

# List of all words across tweets
all_mentions = list(itertools.chain(*mentions_list))

# Create counter
mentions_count_id = collections.Counter(all_mentions)

mentions_count_id.most_common(5)

# Create data frame
mentions_df_id = pd.DataFrame(mentions_count_id.most_common(5),columns=['mentions', 'count'])

##########################
### MENTIONS COUNT CHART
##########################

fig, ax = plt.subplots(figsize=(8, 8))

# Plot horizontal bar graph
mentions_df_id.sort_values(by='count').plot.barh(x='mentions',
    y='count',
    ax=ax)

rects = ax.patches
y_labels = mentions_df_id['count']

# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label. Change to your liking.
    space = 2
    # Vertical alignment for positive values
    ha = 'left'

    # If value of bar is negative: Place label left of bar
    if x_value < 0:
        # Invert space to place label to the left
        space *= -1
        # Horizontally align label at right
        ha = 'right'

    # Use X value as label and format number with one decimal place
    label = "{:.0f}".format(x_value)

    # Create annotation
    plt.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha)                      # Horizontally align label differently for
                                    # positive and negative values.
 
plt.show()

#=============================================================================
#=============================== WORD CLOUD ==================================
#=============================================================================

# Create a function to convert text to lower case
def lowerText(text):
    text = text.lower()
     
    return text

# Convert text to lower case
tilik['Text'] = tilik['Text'].apply(lowerText)
tilik

# Plot the word cloud
allWords = ' '.join([tweets for tweets in tilik['Text']])
wc = WordCloud(width=500, height=300, random_state=42, max_font_size=110).generate(allWords)

plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

#=============================================================================
#==================== SENTIMENT ANALYSIS WITH VADER ==========================
#=============================================================================

# Define VADER lexicon as sid
sid = SentimentIntensityAnalyzer()

# Create a function to get the polarity
def polarity_scores(text):
    return sid.polarity_scores(text)

# Create two new columns
tilik['Scores'] = tilik['Text_en'].apply(polarity_scores)
tilik['Compound'] = tilik['Scores'].apply(lambda d:d['compound'])

tilik.to_csv('sentiment_results_tilik.csv', sep=',')

#=============================================================================
#==================== SENTIMENT ANALYSIS VISUALISATION =======================
#=============================================================================

# Polarity histogram - count of cascades
plt.hist(tilik['Compound'], bins = 50, edgecolor='black')
plt.xlabel('Polarity')
plt.ylabel('Count')
plt.title('Sentiment Polarity Distribution - Tweets Count')
plt.show()

# Polarity histogram - sum of retweets
plt.hist(tilik['Compound'], bins = 50, weights = tilik['RT'],
         edgecolor='black')
plt.xlabel('Polarity')
plt.ylabel('Retweets')
plt.title('Sentiment Polarity Distribution - Total Retweets')
plt.show()

# Sentiment bar chart - count cascades
tilik.groupby('Sentiment')['Text'].count().plot.barh(figsize=(6, 4), rot=0)
plt.setp(plt.xticks()[1], rotation=0, ha='right') 
plt.title('Count of Tweets by Sentiment')

# Sentiment bar chart - sum retweets
tilik.groupby('Sentiment')['RT'].sum().plot.barh(figsize=(6, 4), rot=0)
plt.setp(plt.xticks()[1], rotation=0, ha='right') 
plt.title('Total Retweets by Sentiment')