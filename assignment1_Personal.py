#!/usr/bin/env python
# coding: utf-8

# In[8]:


get_ipython().system('pip install numpy')
get_ipython().system('pip install pandas')
get_ipython().system('pip install sklearn')

from html.parser import HTMLParser
import re, html

import matplotlib.pyplot as plt
import pandas as pd

# Declaring global list variables for classified and unclassified tweets, their related party and sentiment
classified_tweets = []
unclassified_tweets = []
party_affiliations_classified = []
party_affiliations_unclassified = []
classified_tweets_sentiment = []
unclassified_tweets_sentiment = []

def strip_tags(user_input):
	tag_re = re.compile(r'(<!--.*?-->|<[^>]*>)')

# Remove well-formed tags.
	no_tags = tag_re.sub('', user_input)

# Clean up anything else by escaping
	strip_input = html.escape(no_tags)
	return strip_input

# In this function, a tweet text is cleaned. 
# The HTML tags are removed, then the URL is removed followed by the cleaning of punctuation. 
# The remaining text even if empty string is returned.

def clean_text(text):
# Removing HTML tags, by calling strip_tags.
	new_text = strip_tags(text)

# Now, removing URL from the text
# We start by deteecting anything which has a . in between
	url_removed_text = re.sub(r'\S+\.\S+', '', new_text)

# This code is to check if the detection of URL is done correctly.
	if not url_removed_text:
		url_removed_text = re.sub(r'^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$', '', new_text)
# If the remaining text is empty, we return an empty string.
		if not url_removed_text:
			return ""

# Converting text to lower keys
	new_text = url_removed_text.lower()
# Removing punctuation
	new_text = re.sub(r'[^\w\s]','', new_text)

# Removing stop words, using the words in stop_words.txt
	stopwords_file = open("stop_words.txt","r")
	stopwords = stopwords_file.read().split('\n')
	querywords = new_text.split()
	resultwords  = [word for word in querywords if word.lower() not in stopwords]
	result = ' '.join(resultwords)

	return result.strip()

# In this function, a political party related to a tweet is found
# A list of keywords associated with a party are matched with words in the tweet.
# If a tweet contains words from multiple partities, it is flagged as other
# Otherwise tweet is related to the party, which the word belongs to
def find_party(tweet_text):
# Declaring list of words related to each party in a list     
	conservative_party_words = ['conservative', 'conservatives', 'conservativeparty', 'stephen', 'harper', 'harpers', 'stephenharper']
	liberal_party_words = ['liberal', 'liberals', 'liberalparty', 'justin', 'trudeau', 'trudeaus', 'justintrudeau','justnotready','lpc','teamtrudeau','realchange']
	ndp_party_words = ['ndp', 'ndps', 'tom', 'thomas' 'mulcair', 'mulcairs', 'tommulcair', 'thomasmulcair']
    
#   Flag indicating whether the tweet belongs to a party, initialized to false as unknown initially
	party_flags = [False, False, False]

#   After cleaning data, all words are separated by " "
#   Each word is iterated over and checked with the party word list
#   If a word is found in party list, the flag indicating whether tweet relates to that party is set to true
	for word in tweet_text.split(" "):
		if word in liberal_party_words :
			party_flags[0] = True
		if word in conservative_party_words :
			party_flags[1] = True
		if word in ndp_party_words:
			party_flags[2] = True
            
# If the words belongs to more than one party, it will be set as unclarified, as it can't be 
# determined for sure which of the parties it belongs to
	if (party_flags[0] and party_flags[1]) or (party_flags[1] and party_flags[2]) or (party_flags[0] and party_flags[2]):
		return 4
	elif party_flags[0]:
		return 1
	elif party_flags[1]:
		return 2
	elif party_flags[2]:
		return 3

	return 0

# In this function, the corpus optional file is used to find sentiment of a tweet from unclassified data
# Each word from corpus file is checked and then related scoring to that word if found in tweet, is added to 
# current tweet score. At the end it is checked, if score is above 0, then 1 indicating positive emotion is returned.
# Otherwise if a negative or 0 number, 0 indicating negative emotion is returned
def find_sentiment(tweet):
# Initializing sentiment score
	sentiment = 0
    
#   Opening corpus file and reading its data 
	corpus_file = open("corpus.txt","r")
	corpus_data = corpus_file.readlines()
    
#   Checking if each word from corpus is found in tweet, and then its related score is added to current score
	for line in corpus_data:
		corpus = line.strip().split("\t")
		if corpus[0] in tweet:
			sentiment = sentiment + int(corpus[1])
            
# Returns 1 that is positive emotion if positive score, otherwise 0            
	if sentiment > 0:
		return 1
    
	return 0


# This is the main function, in which classified tweets files is read, 
# The tweets are extracted and then cleaned according to requirement 
# The same is done for the unclassified tweets

def main():
#   Using global variables for these lists with the function  
	global classified_tweets, unclassified_tweets, party_affiliations_classified, party_affiliations_unclassified, classified_tweets_sentiment,  unclassified_tweets_sentiment 
	classified_tweets = []
	unclassified_tweets = []
	classified_tweets_sentiment = []
	unclassified_tweets_sentiment = []
	party_affiliations_classified = []
	party_affiliations_unclassified = []
    
# Opening classified_tweets, and extracting each tweet data
	file = open("classified_tweets.txt","r")
	file_data = file.readlines()

# Extracting tweets from the tweet object, cleaning text based on requirment using clean_text
	for tweet_obj in file_data:
		tweet_text = clean_text(tweet_obj.split('","')[5].strip())
		sentiment = tweet_obj.split('","')[0].strip()[1:]
        
		if sentiment == "0":
			sentiment = 0
		else:
			sentiment = 1
        
		classified_tweets_sentiment.append(sentiment)
		classified_tweets.append(tweet_text)
        
	file = open("unclassified_tweets.txt","r")
	file_data = file.readlines()

# Extracting tweets from the tweet object, cleaning text based on requirment using clean_text
	for tweet_obj in file_data:
		tweet_text = clean_text(tweet_obj.strip())
		unclassified_tweets.append(tweet_text)
		sentiment = 1 if find_sentiment(tweet_text) > 0 else 0
		unclassified_tweets_sentiment.append(sentiment)

	party_affiliations_unclassified = []
	for tweet in unclassified_tweets:
		party = find_party(tweet)
		party_affiliations_unclassified.append(party)

	party_affiliations_classified = []
	for tweet in classified_tweets:
		party = find_party(tweet)
		party_affiliations_classified.append(party)
     
main()


# ## 2.2 Graph

# ### This graph will help us with the final analysis as it will show the negative and positive tweets for each party

# In[9]:


import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
    
    
# Intializing data frame objects for classified affiliated party and sentiments and same is done for unclassified.
df_classified_party = pd.DataFrame ({
        'party': party_affiliations_classified,
        'sentimentScore': classified_tweets_sentiment
})

df_unclassified_party = pd.DataFrame ({
        'party': party_affiliations_unclassified,
        'sentimentScore': unclassified_tweets_sentiment
})



#Plot Distribution
NDP = [1]
CPC = [2]
LPC = [3]


# The number of positive tweets per party in classified data is found
NDP_Positive_cl = [len(df_classified_party[(df_classified_party.party == 3) & (df_classified_party.sentimentScore == 1)])]
CPC_Positive_cl = [len(df_classified_party[(df_classified_party.party == 2) & (df_classified_party.sentimentScore == 1)])]
LPC_Positive_cl = [len(df_classified_party[(df_classified_party.party == 1) & (df_classified_party.sentimentScore == 1)])]

# The total number of tweets per party in classified data is found (Used to show negative tweets)
NDP_tweets_cl = [len(df_classified_party[(df_classified_party.party == 3)])]
CPC_tweets_cl = [len(df_classified_party[(df_classified_party.party == 2)])]
LPC_tweets_cl = [len(df_classified_party[(df_classified_party.party == 1)])]

# The number of positive tweets per party in unclassified data is found
NDP_Positive_uncl = [len(df_unclassified_party[(df_unclassified_party.party == 3) & (df_unclassified_party.sentimentScore == 1)])]
CPC_Positive_uncl = [len(df_unclassified_party[(df_unclassified_party.party == 2) & (df_unclassified_party.sentimentScore == 1)])]
LPC_Positive_uncl = [len(df_unclassified_party[(df_unclassified_party.party == 1) & (df_unclassified_party.sentimentScore == 1)])]

# The total number of tweets per party in unclassified data is found (Used to show negative tweets)
NDP_tweets_uncl = [len(df_unclassified_party[(df_unclassified_party.party == 3)])]
CPC_tweets_uncl = [len(df_unclassified_party[(df_unclassified_party.party == 2)])]
LPC_tweets_uncl = [len(df_unclassified_party[(df_unclassified_party.party == 1)])]

# Total number of positive tweets per party is found
NDP_Positive = NDP_Positive_cl[0] + NDP_Positive_uncl[0]
CPC_Positive = CPC_Positive_cl[0] + CPC_Positive_uncl[0]
LPC_Positive = LPC_Positive_cl[0] + LPC_Positive_uncl[0]

# Total number of tweets per party is found
NDP_tweets = NDP_tweets_cl[0] + NDP_tweets_uncl[0]
CPC_tweets = CPC_tweets_cl[0] + CPC_tweets_uncl[0]
LPC_tweets = LPC_tweets_cl[0] + LPC_tweets_uncl[0]

# Plotting total tweets per party for classified data
ax = plt.subplot(111)
ax.bar(NDP, NDP_tweets_cl, width=0.25, color = 'b', align='center', label='Number of tweets')

ax.bar(CPC, CPC_tweets_cl, width=0.25, color = 'b', align='center')

ax.bar(LPC, LPC_tweets_cl, width=0.25, color = 'b', align='center')

ax.set_xticks([1,2,3])
ax.set_xticklabels(['NDP','CPC','LPC'])
ax.legend()
plt.title('Classified Tweets')
plt.xlabel('Elections Parties')
plt.ylabel('Number of Mentions')
plt.show()

# Plotting total tweets per party for unclassified data
ax = plt.subplot(111)
ax.bar(NDP, NDP_tweets_uncl, width=0.25, color = 'b', align='center', label='Number of tweets')

ax.bar(CPC, CPC_tweets_uncl, width=0.25, color = 'b', align='center')

ax.bar(LPC, LPC_tweets_uncl, width=0.25, color = 'b', align='center')

ax.set_xticks([1,2,3])
ax.set_xticklabels(['NDP','CPC','LPC'])
ax.legend()
plt.title('Unclassified Tweets')
plt.xlabel('Elections Parties')
plt.ylabel('Number of Mentions')
plt.show()

# Plotting positive versus negative tweets per party for classified data
ax = plt.subplot(111)
ax.bar(NDP, NDP_tweets_cl, width=0.25, color = 'y', align='center', label='Negative tweets')
ax.bar(NDP, NDP_Positive_cl, width=0.25, color = 'g', align = 'center', label='Positive tweets')

ax.bar(CPC, CPC_tweets_cl, width=0.25, color = 'y', align='center')
ax.bar(CPC, CPC_Positive_cl, width=0.25, color = 'g', align = 'center')

ax.bar(LPC, LPC_tweets_cl, width=0.25, color = 'y', align='center')
ax.bar(LPC, LPC_Positive_cl, width=0.25, color = 'g', align = 'center')

ax.set_xticks([1,2,3])
ax.set_xticklabels(['NDP','CPC','LPC'])
ax.legend()
plt.title('Classified Tweets')
plt.xlabel('Elections Parties')
plt.ylabel('Number of Mentions')
plt.show()

# Plotting positive versus negative tweets per party for unclassified data
ax = plt.subplot(111)
ax.bar(NDP, NDP_tweets_uncl, width=0.25, color = 'y', align='center', label='Negative tweets')
ax.bar(NDP, NDP_Positive_uncl, width=0.25, color = 'g', align = 'center', label='Positive tweets')

ax.bar(CPC, CPC_tweets_uncl, width=0.25, color = 'y', align='center')
ax.bar(CPC, CPC_Positive_uncl, width=0.25, color = 'g', align = 'center')

ax.bar(LPC, LPC_tweets_uncl, width=0.25, color = 'y', align='center')
ax.bar(LPC, LPC_Positive_uncl, width=0.25, color = 'g', align = 'center')

ax.set_xticks([1,2,3])
ax.set_xticklabels(['NDP','CPC','LPC'])
ax.legend()
plt.title('Unclassified Tweets')
plt.xlabel('Elections Parties')
plt.ylabel('Number of Mentions')
plt.show()

# Plotting positive versus negative tweets per party for all data
ax = plt.subplot(111)
ax.bar(NDP, NDP_tweets, width=0.25, color = 'y', align='center', label='Negative tweets')
ax.bar(NDP, NDP_Positive, width=0.25, color = 'g', align = 'center', label='Positive tweets')

ax.bar(CPC, CPC_tweets, width=0.25, color = 'y', align='center')
ax.bar(CPC, CPC_Positive, width=0.25, color = 'g', align = 'center')

ax.bar(LPC, LPC_tweets, width=0.25, color = 'y', align='center')
ax.bar(LPC, LPC_Positive, width=0.25, color = 'g', align = 'center')

ax.set_xticks([1,2,3])
ax.set_xticklabels(['NDP','CPC','LPC'])
ax.legend()
plt.title('All Tweets')
plt.xlabel('Elections Parties')
plt.ylabel('Number of Mentions')
plt.show()


# ## 3.0 Model Preparation

# ### 3.1 Model Preparation for Classified Data

# In[10]:


# Spliting the tweets according to the requirment of the assignment
from sklearn.model_selection import train_test_split

import pandas as pd

# Getting a dataframe object for classified tweet data and its related sentimentScore
df_classified = pd.DataFrame ({
        'tweets': classified_tweets,
        'sentimentScore': classified_tweets_sentiment
})

# Adding length per tweet to the dataframe for logictic regression analysis
df_classified['length'] = df_classified['tweets'].apply(len)
 
# Setting x for logistic regression as length of tweet and sentimentScore (0 o 1) as its y value
x = df_classified[['length']]
y = df_classified['sentimentScore']

# The data is split 70% as training data used to find the model and 30% will be test data used for prediction
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


# ## 4.0 Model Implementation

# ### 4.1 Model Implementation for Classified Data

# In[11]:


# Using Logistic Regression to devide the tweets into different classs (look for the meaning of the chart)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Logistic model is found using the x and y training data
logistic = LogisticRegression()
logistic.fit(X_train, y_train)

# The prediction for y values is done using test x values (30% from previous step)
y_pred = logistic.predict(X_test)

# The predicted y values are compared to actual test y values and accuracy score is found
score = accuracy_score(y_test,y_pred)
print (score*100)

# Classification report based on the prediction and actual y values is printed
print(classification_report(y_test, y_pred))

# Confusion Matrix based on the prediction and actual y values is printed
print (confusion_matrix(y_test,y_pred))


# ### 4.2 Model Implementation Unclassified Data

# In[12]:


# Spliting the tweets according to the requirment of the assignment 70 30
from sklearn.model_selection import train_test_split

import pandas as pd

# Getting a dataframe object for unclassified tweet data and its related sentimentScore
df_unclassified = pd.DataFrame ({
        'tweets': unclassified_tweets,
        'sentimentScore': unclassified_tweets_sentiment
})

# Adding length per tweet to the dataframe for logictic regression analysis
df_unclassified['length'] = df_unclassified['tweets'].apply(len)

# Setting x for logistic regression as length of tweet and sentimentScore (0 o 1) as its y value
x = df_unclassified[['length']]
y = df_unclassified['sentimentScore']

# The data is split 70% as training data used to find the model and 30% will be test data used for prediction
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


# In[13]:


# Using Logistic Regression to devide the tweets into different classs (look for the meaning of the chart)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Logistic model is found using the x and y training data
logistic = LogisticRegression()
logistic.fit(X_train, y_train)

# The prediction for y values is done using test x values (30% from previous step)
y_pred = logistic.predict(X_test)

# The predicted y values are compared to actual test y values and accuracy score is found
score = accuracy_score(y_test,y_pred)
print (score*100)

# Classification report based on the prediction and actual y values is printed
print(classification_report(y_test, y_pred))

# Confusion Matrix based on the prediction and actual y values is printed
print (confusion_matrix(y_test,y_pred))


# ### Testing

# In[14]:


df_unclassified.head()


# ## 5.0 Discussion

# ### 5.1 Classified and Unclassified Data

# 5.1.1
# 
# In classified data, NDP has most tweets whereas liberal has the least. This is surprising due to the fact that liberal won that election. But one of the reasons for this discrepancy could be that when a tweet with two party relates words was found, it was set as unclassified and they could have been talking positively about liberal.
# 
# 5.1.2 
# 
# In unclassified data, results are much more aligned with what we expect as liberals have the most number of tweets, followed by conservatives and NDP. This is similar to actual vote results from that year elections.
# 
# 5.1.3
# 
# In distribution of positive and negative tweets per party for all data, we can see that liberal had the most tweets. This also included most positive mentions but we can see the opposite end of twitter mentions as well. Half of all the tweets for each party is negative.
# 

# ### 5.2 Graphs Analysis

# By doing analysis on the all tweets graph, by comparing NDP to Liberal party, liberal party has higher positive tweets. This is almost as accurate as the actual result of the election in 2015. In the actual result Liberal party won the election with 39.5% compared to Conservative party which they achieved 31.9%. The graph illustrates almost the same percentage as the actual result of the election.

# ### 5.3 Did you gain any potential insights into the political sentiment of the Canadian electorate with respect to the major political parties participating in the 2015 federal election?

# In short answer, we can say yes. The only challenge was that the unclassified data sentiment score was calculated according to a file that calculates the score of the words in each tweet, this can cause an inaccurate analysis compared to the classified data. Another reason for calculating the sentiment score is, logistic regression is used to predict categorical target variable and most often a variable with a binary outcome.

# ### 5.4 How is each political party view in the public eye based on the snetiment value?

# By looking at the positive and negative sentiment score, we can notice that NDP party had higher negative tweets both in unclassified and classified tweets compared to Liberal party. Where in unclassified tweets NDP negative sentiment score was lower than Liberal party. Even though in unclassified data NDP has lower negative tweets compared to Liberal, Liberal had more positive tweets, and by looking at all the data combined, Liberal have higher percentage of positive tweets compared to NDP as mentioned earlier.

# ### 5.5  Logistic Regression Analysis

# In logistic regression analysis, the prediction of classified data is 51.7% compared to unclassified data which 80.8%. By doing further analysis, we can notice that the precession of classified data is 51% for negative and 52% for positive data compared to unclassified data which has precession of 85% to negative and 41% for positive. Prediction could improve by changing the in-depended variable that has been used. For example, instead of the length of the tweet we can use a key word and run this machine learning algorithm.
