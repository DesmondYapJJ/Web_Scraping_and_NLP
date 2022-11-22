<img src="http://imgur.com/1ZcRyrc.png" style="float: left; margin: 20px; height: 55px">

# Project 3: Web APIs & NLP

# Executive Summary 

As a team of young entrepreneurs and data scientists planning to start up a new cafe, we have decided to scrap and analyze data from online forums to gather and inform the business decision of the cafe set-up. The online forum chosen to start off with would be Reddit which is a popular platform for discussions and trending topics from enthusiasts - of which we will be focusing on tea and coffee.([source](https://backlinko.com/reddit-users)) Subsequently, we would be able to identify and sort future feedback from the public into the respective categories for our futher analysis - with the assumption that key label words would not be included - based on the data the model was trained with from Reddit. 

This is a classification problem and we would utilise models such as LogRegression, RandomForest Classifier, Naive Bayers to identify the best model that can be used for future predictions.

Next we will also look into the general sentiments of the public to some of these key topics and gain further insights (positive, neutral, negative) for implementation to the cafe menu or business opportunities. This would be done by utilising sentiment analysis models such as Spacy([source](https://spacy.io/usage/spacy-101)) and twitter roberta base sentiment([source](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)) imported from Hugging Face. 

## Problem Statement

To create a data-driven menu from gathering feedback from the public for implementation into the cafe's tea and coffee menu. 

Key questions to look at include: 
1. Key topics, discussions, unique flavors that can be identified from popular online forum platform Reddit 
2. Which model is the best for predicting / categorising tea and coffee - this can be used for future implementation on feedback received by the cafe 
3. General sentiments of public towards coffee and tea 

# Contents: 
Part 1: Data Collection 

- Webscrapping Subreddit Tea

- Webscrapping Subreddit Coffee

- Summary

Part 2: EDA and Data Cleaning

- Tea

- Coffee

- Initial Identification of Top Words

- Compare Lemmatization and Stemming

- Summary

Part 3: Modelling and Model Evaluation

- Base Model

- CVEC with Log Regression

- TFIDF with Log Regression

- CVEC with Naive Bayes

- TFIDF with Naive Bayes

- CVEC with Random Forest

- TFIDF with Random Forest

Part 4: Sentiment Analysis, Recommendation and Conclusion

- Summary for Tea

- Summary for Coffee

- Recommendation and Conclusion

# Data Collection 

An attempt to scrap a total of 15,000 posts from subreddit tea between a period of Sunday, August 22, 2021 and Saturday, October 1, 2022 yielded 14971 posts. Similarly, an attempt to scrap a total of 15,000 posts from subreddit coffee between a period of Monday, January 3, 2022 and Saturday, October 1, 2022 yielded 14979 posts. 

An excess of posts were scrapped in the event that there are posts that could not be used after cleaning up the data and removal of duplicates etc. 

# EDA and Data Cleaning

**EDA**

There were multiple duplicated rows within the dataset of both tea and coffee and had to be dropped. There were also multiple posts by 'AutoModerator' present which as the name suggests, is by Reddit automoderator. We also removed posts where 'removed_by_category' is not NaN as these posts have been filtered out automatically by Reddit. Subsequently, 'selftext' which is the description of the Reddit post is combined with the 'title' so that we have a more comprehensive title. 

Basic cleaning of data was done before trying to identify the top words of relevance - such as lowercase, removing html links. 

To identify the top words or interesting words, CountVectorizer and TF-IDF Vectorizer were used for unigram, bigram and trigram. This process has been repeated a few times to filter out the words which are not stopwords but of little use in helping us identify the interesting words. Key labels such as 'tea' and 'coffee' were deliberately left in as they hold meaning in identifying the type of unique tea or coffee or other words of interest. 

Bigram and trigram returned interesting results that gave better insight to the popular topics discussed by the public. These words can then carry on into sentiment analysis in Part 4. Taking a closer look at bigram and trigram, we can find words that relate to flavour, methods of making. These words would form the basis of running a sentiment analysis on afterwards. 

**Data Cleaning**

Subsequently, we then proceeded with using the combined 'title' and 'subreddit' for our analysis. We now dropped stopwords, now including key labels suchs as 'tea' and 'coffee' and other customized words as we prepare the data for exportation for modelling. Having an excess of key labels would result in the model easily identifying which classification of the posts it belongs too. Lemmatization and Stemming were applied and compared with one another. As expected, Stemming returns fewer number of features and will be used over lemmatization. 

# Modelling and Model Evaluation 

|Model|Train Accuracy Score|Test Accuracy Score|
|-----|---------------|--------------|
|Base CVEC Logistic Regression|0.9625|0.8803|
|CVEC Logistic Regression|0.9402|0.8784|
|TVEC Logistic Regression|0.9247|0.8846|
|CVEC Naive Bayes|0.8835|0.8672|
|TVEC Naive Bayes|0.9015|0.8756|
|CVEC Random Forest|0.9917|0.8725|
|TVEC Random Forest|0.8106|0.8038|

Logistic Regression with TF-IDF returned the best results amongst the models trained and tested, with a score of 0.8846, meaning that 88% of the future posts fed to the model would be predicted correctly. Most models have minimal overfitting except for base model where no hyperparameter tunings were done and Random Forest where overfitting results in a poor model. 

# Sentiment Analysis, Recommendation and Conclusion

A comparison between sentiment analysis shows that there is generally a positive sentiment towards tea and coffee, which is more evident in spacy's analysis since there is no neutral scoring. Roberta base sentiment may be misleading with a large percentage returning neutral sentiment (ie, coffee returns 27% positive to 24% negative with a large 49% being neutral - this shows up as 84% positive and 16% negative in spacy's sentiment analysis). A simple preprocessing can be done also to convert the emojis to text for sentiment analysis. 

It is also observed that the unique flavours such as earl grey tea and tie guan yin are very low numbers (less than 100) as compared to the more common flavours out of an approximate 11,000 posts on subreddit tea. However, these are exactly what may be important in a business context and what the cafe would want to focus on. 

<h1 align="center">Tea</h1>

|          |   Overall |   Green Tea |   Earl Grey Tea |   Tie Guan Yin |   Black Tea |   Overall_spacy |
|:---------|----------:|------------:|----------------:|---------------:|------------:|----------------:|
| neutral  |      1090 |         359 |              16 |             27 |         313 |             nan |
| positive |       709 |         352 |              14 |             18 |         303 |            1731 |
| negative |       201 |         121 |               6 |              1 |          97 |             269 |

|          |   Overall |   Green Tea |   Earl Grey Tea |   Tie Guan Yin |   Black Tea |   Overall_spacy |
|:---------|----------:|------------:|----------------:|---------------:|------------:|----------------:|
| neutral  |      0.55 |        0.43 |            0.44 |           0.59 |        0.44 |          nan    |
| positive |      0.35 |        0.42 |            0.39 |           0.39 |        0.42 |            0.87 |
| negative |      0.1  |        0.15 |            0.17 |           0.02 |        0.14 |            0.13 |

<h1 align="center">Coffee</h1>

|          |   Overall |   Cold Brew |   French Press |   Moka Pot |   Hand Grinder |   Overall_spacy |
|:---------|----------:|------------:|---------------:|-----------:|---------------:|----------------:|
| neutral  |       972 |         280 |            384 |        236 |            142 |             nan |
| positive |       545 |         161 |            237 |        132 |            108 |            1688 |
| negative |       483 |         137 |            170 |        118 |             47 |             312 |

|          |   Overall |   Cold Brew |   French Press |   Moka Pot |   Hand Grinder |   Overall_spacy |
|:---------|----------:|------------:|---------------:|-----------:|---------------:|----------------:|
| neutral  |      0.49 |        0.48 |           0.49 |       0.49 |           0.48 |          nan    |
| positive |      0.27 |        0.28 |           0.3  |       0.27 |           0.36 |            0.84 |
| negative |      0.24 |        0.24 |           0.21 |       0.24 |           0.16 |            0.16 |

To conclude, we have managed to identify top words in n-grams using CountVectorizer and TF-IDF Vectorizer. From there we have managed to gain some insights to what may be added on to the cafe's menu. We have also identified the best model as Logistic Regression that can be used to predict if a post came from the subreddit tea or coffee. This can be used for sorting out feedback from the public to the cafe and also potentially identifying more key words. 

As one of the largest online forum platform with 52 million daily active users and a total of 430 million number of monthly active users ([source](https://backlinko.com/reddit-users)), Reddit is a good starting point to scrap and analysis data. However, it is also noted that the largest base of users are from the United States of America - which may present a skewed representation of the population and also to consider where the cafe might be set up. 

Subsequently, it would be beneficial to look into more data sets from different sources or more specific subreddit topics. For example, to look in to subreddits such as 'Starbucks' to identify the more unique and popular words, or other platforms such as Facebook groups and Twitter hashtags. 