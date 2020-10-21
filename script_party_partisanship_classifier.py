'''Getting ready for Naive Bayes Classifier'''
'''databases&Tweet_list'''
import pickle
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
import seaborn as sns
import pandas as pd
import random

parties=['Spdde','CDU','Die_Gruenen','fdp','AfD','dieLinke']

with open ('tweets_Spdde.txt', 'rb') as fp:
    tweet_list_spd = pickle.load(fp)
with open ('tweets_CDU.txt', 'rb') as fp:
    tweet_list_cdu = pickle.load(fp)
with open ('tweets_Die_Gruenen.txt', 'rb') as fp:
    tweet_list_die_gruenen = pickle.load(fp)
with open ('tweets_fdp.txt', 'rb') as fp:
    tweet_list_fdp = pickle.load(fp)
with open ('tweets_AfD.txt', 'rb') as fp:
    tweet_list_afd = pickle.load(fp)
with open ('tweets_dieLinke.txt', 'rb') as fp:
    tweet_list_linke = pickle.load(fp)

print(len(tweet_list_spd))
print(len(tweet_list_cdu))
print(len(tweet_list_die_gruenen))
print(len(tweet_list_fdp))
print(len(tweet_list_afd))
print(len(tweet_list_linke))

all_tweets=random.choices(tweet_list_spd, k=1000)+random.choices(tweet_list_cdu, k=1000)+random.choices(tweet_list_die_gruenen, k=1000)+random.choices(tweet_list_fdp, k=1000)+random.choices(tweet_list_afd, k=1000)+random.choices(tweet_list_linke, k=1000)

corpus=all_tweets

def preprocess_text(text):
    #text = text.lower()
    text = re.sub(r'\d+', '', text)
    return text
german_stop_words = stopwords.words('german')
cv = CountVectorizer(analyzer='word', preprocessor=preprocess_text,stop_words = german_stop_words, max_features = 5000)
cv.fit(corpus)

out = cv.transform(corpus)
out.todense()
df_alle_vector_words = pd.DataFrame(out.todense(), columns=cv.get_feature_names())
df_alle_vector_words

tf = TfidfTransformer()
transformed = tf.fit_transform(out)
transformed

tdf = pd.DataFrame(transformed.todense(), columns=cv.get_feature_names())
tdf.round(4)
#df_alle_vector_words.to_csv("Tweets_all_parties_tweets_count.csv", index=False,header=True)
#tdf.to_csv("Tweets_all_parties_tweets_tdf.csv", index = False, header=True)


X = tdf.values
y = ['SPD']*1000+['CDU']*1000+['Buendnis90/Die_Gruenen']*1000+['FDP']*1000+['AfD']*1000+['Die_Linke']*1000

from sklearn.naive_bayes import MultinomialNB

m = MultinomialNB(alpha=0.01)   # high alpha: more regularization
m.fit(X, y)

tweet = input("Enter your tweet here: ")
tweet = [str(tweet)]
counts_1 = cv.transform(tweet)
tfcounts_1 = tf.transform(counts_1)
print(m.predict(tfcounts_1))
