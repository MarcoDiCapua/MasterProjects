from collections import Counter
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer

import pandas as pd
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

nltk.download('stopwords')

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)  
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    return phrase

def clean_data(text):
    print("clean_data")
    preprocessed_reviews = []
    for sentance in text:
        sentance=sentance.lower()
        sentance = decontracted(sentance)
        # sentance = BeautifulSoup(sentance, 'lxml').get_text()
        sentance = re.sub(r"http\S+", "", sentance)
        sentance = re.sub("\S*\d\S*", "", sentance)
        sentance = re.sub('[^A-Za-z]+', ' ', sentance)
        preprocessed_reviews.append(sentance.strip())
   
    
    return preprocessed_reviews

def tokenize(text):
    token=text['clean_text'].apply(word_tokenize)
    return token

def stopword(token):
    stop = stopwords.words('english') 
    stop = [item for item in stop if item not in ["not", "nor", "against"]]
    token = token.apply(lambda x: [item for item in x if item not in stop])
    
    stop= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
                "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
                'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
                'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
                'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
                'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
                'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
                'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
                'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
                's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
                've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
                "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
                "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
                'won', "won't", 'wouldn', "wouldn't",'ň','ě',"ââ"])

    stop = list(stop)

    token = token.apply(lambda x: [item for item in x if item not in stop])

    return token

def remove_punct(token):
    punctuation = string.punctuation

    token = token.apply(lambda x: [item for item in x if item not in punctuation])

    return token

def lemmatizer(token):
    wnl = WordNetLemmatizer()
    token_list = token.tolist()
    # single word lemmatization examples
    for sentences in token_list:
        
        index = token_list.index(sentences)
        token[index] = list(map(wnl.lemmatize, sentences))

    return token

def get_counter(series):
  flat_list = [item for sublist in series for item in sublist]
  c = Counter(flat_list)
  return c