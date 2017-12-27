# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 23:32:42 2017
In The name of Allah,The Beneficent and The Merciful
@author: TINA
"""
"""
Creating own algorithm for text classifier
this is a text classifier for ""sentiment analysis""
an example: positive or negative conversation 

"""
import nltk
import random
from nltk.corpus import movie_reviews


documents=[(list(movie_reviews.words(fileid)),category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)
              ] #this is a one liner code

random.shuffle(documents)
all_words=[]
for w in  movie_reviews.words():
    all_words.append(w.lower())


all_words=nltk.FreqDist(all_words) #ferqdist is ordered from most common word to least common word


word_features=list(all_words.keys())[:1500] #words greater than frequency 1500


def find_features(document):
    words=set(document)
    features={}
    for w in word_features:
        features[w]=(w in words) #this will save a boolean value, eithr it's true or false, so if the words is in the llist of top 1500 words, then it's gonna ssave the value as true
    return features
        



print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets=[(find_features(rev),category) for (rev,category) in documents]
        
    
    
    
    
    
    
    
    
    
    
    



