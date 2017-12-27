# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 22:28:00 2017
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
"""
########full code of this one liner code
documents=[]
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append(list(movie_reviews.words(fileid)),category)
 
"""
random.shuffle(documents)
#print(documents[1])
all_words=[]
for w in  movie_reviews.words():
    all_words.append(w.lower())


all_words=nltk.FreqDist(all_words)
print(all_words.most_common(10))
print("the of word's frequency ->> ",all_words["of"])

 



   

 










































