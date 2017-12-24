# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 09:57:05 2017
In The name of Allah,The Beneficent and The Merciful 
@author: TINA
"""


import nltk
#nltk.download()
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
####Printing all the stopwords provided by the NLTK

stop_words=set(stopwords.words("english"));

print("All the stopwords provided by NLTK for English Language \n ",stop_words)

#######What we can do with StopWords

example_text="This is an example showing off stop word filteration"

words=word_tokenize(example_text)

filtered_sentence=[]
for w in words:
    if w not in stop_words:
        filtered_sentence.append(w)

for w in filtered_sentence:
    print(w)
