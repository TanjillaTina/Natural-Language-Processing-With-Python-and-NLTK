# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 21:52:56 2017
In The name of Allah,The Beneficent and The Merciful
@author: TINA
"""
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer= WordNetLemmatizer()
print(lemmatizer.lemmatize("Cats"))
print(lemmatizer.lemmatize("pupil"))
print(lemmatizer.lemmatize("women"))
print(lemmatizer.lemmatize("saww"))
print(lemmatizer.lemmatize("cities"))
print(lemmatizer.lemmatize("rocking"))



print(lemmatizer.lemmatize("better",pos='a')) # a for adjective
print(lemmatizer.lemmatize("heavier",pos='a'))
print(lemmatizer.lemmatize("dreaming",pos='v')) #v for verb















