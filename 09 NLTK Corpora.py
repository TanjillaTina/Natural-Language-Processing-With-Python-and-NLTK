# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 23:18:07 2017
In The name of Allah,The Beneficent and The Merciful

@author: TINA
"""
"""
Accessing corpuus data
"""
import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

sample=gutenberg.raw('shakespeare-hamlet.txt')
#print(sample)
tok=sent_tokenize(sample)
print(tok[5:15])

