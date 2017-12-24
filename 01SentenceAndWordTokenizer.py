# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 08:05:11 2017
In The name of Allah,The Beneficent and The Merciful 
@author: TINA
"""

import nltk
#nltk.download()
from nltk.tokenize import sent_tokenize, word_tokenize

example_text="Hello, Mr.Anaconda, How Are you?? I will be Glad, If you help me to perform my Natural Language Practice."


print(sent_tokenize(example_text))  #Sentence Tokenizer
print(word_tokenize(example_text))  #Word Tokenizer

for i in word_tokenize(example_text):
    print(i)
    