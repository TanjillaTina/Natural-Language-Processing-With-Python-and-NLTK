# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 08:28:06 2017
In The name of Allah,The Beneficent and The Merciful
@author: TINA
"""

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

p=PorterStemmer()
wo=['Reading','dreamer','Sleeping','failed','hopefully','dreamed','dreaming','chatting','saw','seen']
for w in wo:
    print(p.stem(w))


print('From Sentence')
ntext='It is very important to be pythonly while you are Pythoning with python. All pyhoners have pythoNEd poorly atleast once '
wo=word_tokenize(ntext)
for w in wo:
    print(p.stem(w))
    
    
