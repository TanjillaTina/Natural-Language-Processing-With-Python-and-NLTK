# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 21:52:56 2017
In The name of Allah,The Beneficent and The Merciful
@author: TINA
"""

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

'''
'''

example_text=open('names.txt','r').read()
print(example_text)
def process_content():
   try:
       for i in sent_tokenize(example_text):
           words=nltk.word_tokenize(i)
           tagged=nltk.pos_tag(words)
           # print(tagged)
           NamedEntity=nltk.ne_chunk(tagged)
           NamedEntity.draw()
           print(NamedEntity)
   except Exception as e:
       print(str(e))
    
    
    
    
    
    
    
process_content()   
    
    
    
    















