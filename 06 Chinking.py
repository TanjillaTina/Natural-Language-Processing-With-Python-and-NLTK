# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 18:33:58 2017
In The name of Allah,The Beneficent and The Merciful 
@author: TINA
"""
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

'''
chunking is the basic technique for entity detection
Finding named entities
Finding the subject 
Uses modifieres, regex
'''

example_text=open('HP.txt','r').read()

def process_content():
   try:
       for i in sent_tokenize(example_text):
           words=nltk.word_tokenize(i)
           tagged=nltk.pos_tag(words)
           # print(tagged)
           MyGramr= r"""Chunk: {<.*>+}  
                              }<RB|VBZ>+{"""  ##CHUNK EVERYTHING AND THEN CHINK RB AND VBZ that comes in sequence
           #MyGramr= "NP: {<DT>?<JJ>*<NN>}"
           MyParser=nltk.RegexpParser(MyGramr)
           foundd=MyParser.parse(tagged)
           print(foundd)
           #foundd.draw()
        
   except Exception as e:
       print(str(e))
    
    
    
    
    
    
    
process_content()   
    
    
    
    
