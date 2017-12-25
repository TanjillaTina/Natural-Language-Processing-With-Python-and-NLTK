# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 20:12:58 2017

@author: TINA
"""
import nltk

#################################### Getting help or hint for each taggs################## 
'''
    NLTK provides documentation for each tag, which can be queried using the tag
    '''
nltk.help.upenn_tagset('RB')

text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())

print("\n Similar texts \n",text.similar('the'))


print(nltk.corpus.brown.tagged_words(tagset='universal'))