# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 19:12:04 2017
In The name of Allah,The Beneficent and The Merciful
@author: TINA
"""
import nltk
from nltk.corpus import wordnet
"""
take words, looks synonyms to word,antonyms and definitions and  even context to that word
"""
syns=wordnet.synsets("Program") #getting synonyms , syssets() returns a list
print(syns) 
print("\n Accesing a single element-->> ",syns[2])

print("getting the word only -->> ",syns[2].lemmas()[0].name())

print("\n getting the definition -->> ",syns[2].definition())


print("\n getting the examples -->> ",syns[2].examples())

##### getting antonyms########
synonyms=[]
antonyms=[]
for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        print("l:: ",l )
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())


print("\n printing synonyms -->>" , synonyms) 
print("\n printing antonyms -->>" , antonyms)   

############### Similarity : Sementic Similarity ( between two words) #######################

w1=wordnet.synset("ship.n.01")
w2=wordnet.synset("boat.n.01")
print("\n Comparing the Sementic similarity -->> ",w1.wup_similarity(w2))

w2=wordnet.synset("car.n.01")
print("\n Comparing the Sementic similarity -->> ",w1.wup_similarity(w2))

w2=wordnet.synset("cat.n.01")
print("\n Comparing the Sementic similarity -->> ",w1.wup_similarity(w2))

##############
        
            
    
    
    
    
    
    
    





















