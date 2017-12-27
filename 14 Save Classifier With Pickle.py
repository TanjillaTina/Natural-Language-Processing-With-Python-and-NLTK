# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 08:51:57 2017
In The name of Allah,The Beneficent and The Merciful
@author: TINA
"""

"""
Pickle: A way to save python objects, so, whenever we want to use it,we are going o load in those objects
tht's not a part of NLTK but, STL
"""

import nltk
import random
from nltk.corpus import movie_reviews
import pickle


documents=[(list(movie_reviews.words(fileid)),category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)
              ] #this is a one liner code

random.shuffle(documents)
all_words=[]
for w in  movie_reviews.words():
    all_words.append(w.lower())


all_words=nltk.FreqDist(all_words) #ferqdist is ordered from most common word to least common word


word_features=list(all_words.keys())[:1500] #words greater than frequency 1500


def find_features(document):
    words=set(document)
    features={}
    for w in word_features:
        features[w]=(w in words) #this will save a boolean value, eithr it's true or false, so if the words is in the llist of top 1500 words, then it's gonna ssave the value as true
    return features
        

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets=[(find_features(rev),category) for (rev,category) in documents]
        
    
    
    
trainingset=featuresets[:1900] #first 1900 are gonna train against
testingset=featuresets[1900:] #onword 1900 are gonna test against
#####

#####Naive algorithm: (prior occurances x likelihood)/evidence 

classifier=nltk.NaiveBayesClassifier.train(trainingset)
print("Naive Bayes algorithm accuracy percent ",(nltk.classify.accuracy(classifier,testingset))*100)

classifier.show_most_informative_features(15)

#############pickle code starts from here

######################Saving Classifier as Pickle###################
save_classifier=open("naivebayes.pickle","wb") # write byte
pickle.dump(classifier,save_classifier)
save_classifier.close()
###################### end of Saving Classifier as Pickle################### 

######################Reading the Pickle Classifier###################
print("\n\n Otuput of Reading Classifier\n")
classifier_f=open("naivebayes.pickle","rb") # write byte
classifier=pickle.load(classifier_f)
classifier_f.close()
###################### end of Reading the Pickle Classifier################### 

    
    
    
    




