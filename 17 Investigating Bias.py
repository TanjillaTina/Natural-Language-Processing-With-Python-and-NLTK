# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 03:22:15 2017
In The name of Allah,The Beneficent and The Merciful
@author: TINA
"""
'''
Distribution on accuracy on positive distribution and accuracy on negative distribution
'''
import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB,BernoulliNB #Multinomial distribution and nt a binary distribution
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC
from nltk.classify import ClassifierI #so, we can inherit nltk classifier class
from statistics import mode  #we are gonna are gonna take simple mode of all the results from classifiers


class VoteClassifier(ClassifierI):
    def __init__ (self,*classifiers):
     self._classifiers=classifiers
    
    def classify(self,features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
        choice_votes=votes.count(mode(votes))
        conf=choice_votes/len(votes)
        return conf
        
    

documents=[(list(movie_reviews.words(fileid)),category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)
              ] #this is a one liner code

#random.shuffle(documents)
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
        
    
     
###This is for positive data example  
trainingset=featuresets[:1900] #first 1900 are gonna train against
testingset=featuresets[1900:] #onword 1900 are gonna test against
#####

###This is for negative data example  
trainingset=featuresets[100:] #first 1900 are gonna train against
testingset=featuresets[:100] #onword 1900 are gonna test against
#####



#####Naive algorithm: (prior occurances x likelihood)/evidence 

classifier=nltk.NaiveBayesClassifier.train(trainingset)
print("Original Naive Bayes algorithm accuracy percent ",(nltk.classify.accuracy(classifier,testingset))*100)

classifier.show_most_informative_features(15)

######## Multinomial distribution ###############
MNB_classifier=SklearnClassifier(MultinomialNB())
MNB_classifier.train(trainingset)

print("\n\n MNB_classifier algorithm accuracy percent ",(nltk.classify.accuracy(MNB_classifier,testingset))*100)
    

    
 
###############Bernoulli distribution ###############
BernoulliNB_classifier=SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(trainingset)

print("\n\n Bernoulli_classifier algorithm accuracy percent ",(nltk.classify.accuracy(BernoulliNB_classifier,testingset)*100))



#from sklearn.linear_model import LogisticRegression,SGDClassifier
#from sklearn.svm import SVC,LinearSVC,NuSVC


###############LogisticRegression ###############
LogisticRegression_classifier=SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(trainingset)

print("\n\n LogisticRegression_classifier algorithm accuracy percent ",(nltk.classify.accuracy(LogisticRegression_classifier,testingset)*100))


###############SGDClassifier ###############
SGDClassifier_classifier=SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(trainingset)

print("\n\n SGDClassifier_classifier algorithm accuracy percent ",(nltk.classify.accuracy(SGDClassifier_classifier,testingset)*100))


###############SVC ###############
SVC_classifier=SklearnClassifier(SVC())
SVC_classifier.train(trainingset)

print("\n\n SVC_classifier algorithm accuracy percent ",(nltk.classify.accuracy(SVC_classifier,testingset)*100))


###############LinearSVC###############
LinearSVC_classifier=SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(trainingset)

print("\n\n LinearSVC_classifier algorithm accuracy percent ",(nltk.classify.accuracy(LinearSVC_classifier,testingset)*100))

###############NuSVC###############
NuSVC_classifier=SklearnClassifier(NuSVC())
NuSVC_classifier.train(trainingset)

print("\n\n NuSVC_classifier algorithm accuracy percent ",(nltk.classify.accuracy(NuSVC_classifier,testingset)*100))







voted_classifier=VoteClassifier(classifier,MNB_classifier,BernoulliNB_classifier,LogisticRegression_classifier,SGDClassifier_classifier,SVC_classifier,LinearSVC_classifier)

print("\n\nVoted_classifier accuracy percent ",(nltk.classify.accuracy(voted_classifier,testingset)*100))

#print("Classification : ",voted_classifier.classify(testingset[0][0]),"Confidence ",voted_classifier.confidence(testingset[0][0])*100)
#print("Classification : ",voted_classifier.classify(testingset[1][0]),"Confidence ",voted_classifier.confidence(testingset[1][0])*100)
#print("Classification : ",voted_classifier.classify(testingset[2][0]),"Confidence ",voted_classifier.confidence(testingset[2][0])*100)
#print("Classification : ",voted_classifier.classify(testingset[3][0]),"Confidence ",voted_classifier.confidence(testingset[3][0])*100)
#print("Classification : ",voted_classifier.classify(testingset[4][0]),"Confidence ",voted_classifier.confidence(testingset[4][0])*100)


