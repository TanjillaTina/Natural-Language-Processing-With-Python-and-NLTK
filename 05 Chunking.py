# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 10:13:51 2017
In The Name of Allah, The Beneficent and The Merciful
@author: TINA
"""
'''
A part-of-speech tagger, or POS-tagger, processes a sequence of words, and attaches a part of speech tag to each word
for details: http://www.nltk.org/book/ch05.html

Tag	Meaning	English Examples
ADJ	adjective	new, good, high, special, big, local
ADP	adposition	on, of, at, with, by, into, under
ADV	adverb	really, already, still, early, now
CONJ	conjunction	and, or, but, if, while, although
DET	determiner, article	the, a, some, most, every, no, which
NOUN	noun	year, home, costs, time, Africa
NUM	numeral	twenty-four, fourth, 1991, 14:24
PRT	particle	at, on, out, over per, that, up, with
PRON	pronoun	he, their, her, its, my, I, us
VERB	verb	is, say, told, given, playing, would
.	punctuation marks	. , ; !
//////////////////////////////////////////////
link:https://pythonprogramming.net/natural-language-toolkit-nltk-part-speech-tagging/
POS TAG LIST:
X	other	ersatz, esprit, dunno, gr8, univeristy
CC	coordinating conjunction
CD	cardinal digit
DT	determiner
EX	existential there (like: "there is" ... think of it like "there exists")
FW	foreign word
IN	preposition/subordinating conjunction
JJ	adjective	'big'
JJR	adjective, comparative	'bigger'
JJS	adjective, superlative	'biggest'
LS	list marker	1)
MD	modal	could, will
NN	noun, singular 'desk'
NNS	noun plural	'desks'
NNP	proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	predeterminer	'all the kids'
POS	possessive ending	parent's
PRP	personal pronoun	I, he, she
PRP$	possessive pronoun	my, his, hers
RB	adverb	very, silently,
RBR	adverb, comparative	better
RBS	adverb, superlative	best
RP	particle	give up
TO	to	go 'to' the store.
UH	interjection	errrrrrrrm
VB	verb, base form	take
VBD	verb, past tense	took
VBG	verb, gerund/present participle	taking
VBN	verb, past participle	taken
VBP	verb, sing. present, non-3d	take
VBZ	verb, 3rd person sing. present	takes
WDT	wh-determiner	which
WP	wh-pronoun	who, what
WP$	possessive wh-pronoun	whose
WRB	wh-abverb	where, when
'''
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
           MyGramr= "NP: {<DT>?<JJ>*<NN>}"
           MyParser=nltk.RegexpParser(MyGramr)
           foundd=MyParser.parse(tagged)
           print(foundd)
           #foundd.draw()
        
   except Exception as e:
       print(str(e))
    
    
    
    
    
    
    
process_content()   
    
    
    
    