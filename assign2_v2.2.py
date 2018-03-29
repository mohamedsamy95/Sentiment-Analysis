# -*- coding: utf-8 -*-
import pandas as pd 
from pandas import DataFrame, read_csv
import os
import csv 
import numpy as np
import glob
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from nltk.corpus import stopwords
from gensim import utils
from random import shuffle

import csv


#stopwords = open("imdbvocab.txt", 'r' , encoding="ISO-8859-1").read()
#stopwords = stopwords.split("\n")
#word_list = open('D:\\8th term\\Pattern Recognition\\assign\\assign2\\aclImdb\\imdbvocab.txt', "r")
#stopwords_removed = set(stopwords.words('stopwords'))
import csv


def remove_stopwords(sentence, stopwords):
    sentencewords = sentence.split()
    #tknzr = TweetTokenizer()
    #words=tknzr.tokenize(sentence)
    #print(words)
    #print(sentencewords)
    
    resultwords  = [word for word in sentencewords if word.lower() not in stopwords]
    resultwords  = [word for word in sentencewords if word.lower() not in stopwords]  # commenty el line da wel b3do
    stopwords = set(stopwords.words('english'))
    result = ' '.join(resultwords)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens=tokenizer.tokenize(result)
    #tokens = word_tokenize(result)
    return tokens

stopwords2 = open("imdbvocab.txt", 'r' , encoding="ISO-8859-1").read()
inpath="E:/8th term/Pattern Recognition/assign2/aclImdb/train/"
outpath="E:/8th term/Pattern Recognition/assign2/out/"
stopwords2 = stopwords2.split("\n")
indices = []
text = []
rating = []
i =  0
for filename in os.listdir(inpath+"pos"):
    data = open(inpath+"pos/"+filename, 'r' , encoding="ISO-8859-1").read()    
    data2 = remove_stopwords(data, stopwords2)
    indices.append(i)
    text.append(data2)
    rating.append("1")
    i=i+1
    print(i)
    print(data2)
    break
print("+ve done")

for filename in os.listdir(inpath+"neg"):
        data = open(inpath+"neg/"+filename, 'r' , encoding="ISO-8859-1").read()
        data = remove_stopwords(data, stopwords)
        indices.append(i)
        text.append(data)
        rating.append("0")
        i = i + 1
        print(i)
print("-ve done")        
Dataset = list(zip(indices,text,rating))
df = pd.DataFrame(data = Dataset, columns=['row_Number', 'text', 'polarity'])
df.to_csv(outpath+"imdb_tr.csv", index=False, header=True)
'''-------------------------------------------------------------------------------------------'''

'''data = csv.reader(open('D:/college/term 8/pattern/assign/assign 2/out/imdb_tr.csv', 'rt', encoding="ISO-8859-1"))
indices, reviews , polarity = [],[],[]
i=0
for row in data:
    indices.append(row[0])
    reviews.append(row[1])
    polarity.append(row[2])
    i=i+1
    if i==10:
        break
indices.pop(0)
reviews.pop(0)
polarity.pop(0)
sentences = []

for review , index in zip(reviews ,indices):
    review = review.replace("'", '')
    review = review.replace('[', '')
    review = review.replace(']', '')
    review = review.replace(' ', '')
    sentences.append(LabeledSentence(utils.to_unicode(review).split(","), [index])) 
    
model = Doc2Vec(min_count=1, sample=1e-4, negative=5, workers=8)
model.build_vocab(sentences)
print(sentences)


model.train(sentences,total_examples=model.corpus_count,epochs=5)
model.save('D:/college/term 8/pattern/assign/assign 2/out/model.d2v')'''