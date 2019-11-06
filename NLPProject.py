import csv
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from collections import Counter
from nltk.probability import FreqDist
from collections import defaultdict
from nltk.collocations import *
from nltk.metrics import *
from string import punctuation
import os
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from textblob import Word


#********PHASEI

##1. reading the database file

with open('rev.csv','r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    reviewsList=[]
    for line in csv_reader:
            reviewsList.append(line[3])
    #print(reviewsList)

##2.Find the stem of words in text.

ps = PorterStemmer()
tokenizedList = []
stemsList = []
for r in reviewsList:
    tokenizedList.append(word_tokenize(r))
for r in tokenizedList:
    stems = []
    for word in r:
        stems.append(ps.stem(word))
    stemsList.append(stems)
print(stemsList)



##3. Use stopwords list in English (NLTK).

stop_words = set(stopwords.words('english'))

##4. Eliminate the stopwords in stem list.

all_filtered = []
for r in stemsList:
    filtered_Words = []
    for word in r:
      if word not in stop_words:
          filtered_Words.append(word)

    all_filtered.append(filtered_Words)
#print(all_filtered)

##5. Find the most frequent 10 stems.

mostFreq = []

for r in all_filtered:
    frequency = FreqDist(r)
    mostFreq.append(frequency.most_common(10))
#print(mostFreq)

#********PHASEII

#1.  Write a function named preprocess

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    #punctuations = '''!()-[]{};:\'"\,''<>.../?@#$%^&*``_~�='''
    tokenizedText = []
    textStems = []
    ps = PorterStemmer()
    tokenizedText = word_tokenize(text)
    # for word in tokenizedText:
    #     textStems.append(ps.stem(word))
    filtered_Words = []
    for word in tokenizedText:
      if word not in stop_words:
          filtered_Words.append(word)
    filtered_Words2 = []
    for word in filtered_Words:
      if word not in punctuation:
          filtered_Words2.append(word)
    return filtered_Words2

reviewsText = ' '.join(reviewsList) # for getting all reviews as a one text
#print(preprocess(reviewsText))

#2. Write a function named freqMost

def freqMost(list,n):
        frequency = FreqDist(list)
        return frequency.most_common(n)

#print(freqMost(preprocess(reviewsText),10))

#3. n-gram language models

def listNgrams(list, n ):
    ngrams = []
    for i in range(len(list)-(n-1)):
        ngrams.append(tuple(list[i:i+n]))
    return ngrams

#print(listNgrams(preprocess(reviewsText), 2))
freqBigramList = []
freqBigramList = listNgrams(preprocess(reviewsText), 2)
#print(freqBigramList)

#4

def freq(list):
    freqList=[]
    for words in list:
        freqList.append(' '.join(words))
    return freqList

#print(freq(freqBigramList))


def listFreqBigram(list,frequency,n):
    newList=[]
    finalList=[]
    newList = freq(list)
    counts=0
    for sen in set(newList):
        if newList.count(sen) == frequency:
                finalList.append(tuple(sen.split()))
                counts+=1
                if counts==n:
                    break
                elif sen== range(len(set(newList))-1):
                    for sen in set(newList):
                        if newList.count(sen) > frequency:
                            finalList.append(tuple(sen.split()))
                            counts+=1
                            if counts==n:
                                break
    return finalList
scoreBigramList = []
scoreBigramList = listFreqBigram(freqBigramList,2,10)

#print(scoreBigramList)

#5. Write a function named scoredBigram

def scoredBigram(filteredBigrams):
    splitedList=[]
    newList=[]
    splitedList= freq(filteredBigrams)
    newList= ' '.join(splitedList)
    newList=newList.split()
    bgm    = nltk.collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(newList)
    scored = finder.score_ngrams( bgm.likelihood_ratio )
    return(scored)
#print(scoredBigram(scoreBigramList))

#6. Write a function sortedBigram that takes scored list

def sortedBigram(scoredBigrams):
     splitedList=[]
     for i in scoredBigrams:
         splitedList.append(i[0])
     return splitedList
#print(sortedBigram(scoredBigram(scoreBigramList)))


#********PHASEIII

#1. Use POS-tagger that attaches a part of speech tag to each word.

text = word_tokenize(reviewsText)
taggedList = nltk.pos_tag(text)
#print(taggedList)

#2. numOfTags that takes tagged list and returns only the most common tags.

def numOfTags(taggedlist):
     tag_fd = nltk.FreqDist(tag for (word, tag) in taggedlist)
     return tag_fd.most_common()
#print(numOfTags(taggedList))

#3. findWords that takes tagged list and a pos tag. It returns the most common given pos tag

def findWords(taggedlist,posTag):
    word_tag_fd = nltk.FreqDist(taggedlist)
    return [wt[0] for (wt, _) in word_tag_fd.most_common() if wt[1] == posTag]
wordsList= findWords(taggedList,'PRP')
#print(wordsList)


#********PHASEIV
#
# count_vectorizer = CountVectorizer()
# counts = count_vectorizer.fit_transform(text)
#
# # print(counts)
# #
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(counts)
# X_train_tfidf.shape
# print(X_train_tfidf)
#
#
#
# le = preprocessing.LabelBinarizer()
# labels = le.fit_transform(text)
#
# x_train, x_test=train_test_split(X_train_tfidf,test_size=0.000001)
# y_train, y_test=train_test_split(labels,test_size=0.000001)
#
# clf = MultinomialNB().fit(x_train, y_train)

# b = TextBlob("I havv goood speling!")
# print(b.correct())
