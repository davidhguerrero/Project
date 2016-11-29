#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:21:52 2016

@author: david
"""
import feedparser
import nltk.stem
import gensim

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim import corpora 

import urllib
import numpy as np

import matplotlib.pyplot as plt

s_q = 'search_query=cat:cs.CV'
m_r = 'max_results=2'
url = 'http://export.arxiv.org/api/query?'+ s_q + '&' + m_r
data = urllib.urlopen(url).read()

num_token_single_article = 0
num_token_single_article2 = 0
var1 =[]
var2 =[]

dates = list()
abstracts=list()
tokenize_text=list()
clean_text=list()
clean_abstracts=list()
clean_abstracts_bow=list()
clean_entry=list()
total_corpus_bow=list()
corpus_bow = list()

s=nltk.stem.SnowballStemmer('english')
eng_stopwords = stopwords.words('english')

def get_vector_corpus(clean_abstracts, D):
    """Transform token lists into sparse vectors on the D-space.
    corpus_bow as a sparse_vector. For example, a list of tuples
     [(0, 1), (3, 3), (5,2)]    
    For a dictionary of 10 elements can be represented as a vector
    [1, 0, 0, 3, 0, 2, 0, 0, 0, 0] """

    for n in range(0, len(clean_abstracts)):
        abstracts = clean_abstracts[n]
        corpus = D.doc2bow(abstracts)
        corpus_bow.append(gensim.matutils.sparse2full(corpus, len(D)))
    
    return (corpus_bow)

def count_tokens_in_Dict (corpus_bow):
    """
    Initialize a numpy array that we will use to count tokens.
    token_count[n] should store the number of ocurrences of the n-th token, D[n]
    """
    corpus_bow_flat = [item for sublist in corpus_bow for item in sublist]
    array_corpus = np.asarray(corpus_bow_flat)
    
    token_count = np.zeros(n_tokens)
    
    # sum of all columns of a 2D numpy array (efficiently)
    array_corpus=array_corpus.reshape(feed_entry,len(D))
    # Convert it into a numpy Matrix of a specific shape.
    array_corpus.reshape
    #TODO estoy añadiendo el mismo tipo de variable, habria que dejar una
    token_count = array_corpus.sum(axis=0)   
    return (token_count, array_corpus)
    
def plot_tokens (ids_sorted, tf_sorted):
    plt.rcdefaults()
    n_art = len(feed.entries)
    n_bins = 25
    hot_tokens = [D[i] for i in ids_sorted[n_bins-1::-1]]
    y_pos = np.arange(len(hot_tokens))
    z = tf_sorted[n_bins-1::-1]/n_art
    
    plt.barh(y_pos, z, align='center', alpha=0.4)
    plt.yticks(y_pos, hot_tokens)
    plt.xlabel('Average number of occurrences per article')
    plt.title('Token distribution')
    plt.show()
    
feed = feedparser.parse(data) # All http GET. print feed no está estructurado.
feed_entry = len(feed.entries) # print feed.entries muestra el texto estructurado.
#print feed

for entry in feed.entries: #feed.entries[1].summary, feed.entries[1].publised, etc. muestra el texto
    text = entry.summary #Read 1st summary, is a tag of entry
    dates.append(entry.published[0:4]) #Create a list with years
    tokens = word_tokenize(text) #Split in words
    for token in tokens:
        if token.isalnum():
            tokenize_text.append(token.lower()) #Remove capital letters of each word
            stem_token = s.stem(token) #Select one word
            if stem_token not in eng_stopwords: #if not match in stop words
                clean_text.append(stem_token)
    
    #clean_text = ' '.join(clean_text) 
    clean_abstracts.append(clean_text)
    clean_abstracts_bow.extend(clean_text) #lista de las word bags de todos los textos
    clean_text=list()

#print clean_abstracts_bow
D=gensim.corpora.Dictionary([clean_abstracts_bow])

n_tokens = len(D)

(corpus_bow) = get_vector_corpus(clean_abstracts, D) # Vector to map number of token per article

(token_count, array_corpus) = count_tokens_in_Dict(corpus_bow)
    
ids_sorted = np.argsort(- token_count) # Return the indices would sort an array
tf_sorted = token_count[ids_sorted] #Number of  occurrences of each token

plot_tokens (ids_sorted, tf_sorted)

"""
Count the number of tokens appearing only once, and what is the proportion of them in the token list.
"""
cold_token = np.count_nonzero(tf_sorted == 1)
not_cold_token = sum(tf_sorted[0:(n_tokens-cold_token)]) 
all_token = cold_token+not_cold_token
portion_cold_token = 100*cold_token/all_token

"""Count the number of tokens appearing only in a single article.
"""
for n in range(0, n_tokens):
    column_matrix = array_corpus[:,n]
    if(np.count_nonzero(column_matrix==0) == 9): #Detect tokens appearing once time in a single article
        num_token_single_article = 1 + num_token_single_article
        var1.append(n)
        if ((array_corpus[:,n]).sum(axis=0) < 3): # Detect only one document and less than 2 times
            num_token_single_article2 = 1 + num_token_single_article2 
        else:
            var2.append(n)
            #print n
            
            
from nltk.util import ngrams
sentence = 'this is a foo bar sentences and i want to ngramize it'
sixgrams = ngrams(sentence.split(), 2)
for grams in sixgrams:
    print grams
        
        
"""
Remove all tokens appearing in only one document and less than 2 times.
"""


#token_count=np.asarray(corpus_bow_flat)
#
#plt.rcdefaults()
#n_art = len(feed.entries)
#n_bins = 25
#hot_tokens = [D[i] for i in ids_sorted[n_bins-1::-1]]
#y_pos = np.arange(len(hot_tokens))
#z = tf_sorted[n_bins-1::-1]/n_art
#
#plt.barh(y_pos, z, align='center', alpha=0.4)
#plt.yticks(y_pos, hot_tokens)
#plt.xlabel('Average number of occurrences per article')
#plt.title('Token distribution')
#plt.show()