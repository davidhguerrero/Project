#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 17:06:11 2016

@author: david
"""


# Required imports
from wikitools import wiki
from wikitools import category
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import gensim

import numpy as np
import lda
import lda.datasets

from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import matplotlib.pyplot as plt
import pylab

from test_helper import Test

dictionary = gensim.corpora.Dictionary.load('deerwester.dict')
corpus = gensim.corpora.MmCorpus('deerwester.mm')


tfidf = gensim.models.TfidfModel(corpus)

corpus_tfidf = tfidf[corpus]

# Generate a LSI model with 5 topics for corpus_tfidf and dictionary D
n_topics = 5
lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=5) # initialize an LSI transformation
corpus_lsi = lsi[corpus_tfidf]

"""
from gensim import corpora, models, similarities
if (os.path.exists("/tmp/deerwester.dict")):
    dictionary = corpora.Dictionary.load('/tmp/deerwester.dict')
    corpus = corpora.MmCorpus('/tmp/deerwester.mm')
    print("Used files generated from first tutorial")
else:
    print("Please run first tutorial to generate data set")
"""

"""
  u'0.239*"cat" + 0.196*"ref" + 0.143*"librari" + 0.139*"cite" + 0.135*"anthropolog" + 0.134*"common" + 0.118*"art" + 0.116*"scienc" + 0.116*"human" + 0.108*"social"'
"""