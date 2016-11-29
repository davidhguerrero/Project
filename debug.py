#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 09:42:15 2016

@author: david
"""
x = 0
i = 0

def demo(x):
    for i in range(5):
        print("i={}, x={}".format(i,x))
        x = x + 1
    return (x,i)

(x,i) = demo(0)

import pickle
data = {}
data['x'] = x
data['i'] = i
pickle.dump(data, open("debug.p", "wb"))  

import os.path
from gensim import corpora, models, similarities
if (os.path.exists("/tmp/deerwester.dict")):
    dictionary = corpora.Dictionary.load('/tmp/deerwester.dict')
    corpus = corpora.MmCorpus('/tmp/deerwester.mm')
    print("Used files generated from first tutorial")
else:
    print("Please run first tutorial to generate data set")
    
    

dictionary = corpora.Dictionary.load('/tmp/deerwester.dict')
corpus = corpora.MmCorpus('/tmp/deerwester.mm')
print("Used files generated from first tutorial")