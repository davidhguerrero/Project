#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 00:28:55 2016

@author: david
"""

"""
datasets/mycorpus.txt

"Human machine interface for lab abc computer applications",
"A survey of user opinion of computer system response time",
"The EPS user interface management system",
"System and human system engineering testing of EPS", 
"Relation of user perceived response time to error measurement",
"The generation of random binary unordered trees",
"The intersection graph of paths in trees",
"Graph minors IV Widths of trees and well quasi ordering",
"Graph minors A survey"

"""
from gensim import corpora

class MyCorpus(object):
    def __iter__(self):
        for line in open('datasets/mycorpus.txt'):
            # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split())
            

corpus_memory_friendly = MyCorpus() # doesn't load the corpus into memory!

for vector in corpus_memory_friendly:  # load one vector into memory at a time
    print(vector)
    
from six import iteritems

# collect statistics about all tokens
dictionary = corpora.Dictionary(line.lower().split() for line in open('datasets/mycorpus.txt'))

print (dictionary)

# remove stop words and words that appear only once
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist 
            if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]

# remove stop words and words that appear only once
dictionary.filter_tokens(stop_ids + once_ids)

# remove gaps in id sequence after words that were removed
dictionary.compactify()
print(dictionary)