#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 17:17:56 2016

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

import matplotlib.pyplot as plt

from test_helper import Test

#corpus_bow = list()

num_match_token=0

def check_word_match(word):
    """Check how many times match a token in corpus_clean"""
    num_match_token=0
    for n, list in enumerate(corpus_clean): 
        matching = [s for s in list if  word in s]
        m = len (matching)
        num_match_token += m
    return (num_match_token)


def get_vector_corpus(clean_abstracts, D):
    """Transform token lists into sparse vectors on the D-space.
    corpus_bow as a sparse_vector. For example, a list of tuples
     [(0, 1), (3, 3), (5,2)]    
    For a dictionary of 10 elements can be represented as a vector
    [1, 0, 0, 3, 0, 2, 0, 0, 0, 0] """
   
    for m, tokens_list in enumerate(corpus_clean):
            #abstracts = tokens_list[m]
            #for token_list in corpus_clean
            #corpus = [dictionary.doc2bow(token_list) for token_list in corpus_clean]
            #gensim.corpora.MmCorpus.serialize('deerwester.mm', corpus)
            corpus = D.doc2bow(tokens_list)
            gensim.corpora.MmCorpus.serialize('deerwester.mm', corpus)# transform any list of tokens into a list of tuples (token_id, n), one per each token in token_list
            if m == 0: 
                x = gensim.matutils.sparse2full(corpus, len(D))
            else:
                y = gensim.matutils.sparse2full(corpus, len(D))
                x = np.vstack((x,y))
    return (x)

def count_tokens_in_Dict (corpus_bow):
    """
    Initialize a numpy array that we will use to count tokens.
    token_count[n] should store the number of ocurrences of the n-th token, D[n]
    """
#    corpus_bow_flat = [item for sublist in corpus_bow for item in sublist]
#    array_corpus = np.asarray(corpus_bow_flat)
    
    token_count = np.zeros(n_tokens)
    
    # sum of all columns of a 2D numpy array (efficiently)
    corpus_bow=corpus_bow.reshape(n_art,len(D))
    # Convert it into a numpy Matrix of a specific shape.
    corpus_bow.reshape
    #TODO estoy añadiendo el mismo tipo de variable, habria que dejar una
    token_count = corpus_bow.sum(axis=0)   
    return (token_count)
    
def plot_tokens (ids_sorted, tf_sorted):
    plt.rcdefaults()
    #n_art = len(n_art)
    n_bins = 25
    hot_tokens = [D[i] for i in ids_sorted[n_bins-1::-1]]
    y_pos = np.arange(len(hot_tokens))
    z = tf_sorted[n_bins-1::-1]/n_art
    
    plt.barh(y_pos, z, align='center', alpha=0.4)
    plt.yticks(y_pos, hot_tokens)
    plt.xlabel('Average number of occurrences per article')
    plt.title('Token distribution')
    plt.show()
    

site = wiki.Wiki("https://en.wikipedia.org/w/api.php")
# Select a category with a reasonable number of articles (>100)
#cat = "Culture"
cat = "Games"
print cat


print "Loading category data. This may take a while..."
cat_data = category.Category(site, cat)

corpus_titles = []
corpus_text = []

for n, page in enumerate(cat_data.getAllMembersGen()):
    print "\r Loading article {0}".format(n + 1),
    corpus_titles.append(page.title)
    corpus_text.append(page.getWikiText())

n_art = len(corpus_titles)
print "\nLoaded " + str(n_art) + " articles from category " + cat

corpus_tokens = []
corpus_filtered = []
for n, art in enumerate(corpus_text): 
    print "\rTokenizing article {0} out of {1}".format(n + 1, n_art),
    # This is to make sure that all characters have the appropriate encoding.
    art = art.decode('utf-8')  
    tokens = word_tokenize(art)
    corpus_tokens.append(tokens)

    print "\n The corpus has been tokenized. Let's check some portion of the first article:"
print corpus_tokens[0][0:30]

Test.assertEquals(len(corpus_tokens), n_art, "The number of articles has changed unexpectedly")
Test.assertTrue(len(corpus_tokens) >= 100, 
                "Your corpus_tokens has less than 100 articles. Consider using a larger dataset")

# Select stemmer.
stemmer = nltk.stem.SnowballStemmer('english')
corpus_filtered = [[] for i in range (n_art)]
for n, token_list in enumerate(corpus_tokens):  
    print "\rFiltering article {0} out of {1}".format(n + 1, n_art),
    for token in token_list:
        if token.isalnum():
            corpus_filtered[n].append(token.lower())

print "\nLet's check the first tokens from document 0 after stemming:"
print corpus_filtered[0][0:30]

Test.assertTrue(all([c==c.lower() for c in corpus_filtered[23]]), 'Capital letters have not been removed')
Test.assertTrue(all([c.isalnum() for c in corpus_filtered[13]]), 'Non alphanumeric characters have not been removed')


# Select stemmer.
stemmer = nltk.stem.SnowballStemmer('english')
corpus_stemmed = [[] for i in range (n_art)]

for n, token_list in enumerate(corpus_filtered):
    print "\rStemming article {0} out of {1}".format(n + 1, n_art),
    for token in token_list:
        #stem_token = stemmer.stem(token)
        corpus_stemmed[n].append(stemmer.stem(token)) #Select one word 
    # Apply stemming to all tokens in token_list and save them in stemmed_tokens
    # scode: stemmed_tokens = <FILL IN>
    
    # Add stemmed_tokens to the stemmed corpus
    # scode: <FILL IN>

print "\nLet's check the first tokens from document 0 after stemming:"
print corpus_stemmed[0][0:30]

Test.assertTrue((len([c for c in corpus_stemmed[0] if c!=stemmer.stem(c)]) < 0.1*len(corpus_stemmed[0])), 
                'It seems that stemming has not been applied properly')


wnl = WordNetLemmatizer()
corpus_lemmat = [[] for i in range (n_art)]

for n, token_list in enumerate(corpus_filtered):
    print "\rLemmatizing article {0} out of {1}".format(n + 1, n_art),
    for token in token_list:
        corpus_lemmat[n].append(wnl.lemmatize(token))    
    # scode: lemmat_tokens = <FILL IN>

    # Add art to the stemmed corpus
    # scode: <FILL IN>

print "\nLet's check the first tokens from document 0 after stemming:"
print corpus_lemmat[0][0:30]



corpus_clean = [[] for i in range (n_art)]
stopwords_en = stopwords.words('english')
n = 0
for n, token_list in enumerate(corpus_stemmed):
    print "\rRemoving stopwords from article {0} out of {1}".format(n, n_art),
    for token in token_list:
        if token not in stopwords_en: #if not match in stop words
            corpus_clean[n].append(token)
    # Remove all tokens in the stopwords list and append the result to corpus_clean
    # scode: clean_tokens = <FILL IN>

    # scode: <FILL IN>
    
print "\n Let's check tokens after cleaning:"
print corpus_clean[0][0:30]

Test.assertTrue(len(corpus_clean) == n_art, 'List corpus_clean does not contain the expected number of articles')
Test.assertTrue(len([c for c in corpus_clean[0] if c in stopwords_en])==0, 'Stopwords have not been removed')

corpus_bow = []
D = gensim.corpora.Dictionary()

for n, token_list in enumerate(corpus_clean):
    D.add_documents([corpus_clean[n]]) #converts a collection of words to its bag-of-words representation: a list of (word_id, word_frequency) 2-tuples

D.save('deerwester.dict')
n_tokens = len(D)

#corpus_bow (n_art x len(D)) map number of token per article from sparse vectors on the D-space  
# [(0, 1), (3, 3), (5,2)] -> [1, 0, 0, 3, 0, 2, 0, 0, 0, 0]
(corpus_bow) = get_vector_corpus([corpus_clean], D) # Vector to map number of token per article

print "The dictionary contains {0} tokens".format(n_tokens)
print "First tokens in the dictionary: "
for n in range(10):
    print str(n) + ": " + D[n]

Test.assertTrue(len(corpus_bow)==n_art, 'corpus_bow has not the appropriate size') 

corpus = [D.doc2bow(token_list) for token_list in corpus_clean]
gensim.corpora.MmCorpus.serialize('deerwester.mm', corpus)


# token_count (1 x len(D)) in each row has an element that is a sum of tokens of each article
(token_count) = count_tokens_in_Dict(corpus_bow)
    
ids_sorted = np.argsort(- token_count) # Return the indices would sort an array
tf_sorted = token_count[ids_sorted] #Number of occurrences of each token

plot_tokens (ids_sorted, tf_sorted)


"""
Count the number of tokens appearing only once, and what is the proportion of them in the token list.
"""
cold_token = np.count_nonzero(tf_sorted == 1)
not_cold_token = sum(tf_sorted[0:(n_tokens-cold_token)]) 
all_token = cold_token+not_cold_token
portion_cold_token = 100*cold_token/all_token

"""
Count the number of tokens appearing only in a single article.
Remove all tokens appearing in only one document and less than 2 times.
"""

num_token_single_article = 0
num_only_a_document_and_less_2_times = 0
column_with_single_token =[]
column_with_only_a_document_less2times =[]

"""
from six import iteritems
once_ids = [tokenid for tokenid, docfreq in iteritems(D.dfs) if docfreq == 1]
"""            

for n in range(0, n_tokens):
    column_matrix = corpus_bow[:,n] # Select 1st token of each article, 2nd token ... len[D] token of each article   
    if(np.count_nonzero(column_matrix==0) == (n_art-1)): #Detect tokens appearing once time in a single article
        num_token_single_article = 1 + num_token_single_article
        column_with_single_token.append(n)
        if ((corpus_bow[:,n]).sum(axis=0) < 3): # tokens in only one document and less than 2 times
            num_only_a_document_and_less_2_times = 1 + num_only_a_document_and_less_2_times 
            column_with_only_a_document_less2times.append(n)
            
D.filter_tokens(column_with_single_token + column_with_only_a_document_less2times)
D.compactify()
 

import pickle
data = {}
data['D'] = D
data['corpus_bow'] = corpus_bow
pickle.dump(data, open("wikiresults_games.p", "wb"))       
            
from nltk.util import ngrams
sentence = 'this is a foo bar sentences and i want to ngramize it'
sixgrams = ngrams(sentence.split(), 3)
for grams in sixgrams:
    print grams
        
 
