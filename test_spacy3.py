from __future__ import unicode_literals, print_function
import spacy
print(spacy.__version__)
from collections import defaultdict
from thinc.api import set_gpu_allocator,require_gpu


from spacy.training import Example
import random
import warnings
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from collections import defaultdict
import operator
import string
import numpy as np
from itertools import permutations, islice
import os
import errno
import multiprocessing as mp
from spacy.vectors import Vectors
import time
import sys
import pickle

import random
import warnings
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from collections import defaultdict
import operator
import pickle
import argparse, sys
from datetime import datetime, date
import os
import errno
import itertools
import multiprocessing as mp
import shutil
import numpy as np
import math
from spacy.training import Example
from thinc.api import set_gpu_allocator, require_gpu
from spacy.vocab import Vocab


text = 'My secret is qeytdfd123'
nlp_sm = spacy.load("en_core_web_sm")

tok_sm = nlp_sm.tokenizer
ner_sm = nlp_sm.get_pipe('ner')

nlp_lg =  spacy.load("en_core_web_lg")
tok_lg = nlp_lg.tokenizer
ner_lg = nlp_lg.get_pipe('ner')

vocab_lg = list(nlp_lg.vocab.strings)
print(len(vocab_lg))

vocab_sm = list(nlp_sm.vocab.strings)
print(len(vocab_sm))


count = 0
# for word in vocab_lg:
#     doc = nlp(word)
#     print(word)
#     if doc.has_vector == False:
#         count+=1
docs = tok_lg(text) 
doc = ner_lg(docs)
# doc = nlp_lg(text)
vocab_lg = list(nlp_lg.vocab.strings)
print(len(vocab_lg))

for i in range(len(doc)):
    print(doc[i])
    print(len(doc[i].vector))
    print(doc[i].vector)
    

print(doc.vector)
print(len(doc.vector))
# doc = nlp_lg(docs)
# print(doc.has_vector)
# print(doc.has_vector)

# text = "Rana's secret is rtjcdgfg786"

# doc = nlp(text)
# print(doc.has_vector)
# print(doc.vector)

# print(len(doc[4].vector))
# print(len(doc.vector))

# docs1 = tok_sm(text)
# doc1 = ner_sm(docs1)
# # doc1 = nlp_sm(text)
# vocab_sm = list(nlp_sm.vocab.strings)
# print(len(vocab_sm))

# for i in range(len(doc1)):
#     print(doc1[i].vector)
#     print(len(doc1[i].vector))
# print(doc1.vector)
# print(len(doc1.vector))
# doc1 = nlp_sm(docs1)

# print(doc1.has_vector)
# print(doc1.vector)

# print(len(doc1.vector))

sys.exit()


#================================#

nlp = spacy.load("en_core_web_lg")
# nlp.vocab.to_disk("vocab_original")
vocab_lg = list(nlp.vocab.strings)
test_in_vocabs = vocab_lg
print(len(test_in_vocabs))

nlp = spacy.load("en_core_web_sm")
# nlp.vocab.to_disk("vocab_original")
vocab_sm = list(nlp.vocab.strings)
test_in_vocabs = vocab_sm
print(len(test_in_vocabs))


differ = list(set(vocab_lg) - set(vocab_sm))
# print(list(differ[:100]))


ner = nlp.get_pipe("ner")

tok = nlp.tokenizer

test_word_in = random.sample(vocab_sm, 100)
test_word_out = random.sample(differ, 100)


for i in test_word_in:
    # text = "My name is Tham and I live in "+i
    text = i
    print(text)
    
    for i in range(2):
        # print("i = ", i)
        t0 = time.perf_counter()
        docs = tok(text)
        t1 = time.perf_counter()
        doc = ner(docs)
        t2 = time.perf_counter()
        print("i = {0} \t time for tok : {1}".format(i , t1-t0))
        print("i = {0} \t time for ner : {1}".format(i , t2-t1))
        print("i = {0} \t time for both : {1}".format(i , t2-t0))


for i in test_word_out:
    text = i
    print(text)

    for i in range(2):
        # print("i = ", i)
        t0 = time.perf_counter()
        docs = tok(text)
        t1 = time.perf_counter()
        doc = ner(docs)
        t2 = time.perf_counter()
        print("i = {0} \t time for tok : {1}".format(i , t1-t0))
        print("i = {0} \t time for ner : {1}".format(i , t2-t1))
        print("i = {0} \t time for both : {1}".format(i , t2-t0))




   
# print("time for tokenizer: {}".format(t1-t0))

# doc = ner(text)

# doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

# for ent in doc.ents:
#     print(ent.text, ent.start_char, ent.end_char, ent.label_)

