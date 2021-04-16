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

test_word_in = random.sample(vocab_sm, 10)
test_word_out = random.sample(differ, 10)


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

