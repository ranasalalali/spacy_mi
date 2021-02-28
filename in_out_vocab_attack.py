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

file_name = open("in_out_vocab.txt","a")



iterations = 100
total_in_vocab_time = 0
total_out_vocab_time = 0

count_success = 0

for i in range(iterations):
    nlp = spacy.load('en_core_web_lg')


    ## out vocab
    print("-----OUT vocab-----")
    vocab_string_org = list(nlp.vocab.strings)
    print("len of vocab before query {}".format(len(vocab_string_org)))
    
    text = "dfjgkkd908lkg"
    
    time1 = time.perf_counter()
    doc = nlp(text)
    time_now1 = time.perf_counter()
    vocab_string_after_query = list(nlp.vocab.strings)
    out_vocab_runtime = time_now1 - time1
    print("runtime = ", out_vocab_runtime)

    total_out_vocab_time += out_vocab_runtime

    print("len of vocab before query {}".format(len(vocab_string_after_query)))
    
    diff = list(set(vocab_string_org).symmetric_difference(vocab_string_after_query))
    print("updated elements: ", diff)

    
    ## in vocab
    print("i = ", i)
    print("-----IN vocab-----")
    vocab_string_org = list(nlp.vocab.strings)
    print("len of vocab before query {}".format(len(vocab_string_org)))
    
    text = "password"
    
    time0 = time.perf_counter()
    doc = nlp(text)
    time_now = time.perf_counter()
    vocab_string_after_query = list(nlp.vocab.strings)
    in_vocab_runtime = time_now - time0
    print("runtime = ", in_vocab_runtime)
    total_in_vocab_time += in_vocab_runtime

    print("len of vocab before query {}".format(len(vocab_string_after_query)))

    

    if out_vocab_runtime > in_vocab_runtime:
        count_success +=1
    # print("-------------------")

file_name.write("Number of successs attempts:{}\n".format(count_success))    
file_name.write("======Average======\n") 
if iterations >0:
    file_name.write("runtime with in vocab: {}\n".format(total_in_vocab_time/iterations))
    file_name.write("runtime with out vocab: {}\n".format(total_out_vocab_time/iterations))
    file_name.write("runtime diff: {}\n".format(total_out_vocab_time/iterations - total_in_vocab_time/iterations ))



