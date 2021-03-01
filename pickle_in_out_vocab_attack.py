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


def mkdir_p(path):
    """To make a directory given a path."""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def save_results(results_holder):
    """To save results in a pickle file."""
    now = datetime.now().date()
    now = now.strftime("%Y%m%d")
    folder = 'results/'
    filename = '{}_in_out_vocab.pickle3'.format(now)
    mkdir_p(folder)
    filename = os.path.join(folder, filename)
    save_file = open(filename, 'wb')
    pickle.dump(results_holder, save_file)
    save_file.close()


file_name = open("in_out_vocab.txt","a")



iterations = 10
total_in_vocab_time = 0
total_out_vocab_time = 0

count_success = 0

in_vocab_word = "password"
out_vocab_word = "dfjgkkd908lkg"

file_name.write("=====================")  
file_name.write("In vocab word:{}\n".format(in_vocab_word))  
file_name.write("Out vocab word:{}\n".format(out_vocab_word))    

in_vocab_runtime_list = [None]
out_vocab_runtime_list = [None]

for i in range(iterations):
    
    print("i = ", i)
    nlp = spacy.load('en_core_web_lg')

    

    ## in vocab
    
    print("-----IN vocab-----")
    vocab_string_org = list(nlp.vocab.strings)
    print("len of vocab before query {}".format(len(vocab_string_org)))
    
    text = "password"
    
    time0 = time.perf_counter()
    doc = nlp(text)
    time_now = time.perf_counter()
    vocab_string_after_query = list(nlp.vocab.strings)
    in_vocab_runtime = time_now - time0
    in_vocab_runtime_list.append(in_vocab_runtime)
    
    print(in_vocab_runtime_list)

    print("runtime = ", in_vocab_runtime)
    total_in_vocab_time += in_vocab_runtime

    print("len of vocab before query {}".format(len(vocab_string_after_query)))

    ## out vocab
    nlp = spacy.load('en_core_web_lg')
    print("-----OUT vocab-----")
    vocab_string_org = list(nlp.vocab.strings)
    print("len of vocab before query {}".format(len(vocab_string_org)))
    
    text = "dfjgkkd908lkg"
    
    time1 = time.perf_counter()
    doc = nlp(text)
    time_now1 = time.perf_counter()
    vocab_string_after_query = list(nlp.vocab.strings)
    out_vocab_runtime = time_now1 - time1

    out_vocab_runtime_list.append(out_vocab_runtime)
    
    print(out_vocab_runtime_list)

    print("runtime = ", out_vocab_runtime)

    total_out_vocab_time += out_vocab_runtime

    print("len of vocab before query {}".format(len(vocab_string_after_query)))
    
    diff = list(set(vocab_string_org).symmetric_difference(vocab_string_after_query))
    print("updated elements: ", diff)


    if out_vocab_runtime > in_vocab_runtime:
        count_success +=1
    # print("-------------------")

file_name.write("Number of successs attempts:{}\n".format(count_success))    
file_name.write("======Average======\n") 
if iterations >0:
    file_name.write("runtime with in vocab: {}\n".format(total_in_vocab_time/iterations))
    file_name.write("runtime with out vocab: {}\n".format(total_out_vocab_time/iterations))
    file_name.write("runtime diff: {}\n".format(total_out_vocab_time/iterations - total_in_vocab_time/iterations ))


save_results([in_vocab_runtime_list, out_vocab_runtime_list])
