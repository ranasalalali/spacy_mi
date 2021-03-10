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
from password_generator import PasswordGenerator
# from timing_in_vocab import *


def mkdir_p(path):
    """To make a directory given a path."""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def save_results(results_holder, f_name):
    """To save results in a pickle file."""
    now = datetime.now().date()
    now = now.strftime("%Y%m%d")
    folder = 'compare_timing_1000w_results_{}'.format(now)
    filename = '{}_{}.pickle3'.format(now, f_name)
    mkdir_p(folder)
    filename = os.path.join(folder, filename)
    save_file = open(filename, 'wb')
    pickle.dump(results_holder, save_file)
    save_file.close()




def load_nlp():
    nlp = spacy.load('en_core_web_lg')
    tokeniz = nlp.tokenizer
    tagger = nlp.get_pipe("tagger")
    parser = nlp.get_pipe("parser")
    ner = nlp.get_pipe("ner")
    att_ruler = nlp.get_pipe("attribute_ruler")
    lemmatizer = nlp.get_pipe("lemmatizer")
    return nlp, tokeniz, tagger, parser, ner, att_ruler, lemmatizer


def word_shape(text=None):
    if len(text) >= 100:
        return "LONG"
    shape = []
    last = ""
    shape_char = ""
    seq = 0
    for char in text:
        if char.isalpha():
            if char.isupper():
                shape_char = "X"
            else:
                shape_char = "x"
        elif char.isdigit():
            shape_char = "d"
        else:
            shape_char = char
        if shape_char == last:
            seq += 1
        else:
            seq = 0
            last = shape_char
        if seq < 4:
            shape.append(shape_char)
    return "".join(shape)

def generate_password(lower=1, upper=1, digits=1, special=1, length=8, size=1000):
    
    # prefix = secret[0:knowledge]

    passwords = []

    pwo = PasswordGenerator()
    # pwo.minlen = length-knowledge # (Optional)
    # pwo.maxlen = length-knowledge # (Optional)
    # pwo.minuchars = upper # (Optional)
    # pwo.minlchars = lower # (Optional)
    # pwo.minnumbers = digits # (Optional)
    # #pwo.minschars = special # (Optional)
    # pwo.excludechars = string.punctuation

    # print(type(string.punctuation))

    # for _ in range(size):
    #     passwords.append(pwo.generate())
    passwords =[]
    while len(passwords) < size:
        pw = pwo.generate()
        ### no same suffix and shape
        # if pw[-3:] in vocab or word_shape(pw) in vocab:
        #     pass
        # else:
        #     passwords.append(pw)
            # count +=1
        
        ## having suffix only
        # if pw[-3:] in vocab and word_shape(pw) not in vocab:
        #     passwords.append(pw)
        # else:
        #     pass

        # ## having shape and suffix
        # if pw[-3:] in vocab and word_shape(pw) in vocab:
        #     passwords.append(pw)
        # else:
        #     pass

        ### having whole words
        if pw in vocab:
            passwords.append(pw)
        else:
            pass

    return passwords


def target_nlp_whole(in_vocab, out_vocab, file_name):
    
    total_out_vocab_runtime = 0
    total_in_vocab_runtime = 0
    
    
    file_name.write("======== target whole nlp ==============\n")  
 
    out_vocab_runtime_list = []
    in_vocab_runtime_list = []

    nlp = spacy.load('en_core_web_lg')

    nlp, tokeniz, tagger, parser, ner, att_ruler, lemmatizer = load_nlp()

    for i in out_vocab:
        
        print("text = ", i)
        
        text = i
        
        time0 = time.perf_counter()
        doc = nlp(text)
        time_now = time.perf_counter()
        # vocab_string_after_query = list(nlp.vocab.strings)
        out_vocab_runtime = time_now - time0
        out_vocab_runtime_list.append(out_vocab_runtime)
        
        # print(out_vocab_runtime_list)

        print("runtime = ", out_vocab_runtime)
        total_out_vocab_runtime += out_vocab_runtime

    for i in in_vocab:
        
        print("text = ", i)
        
        text = i
        
        time0 = time.perf_counter()
        doc = nlp(text)
        time_now = time.perf_counter()
        # vocab_string_after_query = list(nlp.vocab.strings)
        in_vocab_runtime = time_now - time0
        in_vocab_runtime_list.append(in_vocab_runtime)
        
        # print(out_vocab_runtime_list)

        print("runtime = ", in_vocab_runtime)
        total_in_vocab_runtime += in_vocab_runtime


    # # out_vocab = "giac7485mo*("
    # time0 = time.perf_counter()
    # doc = tokeniz(out_vocab)
    # doc = tagger(doc)
    # time_now = time.perf_counter()
    # out_vocab_time = time_now - time0
    # file_name.write("runtime of 1 out-vocab (ms): {}\n".format(1000*out_vocab_time))    

    iterations = len(in_vocab)   
    if iterations >0:
        file_name.write("avg runtime with in vocab (ms): {}\n".format(1000*total_in_vocab_runtime/iterations))
        file_name.write("avg runtime with out vocab (ms): {}\n".format(1000*total_out_vocab_runtime/iterations))


    return in_vocab_runtime_list, out_vocab_runtime_list

    # save_results([out_vocab_runtime_list], "target_nlp_whole_in_out_vocab")




def target_nlp_tokenizer(in_vocab, out_vocab, filename):
    total_out_vocab_runtime = 0
    total_in_vocab_runtime = 0

    file_name.write("======== target only tok2vec ==============\n")  
 
    out_vocab_runtime_list = []
    in_vocab_runtime_list = []

    nlp = spacy.load('en_core_web_lg')

    nlp, tokeniz, tagger, parser, ner, att_ruler, lemmatizer = load_nlp()

    for i in out_vocab:
        
        print("text = ", i)
        
        text = i
        
        time0 = time.perf_counter()
        doc = tokeniz(text)
        time_now = time.perf_counter()
        # vocab_string_after_query = list(nlp.vocab.strings)
        out_vocab_runtime = time_now - time0
        out_vocab_runtime_list.append(out_vocab_runtime)
        
        # print(out_vocab_runtime_list)

        print("runtime = ", out_vocab_runtime)
        total_out_vocab_runtime += out_vocab_runtime

    for i in in_vocab:
        
        print("text = ", i)
        
        text = i
        
        time0 = time.perf_counter()
        doc = tokeniz(text)
        
        time_now = time.perf_counter()
        # vocab_string_after_query = list(nlp.vocab.strings)
        in_vocab_runtime = time_now - time0
        in_vocab_runtime_list.append(in_vocab_runtime)
        
        # print(out_vocab_runtime_list)

        print("runtime = ", in_vocab_runtime)
        total_in_vocab_runtime += in_vocab_runtime


    # # out_vocab = "giac7485mo*("
    # time0 = time.perf_counter()
    # doc = tokeniz(out_vocab)
    # doc = tagger(doc)
    # time_now = time.perf_counter()
    # out_vocab_time = time_now - time0
    # file_name.write("runtime of 1 out-vocab (ms): {}\n".format(1000*out_vocab_time))    

    iterations = len(in_vocab)   
    if iterations >0:
        file_name.write("avg runtime with in vocab (ms): {}\n".format(1000*total_in_vocab_runtime/iterations))
        file_name.write("avg runtime with out vocab (ms): {}\n".format(1000*total_out_vocab_runtime/iterations))


    return in_vocab_runtime_list, out_vocab_runtime_list

    
    



def target_ner_tokenizer(in_vocab, out_vocab,  file_name):
    total_out_vocab_runtime = 0
    total_in_vocab_runtime = 0

    file_name.write("======== target tok2vec ner ==============\n")  
 
    out_vocab_runtime_list = []
    in_vocab_runtime_list = []

    nlp = spacy.load('en_core_web_lg')

    nlp, tokeniz, tagger, parser, ner, att_ruler, lemmatizer = load_nlp()

    for i in out_vocab:
        
        print("text = ", i)
        
        text = i
        
        time0 = time.perf_counter()
        doc = tokeniz(text)
        doc = ner(doc)
        time_now = time.perf_counter()
        # vocab_string_after_query = list(nlp.vocab.strings)
        out_vocab_runtime = time_now - time0
        out_vocab_runtime_list.append(out_vocab_runtime)
        
        # print(out_vocab_runtime_list)

        print("runtime = ", out_vocab_runtime)
        total_out_vocab_runtime += out_vocab_runtime

    for i in in_vocab:
        
        print("text = ", i)
        
        text = i
        
        time0 = time.perf_counter()
        doc = tokeniz(text)
        doc = ner(doc)
        time_now = time.perf_counter()
        # vocab_string_after_query = list(nlp.vocab.strings)
        in_vocab_runtime = time_now - time0
        in_vocab_runtime_list.append(in_vocab_runtime)
        
        # print(out_vocab_runtime_list)

        print("runtime = ", in_vocab_runtime)
        total_in_vocab_runtime += in_vocab_runtime


    # # out_vocab = "giac7485mo*("
    # time0 = time.perf_counter()
    # doc = tokeniz(out_vocab)
    # doc = tagger(doc)
    # time_now = time.perf_counter()
    # out_vocab_time = time_now - time0
    # file_name.write("runtime of 1 out-vocab (ms): {}\n".format(1000*out_vocab_time))    

    iterations = len(in_vocab)   
    if iterations >0:
        file_name.write("avg runtime with in vocab (ms): {}\n".format(1000*total_in_vocab_runtime/iterations))
        file_name.write("avg runtime with out vocab (ms): {}\n".format(1000*total_out_vocab_runtime/iterations))


    return in_vocab_runtime_list, out_vocab_runtime_list

      

def target_tagger_tokenizer(in_vocab, out_vocab, filename):
    total_out_vocab_runtime = 0
    total_in_vocab_runtime = 0

    file_name.write("======== target tok2vec tagger ==============\n")  
 
    out_vocab_runtime_list = []
    in_vocab_runtime_list = []

    nlp = spacy.load('en_core_web_lg')

    nlp, tokeniz, tagger, parser, ner, att_ruler, lemmatizer = load_nlp()

    for i in out_vocab:
        
        print("text = ", i)
        
        text = i
        
        time0 = time.perf_counter()
        doc = tokeniz(text)
        doc = tagger(doc)
        time_now = time.perf_counter()
        # vocab_string_after_query = list(nlp.vocab.strings)
        out_vocab_runtime = time_now - time0
        out_vocab_runtime_list.append(out_vocab_runtime)
        
        # print(out_vocab_runtime_list)

        print("runtime = ", out_vocab_runtime)
        total_out_vocab_runtime += out_vocab_runtime

    for i in in_vocab:
        
        print("text = ", i)
        
        text = i
        
        time0 = time.perf_counter()
        doc = tokeniz(text)
        doc = tagger(doc)
        time_now = time.perf_counter()
        # vocab_string_after_query = list(nlp.vocab.strings)
        in_vocab_runtime = time_now - time0
        in_vocab_runtime_list.append(in_vocab_runtime)
        
        # print(out_vocab_runtime_list)

        print("runtime = ", in_vocab_runtime)
        total_in_vocab_runtime += in_vocab_runtime


    # # out_vocab = "giac7485mo*("
    # time0 = time.perf_counter()
    # doc = tokeniz(out_vocab)
    # doc = tagger(doc)
    # time_now = time.perf_counter()
    # out_vocab_time = time_now - time0
    # file_name.write("runtime of 1 out-vocab (ms): {}\n".format(1000*out_vocab_time))    

    iterations = len(in_vocab)   
    if iterations >0:
        file_name.write("avg runtime with in vocab (ms): {}\n".format(1000*total_in_vocab_runtime/iterations))
        file_name.write("avg runtime with out vocab (ms): {}\n".format(1000*total_out_vocab_runtime/iterations))


    return in_vocab_runtime_list, out_vocab_runtime_list

    


def target_parser_tokenizer(in_vocab, out_vocab, filename):
    total_out_vocab_runtime = 0
    total_in_vocab_runtime = 0

    file_name.write("======== target tok2vec parser ==============\n")  
 
    out_vocab_runtime_list = []
    in_vocab_runtime_list = []

    nlp = spacy.load('en_core_web_lg')

    nlp, tokeniz, tagger, parser, ner, att_ruler, lemmatizer = load_nlp()

    for i in out_vocab:
        
        print("text = ", i)
        
        text = i
        
        time0 = time.perf_counter()
        doc = tokeniz(text)
        doc = parser(doc)
        time_now = time.perf_counter()
        # vocab_string_after_query = list(nlp.vocab.strings)
        out_vocab_runtime = time_now - time0
        out_vocab_runtime_list.append(out_vocab_runtime)
        
        # print(out_vocab_runtime_list)

        print("runtime = ", out_vocab_runtime)
        total_out_vocab_runtime += out_vocab_runtime

    for i in in_vocab:
        
        print("text = ", i)
        
        text = i
        
        time0 = time.perf_counter()
        doc = tokeniz(text)
        doc = parser(doc)
        time_now = time.perf_counter()
        # vocab_string_after_query = list(nlp.vocab.strings)
        in_vocab_runtime = time_now - time0
        in_vocab_runtime_list.append(in_vocab_runtime)
        
        # print(out_vocab_runtime_list)

        print("runtime = ", in_vocab_runtime)
        total_in_vocab_runtime += in_vocab_runtime

    # # out_vocab = "giac7485mo*("
    # time0 = time.perf_counter()
    # doc = tokeniz(out_vocab)
    # doc = parser(doc)
    # time_now = time.perf_counter()
    # out_vocab_time = time_now - time0
    # file_name.write("runtime of 1 out-vocab (ms): {}\n".format(1000*out_vocab_time))    

    iterations = len(in_vocab)   
    if iterations >0:
        file_name.write("avg runtime with in vocab (ms): {}\n".format(1000*total_in_vocab_runtime/iterations))
        file_name.write("avg runtime with out vocab (ms): {}\n".format(1000*total_out_vocab_runtime/iterations))


    return in_vocab_runtime_list, out_vocab_runtime_list


    


def target_attRuler_tokenizer(in_vocab, out_vocab,  file_name):
    total_out_vocab_runtime = 0
    total_in_vocab_runtime = 0

    file_name.write("======== target tok2vec attribute_ruler ==============\n")  
 
    out_vocab_runtime_list = []
    in_vocab_runtime_list = []

    nlp = spacy.load('en_core_web_lg')

    nlp, tokeniz, tagger, parser, ner, att_ruler, lemmatizer = load_nlp()

    for i in out_vocab:
        
        print("text = ", i)
        
        text = i
        
        time0 = time.perf_counter()
        doc = tokeniz(text)
        doc = att_ruler(doc)
        time_now = time.perf_counter()
        # vocab_string_after_query = list(nlp.vocab.strings)
        out_vocab_runtime = time_now - time0
        out_vocab_runtime_list.append(out_vocab_runtime)
        
        # print(out_vocab_runtime_list)

        print("runtime = ", out_vocab_runtime)
        total_out_vocab_runtime += out_vocab_runtime

    for i in in_vocab:
        
        print("text = ", i)
        
        text = i
        
        time0 = time.perf_counter()
        doc = tokeniz(text)
        doc = att_ruler(doc)
        time_now = time.perf_counter()
        # vocab_string_after_query = list(nlp.vocab.strings)
        in_vocab_runtime = time_now - time0
        in_vocab_runtime_list.append(in_vocab_runtime)
        
        # print(out_vocab_runtime_list)

        print("runtime = ", in_vocab_runtime)
        total_in_vocab_runtime += in_vocab_runtime


    # # out_vocab = "giac7485mo*("
    # time0 = time.perf_counter()
    # doc = tokeniz(out_vocab)
    # doc = tagger(doc)
    # time_now = time.perf_counter()
    # out_vocab_time = time_now - time0
    # file_name.write("runtime of 1 out-vocab (ms): {}\n".format(1000*out_vocab_time))    

    iterations = len(in_vocab)   
    if iterations >0:
        file_name.write("avg runtime with in vocab (ms): {}\n".format(1000*total_in_vocab_runtime/iterations))
        file_name.write("avg runtime with out vocab (ms): {}\n".format(1000*total_out_vocab_runtime/iterations))


    return in_vocab_runtime_list, out_vocab_runtime_list

    



def target_lemmatizer_tokenizer(in_vocab, out_vocab, filename):
    total_out_vocab_runtime = 0
    total_in_vocab_runtime = 0

    file_name.write("======== target tok2vec lemmatiser ==============\n")  
 
    out_vocab_runtime_list = []
    in_vocab_runtime_list = []

    nlp = spacy.load('en_core_web_lg')

    nlp, tokeniz, tagger, parser, ner, att_ruler, lemmatizer = load_nlp()

    for i in out_vocab:
        
        print("text = ", i)
        
        text = i
        
        time0 = time.perf_counter()
        doc = tokeniz(text)
        doc = lemmatizer(doc)
        time_now = time.perf_counter()
        # vocab_string_after_query = list(nlp.vocab.strings)
        out_vocab_runtime = time_now - time0
        out_vocab_runtime_list.append(out_vocab_runtime)
        
        # print(out_vocab_runtime_list)

        print("runtime = ", out_vocab_runtime)
        total_out_vocab_runtime += out_vocab_runtime

    for i in in_vocab:
        
        print("text = ", i)
        
        text = i
        
        time0 = time.perf_counter()
        doc = tokeniz(text)
        doc = lemmatizer(doc)
        time_now = time.perf_counter()
        # vocab_string_after_query = list(nlp.vocab.strings)
        in_vocab_runtime = time_now - time0
        in_vocab_runtime_list.append(in_vocab_runtime)
        
        # print(out_vocab_runtime_list)

        print("runtime = ", in_vocab_runtime)
        total_in_vocab_runtime += in_vocab_runtime


    # # out_vocab = "giac7485mo*("
    # time0 = time.perf_counter()
    # doc = tokeniz(out_vocab)
    # doc = tagger(doc)
    # time_now = time.perf_counter()
    # out_vocab_time = time_now - time0
    # file_name.write("runtime of 1 out-vocab (ms): {}\n".format(1000*out_vocab_time))    

    iterations = len(in_vocab)   
    if iterations >0:
        file_name.write("avg runtime with in vocab (ms): {}\n".format(1000*total_in_vocab_runtime/iterations))
        file_name.write("avg runtime with out vocab (ms): {}\n".format(1000*total_out_vocab_runtime/iterations))


    return in_vocab_runtime_list, out_vocab_runtime_list

    
if __name__ == "__main__":
    # iterations = 100
    file_name = open("compare_timing_In_Out_vocab_test_have_whole_word.txt","a")
    file_name.write("+++++++++++++++++++++++++++++++++++\n")
    file_name.write("+++++++++++++++++++++++++++++++++++\n")
    # out_vocab = "Gdnam89)k34"

    nlp = spacy.load("en_core_web_lg")
    global vocab
    vocab = list(nlp.vocab.strings)
    in_vocab_words = vocab[10000:11000]
    # print(list(pws))

    pws = generate_password(None,None,None,None,8,1000)
    # print(list(pws))


    in_time_nlp, out_time_nlp = target_nlp_whole(in_vocab_words, pws, file_name)
    in_time_tok2vec, out_time_tok2vec = target_nlp_tokenizer(in_vocab_words, pws,  file_name)
    in_time_tagger, out_time_tagger = target_tagger_tokenizer(in_vocab_words, pws,  file_name)
    in_time_parser, out_time_parser = target_parser_tokenizer(in_vocab_words, pws,   file_name)
    in_time_ner, out_time_ner = target_ner_tokenizer(in_vocab_words, pws,  file_name)
    in_time_attrRuler, out_time_attrRuler = target_attRuler_tokenizer(in_vocab_words, pws,  file_name)
    in_time_lemma, out_time_lemma = target_lemmatizer_tokenizer(in_vocab_words, pws,  file_name)

    save_results([in_time_nlp, out_time_nlp, in_time_tok2vec, out_time_tok2vec, in_time_tagger, out_time_tagger,
                    in_time_parser, out_time_parser, in_time_ner, out_time_ner, in_time_attrRuler, out_time_attrRuler,
                     in_time_lemma, out_time_lemma], "compare_timming_1000_in_vocab_1000_out_vocab_having_whole_word")