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
# from password_generator import PasswordGenerator
import matplotlib.pyplot as plt
from sklearn import metrics


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
    folder = 'timing_results_{}'.format(now)
    filename = '{}_{}.pickle3'.format(now, f_name)
    mkdir_p(folder)
    filename = os.path.join(folder, filename)
    save_file = open(filename, 'wb')
    pickle.dump(results_holder, save_file)
    save_file.close()




def load_nlp():
    # nlp = spacy.load('en_core_web_lg')
    nlp = spacy.load("en_core_web_sm")
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
    passwords =[]
    while len(passwords) < size:
        pw = pwo.generate()
        if pw[-3:] in vocab or word_shape(pw) in vocab:
            pass
        else:
            passwords.append(pw)
            # count +=1

    return passwords



def target_nlp_tokenizer(texts, file_name):
    total_out_vocab_runtime = 0

    file_name.write("======== target only tok2vec ==============\n")  
 
    out_vocab_runtime_list = []

    nlp = spacy.load('en_core_web_lg')

    nlp, tokeniz, tagger, parser, ner, att_ruler, lemmatizer = load_nlp()

    for i in texts:
        
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

    # # out_vocab = "giac7485mo*("
    # time0 = time.perf_counter()
    # doc = tokeniz(text)
    # time_now = time.perf_counter()
    # out_vocab_time = time_now - time0
    # file_name.write("runtime of 1 out-vocab (ms): {}\n".format(1000*out_vocab_time))    
        
    iterations = len(texts)   
    if iterations >0:
        file_name.write("avg runtime with in vocab (ms): {}\n".format(1000*total_out_vocab_runtime/iterations))


    return out_vocab_runtime_list
    
    



def target_ner_tokenizer(texts,  file_name):
    total_out_vocab_runtime = 0

    file_name.write("======== target tok2vec ner ==============\n")  
 
    out_vocab_runtime_list = []

    nlp = spacy.load('en_core_web_lg')

    nlp, tokeniz, tagger, parser, ner, att_ruler, lemmatizer = load_nlp()

    for i in texts:
        
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


    # # out_vocab = "giac7485mo*("
    # time0 = time.perf_counter()
    # doc = tokeniz(out_vocab)
    # doc = ner(doc)
    # time_now = time.perf_counter()
    # out_vocab_time = time_now - time0
    # file_name.write("runtime of 1 out-vocab (ms): {}\n".format(1000*out_vocab_time))    

    iterations = len(texts)   
    if iterations >0:
        file_name.write("avg runtime with in vocab (ms): {}\n".format(1000*total_out_vocab_runtime/iterations))


    return out_vocab_runtime_list


      

def target_tagger_tokenizer(texts, file_name):
    total_out_vocab_runtime = 0

    file_name.write("======== target tok2vec tagger ==============\n")  
 
    out_vocab_runtime_list = []

    nlp = spacy.load('en_core_web_lg')

    nlp, tokeniz, tagger, parser, ner, att_ruler, lemmatizer = load_nlp()

    for i in texts:
        
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


    # # out_vocab = "giac7485mo*("
    # time0 = time.perf_counter()
    # doc = tokeniz(out_vocab)
    # doc = tagger(doc)
    # time_now = time.perf_counter()
    # out_vocab_time = time_now - time0
    # file_name.write("runtime of 1 out-vocab (ms): {}\n".format(1000*out_vocab_time))    

    iterations = len(texts)   
    if iterations >0:
        file_name.write("avg runtime with in vocab (ms): {}\n".format(1000*total_out_vocab_runtime/iterations))


    return out_vocab_runtime_list

    


def target_parser_tokenizer(texts, file_name):
    total_out_vocab_runtime = 0

    file_name.write("======== target tok2vec parser ==============\n")  
 
    out_vocab_runtime_list = []

    nlp = spacy.load('en_core_web_lg')

    nlp, tokeniz, tagger, parser, ner, att_ruler, lemmatizer = load_nlp()

    for i in texts:
        
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

    # # out_vocab = "giac7485mo*("
    # time0 = time.perf_counter()
    # doc = tokeniz(out_vocab)
    # doc = parser(doc)
    # time_now = time.perf_counter()
    # out_vocab_time = time_now - time0
    # file_name.write("runtime of 1 out-vocab (ms): {}\n".format(1000*out_vocab_time))    

    iterations = len(texts)   
    if iterations >0:
        file_name.write("avg runtime with in vocab (ms): {}\n".format(1000*total_out_vocab_runtime/iterations))


    return out_vocab_runtime_list


    


def target_attRuler_tokenizer(texts,  file_name):
    total_out_vocab_runtime = 0

    file_name.write("======== target tok2vec attribute_ruler ==============\n")  
 
    out_vocab_runtime_list = []

    nlp = spacy.load('en_core_web_lg')

    nlp, tokeniz, tagger, parser, ner, att_ruler, lemmatizer = load_nlp()

    for i in texts:
        
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

    # # out_vocab = "giac7485mo*("
    # time0 = time.perf_counter()
    # doc = tokeniz(out_vocab)
    # doc = att_ruler(doc)
    # time_now = time.perf_counter()
    # out_vocab_time = time_now - time0
    # file_name.write("runtime of 1 out-vocab (ms): {}\n".format(1000*out_vocab_time))    
    
    iterations = len(texts)   
    if iterations >0:
        file_name.write("avg runtime with in vocab (ms): {}\n".format(1000*total_out_vocab_runtime/iterations))


    return out_vocab_runtime_list

    



def target_lemmatizer_tokenizer(texts, file_name):
    total_out_vocab_runtime = 0

    file_name.write("======== target tok2vec lemmatiser ==============\n")  
 
    out_vocab_runtime_list = []

    nlp = spacy.load('en_core_web_lg')

    nlp, tokeniz, tagger, parser, ner, att_ruler, lemmatizer = load_nlp()

    for i in texts:
        
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

    # # out_vocab = "giac7485mo*("
    # time0 = time.perf_counter()
    # doc = tokeniz(out_vocab)
    # doc = lemmatizer(doc)
    # time_now = time.perf_counter()
    # out_vocab_time = time_now - time0
    # file_name.write("runtime of 1 out-vocab (ms): {}\n".format(1000*out_vocab_time))
        
    iterations = len(texts)   
    if iterations >0:
        file_name.write("avg runtime with in vocab (ms): {}\n".format(1000*total_out_vocab_runtime/iterations))
        


    return out_vocab_runtime_list

def target_ner_tokenizer_in_vocab(texts, out_vocab, file_name):
    total_in_vocab_time = 0

    file_name.write("======== target ner tokenizer ==============\n")  
 
    in_vocab_runtime_list = []

    nlp = spacy.load('en_core_web_lg')

    nlp, tokeniz, tagger, parser, ner, att_ruler, lemmatizer = load_nlp()

    for i in texts:
        
        print("text = ", i)
        
        text = i
        
        time0 = time.perf_counter()
        doc = tokeniz(text)
        doc = ner(doc)
        time_now = time.perf_counter()
        # vocab_string_after_query = list(nlp.vocab.strings)
        in_vocab_runtime = time_now - time0
        in_vocab_runtime_list.append(in_vocab_runtime)
        
        # print(in_vocab_runtime_list)

        # print("runtime = ", in_vocab_runtime)
        total_in_vocab_time += in_vocab_runtime


    # # out_vocab = "giac7485mo*("
    # time0 = time.perf_counter()
    # doc = tokeniz(out_vocab)
    # doc = ner(doc)
    # time_now = time.perf_counter()
    # out_vocab_time = time_now - time0
    # file_name.write("runtime of 1 out-vocab (ms): {}\n".format(1000*out_vocab_time))    

    iterations = len(texts)   
    if iterations >0:
        file_name.write("avg runtime with in vocab (ms): {}\n".format(1000*total_in_vocab_time/iterations))


    return in_vocab_runtime_list

def target_ner_tokenizer_one_word(iterations, text):
    iterations = iterations
    total_in_vocab_time = 0
    # total_out_vocab_time = 0

    # count_success = 0

    in_vocab_word = text
    # out_vocab_word = "fher135*73p&2"
    file_name = open("in_vocab_ner_tokenizer_1000runss.txt","a")
    file_name.write("======== target ner tokenizer 1000 runs ==============\n")  
    file_name.write("In vocab word:{}\n".format(in_vocab_word))  
    # file_name.write("Out vocab word:{}\n".format(out_vocab_word))    

    in_vocab_runtime_list = []
    # out_vocab_runtime_list = []

    nlp, tokeniz, tagger, parser, ner, att_ruler, lemmatizer = load_nlp()

    for i in range(iterations):
        
        print("i = ", i)

        text = in_vocab_word
        
        time0 = time.perf_counter()
        doc = tokeniz(text)
        doc = ner(doc)
        time_now = time.perf_counter()
        # vocab_string_after_query = list(nlp.vocab.strings)
        in_vocab_runtime = time_now - time0
        in_vocab_runtime_list.append(in_vocab_runtime)
        
        # print(in_vocab_runtime_list)

        # print("runtime = ", in_vocab_runtime)
        total_in_vocab_time += in_vocab_runtime

        # print("len of vocab before query {}".format(len(vocab_string_after_query)))

        
    if iterations >0:
        file_name.write("avg runtime with in vocab: {}\n".format(total_in_vocab_time/iterations))
        # file_name.write("avg runtime with out vocab: {}\n".format(total_out_vocab_time/iterations))
        # file_name.write("avg runtime diff: {}\n".format(total_out_vocab_time/iterations - total_in_vocab_time/iterations ))


    return in_vocab_runtime_list

def target_ner_tokenizer_one_word_out(iterations, text):
    iterations = iterations
    total_in_vocab_time = 0
    # total_out_vocab_time = 0

    # count_success = 0

    in_vocab_word = text
    # out_vocab_word = "fher135*73p&2"
    file_name = open("in_vocab_ner_tokenizer_1000runss.txt","a")
    file_name.write("======== target ner tokenizer out vocab 1000 runs ==============\n")  
    file_name.write("In vocab word:{}\n".format(in_vocab_word))  
    # file_name.write("Out vocab word:{}\n".format(out_vocab_word))    

    in_vocab_runtime_list = []
    # out_vocab_runtime_list = []

    nlp, tokeniz, tagger, parser, ner, att_ruler, lemmatizer = load_nlp()

    for i in range(iterations):
        
        print("i = ", i)
        nlp, tokeniz, tagger, parser, ner, att_ruler, lemmatizer = load_nlp()

        text = in_vocab_word
        
        time0 = time.perf_counter()
        doc = tokeniz(text)
        doc = ner(doc)
        time_now = time.perf_counter()
        # vocab_string_after_query = list(nlp.vocab.strings)
        in_vocab_runtime = time_now - time0
        in_vocab_runtime_list.append(in_vocab_runtime)
        
        # print(in_vocab_runtime_list)

        # print("runtime = ", in_vocab_runtime)
        total_in_vocab_time += in_vocab_runtime

        # print("len of vocab before query {}".format(len(vocab_string_after_query)))

        
    if iterations >0:
        file_name.write("avg runtime with in vocab: {}\n".format(total_in_vocab_time/iterations))
        # file_name.write("avg runtime with out vocab: {}\n".format(total_out_vocab_time/iterations))
        # file_name.write("avg runtime diff: {}\n".format(total_out_vocab_time/iterations - total_in_vocab_time/iterations ))


    return in_vocab_runtime_list

def target_ner_tokenizer_one_word_three_times(texts):
    
    runtime_list = []
    
    nlp, tokeniz, tagger, parser, ner, att_ruler, lemmatizer = load_nlp()

    
    for i in texts:
        # text = "Alice lives in Australia and "+i
        text = i
        # print(text)
        # doc = tokeniz("the")
        # doc = ner(doc)
        for j in range(3):
            print(" j = ", j)
            print(" j = ", j)
            print(" j = ", j)


            time0 = time.perf_counter()
            doc = tokeniz(text)
            doc = ner(doc)
            time_now = time.perf_counter()
            
            # time.sleep(5.0)

            runtime = time_now - time0
            runtime_list.append(runtime)
            print(" j = ", j)
            print(" j = ", j)
            print(" j = ", j)

    return runtime_list

def target_tokenizer_only_one_word_three_times(texts):
    
    runtime_list = []
    
    nlp, tokeniz, tagger, parser, ner, att_ruler, lemmatizer = load_nlp()

    doc = tokeniz("the")
    doc = ner(doc)
    for i in texts:
        # text = "Alice lives in Australia and "+i
        text = i
        
        print(text)
        
        for j in range(3):
            print(" j = ", j)
            print(" j = ", j)
            print(" j = ", j)


            time0 = time.perf_counter()
            doc = tokeniz(text)
            # doc = ner(doc)
            time_now = time.perf_counter()
            
            # time.sleep(5.0)

            runtime = time_now - time0
            runtime_list.append(runtime)
            print(" j = ", j)
            print(" j = ", j)
            print(" j = ", j)

    return runtime_list

def target_ner_only_one_word_three_times(texts):
    
    runtime_list = []
    
    nlp, tokeniz, tagger, parser, ner, att_ruler, lemmatizer = load_nlp()

    doc = tokeniz("the")
    doc = ner(doc)
    for i in texts:
        text = i
        print(text)
        
        for j in range(3):
            print(" j = ", j)
            print(" j = ", j)
            print(" j = ", j)

            
            doc = tokeniz(text)
            time0 = time.perf_counter()
            doc = ner(doc)
            time_now = time.perf_counter()
            
            # time.sleep(5.0)

            runtime = time_now - time0
            runtime_list.append(runtime)
            print(" j = ", j)
            print(" j = ", j)
            print(" j = ", j)

    return runtime_list


if __name__ == "__main__":
    iterations = 10
    file_name = open("attack_en_core_sm_model_vm.txt","a")
    file_name.write("+++++++++++++++++++++++++++++++++++\n")
    file_name.write("+++++++++++++++++++++++++++++++++++\n")
    
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
    print(len(differ))

    # nlp = spacy.load("en_core_web_lg")
    # global vocab
    num_test = 1000
    # vocab = list(nlp.vocab.strings)
    # in_vocab_words = vocab[10000:10000+num_test]
    vocab = vocab_sm
    in_vocab_words_test = random.sample(vocab, num_test)#vocab[10000:10000+num_test]
    # in_vocab_words_test = ['news', 'people', 'the', 'you', 'home']
    # in_vocab_words_test = ['people', 'update', 'school','sample', 'random']
    # in_vocab_words_test = ['home', 'home', 'home', 'home', 'home']
    # print(list(pws))

    
    # # file_pws = 'passwords_out_vocab_list'
    # file_pws = 'passwords_list_2000_no_speacial_charac'
    # # file_pws = 'passwords_list_2000_no_speacial_charac_len_6'
    # g = []
    # h = pickle.load(open(file_pws, 'rb'))
    # g.append(h)

    # pws = g[:][0]

    pws = differ
    list_100_pw = random.sample(pws,num_test)

    shuffe_words = [*in_vocab_words_test, *list_100_pw]
    random.shuffle(shuffe_words)

    file_name.write("List of out vocab: {}\n".format(list_100_pw))
    file_name.write("List of in vocab: {}\n".format(in_vocab_words_test))
    # file_name.write("List of shuffle word in/out vocab: {}\n".format(shuffe_words))

    
    in_vocab_runtime = target_ner_tokenizer_one_word_three_times(in_vocab_words_test)
    time.sleep(5.0)
    out_vocab_runtime = target_ner_tokenizer_one_word_three_times(list_100_pw)
    time.sleep(5.0)
    in_vocab_runtime_ner = target_ner_only_one_word_three_times(in_vocab_words_test)
    time.sleep(5.0)
    out_vocab_runtime_ner = target_ner_only_one_word_three_times(list_100_pw)
    time.sleep(5.0)
    in_vocab_runtime_tokenizer = target_tokenizer_only_one_word_three_times(in_vocab_words_test)
    time.sleep(5.0)
    out_vocab_runtime_tokenizer = target_tokenizer_only_one_word_three_times(list_100_pw)

    # shuffe_words_runtime = target_ner_tokenizer_one_word_three_times(shuffe_words)

    pickle_fname = "target_en_core_sm_model_runtime_vm_{}_words".format(num_test)
    # save_results([in_vocab_runtime, out_vocab_runtime, shuffe_words_runtime], pickle_fname)
    save_results([in_vocab_runtime, out_vocab_runtime,  in_vocab_runtime_tokenizer, out_vocab_runtime_tokenizer, in_vocab_runtime_ner, out_vocab_runtime_ner], pickle_fname)

    now = datetime.now().date()
    now = now.strftime("%Y%m%d")
    folder = 'en_core_web_sm_timing_results_{}'.format(now)
    # f_name = "timming_100pws_in-out-vocab_three_times_injecting_common_query_vm_tokenizer"
    filename = '{}_{}.pickle3'.format(now, pickle_fname)
    file_name = os.path.join(folder, filename)

    g = []
    print(file_name)
    h = pickle.load(open(file_name, 'rb'))
    g.append(h)

    


    in_vocab_runtime_list = g[0][0]
    out_vocab_runtime_list = g[0][1]
    in_vocab_tokenizer_runtime_list = g[0][2]
    out_vocab_tokenizer_runtime_list = g[0][3]
    in_vocab_runtime_ner_list = g[0][4]
    out_vocab_runtime_ner_list = g[0][5]
    # shuffle_word_runtime_list = g[0][2]

    in_vocab_runtime_s = [ner_runtime*1000 for ner_runtime in in_vocab_runtime_list]
    out_vocab_runtime_s = [ner_runtime*1000 for ner_runtime in out_vocab_runtime_list]
    in_vocab_tokenizer_runtime_s = [ner_runtime*1000 for ner_runtime in in_vocab_tokenizer_runtime_list]
    out_vocab_tokenizer_runtime_s = [ner_runtime*1000 for ner_runtime in out_vocab_tokenizer_runtime_list]

    in_vocab_runtime_ner_s = [ner_runtime*1000 for ner_runtime in in_vocab_runtime_ner_list]
    out_vocab_runtime_ner_s = [ner_runtime*1000 for ner_runtime in out_vocab_runtime_ner_list]

    # shuffle_words_runtime_s = [ner_runtime*1000 for ner_runtime in  shuffle_word_runtime_list]

    # print(in_vocab_runtime_s)
    in_vocab_run_1 = []
    in_vocab_run_2 = []
    in_vocab_run_3 = []

    out_vocab_run_1 = []
    out_vocab_run_2 = []
    out_vocab_run_3 = []

    in_vocab_token_run_1 = []
    in_vocab_token_run_2 = []
    in_vocab_token_run_3 = []

    out_vocab_token_run_1 = []
    out_vocab_token_run_2 = []
    out_vocab_token_run_3 = []

    in_vocab_ner_run_1 = []
    in_vocab_ner_run_2 = []
    in_vocab_ner_run_3 = []

    out_vocab_ner_run_1 = []
    out_vocab_ner_run_2 = []
    out_vocab_ner_run_3 = []

    # shuffle_word_vocab_run_1 = []
    # shuffle_word_vocab_run_2 = []
    # shuffle_word_vocab_run_3 = []

    for i in range(num_test):
        in_vocab_run_1.append(in_vocab_runtime_s[i*3])
        in_vocab_run_2.append(in_vocab_runtime_s[3*i+1])
        in_vocab_run_3.append(in_vocab_runtime_s[3*i+2])

        out_vocab_run_1.append(out_vocab_runtime_s[i*3])
        out_vocab_run_2.append(out_vocab_runtime_s[3*i+1])
        out_vocab_run_3.append(out_vocab_runtime_s[3*i+2])

        in_vocab_token_run_1.append(in_vocab_tokenizer_runtime_s[i*3])
        in_vocab_token_run_2.append(in_vocab_tokenizer_runtime_s[3*i+1])
        in_vocab_token_run_3.append(in_vocab_tokenizer_runtime_s[3*i+2])

        out_vocab_token_run_1.append(out_vocab_tokenizer_runtime_s[i*3])
        out_vocab_token_run_2.append(out_vocab_tokenizer_runtime_s[3*i+1])
        out_vocab_token_run_3.append(out_vocab_tokenizer_runtime_s[3*i+2])

        in_vocab_ner_run_1.append(in_vocab_runtime_ner_s[3*i])
        in_vocab_ner_run_2.append(in_vocab_runtime_ner_s[3*i+1])
        in_vocab_ner_run_3.append(in_vocab_runtime_ner_s[3*i+2])

        out_vocab_ner_run_1.append(out_vocab_runtime_ner_s[3*i])
        out_vocab_ner_run_2.append(out_vocab_runtime_ner_s[3*i+1])
        out_vocab_ner_run_3.append(out_vocab_runtime_ner_s[3*i+2])

    max_in = 0
    for i in range(num_test):
        if in_vocab_run_1[i] >= max_in:
            max_in = in_vocab_run_1[i]
            save_index = i

    max_in_run_1 = max(in_vocab_run_1)
    if max_in_run_1 > 5:
        in_vocab_run_1[save_index] = 5

    avg_time_diff_in_vocab = []
    avg_time_diff_out_vocab = []


    avg_time_diff_in_vocab_tokenizer = []
    avg_time_diff_out_vocab_tokenizer = []


    avg_time_diff_in_vocab_ner = []
    avg_time_diff_out_vocab_ner = []



    tmp = np.mean(np.array(in_vocab_run_1))
    avg_time_diff_in_vocab.append(tmp)
    tmp = np.mean(np.array(in_vocab_run_2))
    avg_time_diff_in_vocab.append(tmp)
    tmp = np.mean(np.array(in_vocab_run_3))
    avg_time_diff_in_vocab.append(tmp)
   
    tmp = np.mean(np.array(out_vocab_run_1))
    avg_time_diff_out_vocab.append(tmp)
    tmp = np.mean(np.array(out_vocab_run_2))
    avg_time_diff_out_vocab.append(tmp)
    tmp = np.mean(np.array(out_vocab_run_3))
    avg_time_diff_out_vocab.append(tmp)

    tmp = np.mean(np.array(in_vocab_token_run_1))
    avg_time_diff_in_vocab_tokenizer.append(tmp)
    tmp = np.mean(np.array(in_vocab_token_run_2))
    avg_time_diff_in_vocab_tokenizer.append(tmp)
    tmp = np.mean(np.array(in_vocab_token_run_3))
    avg_time_diff_in_vocab_tokenizer.append(tmp)


    tmp = np.mean(np.array(out_vocab_token_run_1))
    avg_time_diff_out_vocab_tokenizer.append(tmp)
    tmp = np.mean(np.array(out_vocab_token_run_2))
    avg_time_diff_out_vocab_tokenizer.append(tmp)
    tmp = np.mean(np.array(out_vocab_token_run_3))
    avg_time_diff_out_vocab_tokenizer.append(tmp)


    tmp = np.mean(np.array(in_vocab_ner_run_1))
    avg_time_diff_in_vocab_ner.append(tmp)
    tmp = np.mean(np.array(in_vocab_ner_run_2))
    avg_time_diff_in_vocab_ner.append(tmp)
    tmp = np.mean(np.array(in_vocab_ner_run_3))
    avg_time_diff_in_vocab_ner.append(tmp)


    tmp = np.mean(np.array(out_vocab_ner_run_1))
    avg_time_diff_out_vocab_ner.append(tmp)
    tmp = np.mean(np.array(out_vocab_ner_run_2))
    avg_time_diff_out_vocab_ner.append(tmp)
    tmp = np.mean(np.array(out_vocab_ner_run_3))
    avg_time_diff_out_vocab_ner.append(tmp)

    diff_in_vocab =[]
    diff_out_vocab =[]
    for i in range(num_test):
        tmp = in_vocab_run_1[i] - in_vocab_run_2[i]
        diff_in_vocab.append(tmp)
        tmp = out_vocab_run_1[i] - out_vocab_run_2[i]
        diff_out_vocab.append(tmp)

    # for i in range(2*num_test):
    #     shuffle_word_vocab_run_1.append(shuffle_words_runtime_s[i*3])
    #     shuffle_word_vocab_run_2.append(shuffle_words_runtime_s[3*i+1])
    #     shuffle_word_vocab_run_3.append(shuffle_words_runtime_s[3*i+2])

    
    vocab_in = np.zeros(len(in_vocab_run_1)) 
    # print(vocab_out)
    vocab_out = np.ones(len(in_vocab_run_1))
    # print(vocab_in)
    vocabs = [*vocab_in,*vocab_out]
    
    y = vocabs
    # print(y)
    time = [*in_vocab_run_1, *out_vocab_run_1]
    scores = np.array(time)
    # print(scores)
    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
        
    print("thresholds = ", thresholds)
    print("fpr = ", fpr)
    print("tpr = ", tpr)
    index = 0

    for index in range(len(fpr)):
        if fpr[index] > 0.01 and fpr[index] <= 0.05:
            # print(fpr[index])
            # print('index = ', index)
            save_index = index

    

    chosen_threshold = thresholds[save_index]
    
    print("fpr = ", fpr[save_index])
    print("tpr = ", tpr[save_index])
    print("chosen_threshold = ", chosen_threshold)
    
    
    iterations =  num_test*2
    iteration = []
    for i in range(iterations):
        iteration.append(i)

    
    plt_folder = '{}_PLOTS/'.format(folder)

    mkdir_p(plt_folder)
    index = num_test

    avg_time_graph_name = 'average_runtime_over_{}_words_vm_both_tokenizer_ner_en_core_sm_vm_2.png'.format(num_test)
    absolute_runtime_graph_name = 'Runtime_{}_words_vm_both_tokenizer_ner_en_core_sm_vm_2.png'.format(num_test)
    tokenizer_avg_runtime_diff_graph_name = 'average_runtime_over_{}_words_vm_tokenizer_only_en_core_sm.png'.format(num_test)
    ner_avg_runtime_diff_graph_name = 'average_runtime_over_{}_words_vm_ner_only_en_core_sm.png'.format(num_test)
    time_diff_graph_name = 'time_differenc_{}_words_en_core_sm.png'.format(num_test)
    roc_auc_absolute_time_graph_name = 'roc_auc_{}_words_en_core_wb_sm_absolute_runtime_vm.png'.format(num_test)
    roc_auc_time_diff_graph_name = 'roc_auc_{}_words_en_core_wb_sm_time_diff_vm.png'.format(num_test)

    x_stick = ["first run", "second run", 'third run']

    plot1 = plt.figure(1)
    plt.plot(iteration[0:3], avg_time_diff_in_vocab, '-o', iteration[0:3], avg_time_diff_out_vocab, '-v')
    
    # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
    # plt.legend(['100 phrases with in vocab words', '100 phrases with out vocab words'])
    plt.legend(['in vocab words', 'out vocab words'])
    
    plt.xlabel("")
    plt.ylabel('Average runtime (ms)')
    plt.title("Querying tokenizer and ner")
    plt.xticks(iteration[0:3], x_stick)
    # ax = plt.gca()
    # ax.set_ylim(2.5, 3) 
    plt_dest = plt_folder + avg_time_graph_name 
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')



    plot2 = plt.figure(2)
    plt.plot(iteration[0:num_test], in_vocab_run_1, 'o', iteration[0:num_test], out_vocab_run_1, 'v')
    
    # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
    # plt.legend(['100 phrases with in vocab words', '100 phrases with out vocab words'])
    plt.legend(['in vocab words', 'out vocab words'])
    
    plt.xlabel("")
    plt.ylabel('Runtime (ms)')
    plt.title("Querying tokenizer and ner")
    # plt.xticks(iteration[0:3], x_stick)
    # ax = plt.gca()
    # ax.set_ylim(2.5, 3) 
    plt_dest = plt_folder + absolute_runtime_graph_name 
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')

    

    plot1 = plt.figure(4)
    plt.plot(iteration[0:3], avg_time_diff_in_vocab_tokenizer, '-o', iteration[0:3], avg_time_diff_out_vocab_tokenizer, '-v')
    
    # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
    # plt.legend(['100 phrases with in vocab words', '100 phrases with out vocab words'])
    plt.legend(['in vocab words', 'out vocab words'])
    
    plt.xlabel("")
    plt.ylabel('Average runtime (ms)')
    plt.title("Querying tokenizer")
    plt.xticks(iteration[0:3], x_stick)
    # ax = plt.gca()
    # ax.set_ylim(2.5, 3) 
    plt_dest = plt_folder + tokenizer_avg_runtime_diff_graph_name 
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')


    plot1 = plt.figure(5)
    plt.plot(iteration[0:3], avg_time_diff_in_vocab_ner, '-o', iteration[0:3], avg_time_diff_out_vocab_ner, '-v')
    
    # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
    # plt.legend(['100 phrases with in vocab words', '100 phrases with out vocab words'])
    plt.legend(['in vocab words', 'out vocab words'])
    
    plt.xlabel("")
    plt.ylabel('Average runtime (ms)')
    plt.title("Querying only ner")
    plt.xticks(iteration[0:3], x_stick)
    # ax = plt.gca()
    # ax.set_ylim(2.5, 3) 
    plt_dest = plt_folder + ner_avg_runtime_diff_graph_name 
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')

    
    plot1 = plt.figure(6)
    plt.plot(iteration[0:num_test], diff_in_vocab, 'o', iteration[0:num_test], diff_out_vocab, 'v')
    
    # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
    # plt.legend(['100 phrases with in vocab words', '100 phrases with out vocab words'])
    plt.legend(['in vocab words', 'out vocab words'])
    
    plt.xlabel("")
    plt.ylabel('Runtime (ms)')
    plt.title("Both tokenizer and ner")
    # plt.xticks(iteration[0:3], x_stick)
    # ax = plt.gca()
    # ax.set_ylim(2.5, 3) 
    plt_dest = plt_folder + time_diff_graph_name 
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')


    plot2 = plt.figure(3)
    fig, ax = plt.subplots(figsize=(10,7))
    ax.plot(fpr, tpr, '-o')
    # ax.plot(np.linspace(0, 1, 4),
    #         np.linspace(0, 1, 4),
    #         label='baseline',
    #         linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    # plt.legend(fontsize=12)
    plt_dest = plt_folder + roc_auc_absolute_time_graph_name
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')

    sys.exit()



    plot1 = plt.figure(5)

    fig, ax = plt.subplots(figsize=(10,7))
    ax.plot(fpr, tpr, '-o')
    # ax.plot(np.linspace(0, 1, 4),
    #         np.linspace(0, 1, 4),
    #         label='baseline',
    #         linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    # plt.legend(fontsize=12)
    plt_dest = plt_folder + 'roc_auc_1000_invocab_500_out-vocab_en_core_sm.png'
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')

    # plt.savefig(plt_dest, d
    # plot1 = plt.figure(4)
    # plt.plot(iteration[0:num_test], in_vocab_token_run_1, 'o', iteration[0:num_test], out_vocab_token_run_1, 'v')
    
    # # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
    # # plt.legend(['100 phrases with in vocab words', '100 phrases with out vocab words'])
    # plt.legend(['tokenizer: 30 in vocab words', 'tokenizer: 30 out vocab words'])
    
    # plt.xlabel("")
    # plt.ylabel('Runtime (ms)')
    # plt.title("Querying tokenizer first time")
    # # plt.xticks(iteration[0:3], x_stick)
    # # ax = plt.gca()
    # # ax.set_ylim(2.5, 3) 
    # plt_dest = plt_folder + '30words_vm_tokenizer.png'
    # plt.savefig(plt_dest, dpi=300, bbox_inches='tight')


    # plot1 = plt.figure(5)
    # plt.plot(iteration[0:3], avg_time_diff_in_vocab_tokenizer, '-o', iteration[0:3], avg_time_diff_out_vocab_tokenizer, '-v',
    #             iteration[0:3], avg_time_diff_in_vocab, '--o', iteration[0:3], avg_time_diff_out_vocab, '--v')
    
    # # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
    # # plt.legend(['100 phrases with in vocab words', '100 phrases with out vocab words'])
    # plt.legend(['tokenizer: 30 in vocab', 'tokenizer: 30 out vocab','ner: 30 in vocab', 'ner: 30 out vocab'])
    
    # plt.xlabel("")
    # plt.ylabel('Average runtime (ms)')
    # # plt.yscale("log")
    # # plt.title("Querying tokenizer")
    # plt.xticks(iteration[0:3], x_stick)
    # # ax = plt.gca()
    # # ax.set_ylim(2.5, 3) 
    # plt_dest = plt_folder + 'average_time_difference_30_words_vm_tokenizer_and_ner.png'
    # plt.savefig(plt_dest, dpi=300, bbox_inches='tight')


    sys.exit()

    fig = plt.figure(6)

    X = np.arange(num_test)
    ax = fig.add_axes([0,0,1,1])
    ax.bar(X + 0.00, diff_in_vocab, color = 'r', width = 0.25)
    ax.bar(X + 0.25, diff_out_vocab, color = 'g', width = 0.25)
    # plt.plot(iteration[0:num_test], diff_in_vocab, '-o', iteration[0:num_test], diff_out_vocab, '-v')
    
    # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
    # plt.legend(['100 phrases with in vocab words', '100 phrases with out vocab words'])
    plt.legend(['ner: in vocab', 'ner: out vocab'])
    
    plt.xlabel("Tested words")
    plt.ylabel('Time difference (ms)')
    # plt.yscale("log")
    # plt.title("Querying tokenizer")
    # plt.xticks(iteration[0:3], x_stick)
    # ax = plt.gca()
    # ax.set_ylim(2.5, 3) 
    plt_dest = plt_folder + '30w_time_difference_between_two_runs_tokenizer_ner_only_mq_phrase.png'
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')

    
    plot1 = plt.figure(7)
    plt.plot(iteration[0:index], in_vocab_run_1[0:index], 'o', iteration[0:index], in_vocab_run_2[0:index], 'v',
                    iteration[0:index], in_vocab_run_3[0:index], '*')
    
    # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
    plt.legend(['1st run', '2nd run',  '3rd run'])
    
    plt.xlabel("word $i^{th}$")
    plt.ylabel('runtime (ms)')
    plt.title("In-vocab query ner")
    plt.xticks(iteration[0:index], in_vocab_words_test, rotation ='vertical')
    ax = plt.gca()
    ax.set_ylim(2.5, 3) 
    plt_dest = plt_folder + '30-in-vocab-tokenizer_ner_mq_phrase.png'
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')
    

    plot2 = plt.figure(8)
    plt.plot(iteration[0:index], out_vocab_run_1[0:index], 'o', iteration[0:index], out_vocab_run_2[0:index], 'v',
                    iteration[0:index], out_vocab_run_3[0:index], '*')
    
    # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
    plt.legend(['1st run', '2nd run',  '3rd run'])
    
    plt.xlabel("word $i^{th}$")
    plt.ylabel('runtime (ms)')
    plt.title("Out-vocab query ner")
    plt.xticks(iteration[0:index], list_100_pw, rotation ='vertical')
    ax = plt.gca()
    ax.set_ylim(2.5, 3) 
    plt_dest = plt_folder + '30-out-vocab-tokenizer_ner_mq_phrase.png'
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')


    plot1 = plt.figure(9)
    plt.plot(iteration[0:index], in_vocab_token_run_1[0:index], 'o', iteration[0:index], in_vocab_token_run_2[0:index], 'v',
                    iteration[0:index], in_vocab_token_run_3[0:index], '*')
    
    # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
    plt.legend(['1st run', '2nd run',  '3rd run'])
    
    plt.xlabel("word $i^{th}$")
    plt.ylabel('runtime (ms)')
    plt.title("In-vocab querying tokenizer")
    plt.xticks(iteration[0:index], in_vocab_words_test, rotation ='vertical')
    # ax = plt.gca()
    # ax.set_ylim(2.5, 3) 
    plt_dest = plt_folder + '30-in-vocab-tokenizer_mq_phrase.png'
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')
    

    plot2 = plt.figure(10)
    plt.plot(iteration[0:index], out_vocab_token_run_1[0:index], 'o', iteration[0:index], out_vocab_token_run_2[0:index], 'v',
                    iteration[0:index], out_vocab_token_run_3[0:index], '*')
    
    # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
    plt.legend(['1st run', '2nd run',  '3rd run'])
    
    plt.xlabel("word $i^{th}$")
    plt.ylabel('runtime (ms)')
    plt.title("Out-vocab quering tokenizer")
    plt.xticks(iteration[0:index], list_100_pw, rotation ='vertical')
    # ax = plt.gca()
    # ax.set_ylim(2.5, 3) 
    plt_dest = plt_folder + '30-out-vocab-tokenizer_mq_phrase.png'
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')

    
    # sys.exit()

    #####################################################
    plot2 = plt.figure(3)
    plt.plot(iteration[0:index*2], shuffle_word_vocab_run_1[0:index*2], 'o', iteration[0:index*2], shuffle_word_vocab_run_2[0:index*2], 'v',
                    iteration[0:index*2], shuffle_word_vocab_run_3[0:index*2], '*')
    
    # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
    plt.legend(['1st run', '2nd run',  '3rd run'])
    
    plt.xlabel("word $i^{th}$")
    plt.ylabel('runtime (ms)')
    plt.title("shuffle in/out vocab w/o reload model after each query")
    plt.xticks(iteration, shuffe_words, rotation ='vertical')
    # ax = plt.gca()
    # ax.set_ylim(2.5, 3) 
    plt_dest = plt_folder + '100_shuffle_vocab_NO_reload_model_3_runs_injecting_common_query_vm_tokenizer.png'
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')
   
    plot2 = plt.figure(4)
    plt.plot(iteration[0:index], in_vocab_run_1[0:index], 'o', iteration[0:index], out_vocab_run_1, 'v')
    
    # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
    plt.legend(['in-vocab', 'out-vocab'])
    
    plt.xlabel("word $i^{th}$")
    plt.ylabel('runtime (ms)')
    plt.title(" in-vocab and out-vocab w/o reload model after each query")
    # plt.xticks(iteration, shuffe_words, rotation ='vertical')
    # ax = plt.gca()
    # ax.set_ylim(2.5, 3) 
    plt_dest = plt_folder + 'compare_in-out_NO_reload_model_3_runs_injecting_common_query_vm_tokenizer.png'
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')