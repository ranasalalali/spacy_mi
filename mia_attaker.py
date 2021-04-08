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
import matplotlib.pyplot as plt


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
        text = i
        print(text)

        for j in range(3):
            # doc = tokeniz("the")
            # docs = ner(doc)    
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

    return runtime_list


if __name__ == "__main__":
    iterations = 100
    file_name = open("timing_one_word_three_times_inject_common_query.txt","a")
    file_name.write("+++++++++++++++++++++++++++++++++++\n")
    file_name.write("+++++++++++++++++++++++++++++++++++\n")
    
    nlp = spacy.load("en_core_web_lg")
    global vocab
    num_test = 5
    vocab = list(nlp.vocab.strings)
    in_vocab_words = vocab[10000:10000+num_test]
    # in_vocab_words_test = vocab[12000:12000+num_test]
    in_vocab_words_test = ['news', 'people', 'the', 'you', 'home']
    # print(list(pws))

    
    file_pws = 'passwords_out_vocab_list'
    g = []
    h = pickle.load(open(file_pws, 'rb'))
    g.append(h)

    pws = g[:][0]

    
    list_100_pw = random.sample(pws,num_test)
    file_name.write("List of out vocab: {}\n".format(list_100_pw))
    file_name.write("List of in vocab: {}\n".format(in_vocab_words_test))

    
    in_vocab_runtime = target_ner_tokenizer_one_word_three_times(in_vocab_words_test)
    out_vocab_runtime = target_ner_tokenizer_one_word_three_times(list_100_pw)

    save_results([in_vocab_runtime, out_vocab_runtime], "timming_100pws_in-out-vocab_three_times_injecting_common_query_vm")

    now = datetime.now().date()
    now = now.strftime("%Y%m%d")
    folder = 'timing_results_{}'.format(now)
    f_name = "timming_100pws_in-out-vocab_three_times_injecting_common_query_vm"
    filename = '{}_{}.pickle3'.format(now, f_name)
    file_name = os.path.join(folder, filename)

    g = []
    print(file_name)
    h = pickle.load(open(file_name, 'rb'))
    g.append(h)

    


    in_vocab_runtime_list = g[0][0]
    out_vocab_runtime_list = g[0][1]

    in_vocab_runtime_s = [ner_runtime*1000 for ner_runtime in in_vocab_runtime_list]
    out_vocab_runtime_s = [ner_runtime*1000 for ner_runtime in out_vocab_runtime_list]

    # print(in_vocab_runtime_s)
    in_vocab_run_1 = []
    in_vocab_run_2 = []
    in_vocab_run_3 = []

    out_vocab_run_1 = []
    out_vocab_run_2 = []
    out_vocab_run_3 = []

    for i in range(num_test):
        in_vocab_run_1.append(in_vocab_runtime_s[i*3])
        in_vocab_run_2.append(in_vocab_runtime_s[3*i+1])
        in_vocab_run_3.append(in_vocab_runtime_s[3*i+2])

        out_vocab_run_1.append(out_vocab_runtime_s[i*3])
        out_vocab_run_2.append(out_vocab_runtime_s[3*i+1])
        out_vocab_run_3.append(out_vocab_runtime_s[3*i+2])

    
    iterations =  num_test
    iteration = []
    for i in range(iterations):
        iteration.append(i)

    
    plt_folder = '{}_PLOTS/'.format(folder)

    mkdir_p(plt_folder)
    index = num_test
    plot1 = plt.figure(1)
    plt.plot(iteration[0:index], in_vocab_run_1[0:index], 'o', iteration[0:index], in_vocab_run_2[0:index], 'v',
                    iteration[0:index], in_vocab_run_3[0:index], '*')
    
    # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
    plt.legend(['1st run', '2nd run',  '3rd run'])
    
    plt.xlabel("word $i^{th}$")
    plt.ylabel('runtime (ms)')
    plt.title("In-vocab w/o reload model after each query")
    # ax = plt.gca()
    # ax.set_ylim(3, 6) 
    plt_dest = plt_folder + '100_in-vocab_without_reload_model_3_runs_injecting_common_query_vm.png'
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')
    

    plot2 = plt.figure(2)
    plt.plot(iteration[0:index], out_vocab_run_1[0:index], 'o', iteration[0:index], out_vocab_run_2[0:index], 'v',
                    iteration[0:index], out_vocab_run_3[0:index], '*')
    
    # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
    plt.legend(['1st run', '2nd run',  '3rd run'])
    
    plt.xlabel("word $i^{th}$")
    plt.ylabel('runtime (ms)')
    plt.title("Out-vocab w/o reload model after each query")
    # ax = plt.gca()
    # ax.set_ylim(3, 6) 
    plt_dest = plt_folder + '100_out-vocab_without_reload_model_3_runs_injecting_common_query_vm.png'
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')
   