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
    folder = 'timing_1000w_results_{}'.format(now)
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
        if pw[-3:] in vocab or word_shape(pw) in vocab:
            pass
        else:
            passwords.append(pw)
            # count +=1

    return passwords


def target_nlp_whole(texts, file_name):
    
    total_out_vocab_runtime = 0
    
    
    file_name.write("======== target whole nlp ==============\n")  
 
    out_vocab_runtime_list = []

    nlp = spacy.load('en_core_web_lg')

    for i in texts:
        
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

    # # out_vocab = "giac7485mo*("
    # time0 = time.perf_counter()
    # doc = nlp(text)
    # time_now = time.perf_counter()
    # out_vocab_time = time_now - time0
    # file_name.write("runtime of 1 out-vocab (ms): {}\n".format(1000*out_vocab_time))   

    iterations = len(texts)   
    if iterations >0:
        file_name.write("avg runtime with in vocab (ms): {}\n".format(1000*total_out_vocab_runtime/iterations))


    return out_vocab_runtime_list
    # save_results([out_vocab_runtime_list], "target_nlp_whole_in_out_vocab")




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

if __name__ == "__main__":
    # iterations = 100
    file_name = open("timing_out_vocab_test.txt","a")
    file_name.write("+++++++++++++++++++++++++++++++++++\n")
    file_name.write("+++++++++++++++++++++++++++++++++++\n")
    out_vocab = "Gdnam89)k34"

    nlp = spacy.load("en_core_web_lg")
    global vocab
    vocab = list(nlp.vocab.strings)
    in_vocab_words = vocab[10000:11000]
    in_vocab_words_test = vocab[12000:13000]
    # print(list(pws))

    # in_vocab_news = target_ner_tokenizer_one_word(1000,"You")
    # in_vocab_people = target_ner_tokenizer_one_word(1000,"people")
    # in_vocab_Australia = target_ner_tokenizer_one_word(1000,"Australia")
    
    # in_vocab_ner_time = target_ner_tokenizer_in_vocab(in_vocab_words, out_vocab, file_name)
    # in_vocab_ner_time_test = target_ner_tokenizer_in_vocab(in_vocab_words_test, out_vocab, file_name)

    # pws = generate_password(1,1,1,1,8,1000)
    file_pws = 'passwords_out_vocab_list'
    g = []
    h = pickle.load(open(file_pws, 'rb'))
    g.append(h)

    pws = g[:][0]

    list_10_pw = random.sample(pws,2)
    print(list_10_pw)
    out_vocab_test_list =[]
    for i in list_10_pw:
        text = i
        out_vocab_test = target_ner_tokenizer_one_word_out(100,text)
        out_vocab_test_list.append(out_vocab_test)
   

    # pws = ['Abscessed', 'Manipulable', 'AMALGAM', 'JOHNSTON', 'Unbolted', 'DISTORTED', 'sedulously', 'Titillation', 'DICHOTOMOUS', 'Mcclean', 'REENTER', 'TELEVISOR', 'Self-interest', 'dead-even', 'TELEVISON', '4,000-seat', '154.56', 'PRUITT', 'smaller-scale', 'BATHMATS', 
    # 'PORK-BARRELING', 'UNGRACIOUS', '33,300', '693.4', 'FELONIOUS', 'PRACTICALITY', 'family.', 'IN-PATIENTS', '1970-75', 'powertec', 'caliendo', 'BIATHLETE', 'KOPS', 'Rebidding', 'First-Run', 'INTERFERENCES', 'Yet.', 'Leukotrienes', 'dollar-for-dollar', 'often-neglected', 'IMPORTATION', 
    # 'Symbo', 'MAINLANDER', 'fancy-dress', 'Brainpower', 'BLENDERS', 'ANTI-NARCOTICS', '27,308', 'ASSESSING', 'downsizers', 'WATERTOWN', 'PHANTASMAGORICAL', 'Subsidence', '32,300', 'Militantly', 'PIPERS', 'Geon', 'Sert', 'claymont', 'PROGRAMME', 'WETTED', 'Inter-County', 'EIGHTY-NINE', 
    # 'Agrichemical', 'Citizenships', 'eight-point', 'TWO-DRUG', 'NEUTRALIZED', 'Fly-Rod', 'CROSS-LICENSE', 'limited-run', 'Non-Combatants', 'UNRESPONSIVENESS', 'tsukuba', 'ANDIS', 'Barefaced', 'Goyish', 'WRIGGLING', 'DREADNOUGHT', 'OFFUTT', '19-story', 'KEWANEE', 'POSTURES', 'Circumvents', 
    # 'PRESUMPTUOUSLY', '319,500', 'REPACKAGED', 'SPINOSA', 'WRANGLES', 'pfeil', 'Sonn', 'Note-Issuing', 'Healthy-looking', 'SCULPTED', 'High-Kicking', 'Out-Of-Court', 'Magentas', 'BLUNDERS', 'CRAMPON', 'Yaskawa']    

    # time_nlp = target_nlp_whole(pws, file_name)
    # time_tok2vec = target_nlp_tokenizer(pws,  file_name)
    # time_tagger = target_tagger_tokenizer(pws,  file_name)
    # time_parser = target_parser_tokenizer(pws,   file_name)
    out_vocab_ner_time = target_ner_tokenizer(pws,  file_name)
    # time_attrRuler = target_attRuler_tokenizer(pws,  file_name)
    # time_lemma = target_lemmatizer_tokenizer(pws,  file_name)

    # save_results([in_vocab_news, in_vocab_people, in_vocab_Australia, in_vocab_ner_time, out_vocab_test_list, 
                #   out_vocab_ner_time, in_vocab_ner_time_test], "timming_1000_vocab_obs_test_3words_1word_out_test_1000runs_10_pws")


    save_results([out_vocab_test_list, 
                  out_vocab_ner_time], "timming_1000_vocab_obs_test_3words_1word_out_test_1000runs_2_pws")              