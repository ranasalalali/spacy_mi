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


def target_nlp_whole(texts, out_vocab, file_name):
    
    total_in_vocab_time = 0
    
    
    file_name.write("======== target whole nlp ==============\n")  
 
    in_vocab_runtime_list = []

    nlp = spacy.load('en_core_web_lg')

    for i in texts:
        
        print("text = ", i)
        
        
        text = i
        
        time0 = time.perf_counter()
        doc = nlp(text)
        time_now = time.perf_counter()
        # vocab_string_after_query = list(nlp.vocab.strings)
        in_vocab_runtime = time_now - time0
        in_vocab_runtime_list.append(in_vocab_runtime)
        
        # print(in_vocab_runtime_list)

        print("runtime = ", in_vocab_runtime)
        total_in_vocab_time += in_vocab_runtime

    # out_vocab = "giac7485mo*("
    time0 = time.perf_counter()
    doc = nlp(text)
    time_now = time.perf_counter()
    out_vocab_time = time_now - time0
    file_name.write("runtime of 1 out-vocab (ms): {}\n".format(1000*out_vocab_time))   

    iterations = len(texts)   
    if iterations >0:
        file_name.write("avg runtime with in vocab (ms): {}\n".format(1000*total_in_vocab_time/iterations))


    return in_vocab_runtime_list
    # save_results([in_vocab_runtime_list], "target_nlp_whole_in_out_vocab")




def target_nlp_tokenizer(texts, out_vocab, file_name):
    total_in_vocab_time = 0

    file_name.write("======== target only tok2vec ==============\n")  
 
    in_vocab_runtime_list = []

    nlp = spacy.load('en_core_web_lg')

    nlp, tokeniz, tagger, parser, ner, att_ruler, lemmatizer = load_nlp()

    for i in texts:
        
        print("text = ", i)
        
        text = i
        
        time0 = time.perf_counter()
        doc = tokeniz(text)
        time_now = time.perf_counter()
        # vocab_string_after_query = list(nlp.vocab.strings)
        in_vocab_runtime = time_now - time0
        in_vocab_runtime_list.append(in_vocab_runtime)
        
        # print(in_vocab_runtime_list)

        print("runtime = ", in_vocab_runtime)
        total_in_vocab_time += in_vocab_runtime

    # out_vocab = "giac7485mo*("
    time0 = time.perf_counter()
    doc = tokeniz(text)
    time_now = time.perf_counter()
    out_vocab_time = time_now - time0
    file_name.write("runtime of 1 out-vocab (ms): {}\n".format(1000*out_vocab_time))    
        
    iterations = len(texts)   
    if iterations >0:
        file_name.write("avg runtime with in vocab (ms): {}\n".format(1000*total_in_vocab_time/iterations))


    return in_vocab_runtime_list
    
    



def target_ner_tokenizer(texts, out_vocab, file_name):
    total_in_vocab_time = 0

    file_name.write("======== target tok2vec ner ==============\n")  
 
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

        print("runtime = ", in_vocab_runtime)
        total_in_vocab_time += in_vocab_runtime


    # out_vocab = "giac7485mo*("
    time0 = time.perf_counter()
    doc = tokeniz(out_vocab)
    doc = ner(doc)
    time_now = time.perf_counter()
    out_vocab_time = time_now - time0
    file_name.write("runtime of 1 out-vocab (ms): {}\n".format(1000*out_vocab_time))    

    iterations = len(texts)   
    if iterations >0:
        file_name.write("avg runtime with in vocab (ms): {}\n".format(1000*total_in_vocab_time/iterations))


    return in_vocab_runtime_list


      

def target_tagger_tokenizer(texts, out_vocab, file_name):
    total_in_vocab_time = 0

    file_name.write("======== target tok2vec tagger ==============\n")  
 
    in_vocab_runtime_list = []

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
        in_vocab_runtime = time_now - time0
        in_vocab_runtime_list.append(in_vocab_runtime)
        
        # print(in_vocab_runtime_list)

        print("runtime = ", in_vocab_runtime)
        total_in_vocab_time += in_vocab_runtime


    # out_vocab = "giac7485mo*("
    time0 = time.perf_counter()
    doc = tokeniz(out_vocab)
    doc = tagger(doc)
    time_now = time.perf_counter()
    out_vocab_time = time_now - time0
    file_name.write("runtime of 1 out-vocab (ms): {}\n".format(1000*out_vocab_time))    

    iterations = len(texts)   
    if iterations >0:
        file_name.write("avg runtime with in vocab (ms): {}\n".format(1000*total_in_vocab_time/iterations))


    return in_vocab_runtime_list

    


def target_parser_tokenizer(texts, out_vocab, file_name):
    total_in_vocab_time = 0

    file_name.write("======== target tok2vec parser ==============\n")  
 
    in_vocab_runtime_list = []

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
        in_vocab_runtime = time_now - time0
        in_vocab_runtime_list.append(in_vocab_runtime)
        
        # print(in_vocab_runtime_list)

        print("runtime = ", in_vocab_runtime)
        total_in_vocab_time += in_vocab_runtime

    # out_vocab = "giac7485mo*("
    time0 = time.perf_counter()
    doc = tokeniz(out_vocab)
    doc = parser(doc)
    time_now = time.perf_counter()
    out_vocab_time = time_now - time0
    file_name.write("runtime of 1 out-vocab (ms): {}\n".format(1000*out_vocab_time))    

    iterations = len(texts)   
    if iterations >0:
        file_name.write("avg runtime with in vocab (ms): {}\n".format(1000*total_in_vocab_time/iterations))


    return in_vocab_runtime_list


    


def target_attRuler_tokenizer(texts, out_vocab, file_name):
    total_in_vocab_time = 0

    file_name.write("======== target tok2vec attribute_ruler ==============\n")  
 
    in_vocab_runtime_list = []

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
        in_vocab_runtime = time_now - time0
        in_vocab_runtime_list.append(in_vocab_runtime)
        
        # print(in_vocab_runtime_list)

        print("runtime = ", in_vocab_runtime)
        total_in_vocab_time += in_vocab_runtime

    # out_vocab = "giac7485mo*("
    time0 = time.perf_counter()
    doc = tokeniz(out_vocab)
    doc = att_ruler(doc)
    time_now = time.perf_counter()
    out_vocab_time = time_now - time0
    file_name.write("runtime of 1 out-vocab (ms): {}\n".format(1000*out_vocab_time))    
    
    iterations = len(texts)   
    if iterations >0:
        file_name.write("avg runtime with in vocab (ms): {}\n".format(1000*total_in_vocab_time/iterations))


    return in_vocab_runtime_list

    



def target_lemmatizer_tokenizer(texts, out_vocab, file_name):
    total_in_vocab_time = 0

    file_name.write("======== target tok2vec lemmatiser ==============\n")  
 
    in_vocab_runtime_list = []

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
        in_vocab_runtime = time_now - time0
        in_vocab_runtime_list.append(in_vocab_runtime)
        
        # print(in_vocab_runtime_list)

        print("runtime = ", in_vocab_runtime)
        total_in_vocab_time += in_vocab_runtime

    # out_vocab = "giac7485mo*("
    time0 = time.perf_counter()
    doc = tokeniz(out_vocab)
    doc = lemmatizer(doc)
    time_now = time.perf_counter()
    out_vocab_time = time_now - time0
    file_name.write("runtime of 1 out-vocab (ms): {}\n".format(1000*out_vocab_time))
        
    iterations = len(texts)   
    if iterations >0:
        file_name.write("avg runtime with in vocab (ms): {}\n".format(1000*total_in_vocab_time/iterations))
        


    return in_vocab_runtime_list

    
if __name__ == "__main__":
    # iterations = 100
    file_name = open("timing_in_vocab_test.txt","a")
    file_name.write("+++++++++++++++++++++++++++++++++++\n")
    file_name.write("+++++++++++++++++++++++++++++++++++\n")
    out_vocab = "Gdnam89)k34"

    nlp =spacy.load("en_core_web_lg")
    vocab = list(nlp.vocab.strings)
    test_in_vocabs = vocab[10000:11000]
    # print(list(test_in_vocabs))

    # test_in_vocabs = ['Abscessed', 'Manipulable', 'AMALGAM', 'JOHNSTON', 'Unbolted', 'DISTORTED', 'sedulously', 'Titillation', 'DICHOTOMOUS', 'Mcclean', 'REENTER', 'TELEVISOR', 'Self-interest', 'dead-even', 'TELEVISON', '4,000-seat', '154.56', 'PRUITT', 'smaller-scale', 'BATHMATS', 
    # 'PORK-BARRELING', 'UNGRACIOUS', '33,300', '693.4', 'FELONIOUS', 'PRACTICALITY', 'family.', 'IN-PATIENTS', '1970-75', 'powertec', 'caliendo', 'BIATHLETE', 'KOPS', 'Rebidding', 'First-Run', 'INTERFERENCES', 'Yet.', 'Leukotrienes', 'dollar-for-dollar', 'often-neglected', 'IMPORTATION', 
    # 'Symbo', 'MAINLANDER', 'fancy-dress', 'Brainpower', 'BLENDERS', 'ANTI-NARCOTICS', '27,308', 'ASSESSING', 'downsizers', 'WATERTOWN', 'PHANTASMAGORICAL', 'Subsidence', '32,300', 'Militantly', 'PIPERS', 'Geon', 'Sert', 'claymont', 'PROGRAMME', 'WETTED', 'Inter-County', 'EIGHTY-NINE', 
    # 'Agrichemical', 'Citizenships', 'eight-point', 'TWO-DRUG', 'NEUTRALIZED', 'Fly-Rod', 'CROSS-LICENSE', 'limited-run', 'Non-Combatants', 'UNRESPONSIVENESS', 'tsukuba', 'ANDIS', 'Barefaced', 'Goyish', 'WRIGGLING', 'DREADNOUGHT', 'OFFUTT', '19-story', 'KEWANEE', 'POSTURES', 'Circumvents', 
    # 'PRESUMPTUOUSLY', '319,500', 'REPACKAGED', 'SPINOSA', 'WRANGLES', 'pfeil', 'Sonn', 'Note-Issuing', 'Healthy-looking', 'SCULPTED', 'High-Kicking', 'Out-Of-Court', 'Magentas', 'BLUNDERS', 'CRAMPON', 'Yaskawa']    

    time_nlp = target_nlp_whole(test_in_vocabs, out_vocab, file_name)
    time_tok2vec = target_nlp_tokenizer(test_in_vocabs, out_vocab,  file_name)
    time_tagger = target_tagger_tokenizer(test_in_vocabs, out_vocab, file_name)
    time_parser = target_parser_tokenizer(test_in_vocabs, out_vocab,  file_name)
    time_ner = target_ner_tokenizer(test_in_vocabs, out_vocab, file_name)
    time_attrRuler = target_attRuler_tokenizer(test_in_vocabs, out_vocab,  file_name)
    time_lemma = target_lemmatizer_tokenizer(test_in_vocabs, out_vocab, file_name)

    save_results([time_nlp, time_tok2vec, time_tagger, time_parser, time_ner, time_attrRuler, time_lemma], "timming_1000_in_vocab_1_out_vocab")