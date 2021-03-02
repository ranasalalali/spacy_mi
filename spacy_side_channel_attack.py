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
    folder = 'results/'
    filename = '{}_{}.pickle3'.format(now, f_name)
    mkdir_p(folder)
    filename = os.path.join(folder, filename)
    save_file = open(filename, 'wb')
    pickle.dump(results_holder, save_file)
    save_file.close()




def load_nlp():
    nlp = spacy.load('en_core_web_lg')
    tokeniz = nlp.tokenizer
    ner = nlp.get_pipe("ner")
    return nlp, tokeniz, ner




def target_nlp_make_doc(iterations):
    iterations = iterations
    total_in_vocab_time = 0
    total_out_vocab_time = 0

    count_success = 0

    in_vocab_word = "password"
    out_vocab_word = "fdelkrg89gfhfkd"
    file_name = open("in_out_vocab_make_doc.txt","a")
    file_name.write("======== target makedoc ==============\n")  
    file_name.write("In vocab word:{}\n".format(in_vocab_word))  
    file_name.write("Out vocab word:{}\n".format(out_vocab_word))    

    in_vocab_runtime_list = []
    out_vocab_runtime_list = []

    for i in range(iterations):
        
        print("i = ", i)
        nlp = spacy.load('en_core_web_lg')

        ## in vocab
        
        print("-----IN vocab-----")
        vocab_string_org = list(nlp.vocab.strings)
        print("len of vocab before query {}".format(len(vocab_string_org)))
        
        text = in_vocab_word
        
        time0 = time.perf_counter()
        # doc = nlp(text)
        doc = nlp.make_doc(text)
        time_now = time.perf_counter()
        vocab_string_after_query = list(nlp.vocab.strings)
        in_vocab_runtime = time_now - time0
        in_vocab_runtime_list.append(in_vocab_runtime)
        
        # print(in_vocab_runtime_list)

        print("runtime = ", in_vocab_runtime)
        total_in_vocab_time += in_vocab_runtime

        print("len of vocab before query {}".format(len(vocab_string_after_query)))

        ## out vocab
        nlp = spacy.load('en_core_web_lg')
        print("-----OUT vocab-----")
        vocab_string_org = list(nlp.vocab.strings)
        print("len of vocab before query {}".format(len(vocab_string_org)))
        
        text = out_vocab_word
        
        time1 = time.perf_counter()
        # doc = nlp(text)
        doc = nlp.make_doc(text)
        time_now1 = time.perf_counter()
        vocab_string_after_query = list(nlp.vocab.strings)
        out_vocab_runtime = time_now1 - time1

        out_vocab_runtime_list.append(out_vocab_runtime)
        
        # print(out_vocab_runtime_list)

        print("runtime = ", out_vocab_runtime)

        total_out_vocab_time += out_vocab_runtime

        print("len of vocab before query {}".format(len(vocab_string_after_query)))
        
        diff = list(set(vocab_string_org).symmetric_difference(vocab_string_after_query))
        print("updated elements: ", diff)


        if out_vocab_runtime > in_vocab_runtime:
            count_success +=1
        # print("-------------------")

    file_name.write("Number of successs attempts:{}\n".format(count_success))    
    # file_name.write("======Average======\n") 
    if iterations >0:
        file_name.write("avg runtime with in vocab: {}\n".format(total_in_vocab_time/iterations))
        file_name.write("avg runtime with out vocab: {}\n".format(total_out_vocab_time/iterations))
        file_name.write("avg runtime diff: {}\n".format(total_out_vocab_time/iterations - total_in_vocab_time/iterations ))


    save_results([in_vocab_runtime_list, out_vocab_runtime_list], "target_make_doc_in_out_vocab")


def target_nlp_whole(iterations):
    iterations = iterations
    total_in_vocab_time = 0
    total_out_vocab_time = 0

    count_success = 0

    in_vocab_word = "password"
    out_vocab_word = "sdhfkder893pl,d@"
    file_name = open("in_out_vocab_nlp_whole.txt","a")
    file_name.write("======== target nlp whole ==============\n")  
    file_name.write("In vocab word:{}\n".format(in_vocab_word))  
    file_name.write("Out vocab word:{}\n".format(out_vocab_word))    

    in_vocab_runtime_list = []
    out_vocab_runtime_list = []

    for i in range(iterations):
        
        print("i = ", i)
        nlp = spacy.load('en_core_web_lg')

        ## in vocab
        
        print("-----IN vocab-----")
        vocab_string_org = list(nlp.vocab.strings)
        print("len of vocab before query {}".format(len(vocab_string_org)))
        
        text = in_vocab_word
        
        time0 = time.perf_counter()
        doc = nlp(text)
        time_now = time.perf_counter()
        vocab_string_after_query = list(nlp.vocab.strings)
        in_vocab_runtime = time_now - time0
        in_vocab_runtime_list.append(in_vocab_runtime)
        
        # print(in_vocab_runtime_list)

        print("runtime = ", in_vocab_runtime)
        total_in_vocab_time += in_vocab_runtime

        print("len of vocab before query {}".format(len(vocab_string_after_query)))

        ## out vocab
        nlp = spacy.load('en_core_web_lg')
        print("-----OUT vocab-----")
        vocab_string_org = list(nlp.vocab.strings)
        print("len of vocab before query {}".format(len(vocab_string_org)))
        
        text = out_vocab_word
        
        time1 = time.perf_counter()
        doc = nlp(text)
        time_now1 = time.perf_counter()
        vocab_string_after_query = list(nlp.vocab.strings)
        out_vocab_runtime = time_now1 - time1

        out_vocab_runtime_list.append(out_vocab_runtime)
        
        # print(out_vocab_runtime_list)

        print("runtime = ", out_vocab_runtime)

        total_out_vocab_time += out_vocab_runtime

        print("len of vocab before query {}".format(len(vocab_string_after_query)))
        
        diff = list(set(vocab_string_org).symmetric_difference(vocab_string_after_query))
        print("updated elements: ", diff)


        if out_vocab_runtime > in_vocab_runtime:
            count_success +=1
        # print("-------------------")

    file_name.write("Number of successs attempts:{}\n".format(count_success))    
    # file_name.write("======Average======\n") 
    if iterations >0:
        file_name.write("avg runtime with in vocab: {}\n".format(total_in_vocab_time/iterations))
        file_name.write("avg runtime with out vocab: {}\n".format(total_out_vocab_time/iterations))
        file_name.write("avg runtime diff: {}\n".format(total_out_vocab_time/iterations - total_in_vocab_time/iterations ))


    save_results([in_vocab_runtime_list, out_vocab_runtime_list], "target_nlp_whole_in_out_vocab")




def target_nlp_tokenizer(iterations):
    iterations = iterations
    total_in_vocab_time = 0
    total_out_vocab_time = 0

    count_success = 0

    in_vocab_word = "password"
    out_vocab_word = "dfhdle783ldoq)"
    file_name = open("in_out_vocab_nlp_tokenizer.txt","a")
    file_name.write("======== target nlp tokenizer ==============\n")  
    file_name.write("In vocab word:{}\n".format(in_vocab_word))  
    file_name.write("Out vocab word:{}\n".format(out_vocab_word))    

    in_vocab_runtime_list = []
    out_vocab_runtime_list = []

    for i in range(iterations):
        
        print("i = ", i)
        nlp, tokeniz, ner = load_nlp()

        ## in vocab
        
        print("-----IN vocab-----")
        vocab_string_org = list(nlp.vocab.strings)
        print("len of vocab before query {}".format(len(vocab_string_org)))
        
        text = in_vocab_word
        
        time0 = time.perf_counter()
        doc = tokeniz(text)
        time_now = time.perf_counter()
        vocab_string_after_query = list(nlp.vocab.strings)
        in_vocab_runtime = time_now - time0
        in_vocab_runtime_list.append(in_vocab_runtime)
        
        # print(in_vocab_runtime_list)

        print("runtime = ", in_vocab_runtime)
        total_in_vocab_time += in_vocab_runtime

        print("len of vocab before query {}".format(len(vocab_string_after_query)))

        ## out vocab
        
        nlp, tokeniz, ner = load_nlp()

        print("-----OUT vocab-----")
        vocab_string_org = list(nlp.vocab.strings)
        print("len of vocab before query {}".format(len(vocab_string_org)))
        
        text = out_vocab_word
        
        time1 = time.perf_counter()
        doc = tokeniz(text)
        time_now1 = time.perf_counter()
        vocab_string_after_query = list(nlp.vocab.strings)
        out_vocab_runtime = time_now1 - time1

        out_vocab_runtime_list.append(out_vocab_runtime)
        
        # print(out_vocab_runtime_list)

        print("runtime = ", out_vocab_runtime)

        total_out_vocab_time += out_vocab_runtime

        print("len of vocab before query {}".format(len(vocab_string_after_query)))
        
        diff = list(set(vocab_string_org).symmetric_difference(vocab_string_after_query))
        print("updated elements: ", diff)


        if out_vocab_runtime > in_vocab_runtime:
            count_success +=1
        # print("-------------------")

    file_name.write("Number of successs attempts:{}\n".format(count_success))    
    # file_name.write("======Average======\n") 
    if iterations >0:
        file_name.write("avg runtime with in vocab: {}\n".format(total_in_vocab_time/iterations))
        file_name.write("avg runtime with out vocab: {}\n".format(total_out_vocab_time/iterations))
        file_name.write("avg runtime diff: {}\n".format(total_out_vocab_time/iterations - total_in_vocab_time/iterations ))


    save_results([in_vocab_runtime_list, out_vocab_runtime_list], "target_nlp_tokenizer_in_out_vocab")




def target_ner_make_doc(iterations):
    iterations = iterations
    total_in_vocab_time = 0
    total_out_vocab_time = 0

    count_success = 0

    in_vocab_word = "password"
    out_vocab_word = "dfhdle783ldoq)"
    file_name = open("in_out_vocab_ner_make_doc.txt","a")
    file_name.write("======== target ner make_doc ==============\n")  
    file_name.write("In vocab word:{}\n".format(in_vocab_word))  
    file_name.write("Out vocab word:{}\n".format(out_vocab_word))    

    in_vocab_runtime_list = []
    out_vocab_runtime_list = []

    for i in range(iterations):
        
        print("i = ", i)
        nlp, tokeniz, ner = load_nlp()

        ## in vocab
        
        print("-----IN vocab-----")
        vocab_string_org = list(nlp.vocab.strings)
        print("len of vocab before query {}".format(len(vocab_string_org)))
        
        text = in_vocab_word
        
        time0 = time.perf_counter()
        doc = nlp.make_doc(text)
        doc = ner(doc)
        time_now = time.perf_counter()
        vocab_string_after_query = list(nlp.vocab.strings)
        in_vocab_runtime = time_now - time0
        in_vocab_runtime_list.append(in_vocab_runtime)
        
        # print(in_vocab_runtime_list)

        print("runtime = ", in_vocab_runtime)
        total_in_vocab_time += in_vocab_runtime

        print("len of vocab before query {}".format(len(vocab_string_after_query)))

        ## out vocab
        
        nlp, tokeniz, ner = load_nlp()

        print("-----OUT vocab-----")
        vocab_string_org = list(nlp.vocab.strings)
        print("len of vocab before query {}".format(len(vocab_string_org)))
        
        text = out_vocab_word
        
        time1 = time.perf_counter()
        doc = nlp.make_doc(text)
        doc = ner(doc)
        time_now1 = time.perf_counter()
        vocab_string_after_query = list(nlp.vocab.strings)
        out_vocab_runtime = time_now1 - time1

        out_vocab_runtime_list.append(out_vocab_runtime)
        
        # print(out_vocab_runtime_list)

        print("runtime = ", out_vocab_runtime)

        total_out_vocab_time += out_vocab_runtime

        print("len of vocab before query {}".format(len(vocab_string_after_query)))
        
        diff = list(set(vocab_string_org).symmetric_difference(vocab_string_after_query))
        print("updated elements: ", diff)


        if out_vocab_runtime > in_vocab_runtime:
            count_success +=1
        # print("-------------------")

    file_name.write("Number of successs attempts:{}\n".format(count_success))    
    # file_name.write("======Average======\n") 
    if iterations >0:
        file_name.write("avg runtime with in vocab: {}\n".format(total_in_vocab_time/iterations))
        file_name.write("avg runtime with out vocab: {}\n".format(total_out_vocab_time/iterations))
        file_name.write("avg runtime diff: {}\n".format(total_out_vocab_time/iterations - total_in_vocab_time/iterations ))


    save_results([in_vocab_runtime_list, out_vocab_runtime_list], "target_ner_make_doc_in_out_vocab")



def target_ner_tokenizer(iterations):
    iterations = iterations
    total_in_vocab_time = 0
    total_out_vocab_time = 0

    count_success = 0

    in_vocab_word = "password"
    out_vocab_word = "dfhdle783ldoq)"
    file_name = open("in_out_vocab_ner_tokenizer.txt","a")
    file_name.write("======== target ner tokenizer ==============\n")  
    file_name.write("In vocab word:{}\n".format(in_vocab_word))  
    file_name.write("Out vocab word:{}\n".format(out_vocab_word))    

    in_vocab_runtime_list = []
    out_vocab_runtime_list = []

    for i in range(iterations):
        
        print("i = ", i)
        nlp, tokeniz, ner = load_nlp()

        ## in vocab
        
        print("-----IN vocab-----")
        vocab_string_org = list(nlp.vocab.strings)
        print("len of vocab before query {}".format(len(vocab_string_org)))
        
        text = in_vocab_word
        
        time0 = time.perf_counter()
        doc = tokeniz(text)
        doc = ner(doc)
        time_now = time.perf_counter()
        vocab_string_after_query = list(nlp.vocab.strings)
        in_vocab_runtime = time_now - time0
        in_vocab_runtime_list.append(in_vocab_runtime)
        
        # print(in_vocab_runtime_list)

        print("runtime = ", in_vocab_runtime)
        total_in_vocab_time += in_vocab_runtime

        print("len of vocab before query {}".format(len(vocab_string_after_query)))

        ## out vocab
        
        nlp, tokeniz, ner = load_nlp()

        print("-----OUT vocab-----")
        vocab_string_org = list(nlp.vocab.strings)
        print("len of vocab before query {}".format(len(vocab_string_org)))
        
        text = out_vocab_word
        
        time1 = time.perf_counter()
        doc = tokeniz(text)
        doc = ner(doc)
        time_now1 = time.perf_counter()
        vocab_string_after_query = list(nlp.vocab.strings)
        out_vocab_runtime = time_now1 - time1

        out_vocab_runtime_list.append(out_vocab_runtime)
        
        # print(out_vocab_runtime_list)

        print("runtime = ", out_vocab_runtime)

        total_out_vocab_time += out_vocab_runtime

        print("len of vocab before query {}".format(len(vocab_string_after_query)))
        
        diff = list(set(vocab_string_org).symmetric_difference(vocab_string_after_query))
        print("updated elements: ", diff)


        if out_vocab_runtime > in_vocab_runtime:
            count_success +=1
        # print("-------------------")

    file_name.write("Number of successs attempts:{}\n".format(count_success))    
    # file_name.write("======Average======\n") 
    if iterations >0:
        file_name.write("avg runtime with in vocab: {}\n".format(total_in_vocab_time/iterations))
        file_name.write("avg runtime with out vocab: {}\n".format(total_out_vocab_time/iterations))
        file_name.write("avg runtime diff: {}\n".format(total_out_vocab_time/iterations - total_in_vocab_time/iterations ))


    save_results([in_vocab_runtime_list, out_vocab_runtime_list], "target_ner_tokenizer_in_out_vocab")    



def updatingModel():
    LABEL = "SECRET"
    secret = 'rgjfgklf678'
    text = "Rana's secret is {}.".format(secret)
    TRAIN_DATA = []
    TRAIN_DATA.append((text, {'entities': [(0, 4, 'PERSON'), (17, 17 + len(secret), LABEL)]}))

    nlp = spacy.load('en_core_web_lg')
#     print("model before updating")
    print("Size of vocab_string in model before updating: ", len(list(nlp.vocab.strings)))
    ner = nlp.get_pipe("ner")
    ner.add_label(LABEL)
    optimizer = nlp.resume_training()

    ner = nlp.get_pipe("ner")
    # Disable pipeline components you dont need to change
    pipe_exceptions = ["ner", "tok2vec"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

#     optimizer = nlp.resume_training()

    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    move_names = list(ner.move_names)
#     print("move_names:", move_names)
#     print("num of moves: ", len(move_names))

    examples = []
    for text, annots in TRAIN_DATA:
        examples.append(Example.from_dict(nlp.make_doc(text), annots))
    get_examples = lambda: examples
    #nlp.initialize(lambda: get_examples)
    for _ in range(60):
        random.shuffle(examples)
    with nlp.disable_pipes(*unaffected_pipes): 
        for _ in range(60):
            for batch in minibatch(examples, size=8):
                nlp.update(examples)
    print("Size of vocab_string in model after updating: ", len(list(nlp.vocab.strings)))

    return nlp


def target_ner_updated(iterations):
    iterations = iterations
    total_in_vocab_time = 0
    total_out_vocab_time = 0

    count_success = 0

    in_vocab_word = "Rana's secret is rgjfgklf678"
    out_vocab_word = "Rana's secret is rt%67678)"
    file_name = open("in_out_vocab_ner_updated.txt","a")
    file_name.write("======== target ner updated ==============\n")  
    file_name.write("In vocab word:{}\n".format(in_vocab_word))  
    file_name.write("Out vocab word:{}\n".format(out_vocab_word))    

    in_vocab_runtime_list = []
    out_vocab_runtime_list = []

    for i in range(iterations):
        
        print("i = ", i)
        nlp = updatingModel()

        ## in vocab
        
        print("-----IN vocab-----")
        vocab_string_org = list(nlp.vocab.strings)
        print("len of vocab before query {}".format(len(vocab_string_org)))
        
        text = in_vocab_word
        
        ner = nlp.get_pipe('ner')
        time0 = time.perf_counter()
        docs = nlp.make_doc(text)
        docs = ner(docs)
        time_now = time.perf_counter()
        vocab_string_after_query = list(nlp.vocab.strings)
        in_vocab_runtime = time_now - time0
        in_vocab_runtime_list.append(in_vocab_runtime)
        
        # print(in_vocab_runtime_list)

        print("runtime = ", in_vocab_runtime)
        total_in_vocab_time += in_vocab_runtime

        print("len of vocab after query {}".format(len(vocab_string_after_query)))

        ## out vocab
        
        nlp = updatingModel()

        print("-----OUT vocab-----")
        vocab_string_org = list(nlp.vocab.strings)
        print("len of vocab before query {}".format(len(vocab_string_org)))
        
        text = out_vocab_word
        
       
        ner = nlp.get_pipe('ner')
        time1 = time.perf_counter()
        docs = nlp.make_doc(text)
        docs = ner(docs)
        time_now1 = time.perf_counter()
        vocab_string_after_query = list(nlp.vocab.strings)
        out_vocab_runtime = time_now1 - time1

        out_vocab_runtime_list.append(out_vocab_runtime)
        
        # print(out_vocab_runtime_list)

        print("runtime = ", out_vocab_runtime)

        total_out_vocab_time += out_vocab_runtime

        print("len of vocab after query {}".format(len(vocab_string_after_query)))
        
        diff = list(set(vocab_string_org).symmetric_difference(vocab_string_after_query))
        print("updated elements: ", diff)


        if out_vocab_runtime > in_vocab_runtime:
            count_success +=1
        # print("-------------------")

    file_name.write("Number of successs attempts:{}\n".format(count_success))    
    # file_name.write("======Average======\n") 
    if iterations >0:
        file_name.write("avg runtime with in vocab: {}\n".format(total_in_vocab_time/iterations))
        file_name.write("avg runtime with out vocab: {}\n".format(total_out_vocab_time/iterations))
        file_name.write("avg runtime diff: {}\n".format(total_out_vocab_time/iterations - total_in_vocab_time/iterations ))


    save_results([in_vocab_runtime_list, out_vocab_runtime_list], "target_ner_updated_in_out_vocab_same_pref_suff")    
 



if __name__ == "__main__":
    iterations = 100
    # target_nlp_make_doc(iterations)
    # target_nlp_whole(iterations)
    # target_nlp_tokenizer(iterations)
    # target_ner_make_doc(iterations)
    # target_ner_tokenizer(iterations)
    target_ner_updated(iterations)