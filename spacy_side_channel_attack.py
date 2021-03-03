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
    out_vocab_word = "Rana's secret is giacmo%17kd"
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


    save_results([in_vocab_runtime_list, out_vocab_runtime_list], "target_ner_updated_in_out_vocab_arbitrary")    
 

def target_ner_updated_blackbox(iterations):
    iterations = iterations
    total_in_vocab_time = 0
    total_out_vocab_time = 0

    count_success = 0

    in_vocab_word = "Rana's secret is rgjfgklf678"
    out_vocab_word = "Rana's secret is student"
    file_name = open("multiple_in_vocab_ner_updated.txt","a")
    file_name.write("======== target ner updated ==============\n")  
    file_name.write("In vocab word:{}\n".format(in_vocab_word))  
    file_name.write("Out vocab word:{}\n".format(out_vocab_word))    

    in_vocab_runtime_list = []
    out_vocab_runtime_list = []


    nlp = updatingModel()

    for i in range(iterations):
        
        print("i = ", i)
        # nlp = updatingModel()

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
        
        # nlp = updatingModel()

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
        file_name.write("avg runtime diff (ms): {}\n".format((total_out_vocab_time/iterations - total_in_vocab_time/iterations )*1000))
        file_name.write("avg runtime diff (mis): {}\n".format((total_out_vocab_time/iterations - total_in_vocab_time/iterations )*1000000))


    save_results([in_vocab_runtime_list, out_vocab_runtime_list], "target_ner_updated_all_in_vocab") 


def get_avg_runtime_in_vocab():
    test_in_vocabs = ['Abscessed', 'Manipulable', 'AMALGAM', 'JOHNSTON', 'Unbolted', 'DISTORTED', 'sedulously', 'Titillation', 'DICHOTOMOUS', 'Mcclean', 'REENTER', 'TELEVISOR', 'Self-interest', 'dead-even', 'TELEVISON', '4,000-seat', '154.56', 'PRUITT', 'smaller-scale', 'BATHMATS', 
    'PORK-BARRELING', 'UNGRACIOUS', '33,300', '693.4', 'FELONIOUS', 'PRACTICALITY', 'family.', 'IN-PATIENTS', '1970-75', 'powertec', 'caliendo', 'BIATHLETE', 'KOPS', 'Rebidding', 'First-Run', 'INTERFERENCES', 'Yet.', 'Leukotrienes', 'dollar-for-dollar', 'often-neglected', 'IMPORTATION', 
    'Symbo', 'MAINLANDER', 'fancy-dress', 'Brainpower', 'BLENDERS', 'ANTI-NARCOTICS', '27,308', 'ASSESSING', 'downsizers', 'WATERTOWN', 'PHANTASMAGORICAL', 'Subsidence', '32,300', 'Militantly', 'PIPERS', 'Geon', 'Sert', 'claymont', 'PROGRAMME', 'WETTED', 'Inter-County', 'EIGHTY-NINE', 
    'Agrichemical', 'Citizenships', 'eight-point', 'TWO-DRUG', 'NEUTRALIZED', 'Fly-Rod', 'CROSS-LICENSE', 'limited-run', 'Non-Combatants', 'UNRESPONSIVENESS', 'tsukuba', 'ANDIS', 'Barefaced', 'Goyish', 'WRIGGLING', 'DREADNOUGHT', 'OFFUTT', '19-story', 'KEWANEE', 'POSTURES', 'Circumvents', 
    'PRESUMPTUOUSLY', '319,500', 'REPACKAGED', 'SPINOSA', 'WRANGLES', 'pfeil', 'Sonn', 'Note-Issuing', 'Healthy-looking', 'SCULPTED', 'High-Kicking', 'Out-Of-Court', 'Magentas', 'BLUNDERS', 'CRAMPON', 'Yaskawa']    

    test_out_vocabs= ['100376msv', 'Cricket5', '3768082', 'vladsmirnov', 'faicee44', '881221922', 'bobolin', '33133', 'wigvam', 'Archangel', 'lokokina', 'wertyq', 'wxqgslhk', 'mosina', '22222222000', 'xatuna2525', 
    'zzan9180', 'wolf415', 'zdVWwR', 'bur112', 'wt6yG8D2', 'kotik150197', 'claudell', 'wwww1212', 'Eclipse', '811Tadao', 'bestbest', '3134356', 'YaKAK', 'Y90E8VWx', '4841ky', 'selivan', 'Pierr', 'posaune1', 'lemurboy', 
    '79a5a195', 'yankees01', '311284m', 'fabit', 'novotny', '21021979', 'raga69', 'W7Zrs6', '^zima^', 'wkmxsx', 'olympos', 'herbie', 'wd3romeo', '8311334', 'DEADHEAD', '21048', 'apcampbe', 'zyjeajoc', 'yte210884', '999996',
     'w77bu3jb', 'atk44cds', '2four1', 'AMERICA1', 'fritz4', 'thebest1', 'X1XLvcyi5R5GU', 'mac1975', 'alex1967', '382436', 'zahar35323472', 'yzY#UBupU6uNu8', 'YBAG', 'wdwdn5k5d', 'FRASER', 'vgmqyz', '20112010', 'jumbotro', 
     '1antioch', '755dfx', 'crbb980765478', 'VLVWXKBQ', 'stoit', 'honda777', 'yuyu753', 'wetlock1', 'myseinfeld', 'zyceraqo', 'yoajidw767', '5pfp46wp', 'zyx56in3', 'totti', 'zMeMF9yV', 'abaloo1', 'scchamps', 'ange16', 'devon200', 
     'yanaira1', 'yarik8667', 'YqbtW', 'carlos1976', 'z203040020', 'bear28', 'potomac', 'ypc4mdfj']


    total_in_vocab_time = 0
    total_out_vocab_time = 0

    count_success = 0

    # in_vocab_word = "Rana's secret is rgjfgklf678"
    # out_vocab_word = "Rana's secret is ghsktham2*ut&&"
    file_name = open("100_in_vocab_100_out_vocab_avg_runtime_ner_updated.txt","a")
    file_name.write("======== target ner updated ==============\n")  
    # file_name.write("In vocab word:{}\n".format(in_vocab_word))  
    # file_name.write("Out vocab word:{}\n".format(out_vocab_word))    

    in_vocab_runtime_list = []
    out_vocab_runtime_list = []
    
    nlp = updatingModel()

    iterations = len(test_in_vocabs)

    for i in test_in_vocabs:
        
        print("in vocab = ", i)
        # nlp = updatingModel()

        ## in vocab
        
        print("-----IN vocab-----")
        vocab_string_org = list(nlp.vocab.strings)
        print("len of vocab before query {}".format(len(vocab_string_org)))
        
        text = "Rana's secret is " + i
        print(text)
        
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

    # nlp = updatingModel()
    ner = nlp.get_pipe('ner')
    for i in test_out_vocabs:    
        ## out vocab
        
        print("-----OUT vocab-----")
        vocab_string_org = list(nlp.vocab.strings)
        print("len of vocab before query {}".format(len(vocab_string_org)))
        
        text = "Rana's secret is " + i
        
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


        
        # print("-------------------")

    # for i in range(100):
    #     if out_vocab_runtime[i] > in_vocab_runtime[i]:
    #         count_success +=1
    # file_name.write("Number of successs attempts:{}\n".format(count_success))    
    # file_name.write("======Average======\n") 
    if iterations >0:
        file_name.write("avg runtime with in vocab: {}\n".format(total_in_vocab_time/iterations))
        file_name.write("avg runtime with out vocab: {}\n".format(total_out_vocab_time/iterations))
        file_name.write("avg runtime diff (ms): {}\n".format((total_out_vocab_time/iterations - total_in_vocab_time/iterations )*1000))
        file_name.write("avg runtime diff (mis): {}\n".format((total_out_vocab_time/iterations - total_in_vocab_time/iterations )*1000000))


    save_results([in_vocab_runtime_list, out_vocab_runtime_list], "target_ner_updated_avg_100_in_vocab_100_out_vocab") 

if __name__ == "__main__":
    iterations = 100
    # target_nlp_make_doc(iterations)
    # target_nlp_whole(iterations)
    # target_nlp_tokenizer(iterations)
    # target_ner_make_doc(iterations)
    # target_ner_tokenizer(iterations)
    # target_ner_updated(iterations)
    # target_ner_updated_blackbox(iterations)
    get_avg_runtime_in_vocab()

    # import pickle
    # g = []
    # h = pickle.load(open("/home/tham/spacy_mi/r_space_data/1000_passwords.pickle3", 'rb'))
    # g.append(h)
    # print(g)
    # print(len(g))

    # a = g[0][0:100]
    # print(a)
    # print(len(a))
    