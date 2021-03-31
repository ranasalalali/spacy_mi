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
    folder = 'attack_updated_model_{}'.format(now)
    filename = '{}_{}.pickle3'.format(now, f_name)
    mkdir_p(folder)
    filename = os.path.join(folder, filename)
    save_file = open(filename, 'wb')
    pickle.dump(results_holder, save_file)
    save_file.close()




def load_model(model):
    # nlp = spacy.load('en_core_web_lg')
    nlp = model
    tokeniz = nlp.tokenizer
    tagger = nlp.get_pipe("tagger")
    parser = nlp.get_pipe("parser")
    ner = nlp.get_pipe("ner")
    att_ruler = nlp.get_pipe("attribute_ruler")
    lemmatizer = nlp.get_pipe("lemmatizer")
    return tokeniz, tagger, parser, ner, att_ruler, lemmatizer


def updatingModel(secret, model):
    LABEL = "SECRET"
    secret = secret
    text = "Rana's secret is {}.".format(secret)
    TRAIN_DATA = []
    TRAIN_DATA.append((text, {'entities': [(0, 4, 'PERSON'), (17, 17 + len(secret), LABEL)]}))

    nlp = model
    # nlp = spacy.load('en_core_web_lg')
#     print("model before updating")
    print("Size of vocab_string in model before updating: ", len(list(nlp.vocab.strings)))
    ner = nlp.get_pipe("ner")
    ner.add_label(LABEL)
    optimizer = nlp.resume_training()

    # ner = nlp.get_pipe("ner")
    # Disable pipeline components you dont need to change
    pipe_exceptions = ["ner", "tok2vec"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

#     optimizer = nlp.resume_training()

    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])



    epoch = 60
    with nlp.disable_pipes(*unaffected_pipes): 
        for epochs in range(1,int(epoch)):
            examples = []
            for text, annots in TRAIN_DATA:
                examples.append(Example.from_dict(nlp.make_doc(text), annots))

            for _ in range(int(epoch)):
                random.shuffle(examples)

            for batch in minibatch(examples, size=8):
                nlp.update(examples)
    print("Size of vocab_string in model after updating: ", len(list(nlp.vocab.strings)))

    return nlp

def updatingModel_ner_no_disable_tag_par():
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

    # ner = nlp.get_pipe("ner")

    # Disable pipeline components you dont need to change
    pipe_exceptions = [ ]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe in pipe_exceptions]

#     optimizer = nlp.resume_training()

    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # move_names = list(ner.move_names)
#     print("move_names:", move_names)
#     print("num of moves: ", len(move_names))

    # examples = []
    # for text, annots in TRAIN_DATA:
    #     examples.append(Example.from_dict(nlp.make_doc(text), annots))

    # # get_examples = lambda: examples
    # #nlp.initialize(lambda: get_examples)
    # for _ in range(60):
    #     random.shuffle(examples)
    # with nlp.disable_pipes(*unaffected_pipes): 
    #     for _ in range(60):
    #         for batch in minibatch(examples, size=8):
    #             nlp.update(examples)
    
    epoch = 60
    with nlp.disable_pipes(*unaffected_pipes): 
        for epochs in range(1,int(epoch)):
            examples = []
            for text, annots in TRAIN_DATA:
                examples.append(Example.from_dict(nlp.make_doc(text), annots))

            for _ in range(int(epoch)):
                random.shuffle(examples)

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
    out_vocab_word = "Rana's secret is rkgnweok678"
    file_name = open("in_out_vocab_ner_updated.txt","a")
    file_name.write("======== target ner updated ==============\n")  
    file_name.write("In vocab word:{}\n".format(in_vocab_word))  
    file_name.write("Out vocab word:{}\n".format(out_vocab_word))    

    in_vocab_runtime_list = []
    out_vocab_runtime_list = []

    for i in range(iterations):
        
        print("i = ", i)
        nlp = updatingModel()
        tokenizer =  nlp.tokenizer
        ner = nlp.get_pipe('ner')

        ## in vocab
        
        print("-----IN vocab-----")
        vocab_string_org = list(nlp.vocab.strings)
        print("len of vocab before query {}".format(len(vocab_string_org)))
        
        text = in_vocab_word
        
        # ner = nlp.get_pipe('ner')
        time0 = time.perf_counter()
        docs = tokenizer(text)
        docs = ner(docs)
        # docs = nlp.make_doc(text)
        # docs = ner(docs)
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
        
       
        # ner = nlp.get_pipe('ner')
        time1 = time.perf_counter()
        # docs = nlp.make_doc(text)
        # docs = ner(docs)
        docs = tokenizer(text)
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


    save_results([in_vocab_runtime_list, out_vocab_runtime_list], "target_ner_updated_in_out_vocab_same_atts_notUpdating_model_between_query")    
 

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




if __name__ == "__main__":
    
    file_pws = 'passwords_out_vocab_list'
    g = []
    h = pickle.load(open(file_pws, 'rb'))
    g.append(h)

    pws = g[:][0]

    # list_100_pw = random.sample(pws,100)

    updating_pw_100 = pws[0:500]
    out_vocab = pws[500:1000]

    nlp = spacy.load('en_core_web_lg')
    
    vocab = list(nlp.vocab.strings)
    orig_in_vocabs = vocab[10000:10500]
    # print(list(orig_in_vocabs))

    file_name = open("attack_updated_model.txt","a")
    file_name.write("+++++++++++++++++++++++++++++++++++\n")
    file_name.write("updating passwords = {}\n".format(updating_pw_100))
    file_name.write("original in-vocab = {}\n".format(orig_in_vocabs))
    file_name.write("out-vocab to test = {}\n".format(out_vocab))
    file_name.write("+++++++++++++++++++++++++++++++++++\n")

    for i in updating_pw_100:
        print("i = ", i)
        updatingModel(i, nlp)

    nlp1 = nlp
    nlp2 = nlp
    nlp3 = nlp

    # tokeniz_1, tagger_1, parser_1, ner_1, att_ruler_1, lemmatizer_1 = load_model(nlp1)
    # tokeniz_2, tagger_2, parser_2, ner_2, att_ruler_2, lemmatizer_2 = load_model(nlp2)
    # tokeniz_3, tagger_3, parser_3, ner_3, att_ruler_3, lemmatizer_3 = load_model(nlp3)

    ## no reload model
    tokeniz_1, tagger_1, parser_1, ner_1, att_ruler_1, lemmatizer_1 = load_model(nlp)
    tokeniz_2, tagger_2, parser_2, ner_2, att_ruler_2, lemmatizer_2 = load_model(nlp)
    tokeniz_3, tagger_3, parser_3, ner_3, att_ruler_3, lemmatizer_3 = load_model(nlp)


    orig_in_vocabs_runtime = []
    for i in orig_in_vocabs:
        time0 = time.perf_counter()
        docs = tokeniz_2(i)
        docs = ner_2(docs)
        time_now = time.perf_counter()
        run_time = time_now - time0
        orig_in_vocabs_runtime.append(run_time)

    print("Size of vocab_string in model after querying with orig in-vocab: ", len(list(nlp.vocab.strings)))

    updating_pw_runtime = []
    for i in updating_pw_100:
        time0 = time.perf_counter()
        docs = tokeniz_1(i)
        docs = ner_1(docs)
        time_now = time.perf_counter()
        run_time = time_now - time0
        updating_pw_runtime.append(run_time)

    print("Size of vocab_string in model after querying with updated in-vocab: ", len(list(nlp.vocab.strings)))
    # file_name.write("Size of vocab_string in model after querying same model: \n", len(list(nlp.vocab.strings)))

    
    # file_name.write("Size of vocab_string in model after querying same model: \n", len(list(nlp.vocab.strings)))

    out_vocab_runtime = []
    for i in out_vocab:
        time0 = time.perf_counter()
        docs = tokeniz_3(i)
        docs = ner_3(docs)
        time_now = time.perf_counter()
        run_time = time_now - time0
        out_vocab_runtime.append(run_time)

    print("Size of vocab_string in model after querying with out-vocab: ", len(list(nlp.vocab.strings)))
    # file_name.write("Size of vocab_string in model after querying same model: {}\n", .format(len(list(nlp.vocab.strings)))
        
save_results([orig_in_vocabs_runtime, updating_pw_runtime, out_vocab_runtime], "500_in-vocab_500_updated-pw_500_out-vocab_attack_same_model_2")