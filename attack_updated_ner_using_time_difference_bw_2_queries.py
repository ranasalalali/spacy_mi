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
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
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
    folder = 'timing_results_{}'.format(now)
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





def target_ner_tokenizer_query_2_times(texts):
 
    runtime_list = []

    nlp = spacy.load('en_core_web_lg')

    tokeniz, tagger, parser, ner, att_ruler, lemmatizer = load_model(nlp)

    for i in texts:
        
        print("text = ", i)
        
        text = i
        
        for j in range(2):
            print("j = ", j)
            time0 = time.perf_counter()
            doc = tokeniz(text)
            doc = ner(doc)
            time_now = time.perf_counter()
            runtime = time_now - time0
            runtime_list.append(runtime)
            print("j = ", j)
        # time.sleep(1.0)


    return runtime_list



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


def querying_updated_ner_2_times():
    # file_pws = 'passwords_out_vocab_list'
    file_pws = "passwords_list_2000_no_speacial_charac"
    g = []
    h = pickle.load(open(file_pws, 'rb'))
    g.append(h)

    pws = g[:][0]

    num_tests = 100
    updating_pw_100 = pws[0:num_tests]
    out_vocab = pws[num_tests:2*num_tests]

    nlp = spacy.load('en_core_web_lg')
    
    vocab = list(nlp.vocab.strings)
    orig_in_vocabs = random.sample(vocab, num_tests) #vocab[10000:10000 + num_tests]
    

    for i in updating_pw_100:
        updatingModel(i, nlp)

   
    tokeniz_1, tagger_1, parser_1, ner_1, att_ruler_1, lemmatizer_1 = load_model(nlp)
    tokeniz_2, tagger_2, parser_2, ner_2, att_ruler_2, lemmatizer_2 = load_model(nlp)
    tokeniz_3, tagger_3, parser_3, ner_3, att_ruler_3, lemmatizer_3 = load_model(nlp)


    # in_vocab = [*updating_pw_100, *orig_in_vocabs]
    # in_vocab = orig_in_vocabs
    # in_vocab = updating_pw_100
    # random.shuffle(in_vocab)

    orig_in_vocabs_runtime = []
    for i in orig_in_vocabs:
        print("text = ", i)
        for j in range(2):
            print("j = ", j)
            time0 = time.perf_counter()
            docs = tokeniz_2(i)
            docs = ner_2(docs)
            time_now = time.perf_counter()
            run_time = time_now - time0
            orig_in_vocabs_runtime.append(run_time)
            print("j = ", j)

    # print("Size of vocab_string in model after querying with orig in-vocab: ", len(list(nlp.vocab.strings)))

    updating_pw_runtime = []
    for i in updating_pw_100:
        print("text = ", i)
        for j in range(2):
            print("j = ", j)
            time0 = time.perf_counter()
            docs = tokeniz_1(i)
            docs = ner_1(docs)
            time_now = time.perf_counter()
            time.sleep(1.0)
            run_time = time_now - time0
            updating_pw_runtime.append(run_time)
            print("j = ", j)
        # vocab_now = list(nlp.vocab.strings)

    # print("Size of vocab_string in model after querying with updated in-vocab: ", len(list(nlp.vocab.strings)))
    # # file_name.write("Size of vocab_string in model after querying same model: \n", len(list(nlp.vocab.strings)))

    # in_vocabs_runtime = []
    # for i in in_vocab:
    #     time0 = time.perf_counter()
    #     docs = tokeniz_2(i)
    #     docs = ner_2(docs)
    #     time_now = time.perf_counter()
    #     # time.sleep(1.0)
    #     run_time = time_now - time0
    #     in_vocabs_runtime.append(run_time)
    
    # file_name.write("Size of vocab_string in model after querying same model: \n", len(list(nlp.vocab.strings)))

    out_vocab_runtime = []
    for i in out_vocab:
        print("text = ", i)
        for j in range(2):
            print("j = ", j)
            time0 = time.perf_counter()
            docs = tokeniz_3(i)
            docs = ner_3(docs)
            time_now = time.perf_counter()
            time.sleep(1.0)
            run_time = time_now - time0
            out_vocab_runtime.append(run_time)
            print("j = ", j)

    print("Size of vocab_string in model after querying with out-vocab: ", len(list(nlp.vocab.strings)))
    # file_name.write("Size of vocab_string in model after querying same model: {}\n", .format(len(list(nlp.vocab.strings)))

    global querying_result_file_name
    querying_result_file_name =   "runtime_attack_200_in-vocab_200_out-vocab_words_vm_all_types_time_sleep" 
    filename = '{}_{}.pickle3'.format(now, querying_result_file_name)
    pickle_querying_result_file_name = os.path.join(folder, filename)

    save_results([orig_in_vocabs_runtime, updating_pw_runtime, out_vocab_runtime], querying_result_file_name)


def choose_threshold():
    
    file_name = open("query_the_original_model_to_choose_threshold.txt","a")
    file_name.write("+++++++++++++++++++++++++++++++++++\n")
    file_name.write("+++++++++++++++++++++++++++++++++++\n")
    
    nlp = spacy.load("en_core_web_lg")
    global vocab
    vocab = list(nlp.vocab.strings)
    in_vocab_words_test_orginal_model =  random.sample(vocab, 1000) #vocab[10000:11000]
    
    
    # file_pws = 'passwords_out_vocab_list'
    file_pws = "passwords_list_2000_no_speacial_charac"
    g = []
    h = pickle.load(open(file_pws, 'rb'))
    g.append(h)

    pws = g[:][0]

    out_vocab_test_orginal_model = random.sample(pws, 1000)
    file_name.write("List of out-vocab test original model: {}\n".format(out_vocab_test_orginal_model))
    file_name.write("List of in-vocab test original model: {}\n".format(in_vocab_words_test_orginal_model))

    
    in_vocab_runtime_test_orignal_model = target_ner_tokenizer_query_2_times(in_vocab_words_test_orginal_model)
    out_vocab_runtime_test_original_model = target_ner_tokenizer_query_2_times(out_vocab_test_orginal_model)

    save_results([in_vocab_runtime_test_orignal_model, out_vocab_runtime_test_original_model], thesh_choose_pickle_file_name)


    # now = datetime.now().date()
    # now = now.strftime("%Y%m%d")
    now = datetime.now().date()
    now = now.strftime("%Y%m%d")
    folder = 'attack_based_time_different_results_{}'.format(now)
    # global thesh_choose_pickle_file_name
    thesh_choose_pickle_file_name = "runtime_to_choose_threshold_1000_in-vocab_1000_out-vocab_vm_2"
    filename = '{}_{}.pickle3'.format(now, thesh_choose_pickle_file_name)
    # global thesh_choose_filename
    thesh_choose_filename = os.path.join(folder, filename)


    g = []
    print(thesh_choose_filename)
    h = pickle.load(open(thesh_choose_filename, 'rb'))
    g.append(h)

    

    in_vocab_runtime = g[0][0]
    # print(in_vocab_runtime)
    out_vocab_runtime = g[0][1]
    # print(out_vocab_runtime)
    orig_in_vocab = [ner_runtime*1000 for ner_runtime in in_vocab_runtime]
    orig_out_vocab = [ner_runtime*1000 for ner_runtime in out_vocab_runtime]

    orig_in_vocab_run_1 = []
    orig_in_vocab_run_2 = []
    orig_out_vocab_run_1 = []
    orig_out_vocab_run_2 = []

    for i in range(1000):
        tmp = orig_in_vocab[2*1]
        orig_in_vocab_run_1.append(tmp)
        tmp = orig_in_vocab[2*i + 1]
        orig_in_vocab_run_2.append(tmp)

        tmp = orig_out_vocab[2*1]
        orig_out_vocab_run_1.append(tmp)
        tmp = orig_out_vocab[2*i + 1]
        orig_out_vocab_run_2.append(tmp)



    
    in_mean_run_1 = np.mean(np.array(orig_in_vocab_run_1))
    in_std_run_1 = np.std(np.array(orig_in_vocab_run_1))


    for index in range(len(orig_in_vocab_run_1)):
        if abs(orig_in_vocab_run_1[index] - in_mean_run_1) >= (3*in_std_run_1):
            orig_in_vocab_run_1[index] = in_mean_run_1

    in_mean_run_2 = np.mean(np.array(orig_in_vocab_run_2))
    in_std_run_2 = np.std(np.array(orig_in_vocab_run_2))


    for index in range(len(orig_in_vocab_run_2)):
        if abs(orig_in_vocab_run_2[index] - in_mean_run_2) >= (3*in_std_run_2):
            orig_in_vocab_run_2[index] = in_mean_run_2
    

    out_std_run_1 = np.std(np.array(orig_out_vocab_run_1))
    out_mean_run_1 = np.mean(np.array(orig_out_vocab_run_1))


    for index in range(len(orig_out_vocab_run_1)):
        if abs(orig_out_vocab_run_1[index] - out_mean_run_1) >= (3*out_std_run_1):
            orig_out_vocab_run_1[index] = out_mean_run_1

    out_std_run_2 = np.std(np.array(orig_out_vocab_run_2))
    out_mean_run_2 = np.mean(np.array(orig_out_vocab_run_2))


    for index in range(len(orig_out_vocab_run_2)):
        if abs(orig_out_vocab_run_2[index] - out_mean_run_2) >= (3*out_std_run_2):
            orig_out_vocab_run_2[index] = out_mean_run_2
    
    time_diff_in_vocab = []
    time_diff_out_vocab = []

    for i in range(1000):
        tmp = orig_in_vocab_run_1[i] - orig_in_vocab_run_2[i]
        time_diff_in_vocab.append(tmp)
        tim = orig_out_vocab_run_1[i] - orig_out_vocab_run_2[i]
        time_diff_out_vocab.append(tmp)



    vocab_in = np.zeros(len(orig_in_vocab)) 
    # print(vocab_out)
    vocab_out = np.ones(len(orig_out_vocab))
    # print(vocab_in)
    vocabs = [*vocab_in,*vocab_out]
    
    y = vocabs
    # print(y)
    time = [*time_diff_in_vocab, *time_diff_out_vocab]
    scores = np.array(time)
    # print(scores)
    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
        
    
    index = 0
    
    for index in range(len(fpr)):
        if fpr[index] > 0.05 and fpr[index] <= 0.1:
            # print(fpr[index])
            # print('index = ', index)
            save_index = index
    
    # for index in range(len(tpr)):
    #     if tpr[index] > 0.8 and tpr[index] <= 0.9:
    #         # print(fpr[index])
    #         # print('index = ', index)
    #         save_index = index

    chosen_threshold = thresholds[save_index]
    print("fpr = ", fpr[save_index])
    print("tpr = ", tpr[save_index])
    print("chosen_threshold = ", chosen_threshold)

    folder = 'vm_entire_attack_time_diff_{}'.format(now)
    plt_folder = '{}_PLOTS/'.format(folder)

    mkdir_p(plt_folder)

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
    plt_dest = plt_folder + 'roc_auc_1000_invocab_1000_out-vocab_wo_reload_timesleep.png'
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')   

    return chosen_threshold




if __name__ == "__main__":
    
    now = datetime.now().date()
    now = now.strftime("%Y%m%d")
    folder = 'attack_based_time_different_results_{}'.format(now)
    # global thesh_choose_pickle_file_name
    thesh_choose_pickle_file_name = "runtime_to_choose_threshold_1000_in-vocab_1000_out-vocab_vm_2"
    filename = '{}_{}.pickle3'.format(now, thesh_choose_pickle_file_name)
    # global thesh_choose_filename
    thesh_choose_filename = os.path.join(folder, filename)

    threshold = choose_threshold()
    
    querying_updated_ner_2_times()

    
    # global querying_result_file_name
    querying_result_file_name =   "runtime_attack_200_in-vocab_200_out-vocab_words_vm_all_types_time_sleep" 
    filename = '{}_{}.pickle3'.format(now, querying_result_file_name)
    pickle_querying_result_file_name = os.path.join(folder, filename)

    g = []
    # print(file_name)
    h = pickle.load(open(pickle_querying_result_file_name, 'rb'))
    g.append(h)

    # in_vocab_runtime = g[0][0]
    # # print(in_vocab_runtime)
    # out_vocab_runtime = g[0][1]
    # # print(out_vocab_runtime)
    orig_vocab_runtime = h[0]
    updating_word_runtime = h[1]
    # print(in_vocab_runtime)
    out_vocab_runtime = h[2]

    # print(out_vocab_runtime)
    orig_vocab = [ner_runtime*1000 for ner_runtime in orig_vocab_runtime]
    updating_vocab = [ner_runtime*1000 for ner_runtime in updating_word_runtime]
    out_vocab = [ner_runtime*1000 for ner_runtime in out_vocab_runtime]

    
    orig_vocab_run_1 = []
    orig_vocab_run_2 = []

    updating_vocab_run_1 = []
    updating_vocab_run_2 = []

    out_vocab_run_1 = []
    out_vocab_run_2 = []

    for i in range(100):
        tmp = orig_vocab[2*i]
        orig_vocab_run_1.append(tmp)
        tmp = orig_vocab[2*i +1]
        orig_vocab_run_2.append(tmp)

        tmp = updating_vocab[2*i]
        updating_vocab_run_1.append(tmp)
        tmp = updating_vocab[2*i +1]
        updating_vocab_run_2.append(tmp)

        tmp = out_vocab[2*i]
        out_vocab_run_1.append(tmp)
        tmp = out_vocab[2*i +1]
        out_vocab_run_2.append(tmp)

    query_time_diff_in_vocab = []
    query_time_diff_out_vocab = []
    query_time_diff_update_vocab = []

    for i in range(100):
        tmp = orig_vocab_run_1[i] - orig_vocab_run_2[i]
        query_time_diff_in_vocab.append(tmp)
        tmp = out_vocab_run_1[i] - out_vocab_run_2[i]
        query_time_diff_out_vocab.append(tmp)
        tmp = updating_vocab_run_1[i] - updating_vocab_run_2[i]
        query_time_diff_update_vocab.append(tmp)


    thre = threshold
    # thre = 3.5
    count_out = 0
    for i in query_time_diff_out_vocab:
        if i > thre:
            count_out +=1

    count_orig = 0
    for i in query_time_diff_in_vocab:
        if i < thre:
            count_orig +=1

    count_update = 0
    for i in query_time_diff_update_vocab:
        if i < thre:
            count_update +=1

    accuracy_out = count_out/len(query_time_diff_out_vocab)
    accuracy_orig = count_orig/len(query_time_diff_out_vocab)
    accuracy_update = count_update/len(query_time_diff_out_vocab)
    
    print(threshold)
    print("success out-vocab = {}".format(accuracy_out))
    print("success in-vocab = {}".format(accuracy_orig))
    print("success updating-vocab = {}".format(accuracy_update))
    

    # recall = (count_in)/((count_in)+(len(out_vocab) - count_out))
    # print("recall = ", recall)
    # precision = (count_in)/len(in_vocab)
    # print("precision = ", precision)

    title = "accuracy of classifying: out-vocab = {0}; in-vocab = {1}; updating-vocab = {2}".format(accuracy_out, accuracy_orig, accuracy_update)
    print(title)

    iterations =  len(out_vocab)
    print(iterations)
    iteration = []
    for i in range(iterations):
        iteration.append(i)

    # thre = threshold
    thresholds = []
    
    for i in range(iterations):
        thresholds.append(thre)

    folder = 'vm_entire_attack_time_diff_{}'.format(now)
    plt_folder = '{}_PLOTS/'.format(folder)

    mkdir_p(plt_folder)

    threshold_legend = 'threshold = {}'.format(thre)
    plot1 = plt.figure(2)
    plt.plot(iteration, orig_vocab, 'o', iteration, updating_vocab, '*', iteration, out_vocab, 'v', iteration, thresholds, '-')
    
    # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
    plt.legend(['original-vocab words', 'updating words', 'out-vocab words', threshold_legend])
    
    plt.xlabel("word $i^{th}$")
    plt.ylabel('runtime (ms)')
    plt.title(title)
    # ax = plt.gca()
    # ax.set_ylim(3, 6) 
    plt_dest = plt_folder + 'attack_result_200_in-out-vocab_vm_all_types_words_time_sleep.png'
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')
    
    # recall = (count_in)/((count_in)+(len(out_vocab) - count_out))
    # print("recall = ", recall)
    # precision = (count_in)/len(in_vocab)
    # print("precision = ", precision)

    # title = "accuracy of classifying: out-vocab = {0}; in-vocab = {1}".format(accuracy_out, accuracy_in)
    # print(title)

    # iterations =  len(out_vocab)
    # print(iterations)
    # iteration = []
    # for i in range(iterations):
    #     iteration.append(i)

    # thre = threshold
    # thresholds = []
    
    # for i in range(iterations):
    #     thresholds.append(thre)

    # folder = 'vm_entire_attack_{}'.format(now)
    # plt_folder = '{}_PLOTS/'.format(folder)

    # mkdir_p(plt_folder)

    # threshold_legend = 'threshold = {}'.format(thre)
    # plot1 = plt.figure(2)
    # plt.plot(iteration, in_vocab, 'o', iteration, out_vocab, 'v', iteration, thresholds, '-')
    
    # # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
    # plt.legend(['in-vocab words', 'out-vocab words', threshold_legend])
    
    # plt.xlabel("word $i^{th}$")
    # plt.ylabel('runtime (ms)')
    # plt.title(title)
    # ax = plt.gca()
    # ax.set_ylim(3, 6) 
    # plt_dest = plt_folder + 'attack_result_200_in-out-vocab_vm_updating_words.png'
    # plt.savefig(plt_dest, dpi=300, bbox_inches='tight')
