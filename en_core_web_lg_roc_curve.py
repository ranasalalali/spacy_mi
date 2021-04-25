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
from sklearn.metrics import roc_auc_score


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
    folder = 'new_pws_list_11116_en_core_web_lg_timing_results_ROC_{}'.format(now)
    filename = '{}_{}.pickle3'.format(now, f_name)
    mkdir_p(folder)
    filename = os.path.join(folder, filename)
    save_file = open(filename, 'wb')
    pickle.dump(results_holder, save_file)
    save_file.close()




def load_nlp():
    nlp = spacy.load('en_core_web_lg')
    # nlp = spacy.load("en_core_web_sm")
    tokeniz = nlp.tokenizer
    tagger = nlp.get_pipe("tagger")
    parser = nlp.get_pipe("parser")
    ner = nlp.get_pipe("ner")
    att_ruler = nlp.get_pipe("attribute_ruler")
    lemmatizer = nlp.get_pipe("lemmatizer")
    return nlp, tokeniz, tagger, parser, ner, att_ruler, lemmatizer
 



def target_ner_tokenizer_multiple_words_one_run(text):
    
    nlp, tokeniz, tagger, parser, ner, att_ruler, lemmatizer = load_nlp()
    runtime_list = []
    for i in text: #range(iterations):
        
        print("i = ", i)

        text = i
        
        time0 = time.perf_counter()
        docs = tokeniz(text)
        doc = ner(docs)
        time_now = time.perf_counter()
        runtime = time_now - time0
        runtime_list.append(runtime)
       
    return runtime_list


def target_ner_tokenizer_one_word_multiple_times(texts, iterations):
    
    avg_runtime_list = []
    
    nlp, tokeniz, tagger, parser, ner, att_ruler, lemmatizer = load_nlp()

    
    for i in texts:
        # text = "Alice lives in Australia and "+i
        text = i
        # print(text)
        # doc = tokeniz("the")
        # doc = ner(doc)
        total_runtime = 0
        for j in range(iterations):
           
            time0 = time.perf_counter()
            docs = tokeniz(text)
            doc = ner(docs)
            time_now = time.perf_counter()
            
            runtime = time_now - time0
            total_runtime += runtime
            
            print(" j = ", j)

        avg_runtime_list.append(total_runtime/iterations)  

    return avg_runtime_list

def target_tokenizer_ner_time_diff(texts):
    
    runtime_diff_list = []
    
    nlp, tokeniz, tagger, parser, ner, att_ruler, lemmatizer = load_nlp()

   
    for i in texts:
        text = i
        print(text)
        
        for j in range(2):

            time0 = time.perf_counter()           
            docs = tokeniz(text)
            doc = ner(docs)
            time_now = time.perf_counter()
            
            # time.sleep(5.0)

            runtime = time_now - time0
            runtime_diff_list.append(runtime)
            print(" j = ", j)
            

    return runtime_diff_list


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
    file_name = open("attack_en_core_lg_model_ROC_vm.txt","a")
    file_name.write("+++++++++++++++++++++++++++++++++++\n")
    file_name.write("+++++++++++++++++++++++++++++++++++\n")
    
    nlp = spacy.load("en_core_web_lg")
    # nlp.vocab.to_disk("vocab_original")
    vocab_lg = list(nlp.vocab.strings)
    test_in_vocabs = vocab_lg
    print(len(test_in_vocabs))

    # nlp = spacy.load("en_core_web_sm")
    # # nlp.vocab.to_disk("vocab_original")
    # vocab_sm = list(nlp.vocab.strings)
    # test_in_vocabs = vocab_sm
    # print(len(test_in_vocabs))

    # differ = list(set(vocab_lg) - set(vocab_sm))
    # print(len(differ))



    # nlp = spacy.load("en_core_web_lg")
    # global vocab
    num_test = 2000
    # vocab = list(nlp.vocab.strings)
    # in_vocab_words = vocab[10000:10000+num_test]
    vocab = vocab_lg
    in_vocab_words_test = random.sample(vocab, num_test)#vocab[10000:10000+num_test]#vocab[1000:1000+num_test]#random.sample(vocab, num_test)#vocab[10000:10000+num_test]
    # in_vocab_words_test = ['news', 'people', 'the', 'you', 'home']
    # in_vocab_words_test = ['people', 'update', 'school','sample', 'random']
    # in_vocab_words_test = ['home', 'home', 'home', 'home', 'home']
    # print(list(pws))

    
    # file_pws = 'passwords_out_vocab_list'
    # file_pws = 'passwords_list_5000_min_lower_1_min_upper_1_min_digit_1_min_spec_0_min_len_6'
    file_pws = 'passwords_list_5000_min_lower_1_min_upper_1_min_digit_1_min_spec_1_min_len_6'#'passwords_list_2000_no_speacial_charac'
    # file_pws = 'passwords_list_5000_no_speacial_charac_len_10'
    g = []
    h = pickle.load(open(file_pws, 'rb'))
    g.append(h)

    pws = g[:][0]

    out_vocab_words = random.sample(pws,num_test) #pws[0:num_test] 

    
    file_name.write("List of out vocab: {}\n".format(out_vocab_words))
    file_name.write("+++++++++++++++++++++++++++++++++++\n")
    file_name.write("List of in vocab: {}\n".format(in_vocab_words_test))
    # file_name.write("List of shuffle word in/out vocab: {}\n".format(shuffe_words))

    iterations = 10
    in_vocab_runtime_one_run = target_ner_tokenizer_multiple_words_one_run(in_vocab_words_test)
    time.sleep(5.0)
    out_vocab_runtime_one_run = target_ner_tokenizer_multiple_words_one_run(out_vocab_words)
    time.sleep(5.0)
    in_vocab_runtime_avg = target_ner_tokenizer_one_word_multiple_times(in_vocab_words_test, iterations)
    time.sleep(5.0)
    out_vocab_runtime_avg = target_ner_tokenizer_one_word_multiple_times(out_vocab_words, iterations)
    time.sleep(5.0)
    in_vocab_runtime_time_diff = target_tokenizer_ner_time_diff(in_vocab_words_test)
    time.sleep(5.0)
    out_vocab_runtime_time_diff = target_tokenizer_ner_time_diff(out_vocab_words)


    now = datetime.now().date()
    now = now.strftime("%Y%m%d")
    now1 = datetime.now()
    now1 = now1.strftime("%d-%m-%Y-%H-%M-%S")

    folder = 'new_pws_list_11116_en_core_web_lg_timing_results_ROC_{}'.format(now)

    # shuffe_words_runtime = target_ner_tokenizer_one_word_three_times(shuffe_words)

    # pickle_fname = "target_en_core_sm_model_runtime_vm_ROC_{0}_words_{1}".format(num_test, now1)
    pickle_fname = "target_en_core_sm_model_runtime_vm_ROC_{}_words".format(num_test)
    # save_results([in_vocab_runtime, out_vocab_runtime, shuffe_words_runtime], pickle_fname)
    save_results([in_vocab_runtime_one_run, out_vocab_runtime_one_run,  in_vocab_runtime_avg, out_vocab_runtime_avg, in_vocab_runtime_time_diff, out_vocab_runtime_time_diff], pickle_fname)

    
    # f_name = "timming_100pws_in-out-vocab_three_times_injecting_common_query_vm_tokenizer"
    filename = '{}_{}.pickle3'.format(now, pickle_fname)
    file_name = os.path.join(folder, filename)

    g = []
    print(file_name)
    h = pickle.load(open(file_name, 'rb'))
    g.append(h)

    


    in_vocab_runtime_one_run_list = g[0][0]
    out_vocab_runtime_one_run_list = g[0][1]
    in_vocab_runtime_avg_list = g[0][2]
    out_vocab_runtime_avg_list = g[0][3]
    in_vocab_runtime_time_diff_list = g[0][4]
    out_vocab_runtime_time_diff_list = g[0][5]
    # shuffle_word_runtime_list = g[0][2]

    in_vocab_runtime_one_run_list_s = [ner_runtime*1000 for ner_runtime in in_vocab_runtime_one_run_list]
    out_vocab_runtime_one_run_list_s = [ner_runtime*1000 for ner_runtime in out_vocab_runtime_one_run_list]
    in_vocab_runtime_avg_list_s = [ner_runtime*1000 for ner_runtime in in_vocab_runtime_avg_list]
    out_vocab_runtime_avg_list_s = [ner_runtime*1000 for ner_runtime in out_vocab_runtime_avg_list]

    in_vocab_runtime_time_diff_list_s = [ner_runtime*1000 for ner_runtime in in_vocab_runtime_time_diff_list]
    out_vocab_runtime_time_diff_list_s = [ner_runtime*1000 for ner_runtime in out_vocab_runtime_time_diff_list]

    # shuffle_words_runtime_s = [ner_runtime*1000 for ner_runtime in  shuffle_word_runtime_list]

    # print(in_vocab_runtime_s)
    in_vocab_run_1 = []
    in_vocab_run_2 = []
    # in_vocab_run_3 = []

    out_vocab_run_1 = []
    out_vocab_run_2 = []
    # out_vocab_run_3 = []

    

    for i in range(num_test):
        in_vocab_run_1.append(in_vocab_runtime_time_diff_list_s [i*2])
        in_vocab_run_2.append(in_vocab_runtime_time_diff_list_s[2*i+1])
        # in_vocab_run_3.append(in_vocab_runtime_s[3*i+2])

        out_vocab_run_1.append(out_vocab_runtime_time_diff_list_s[i*2])
        out_vocab_run_2.append(out_vocab_runtime_time_diff_list_s[2*i+1])
        # out_vocab_run_3.append(out_vocab_runtime_s[3*i+2])

        
    # max_in = 0
    # for i in range(num_test):
    #     if in_vocab_run_1[i] >= max_in:
    #         max_in = in_vocab_run_1[i]
    #         save_index = i

    # max_in_run_1 = max(in_vocab_run_1)
    # if max_in_run_1 > 5:
    #     in_vocab_run_1[save_index] = 5

    
    
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

    
    # vocab_in = np.zeros(len(in_vocab_run_1)) 
    # # print(vocab_out)
    # vocab_out = np.ones(len(in_vocab_run_1))
    # # print(vocab_in)
    # vocabs = [*vocab_in,*vocab_out]
    
    # y = vocabs
    # # print(y)
    # time = [*in_vocab_runtime_one_run_list_s, *out_vocab_runtime_one_run_list_s]
    # scores = np.array(time)
    # # print(scores)
    # fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
        
    # print("thresholds = ", thresholds)
    # print("fpr = ", fpr)
    # print("tpr = ", tpr)
    # index = 0

    # for index in range(len(fpr)):
    #     if fpr[index] > 0.01 and fpr[index] <= 0.05:
    #         # print(fpr[index])
    #         # print('index = ', index)
    #         save_index = index

    

    # chosen_threshold = thresholds[save_index]
    
    # print("fpr = ", fpr[save_index])
    # print("tpr = ", tpr[save_index])
    # print("chosen_threshold = ", chosen_threshold)
    
    
    iterations =  num_test*2
    iteration = []
    for i in range(iterations):
        iteration.append(i)

    
    plt_folder = '{}_PLOTS/'.format(folder)

    mkdir_p(plt_folder)
    index = num_test

    avg_time_graph_name = 'average_runtime_over_{}_words_vm_both_tokenizer_ner_en_core_lg_vm_2.png'.format(num_test)
    absolute_runtime_graph_name = 'Runtime_{}_words_vm_both_tokenizer_ner_en_core_lg_vm_2.png'.format(num_test)
    tokenizer_avg_runtime_diff_graph_name = 'average_runtime_over_{}_words_vm_tokenizer_only_en_core_lg.png'.format(num_test)
    ner_avg_runtime_diff_graph_name = 'average_runtime_over_{}_words_vm_ner_only_en_core_lg.png'.format(num_test)
    time_diff_graph_name = 'time_differenc_{}_words_en_core_lg.png'.format(num_test)
    
    roc_auc_absolute_time_graph_name = 'roc_auc_{}_words_en_core_wb_lg_absolute_runtime_vm.png'.format(num_test)
    roc_auc_time_diff_graph_name = 'roc_auc_{}_words_en_core_wb_lg_time_diff_vm.png'.format(num_test)
    roc_auc_avg_time_graph_name = 'roc_auc_{}_words_en_core_wb_lg_avg_time_vm.png'.format(num_test)
    all_roc_graph_name = 'all_roc_auc_{}_words_en_core_lg.png'.format(num_test)
    all_roc_graph_name_pdf = 'all_roc_auc_{}_words_en_core_lg.pdf'.format(num_test)
    
    ner_runtime_three_runs_IN = 'IN_runtime_three_run_{}_words_ner_en_core_web_lg_vm.png'.format(num_test)
    ner_runtime_three_runs_OUT = 'OUT_runtime_three_run_{}_words_ner_en_core_web_lg_vm.png'.format(num_test)


    
    
    
    plot2 = plt.figure(1)
    plt.plot(iteration[0:num_test], in_vocab_runtime_one_run_list_s, 'o', iteration[0:num_test], out_vocab_runtime_one_run_list_s, 'v')
    
    # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
    # plt.legend(['100 phrases with in vocab words', '100 phrases with out vocab words'])
    plt.legend(['in vocab words', 'out vocab words'])
    
    plt.xlabel("")
    plt.ylabel('Runtime (ms)')
    plt.title("Querying tokenizer and ner")
    # plt.xticks(iteration[0:3], x_stick)
    ax = plt.gca()
    ax.set_ylim(2, 5) 
    plt_dest = plt_folder + absolute_runtime_graph_name 
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')


    plot2 = plt.figure(2)
    plt.plot(iteration[0:num_test], in_vocab_runtime_avg_list_s, 'o', iteration[0:num_test], out_vocab_runtime_avg_list_s, 'v')
    
    # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
    # plt.legend(['100 phrases with in vocab words', '100 phrases with out vocab words'])
    plt.legend(['in vocab words', 'out vocab words'])
    
    plt.xlabel("")
    plt.ylabel('Runtime (ms)')
    plt.title("Querying tokenizer and ner")
    # plt.xticks(iteration[0:3], x_stick)
    ax = plt.gca()
    ax.set_ylim(2, 5) 
    plt_dest = plt_folder + avg_time_graph_name 
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')


    plot2 = plt.figure(4)
    plt.plot(iteration[0:num_test], diff_in_vocab, 'o', iteration[0:num_test], diff_out_vocab, 'v')
    
    # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
    # plt.legend(['100 phrases with in vocab words', '100 phrases with out vocab words'])
    plt.legend(['in vocab words', 'out vocab words'])
    
    plt.xlabel("")
    plt.ylabel('Runtime (ms)')
    plt.title("Querying tokenizer and ner")
    # plt.xticks(iteration[0:3], x_stick)
    # ax = plt.gca()
    # ax.set_ylim(2, 3) 
    plt_dest = plt_folder + time_diff_graph_name
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')


    vocab_in = np.zeros(len(in_vocab_run_1)) 
    # print(vocab_out)
    vocab_out = np.ones(len(in_vocab_run_1))
    # print(vocab_in)
    vocabs = [*vocab_in,*vocab_out]
    
    y = vocabs
    # print(y)
    time_absolute = [*in_vocab_runtime_one_run_list_s, *out_vocab_runtime_one_run_list_s]
    scores = np.array(time_absolute)
    # print(scores)
    fpr_abs, tpr_abs, thresholds_abs = metrics.roc_curve(y, scores, pos_label=1)
    auc_abs = roc_auc_score(y, scores)
    print('AUC: %.2f' % auc_abs)
        
    time_avg = [*in_vocab_runtime_avg_list_s, *out_vocab_runtime_avg_list_s]

    scores = np.array(time_avg)
    fpr_avg, tpr_avg, thresholds_avg = metrics.roc_curve(y, scores, pos_label=1)
    auc_avg = roc_auc_score(y, scores)
    print('AUC: %.2f' % auc_avg)


    time_diff = [*diff_in_vocab, *diff_out_vocab]

    scores = np.array(time_diff)
    fpr_diff, tpr_diff, thresholds_diff = metrics.roc_curve(y, scores, pos_label=1)
    auc_diff = roc_auc_score(y, scores)
    print('AUC: %.2f' % auc_diff)
    

    plot2 = plt.figure(3)
    fig, ax = plt.subplots(figsize=(10,7))
    ax.plot(fpr_abs, tpr_abs, '-o', fpr_avg, tpr_avg, '-v', fpr_diff, tpr_diff, '-*')
    # ax.plot(np.linspace(0, 1, 4),
    #         np.linspace(0, 1, 4),
    #         label='baseline',
    #         linestyle='--')
    plt.title('member: words in vocab; non-member: generated passwords', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    legend_1 = 'One run: AUC = {}'.format(auc_abs, '.3f')
    legend_2 = 'Average runtime: AUC = {}'.format(auc_avg, '.3f')
    legend_3 = 'Time difference between two runs: AUC = {}'.format(auc_diff, '.3f')
    plt.legend([legend_1, legend_2, legend_3])
    plt_dest = plt_folder + all_roc_graph_name
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')
    plt_dest = plt_folder + all_roc_graph_name_pdf
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')






    
    sys.exit()

    ##############################################


    plot2 = plt.figure(10)
    plt.plot(iteration[0:num_test], in_vocab_ner_run_1, 'o', iteration[0:num_test], in_vocab_ner_run_2, 'v', iteration[0:num_test], in_vocab_ner_run_3, '*')
    
    # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
    # plt.legend(['100 phrases with in vocab words', '100 phrases with out vocab words'])
    plt.legend(['1st run', '2nd run', '3rd run'])
    
    plt.xlabel("")
    plt.ylabel('Runtime (ms)')
    plt.title("NER: in vocab")
    # plt.xticks(iteration[0:num_test], in_vocab_words_test, rotation ='vertical')
    ax = plt.gca()
    ax.set_ylim(2, 3) 
    plt_dest = plt_folder + ner_runtime_three_runs_IN 
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')


    plot2 = plt.figure(20)
    plt.plot(iteration[0:num_test], out_vocab_ner_run_1, 'o', iteration[0:num_test], out_vocab_ner_run_2, 'v', iteration[0:num_test], out_vocab_ner_run_3, '*')
    
    # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
    # plt.legend(['100 phrases with in vocab words', '100 phrases with out vocab words'])
    plt.legend(['1st run', '2nd run', '3rd run'])
    
    plt.xlabel("")
    plt.ylabel('Runtime (ms)')
    plt.title("NER: out vocab")
    # plt.xticks(iteration[0:num_test], out_vocab_words, rotation ='vertical')
    ax = plt.gca()
    ax.set_ylim(2, 3) 
    plt_dest = plt_folder + ner_runtime_three_runs_OUT 
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')




    # sys.exit()



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


    vocab_in = np.zeros(len(diff_in_vocab)) 
    # print(vocab_out)
    vocab_out = np.ones(len(diff_in_vocab))
    # print(vocab_in)
    vocabs = [*vocab_in,*vocab_out]
    
    y = vocabs
    # print(y)
    time = [*diff_in_vocab, *diff_out_vocab]
    scores = np.array(time)
    # print(scores)
    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
        
    print("thresholds = ", thresholds)
    print("fpr = ", fpr)
    print("tpr = ", tpr)
    index = 0

    plot2 = plt.figure(7)
    fig1, ax1 = plt.subplots(figsize=(10,7))
    ax1.plot(fpr, tpr, '-o')
    # ax.plot(np.linspace(0, 1, 4),
    #         np.linspace(0, 1, 4),
    #         label='baseline',
    #         linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    # plt.legend(fontsize=12)
    plt_dest = plt_folder + roc_auc_time_diff_graph_name
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
    plt.xticks(iteration[0:index], out_vocab_words, rotation ='vertical')
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
    plt.xticks(iteration[0:index], out_vocab_words, rotation ='vertical')
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