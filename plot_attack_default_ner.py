import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import math
import statistics as st
import heapq
import operator
import errno
from itertools import islice
from password_strength import PasswordStats
import argparse
from mpl_toolkits.mplot3d import Axes3D
import re
from Levenshtein import distance as levenshtein_distance
import statistics 
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

def format_string(s):
    escaped = re.escape(s)
    return escaped

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--loc', type=str, help='Location of Results')

    args = parser.parse_args()

    loc = args.loc

    folder = loc

    # Load results for plotting
    #res_folder = 'Results/results_{}_len/'.format(secret_len)

    res_folder = '{}/'.format(folder)
    print(res_folder)

    files = os.listdir(res_folder)
    print(files)
    

    g = []
    br = 0
    g_obs = []
    for file_name in files:
        br += 1
        print(file_name)
        if file_name.split("_")[0] == "20210311":
            file_path = os.path.join(res_folder, file_name)
            h = pickle.load(open(file_path, 'rb'))
            g.append(h)
        elif file_name.split("_")[0] == "20210316":
            file_path = os.path.join(res_folder, file_name)
            h1 = pickle.load(open(file_path, 'rb'))
            g_obs.append(h1)

    # print('Read Disk')
    # print('{} FILES FOUND'.format(len(g)))

    plt_folder = '{}_PLOTS/'.format(folder)

    mkdir_p(plt_folder)


    

    
    iterations = 1000
    iteration = []
    for i in range(iterations):
        iteration.append(i)

    

    plot_names = []
    for plot_name in files:
        plot_name = plot_name.split('.')[0]
        plot_names.append(plot_name)
    print(plot_names)

    
    global in_mean_ner 
    global in_std_ner 

    g_obs_names = ['20210316_target_nlp_whole_in_out_vocab', '20210316_target_ner_tokenizer_in_out_vocab']
    for i in range(2):
        print(i)
        # plot1 = plt.figure(i)
        in_vocab_runtimes = g_obs[:][:][i][0]
        in_vocab_runtime = [in_vocab_runtime*1000 for in_vocab_runtime in in_vocab_runtimes]
        print(g_obs_names[i])
        print(g_obs_names[i].split("_")[2])
        if g_obs_names[i].split("_")[2]=="ner":
            in_std_ner = np.std(np.array(in_vocab_runtime))
            in_mean_ner = np.mean(np.array(in_vocab_runtime))
            print("in_mean_ner = ", in_mean_ner)
            print("in_std_ner = ", in_std_ner)
        elif g_obs_names[i].split("_")[2]=="nlp":
            in_std_nlp = np.std(np.array(in_vocab_runtime))
            in_mean_nlp = np.mean(np.array(in_vocab_runtime))

        # for index in range(len(in_vocab_runtime)):
        #     if in_vocab_runtime[index] - in_mean >= (3*in_std):
        #         in_vocab_runtime[index] = in_mean

        
    g_names = ['20210311_timming_1000_out_vocab_all_components', '20210311_timming_1000_in_vocab_all_components']  
    in_vocab_ner_runtime = []
    out_vocab_ner_runtime = []
    for i in range(len(g)):
        print(i)
        plot1 = plt.figure(i)
        # whole_nlp_runtimes = g[:][:][i][0]
        # whole_nlb_runtime = [nlp_runtime*1000 for nlp_runtime in whole_nlp_runtimes]

        # tokeniser_runtimes = g[:][:][i][1]
        # tokeniser_runtime = [tokenizer_runtime*1000 for tokenizer_runtime in tokeniser_runtimes]

        # tok2vec_runtimes = g[:][:][i][2]
        # tok2vec_runtime = [tok2vec_runtime*1000 for tok2vec_runtime in tok2vec_runtimes]

        # tagger_runtimes = g[:][:][i][3]
        # tagger_runtime = [tagger_runtime*1000 for tagger_runtime in tagger_runtimes]

        # parser_runtimes = g[:][:][i][4]
        # parser_runtime = [parser_runtime*1000 for parser_runtime in parser_runtimes]

        ner_runtimes = g[:][:][i][5]
        print(g_names[i])
        print(g_names[i].split("_")[3])
        if g_names[i].split("_")[3]=="in":
            in_vocab_ner_runtime = [ner_runtime*1000 for ner_runtime in ner_runtimes]
        elif g_names[i].split("_")[3]=="out":
            out_vocab_ner_runtime = [ner_runtime*1000 for ner_runtime in ner_runtimes]

        # attr_runtimes = g[:][:][i][6]
        # attr_runtime = [attr_runtime*1000 for attr_runtime in attr_runtimes]

        # lemma_runtimes = g[:][:][i][7]
        # lemma_runtime = [lemma_runtime*1000 for lemma_runtime in lemma_runtimes]
    
    in_std = np.std(np.array(in_vocab_ner_runtime))
    in_mean = np.mean(np.array(in_vocab_ner_runtime))

    for index in range(len(in_vocab_ner_runtime)):
        if in_vocab_ner_runtime[index] - in_mean >= (3*in_std):
            in_vocab_ner_runtime[index] = in_mean

    # out_vocab_runtimes = g[:][:][i][1]
    # out_vocab_runtime = [out_vocab_runtime*1000 for out_vocab_runtime in out_vocab_runtimes]


    out_std = np.std(np.array(in_vocab_ner_runtime))
    out_mean = np.mean(np.array(in_vocab_ner_runtime))

    for index in range(len(in_vocab_ner_runtime)):
        if in_vocab_ner_runtime[index] - out_mean >= (3*out_std):
            in_vocab_ner_runtime[index] = out_mean
    plot1 = plt.figure(1)
    plt.plot(iteration, in_vocab_ner_runtime, 'o', iteration, out_vocab_ner_runtime, 'v')
    # plt.plot(iteration, in_vocab_runtimes, 'o', iteration, out_vocab_runtimes, 'v')
    plt.legend(['ner: in vocab', 'ner: out vocab'])
    plt.xlabel("word $i^{th}$")
    plt.ylabel('runtime (ms)')
    # ax = plt.gca()
    # ax.set_ylim(0, 9) 
    plt_dest = plt_folder + 'in_out_runtime_1000_words.png'
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')


    TP = 0
    TN = 0
    FP = 0
    FN = 0
    # for i in range(len(in_vocab_ner_runtime)):
    #     if abs(in_vocab_ner_runtime[i] - in_mean_ner) < 3*in_std_ner/2:
    #         TP +=1
    #     else:
    #         FP +=1
    # for i in range(len(out_vocab_ner_runtime)):
    #     if abs(out_vocab_ner_runtime[i] - in_mean_ner) >= 3*in_std_ner/2:
    #         TN +=1
    #     else:
    #         FN +=1

    for i in range(len(in_vocab_ner_runtime)):
        if abs(in_vocab_ner_runtime[i] - in_mean_ner) < 0:
            TP +=1
        else:
            FP +=1
    for i in range(len(out_vocab_ner_runtime)):
        if out_vocab_ner_runtime[i] > in_mean_ner:
            TN +=1
        else:
            FN +=1
    
    print("FP = ", FP)
    print("TP = ", TP)
    print("FN = ", FN)
    print("TN = ", TN)
    print("precision = ", (TP/(TP+FP)))
    print("recall = ", (TP/(TP+FN)))





