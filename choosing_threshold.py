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
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
import sys


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
        # if file_name.split("_")[6] == "3words.pickle3":
        if file_name == "20210325_test_1000_out_vocab_reload_noreload_model.pickle3":
            file_path = os.path.join(res_folder, file_name)
            print(file_path)
            h = pickle.load(open(file_path, 'rb'))
            g.append(h)
        else:
            pass

    # print('Read Disk')
    # print('{} FILES FOUND'.format(len(g)))

    plt_folder = '{}_PLOTS/'.format(folder)

    mkdir_p(plt_folder)

    
    
    plot_names = []
    for plot_name in files:
        plot_name = plot_name.split('.')[0]
        plot_names.append(plot_name)
    print(plot_names)

    
    global in_mean_ner 
    global in_std_ner 

    # g_obs_names = ['20210316_timming_1000_vocab_obs_test_3words']
    in_vocab_test_runtime = []
    out_vocab_test_runtime = []
    for i in range(len(g)):
        
        in_vocab_test_runtimes = g[:][:][i][0]
        in_vocab_test_runtime = [ner_runtime*1000 for ner_runtime in in_vocab_test_runtimes]



        out_vocab_test_runtimes_list_no_reload = g[:][:][i][2]
        out_vocab_test_runtimes_list_no_reload_s = [ner_runtime*1000 for ner_runtime in out_vocab_test_runtimes_list_no_reload]




        out_vocab_test_runtimes_list_reload = g[:][:][i][1]
        out_vocab_test_runtimes_list_reload_s = [ner_runtime*1000 for ner_runtime in out_vocab_test_runtimes_list_reload]

        # out_vocab_test_NO_reload_runtimes_list = g[:][:][i][6]

        # out_vocab_100_password_runtimes = g[:][:][i][5]
        # out_vocab_1000_password_runtime = [ner_runtime*1000 for ner_runtime in out_vocab_1000_password_runtimes]


        in_vocab_test2_runtimes = g[:][:][i][3]
        in_vocab_test2_runtime = [ner_runtime*1000 for ner_runtime in in_vocab_test2_runtimes]


    in_std_ = np.std(np.array(out_vocab_test_runtimes_list_no_reload_s))
    in_mean_ = np.mean(np.array(out_vocab_test_runtimes_list_no_reload_s))

    # print("in_mean_test2: ", in_mean_test2)
    # print("in_std_test2: ", in_std_test2)

    for index in range(len(out_vocab_test_runtimes_list_no_reload_s)):
        if abs(out_vocab_test_runtimes_list_no_reload_s[index] - in_mean_) >= (3*in_std_):
            out_vocab_test_runtimes_list_no_reload_s[index] = in_mean_
    

    in_std_ = np.std(np.array(out_vocab_test_runtimes_list_reload_s))
    in_mean_ = np.mean(np.array(out_vocab_test_runtimes_list_reload_s))

    # print("in_mean_test2: ", in_mean_test2)
    # print("in_std_test2: ", in_std_test2)

    for index in range(len(out_vocab_test_runtimes_list_reload_s)):
        if abs(out_vocab_test_runtimes_list_reload_s[index] - in_mean_) >= (3*in_std_):
            out_vocab_test_runtimes_list_reload_s[index] = in_mean_
    

    print(in_vocab_test_runtimes)
    in_std = np.std(np.array(in_vocab_test_runtime))
    in_mean = np.mean(np.array(in_vocab_test_runtime))

    print("in_mean: ", in_mean)
    print("in_std: ", in_std)

    for index in range(len(in_vocab_test_runtime)):
        if abs(in_vocab_test_runtime[index] - in_mean) >= (3*in_std):
            in_vocab_test_runtime[index] = in_mean

    

   
    in_std_test2 = np.std(np.array(in_vocab_test2_runtime))
    in_mean_test2 = np.mean(np.array(in_vocab_test2_runtime))

    print("in_mean_test2: ", in_mean_test2)
    print("in_std_test2: ", in_std_test2)

    for index in range(len(in_vocab_test2_runtime)):
        if abs(in_vocab_test2_runtime[index] - in_mean_test2) >= (3*in_std_test2):
            in_vocab_test2_runtime[index] = in_mean_test2

   

    iterations =  len(in_vocab_test2_runtime)
    iteration = []
    for i in range(iterations):
        iteration.append(i)
    means = []
    stds = []
    for i in range(len(in_vocab_test2_runtime)):
        means.append(in_mean)
    for i in range(len(in_vocab_test2_runtime)):
        stds.append(in_std)

    mean = np.array(means, dtype=np.float64)
    std = np.array(stds, dtype=np.float64)


    clrs = sns.color_palette("husl", 1)
    with sns.axes_style("darkgrid"):

        plot1 = plt.figure(1)
       
        plt.plot(iteration, in_vocab_test2_runtime, 'o', iteration, out_vocab_test_runtimes_list_no_reload_s, 'v',
                      iteration, means, '-')
        
        plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
        plt.legend(['ner: 1000w in-vocab', 'ner: 1000 out-vocab',  '1000w in-vocab mean +/- std'])
        
        plt.xlabel("word $i^{th}$")
        plt.ylabel('runtime (ms)')
        # ax = plt.gca()
        # ax.set_ylim(2.5, 3.8) 
        plt.title("Without reload model after each query")
        plt_dest = plt_folder + '1000_out-vocab_without_reload_model_3103.png'
        plt.savefig(plt_dest, dpi=300, bbox_inches='tight')


        plot2 = plt.figure(2)
        
        plt.plot(iteration, in_vocab_test2_runtime, 'o', iteration, out_vocab_test_runtimes_list_reload_s, 'v', iteration, means, '-')
        plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
        plt.legend(['ner: 1000w in-vocab', 'ner: 1000w out-vocab', 'mean +/- std (1000w in-vocab)'])
        plt.xlabel("word $i^{th}$")
        plt.ylabel('runtime (ms)')
        # ax = plt.gca()
        # ax.set_ylim(0, 9) 
        plt.title("Reload model after each query")
        plt_dest = plt_folder + '11000_out-vocab_reload_model_3103.png'
        plt.savefig(plt_dest, dpi=300, bbox_inches='tight')


    ### ROC curve    
    
    import numpy as np
    from sklearn import metrics
    vocab_in = np.zeros(len(in_vocab_test2_runtime)) 
    # print(vocab_out)
    vocab_out = np.ones(len(in_vocab_test2_runtime))
    # print(vocab_in)
    vocabs = [*vocab_in,*vocab_out]
    
    y = vocabs
    # print(y)
    time = [*in_vocab_test2_runtime, *out_vocab_test_runtimes_list_no_reload_s]
    scores = np.array(time)
    # print(scores)
    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
        
    print(thresholds)
    print(fpr)
    print(tpr)

    index = 0
    for index in range(len(fpr)):
        if fpr[index] > 0.1 and fpr[index] <= 0.2:
            print(fpr[index])
            print('index = ', index)
            save_index = index
        # index +=1
    print("chosen threshold = ", thresholds[save_index])
    print(len(thresholds))
    print(len(fpr))
    print(len(tpr))

    file_name = open("chosen_thresholds.txt","a")
    file_name.write("+++++++++++++++++++++++++++++++++++\n")
    file_name.write("fpr = {}\n".format(fpr[save_index]))
    file_name.write("tpr = {}\n".format(tpr[save_index]))
    file_name.write("chosen threshold = {}\n".format(thresholds[save_index]))
    file_name.write("+++++++++++++++++++++++++++++++++++\n")
    
    # roc_auc = metrics.auc(fpr, tpr)
    # print(roc_auc)
    
    # random_guess = [0 for _ in range(len(y))]
    # rg_fpr, rg_tpr, rg_thresholds = metrics.roc_curve(y, random_guess, pos_label=1)
    
    # print(rg_thresholds)
    # print(rg_fpr)
    # print(rg_tpr)

    # fig, ax = plt.subplots(figsize=(10,7))
    # ax.plot(fpr, tpr, '-o', rg_fpr, rg_tpr, '--')
    # # ax.plot(np.linspace(0, 1, 4),
    # #         np.linspace(0, 1, 4),
    # #         label='baseline',
    # #         linestyle='--')
    # plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=18)
    # plt.ylabel('True Positive Rate', fontsize=16)
    # plt.xlabel('False Positive Rate', fontsize=16)
    # # plt.legend(fontsize=12)
    # plt_dest = plt_folder + 'roc_auc_1000_invocab_1000_out_vocab_without_reload3103.png'
    # plt.savefig(plt_dest, dpi=300, bbox_inches='tight')    


    # # 1 word, 1000 runs
    # time2 = [*in_vocab_test2_runtime, *out_vocab_test_runtimes_list_reload_s]
    # scores = np.array(time2)
    # print(scores)
    # fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
        
    # print(thresholds)
    # print(fpr)
    # print(tpr)

    # roc_auc = metrics.auc(fpr, tpr)
    # print(roc_auc)
    
    # random_guess = [0 for _ in range(len(y))]
    # rg_fpr, rg_tpr, rg_thresholds = metrics.roc_curve(y, random_guess, pos_label=1)
    
    # # print(rg_thresholds)
    # # print(rg_fpr)
    # # print(rg_tpr)

    # fig, ax = plt.subplots(figsize=(10,7))
    # ax.plot(fpr, tpr, '-o', rg_fpr, rg_tpr, '--')
    # # ax.plot(np.linspace(0, 1, 4),
    # #         np.linspace(0, 1, 4),
    # #         label='baseline',
    # #         linestyle='--')
    # plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=18)
    # plt.ylabel('True Positive Rate', fontsize=16)
    # plt.xlabel('False Positive Rate', fontsize=16)
    # # plt.legend(fontsize=12)
    # plt_dest = plt_folder + 'roc_auc_1000_invocab_1000_out-vocab_reload_3103.png'
    # plt.savefig(plt_dest, dpi=300, bbox_inches='tight')   





