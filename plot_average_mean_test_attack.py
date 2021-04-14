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
        if file_name == "20210317_timming_1000_vocab_obs_test_3words_1word_out_test_1000in_2.pickle3":
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

    # print(g)


    

    
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

    g_obs_names = ['20210316_timming_1000_vocab_obs_test_3words']
    in_vocab_test_runtime = []
    out_vocab_test_runtime = []
    for i in range(len(g)):
        # print(g)
        # plot1 = plt.figure(i)
        in_vocab_runtimes_1 = g[:][:][i][0]
        in_vocab_runtime_1 = [in_vocab_runtime*1000 for in_vocab_runtime in in_vocab_runtimes_1]
        
        in_std_ner_1 = np.std(np.array(in_vocab_runtime_1))
        in_mean_ner_1 = np.mean(np.array(in_vocab_runtime_1))

        print("mean news: ", in_mean_ner_1)
        print("std news: ", in_std_ner_1)

        in_vocab_runtimes_2 = g[:][:][i][1]
        in_vocab_runtime_2 = [in_vocab_runtime*1000 for in_vocab_runtime in in_vocab_runtimes_2]
        
        in_std_ner_2 = np.std(np.array(in_vocab_runtime_2))
        in_mean_ner_2 = np.mean(np.array(in_vocab_runtime_2))
        print("mean people: ", in_mean_ner_2)
        print("std people: ", in_std_ner_2)
        
        in_vocab_runtimes_3 = g[:][:][i][2]
        in_vocab_runtime_3 = [in_vocab_runtime*1000 for in_vocab_runtime in in_vocab_runtimes_3]
        
        in_std_ner_3 = np.std(np.array(in_vocab_runtime_3))
        in_mean_ner_3 = np.mean(np.array(in_vocab_runtime_3))
        print("mean Australia: ", in_mean_ner_3)
        print("std Australia: ", in_std_ner_3)

        avg_std_ner_in_vocab = (in_std_ner_1+in_std_ner_2+in_std_ner_3)/3
        avg_mean_ner_in_vocab = (in_mean_ner_1+in_mean_ner_2+in_mean_ner_3)/3

        print("avg_mean_ner = ", avg_mean_ner_in_vocab)
        print("avg_std_ner = ", avg_std_ner_in_vocab)

        in_vocab_test_runtimes = g[:][:][i][3]
        in_vocab_test_runtime = [ner_runtime*1000 for ner_runtime in in_vocab_test_runtimes]



        out_vocab_test_runtimes = g[:][:][i][4]
        out_vocab_test_runtime = [ner_runtime*1000 for ner_runtime in out_vocab_test_runtimes]
        

    
    in_std = np.std(np.array(in_vocab_test_runtime))
    in_mean = np.mean(np.array(in_vocab_test_runtime))

    print("in_mean: ", in_mean)
    print("in_std: ", in_std)

    for index in range(len(in_vocab_test_runtime)):
        if abs(in_vocab_test_runtime[index] - in_mean) >= (3*in_std):
            in_vocab_test_runtime[index] = in_mean

    
    out_std = np.std(np.array(out_vocab_test_runtime))
    out_mean = np.mean(np.array(out_vocab_test_runtime))
    print("out_mean: ", out_mean)
    print("out_std: ", out_std)


    for index in range(len(out_vocab_test_runtime)):
        if abs(out_vocab_test_runtime[index] - out_mean) >= (3*out_std):
            out_vocab_test_runtime[index] = out_mean


    
    means = []
    stds = []
    for i in range(len(in_vocab_test_runtime)):
        means.append(avg_mean_ner_in_vocab)
    for i in range(len(in_vocab_test_runtime)):
        stds.append(avg_std_ner_in_vocab)

    mean = np.array(means, dtype=np.float64)
    std = np.array(stds, dtype=np.float64)

    clrs = sns.color_palette("husl", 1)
    with sns.axes_style("darkgrid"):
        plot1 = plt.figure(1)
        # fig, ax = plt.subplots()
        plt.plot(iteration, in_vocab_test_runtime, 'o', iteration, out_vocab_test_runtime, 'v', iteration, means, '-')
        plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
        plt.legend(['ner: in vocab', 'ner: out vocab', 'in vocab mean +/- std'])
        plt.xlabel("word $i^{th}$")
        plt.ylabel('runtime (ms)')
        # ax = plt.gca()
        # ax.set_ylim(0, 9) 
        plt_dest = plt_folder + 'in_out_runtime_1000_words_20210317_4.png'
        plt.savefig(plt_dest, dpi=300, bbox_inches='tight')



    means = []
    stds = []
    for i in range(len(in_vocab_test_runtime)):
        means.append(in_mean)
    for i in range(len(in_vocab_test_runtime)):
        stds.append(in_std)

    mean = np.array(means, dtype=np.float64)
    std = np.array(stds, dtype=np.float64)

    clrs = sns.color_palette("husl", 1)
    with sns.axes_style("darkgrid"):
        plot2 = plt.figure(2)
        # fig, ax = plt.subplots()
        plt.plot(iteration, in_vocab_test_runtime, 'o', iteration, out_vocab_test_runtime, 'v', iteration, means, '-')
        plt.fill_between(iteration, mean-std, mean+std, alpha=0.5, facecolor=clrs[0])
        plt.legend(['ner: in vocab', 'ner: out vocab', 'in_vocab mean +/- std'])
        plt.xlabel("word $i^{th}$")
        plt.ylabel('runtime (ms)')
        # ax = plt.gca()
        # ax.set_ylim(0, 9) 
        plt_dest = plt_folder + 'compare_1000_in_runtime_1000_words_20210317_4.png'
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

    # for i in range(len(in_vocab_test_runtime)):
    #     if abs(in_vocab_test_runtime[i] - avg_mean_ner_in_vocab) < avg_std_ner_in_vocab/2:
    #         TP +=1
    #     else:
    #         FP +=1
    # for i in range(len(out_vocab_test_runtime)):
    #     if abs(out_vocab_test_runtime[i] - avg_mean_ner_in_vocab) >= avg_std_ner_in_vocab/2:
    #         TN +=1
    #     else:
    #         FN +=1
    
    thres_ind = [0.5, 1, 1.5, 2]
    TPS = []
    FPS = []
    TNS = []
    FNS = []
    for index in thres_ind:
        for i in range(len(in_vocab_test_runtime)):
            if abs(in_vocab_test_runtime[i] - in_mean) < index*in_std:
                TP +=1
            else:
                FP +=1
        for i in range(len(out_vocab_test_runtime)):
            if abs(out_vocab_test_runtime[i] - in_mean) >= index*in_std:
                TN +=1
            else:
                FN +=1
        
        TPS.append(TP)
        FPS.append(FP)
        TNS.append(TN)
        FNS.append(FN)
        TP=0
        FP=0
        TN=0
        FN=0

    print("FPS = ", FPS)
    print("TPS = ", TPS)
    print("FNS = ", FNS)
    print("TNS = ", TNS)
    # print("precision = ", (TP/(TP+FP)))
    # print("recall = ", (TP/(TP+FN)))

    roc_values = []
    for i in range(4):
        tpr = TPS[i]/(TPS[i]+FNS[i])
        fpr = FPS[i]/(FPS[i]+TNS[i])
        roc_values.append([tpr, fpr])
    tpr_values, fpr_values = zip(*roc_values)

    fig, ax = plt.subplots(figsize=(10,7))
    ax.plot(fpr_values, tpr_values, '-o')
    # ax.plot(np.linspace(0, 1, 4),
    #         np.linspace(0, 1, 4),
    #         label='baseline',
    #         linestyle='--')
    plt.title('Receiver Operating Characteristic Curve', fontsize=18)
    plt.ylabel('TPR', fontsize=16)
    plt.xlabel('FPR', fontsize=16)
    # plt.legend(fontsize=12)
    plt_dest = plt_folder + 'roc_auc_1000_in_runtime_1000_words_20210317_4.png'
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')    
    



