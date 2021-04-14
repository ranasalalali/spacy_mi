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
        if file_name == "20210407_timming_100pws_in-out-vocab_three_times_injecting_common_query.pickle3":
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

    print(len(g))
    # print(g[0][0])

    
    # orig_in_vocabs = g[0][0]
    # updated_vocabs = g[0][1]
    # out_vocabs = g[0][2]

    # orig_in_vocab = [ner_runtime*1000 for ner_runtime in orig_in_vocabs]
    # updated_vocab = [ner_runtime*1000 for ner_runtime in updated_vocabs]
    # out_vocab = [ner_runtime*1000 for ner_runtime in out_vocabs]


    # print(updated_vocab)

    # # print(out_vocab)

   
    # thre = 2.7691549621522427
    # count_out = 0
    # for i in out_vocab:
    #     if i > thre:
    #         count_out +=1
    # count_in_updated = 0
    # for i in updated_vocab:
    #     if i < thre:
    #         count_in_updated +=1

    # count_in_orig = 0
    # for i in orig_in_vocab:
    #     if i < thre:
    #         count_in_orig +=1

    # success_out = count_out/len(out_vocab)
    # success_in_update = count_in_updated/len(updated_vocab)
    # success_in_orig = count_in_orig/len(orig_in_vocab)

    # print("success out-vocab = {}".format(success_out))
    # print("success in-updated = {}".format(success_in_update))
    # print("success in-orig = {}".format(success_in_orig)) 

    # success_out = count_out
    # success_in_update = count_in_updated
    # success_in_orig = count_in_orig

    # recall = (count_in_orig + count_in_updated)/((count_in_orig + count_in_updated)+(len(out_vocab) - count_out))
    # print("recall = ", recall)
    # precision = (count_in_orig + count_in_updated)/(len(orig_in_vocab) + len(updated_vocab))
    # print("precision = ", precision)


    # # title = "accuracy of classifying: out-vocab = {0}; in-updated vocab = {1}; in-orig vocab = {2}".format(success_out, success_in_update, success_in_orig)

    
    # sys.exit()
    # index = len(g[0][2])
    # in_vo = [*orig_in_vocab, *updated_vocab]

    # iterations =  len(g[0][2])
    # iteration = []
    # for i in range(iterations):
    #     iteration.append(i)

    # threshold = []
    
    # for i in range(iterations):
    #     threshold.append(thre)

    # # plot1 = plt.figure(1)
    # # plt.plot(iteration[0:index], orig_in_vocab[0:index], 'o', iteration[0:index], updated_vocab[0:index], 'v',
    # #                 iteration[0:index], out_vocab[0:index], '*', iteration[0:index], threshold[0:index], '-')
    
    # # # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
    # # plt.legend(['in-vocab words', 'updating-words',  'out-vocab words', 'threshold'])
    
    # # plt.xlabel("word $i^{th}$")
    # # plt.ylabel('runtime (ms)')
    # # plt.title(title)
    # # # ax = plt.gca()
    # # # ax.set_ylim(1, 5) 
    # # plt_dest = plt_folder + 'attack_result_500words_3.png'
    # # plt.savefig(plt_dest, dpi=300, bbox_inches='tight')

    # plot1 = plt.figure(1)
    # title = "precision = {0}; recall = {1}".format(precision, recall)
    # plt.plot(iteration, in_vo, 'o', iteration, out_vocab, '*', iteration, threshold, '-')
    
    # # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
    # plt.legend(['in-vocab words', 'out-vocab words', 'threshold'])
    
    # plt.xlabel("word $i^{th}$")
    # plt.ylabel('runtime (ms)')
    # plt.title(title)
    # # ax = plt.gca()
    # # ax.set_ylim(1, 5) 
    # plt_dest = plt_folder + 'attack_result_1000words.png'
    # plt.savefig(plt_dest, dpi=300, bbox_inches='tight')



    # sys.exit()

    in_vocab_runtime_list = g[0][0]
    out_vocab_runtime_list = g[0][1]

    in_vocab_runtime_s = [ner_runtime*1000 for ner_runtime in in_vocab_runtime_list]
    out_vocab_runtime_s = [ner_runtime*1000 for ner_runtime in out_vocab_runtime_list]

    # print(in_vocab_runtime_s)
    in_vocab_run_1 = []
    in_vocab_run_2 = []
    in_vocab_run_3 = []

    out_vocab_run_1 = []
    out_vocab_run_2 = []
    out_vocab_run_3 = []

    for i in range(100):
        in_vocab_run_1.append(in_vocab_runtime_s[i*3])
        in_vocab_run_2.append(in_vocab_runtime_s[3*i+1])
        in_vocab_run_3.append(in_vocab_runtime_s[3*i+2])

        out_vocab_run_1.append(out_vocab_runtime_s[i*3])
        out_vocab_run_2.append(out_vocab_runtime_s[3*i+1])
        out_vocab_run_3.append(out_vocab_runtime_s[3*i+2])


    # print(in_vocab_run_1)
    # sample_in_vocab_run_1 = np.array(in_vocab_run_1[0:10])
    # sys.exit()
    # print(len(g[:][:][:][0]))
    # print(len(g[:][:][:][0][0]))
    # print(g[:][:][:][0][0][0])
    # print(g[:][:][:][0][0][1])
    # print(len(g[:][:][:][0][0][0]))


    # pws_list = ['62r?EM', 'W&wy$&8', 'CP5LK*.tW$?#k', '5cUvl$Y', ')9yVE_m$']
    # sys.exit()
    
    iterations =  100
    iteration = []
    for i in range(iterations):
        iteration.append(i)

    index = 100
    plot1 = plt.figure(1)
    plt.plot(iteration[0:index], in_vocab_run_1[0:index], 'o', iteration[0:index], in_vocab_run_2[0:index], 'v',
                    iteration[0:index], in_vocab_run_3[0:index], '*')
    
    # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
    plt.legend(['1st run', '2nd run',  '3rd run'])
    
    plt.xlabel("word $i^{th}$")
    plt.ylabel('runtime (ms)')
    plt.title("In-vocab w/o reload model after each query")
    ax = plt.gca()
    ax.set_ylim(3, 6) 
    plt_dest = plt_folder + '100_in-vocab_without_reload_model_3_runs_each_time_sleep_5sec.png'
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')
    

    plot2 = plt.figure(2)
    plt.plot(iteration[0:index], out_vocab_run_1[0:index], 'o', iteration[0:index], out_vocab_run_2[0:index], 'v',
                    iteration[0:index], out_vocab_run_3[0:index], '*')
    
    # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
    plt.legend(['1st run', '2nd run',  '3rd run'])
    
    plt.xlabel("word $i^{th}$")
    plt.ylabel('runtime (ms)')
    plt.title("Out-vocab w/o reload model after each query")
    ax = plt.gca()
    ax.set_ylim(3, 6) 
    plt_dest = plt_folder + '100_out-vocab_without_reload_model_3_runs_each_time_sleep_5sec.png'
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')


    sys.exit()


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
    for i in range(len(g[0])):
        # print(g)
        # plot1 = plt.figure(i)
        # in_vocab_runtimes_1 = g[:][:][i][0]
        # in_vocab_runtime_1 = [in_vocab_runtime*1000 for in_vocab_runtime in in_vocab_runtimes_1]
        
        # in_std_ner_1 = np.std(np.array(in_vocab_runtime_1))
        # in_mean_ner_1 = np.mean(np.array(in_vocab_runtime_1))

        # print("mean news: ", in_mean_ner_1)
        # print("std news: ", in_std_ner_1)

        # in_vocab_runtimes_2 = g[:][:][i][1]
        # in_vocab_runtime_2 = [in_vocab_runtime*1000 for in_vocab_runtime in in_vocab_runtimes_2]
        
        # in_std_ner_2 = np.std(np.array(in_vocab_runtime_2))
        # in_mean_ner_2 = np.mean(np.array(in_vocab_runtime_2))
        # print("mean people: ", in_mean_ner_2)
        # print("std people: ", in_std_ner_2)
        
        # in_vocab_runtimes_3 = g[:][:][i][2]
        # in_vocab_runtime_3 = [in_vocab_runtime*1000 for in_vocab_runtime in in_vocab_runtimes_3]
        
        # in_std_ner_3 = np.std(np.array(in_vocab_runtime_3))
        # in_mean_ner_3 = np.mean(np.array(in_vocab_runtime_3))
        # print("mean Australia: ", in_mean_ner_3)
        # print("std Australia: ", in_std_ner_3)

        # avg_std_ner_in_vocab = (in_std_ner_1+in_std_ner_2+in_std_ner_3)/3
        # avg_mean_ner_in_vocab = (in_mean_ner_1+in_mean_ner_2+in_mean_ner_3)/3

        # print("avg_mean_ner = ", avg_mean_ner_in_vocab)
        # print("avg_std_ner = ", avg_std_ner_in_vocab)


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

    
    # ### List of 5 passwords with reload model
    # ## password 0
    # out_vocab_password_runtimes = out_vocab_test_runtimes_list[0]
    # out_vocab_password_runtime = [ner_runtime*1000 for ner_runtime in out_vocab_password_runtimes]

    # out_std_pw0 = np.std(np.array(out_vocab_password_runtime))
    # out_mean_pw0 = np.mean(np.array(out_vocab_password_runtime))
    # print("out_mean: ", out_mean_pw0)
    # print("out_std: ", out_std_pw0)


    # for index in range(len(out_vocab_password_runtime)):
    #     if abs(out_vocab_password_runtime[index] - out_mean_pw0) >= (3*out_std_pw0):
    #         out_vocab_password_runtime[index] = out_mean_pw0

    # # ## password 1
    # out_vocab_password_1_runtimes = out_vocab_test_runtimes_list[1]
    # out_vocab_password_1_runtime = [ner_runtime*1000 for ner_runtime in out_vocab_password_1_runtimes]

    # out_std_pw1 = np.std(np.array(out_vocab_password_1_runtime))
    # out_mean_pw1 = np.mean(np.array(out_vocab_password_1_runtime))
    # print("out_mean: ", out_mean_pw1)
    # print("out_std: ", out_std_pw1)


    # for index in range(len(out_vocab_password_1_runtime)):
    #     if abs(out_vocab_password_1_runtime[index] - out_mean_pw1) >= (3*out_std_pw1):
    #         out_vocab_password_1_runtime[index] = out_mean_pw1

    # ## password 2
    # out_vocab_password_2_runtimes = out_vocab_test_runtimes_list[2]
    # out_vocab_password_2_runtime = [ner_runtime*1000 for ner_runtime in out_vocab_password_2_runtimes]

    # out_std_pw2 = np.std(np.array(out_vocab_password_2_runtime))
    # out_mean_pw2 = np.mean(np.array(out_vocab_password_2_runtime))
    # print("out_mean: ", out_mean_pw2)
    # print("out_std: ", out_std_pw2)


    # for index in range(len(out_vocab_password_2_runtime)):
    #     if abs(out_vocab_password_2_runtime[index] - out_mean_pw2) >= (3*out_std_pw2):
    #         out_vocab_password_2_runtime[index] = out_mean_pw2

    # ## password 3
    # out_vocab_password_3_runtimes = out_vocab_test_runtimes_list[3]
    # out_vocab_password_3_runtime = [ner_runtime*1000 for ner_runtime in out_vocab_password_3_runtimes]

    # out_std_pw3 = np.std(np.array(out_vocab_password_3_runtime))
    # out_mean_pw3 = np.mean(np.array(out_vocab_password_3_runtime))
    # print("out_mean: ", out_mean_pw3)
    # print("out_std: ", out_std_pw3)


    # for index in range(len(out_vocab_password_3_runtime)):
    #     if abs(out_vocab_password_3_runtime[index] - out_mean_pw3) >= (3*out_std_pw3):
    #         out_vocab_password_3_runtime[index] = out_mean_pw3

    # ## password 4
    # out_vocab_password_4_runtimes = out_vocab_test_runtimes_list[4]
    # out_vocab_password_4_runtime = [ner_runtime*1000 for ner_runtime in out_vocab_password_4_runtimes]

    # out_std_pw4 = np.std(np.array(out_vocab_password_4_runtime))
    # out_mean_pw4 = np.mean(np.array(out_vocab_password_4_runtime))
    # print("out_mean: ", out_mean_pw4)
    # print("out_std: ", out_std_pw4)


    # for index in range(len(out_vocab_password_4_runtime)):
    #     if abs(out_vocab_password_4_runtime[index] - out_mean_pw4) >= (3*out_std_pw4):
    #         out_vocab_password_4_runtime[index] = out_mean_pw4

    
    # tmp = np.add(out_vocab_test_runtimes_list[0],out_vocab_test_runtimes_list[1])
    # tmp1 = np.add(tmp,out_vocab_test_runtimes_list[2])
    # tmp2 = np.add(tmp1,out_vocab_test_runtimes_list[3])
    # avg_out_vocab_1_runtimes = np.add(tmp2,out_vocab_test_runtimes_list[4])

    # avg_out_vocab_1_runtime = [ner_runtime*200 for ner_runtime in avg_out_vocab_1_runtimes]

    # out_std_100_password = np.std(np.array(avg_out_vocab_1_runtime))
    # out_mean_100_password = np.mean(np.array(avg_out_vocab_1_runtime))
    # print("out_mean_password: ", out_mean_100_password)
    # print("out_std_password: ", out_std_100_password)

    # for index in range(len(avg_out_vocab_1_runtime)):
    #     if abs(avg_out_vocab_1_runtime[index] - out_mean_100_password) >=(3*out_std_100_password):
    #         avg_out_vocab_1_runtime[index] = out_mean_100_password  

    
    # # ### List of 5 passwords without reload model
    # # ## password 0
    # # out_vocab_password_No_reload_runtimes = out_vocab_test_NO_reload_runtimes_list[0]
    # # out_vocab_password_NO_reload_runtime = [ner_runtime*1000 for ner_runtime in out_vocab_password_No_reload_runtimes]

    # # out_std_pw0 = np.std(np.array(out_vocab_password_NO_reload_runtime))
    # # out_mean_pw0 = np.mean(np.array(out_vocab_password_NO_reload_runtime))
    # # print("out_mean: ", out_mean_pw0)
    # # print("out_std: ", out_std_pw0)


    # # for index in range(len(out_vocab_password_NO_reload_runtime)):
    # #     if abs(out_vocab_password_NO_reload_runtime[index] - out_mean_pw0) >= (3*out_std_pw0):
    # #         out_vocab_password_NO_reload_runtime[index] = out_mean_pw0

    # # # ## password 1
    # # out_vocab_password_No_reload_runtimes_1 = out_vocab_test_NO_reload_runtimes_list[1]
    # # out_vocab_password_NO_reload_runtime_1 = [ner_runtime*1000 for ner_runtime in out_vocab_password_No_reload_runtimes_1]

    # # out_std_pw0 = np.std(np.array(out_vocab_password_NO_reload_runtime_1))
    # # out_mean_pw0 = np.mean(np.array(out_vocab_password_NO_reload_runtime_1))
    # # print("out_mean pw1: ", out_mean_pw0)
    # # print("out_std pw1: ", out_std_pw0)


    # # for index in range(len(out_vocab_password_NO_reload_runtime_1)):
    # #     if abs(out_vocab_password_NO_reload_runtime_1[index] - out_mean_pw0) >= (3*out_std_pw0):
    # #         out_vocab_password_NO_reload_runtime_1[index] = out_mean_pw0

    # # ## password 2
    # # out_vocab_password_No_reload_runtimes_2 = out_vocab_test_NO_reload_runtimes_list[2]
    # # out_vocab_password_NO_reload_runtime_2 = [ner_runtime*1000 for ner_runtime in out_vocab_password_No_reload_runtimes_2]

    # # out_std_pw0 = np.std(np.array(out_vocab_password_NO_reload_runtime_2))
    # # out_mean_pw0 = np.mean(np.array(out_vocab_password_NO_reload_runtime_2))
    # # print("out_mean: ", out_mean_pw0)
    # # print("out_std: ", out_std_pw0)


    # # for index in range(len(out_vocab_password_NO_reload_runtime_2)):
    # #     if abs(out_vocab_password_NO_reload_runtime_2[index] - out_mean_pw0) >= (3*out_std_pw0):
    # #         out_vocab_password_NO_reload_runtime_2[index] = out_mean_pw0

    # # ## password 3
    # # out_vocab_password_No_reload_runtimes_3 = out_vocab_test_NO_reload_runtimes_list[3]
    # # out_vocab_password_NO_reload_runtime_3 = [ner_runtime*1000 for ner_runtime in out_vocab_password_No_reload_runtimes_3]

    # # out_std_pw0 = np.std(np.array(out_vocab_password_NO_reload_runtime_3))
    # # out_mean_pw0 = np.mean(np.array(out_vocab_password_NO_reload_runtime_3))
    # # print("out_mean: ", out_mean_pw0)
    # # print("out_std: ", out_std_pw0)


    # # for index in range(len(out_vocab_password_NO_reload_runtime_3)):
    # #     if abs(out_vocab_password_NO_reload_runtime_3[index] - out_mean_pw0) >= (3*out_std_pw0):
    # #         out_vocab_password_NO_reload_runtime_3[index] = out_mean_pw0

    # # ## password 4
    # # out_vocab_password_No_reload_runtimes_4 = out_vocab_test_NO_reload_runtimes_list[4]
    # # out_vocab_password_NO_reload_runtime_4 = [ner_runtime*1000 for ner_runtime in out_vocab_password_No_reload_runtimes_4]

    # # out_std_pw0 = np.std(np.array(out_vocab_password_NO_reload_runtime_4))
    # # out_mean_pw0 = np.mean(np.array(out_vocab_password_NO_reload_runtime_4))
    # # print("out_mean: ", out_mean_pw0)
    # # print("out_std: ", out_std_pw0)


    # # for index in range(len(out_vocab_password_NO_reload_runtime_4)):
    # #     if abs(out_vocab_password_NO_reload_runtime_4[index] - out_mean_pw0) >= (3*out_std_pw0):
    # #         out_vocab_password_NO_reload_runtime_4[index] = out_mean_pw0

    
    # # tmp = np.add(out_vocab_test_NO_reload_runtimes_list[0],out_vocab_test_NO_reload_runtimes_list[1])
    # # tmp1 = np.add(tmp,out_vocab_test_NO_reload_runtimes_list[2])
    # # tmp2 = np.add(tmp1,out_vocab_test_NO_reload_runtimes_list[3])
    # # avg_out_vocab_1_no_reload_runtimes = np.add(tmp2,out_vocab_test_NO_reload_runtimes_list[4])

    # # avg_out_vocab_1_no_reload_runtime = [ner_runtime*200 for ner_runtime in avg_out_vocab_1_no_reload_runtimes]

    # # out_std_100_password = np.std(np.array(avg_out_vocab_1_no_reload_runtime))
    # # out_mean_100_password = np.mean(np.array(avg_out_vocab_1_no_reload_runtime))
    # # print("out_mean_password: ", out_mean_100_password)
    # # print("out_std_password: ", out_std_100_password)

    # # for index in range(len(avg_out_vocab_1_runtimes)):
    # #     if abs(avg_out_vocab_1_no_reload_runtime[index] - out_mean_100_password) >=(3*out_std_100_password):
    # #         avg_out_vocab_1_no_reload_runtime[index] = out_mean_100_password  

    ### 1000w in-vocab test
    in_std_test2 = np.std(np.array(in_vocab_test2_runtime))
    in_mean_test2 = np.mean(np.array(in_vocab_test2_runtime))

    print("in_mean_test2: ", in_mean_test2)
    print("in_std_test2: ", in_std_test2)

    for index in range(len(in_vocab_test2_runtime)):
        if abs(in_vocab_test2_runtime[index] - in_mean_test2) >= (3*in_std_test2):
            in_vocab_test2_runtime[index] = in_mean_test2

    
    # # ### runtime of 100 passwords over 1 run
    # # 0
    # out_vocab_100_password_runtime = [ner_runtime*1000 for ner_runtime in out_vocab_100_password_runtimes[0]]
    # out_std_100_password = np.std(np.array(out_vocab_100_password_runtime))
    # out_mean_100_password = np.mean(np.array(out_vocab_100_password_runtime))
    # print("out_mean_password: ", out_mean_100_password)
    # print("out_std_password: ", out_std_100_password)

    # for index in range(len(out_vocab_100_password_runtime)):
    #     if abs(out_vocab_100_password_runtime[index] - out_mean_100_password) >=(3*out_std_100_password):
    #         out_vocab_100_password_runtime[index] = out_mean_100_password

   
    # # 1
    # out_vocab_100_password_runtime1 = [ner_runtime*1000 for ner_runtime in out_vocab_100_password_runtimes[1]]
    # out_std_100_password = np.std(np.array(out_vocab_100_password_runtime1))
    # out_mean_100_password = np.mean(np.array(out_vocab_100_password_runtime1))
    # print("out_mean_password: ", out_mean_100_password)
    # print("out_std_password: ", out_std_100_password)

    # for index in range(len(out_vocab_100_password_runtime1)):
    #     if abs(out_vocab_100_password_runtime1[index] - out_mean_100_password) >=(3*out_std_100_password):
    #         out_vocab_100_password_runtime1[index] = out_mean_100_password
   
    # # 2
    # out_vocab_100_password_runtime2 = [ner_runtime*1000 for ner_runtime in out_vocab_100_password_runtimes[2]]
    # out_std_100_password = np.std(np.array(out_vocab_100_password_runtime2))
    # out_mean_100_password = np.mean(np.array(out_vocab_100_password_runtime2))
    # print("out_mean_password: ", out_mean_100_password)
    # print("out_std_password: ", out_std_100_password)

    # for index in range(len(out_vocab_100_password_runtime2)):
    #     if abs(out_vocab_100_password_runtime2[index] - out_mean_100_password) >=(3*out_std_100_password):
    #         out_vocab_100_password_runtime2[index] = out_mean_100_password

    # # 3
    # out_vocab_100_password_runtime3 = [ner_runtime*1000 for ner_runtime in out_vocab_100_password_runtimes[3]]
    # out_std_100_password = np.std(np.array(out_vocab_100_password_runtime3))
    # out_mean_100_password = np.mean(np.array(out_vocab_100_password_runtime3))
    # print("out_mean_password: ", out_mean_100_password)
    # print("out_std_password: ", out_std_100_password)

    # for index in range(len(out_vocab_100_password_runtime3)):
    #     if abs(out_vocab_100_password_runtime3[index] - out_mean_100_password) >=(3*out_std_100_password):
    #         out_vocab_100_password_runtime3[index] = out_mean_100_password   



    # # 4
    # out_vocab_100_password_runtime4 = [ner_runtime*1000 for ner_runtime in out_vocab_100_password_runtimes[4]]
    # out_std_100_password = np.std(np.array(out_vocab_100_password_runtime4))
    # out_mean_100_password = np.mean(np.array(out_vocab_100_password_runtime4))
    # print("out_mean_password: ", out_mean_100_password)
    # print("out_std_password: ", out_std_100_password)

    # for index in range(len(out_vocab_100_password_runtime4)):
    #     if abs(out_vocab_100_password_runtime4[index] - out_mean_100_password) >=(3*out_std_100_password):
    #         out_vocab_100_password_runtime4[index] = out_mean_100_password        

    
    
    
    # # average 1000pws runtime over 5 runs
    # tmp2 = np.add(out_vocab_100_password_runtimes[0],out_vocab_100_password_runtimes[1])
    # tmp = np.add(tmp2,out_vocab_100_password_runtimes[2])
    # tmp1 = np.add(tmp,out_vocab_100_password_runtimes[3])
    # avg_out_vocab_100_password_runtimes = np.add(tmp1,out_vocab_100_password_runtimes[4])

    # avg_out_vocab_100_password_runtime = [ner_runtime*200 for ner_runtime in avg_out_vocab_100_password_runtimes]

    # out_std_100_password = np.std(np.array(avg_out_vocab_100_password_runtime))
    # out_mean_100_password = np.mean(np.array(avg_out_vocab_100_password_runtime))
    # print("out_mean_password: ", out_mean_100_password)
    # print("out_std_password: ", out_std_100_password)

    # for index in range(len(avg_out_vocab_100_password_runtime)):
    #     if abs(avg_out_vocab_100_password_runtime[index] - out_mean_100_password) >=(3*out_std_100_password):
    #         avg_out_vocab_100_password_runtime[index] = out_mean_100_password  


    # sys.exit()
    # file_pws = 'passwords_out_vocab_list'
    # g = []
    # h = pickle.load(open(file_pws, 'rb'))

    # g.append(h)

    # pws = g[:][0]

    # for i in range(len(pws)):
    #     if pws[i] == "74QR+H?bQ)xf":
    #         print("i = ", i)
    #         index = i

    # print(out_vocab_1000_password_runtime[index])
    # if out_vocab_1000_password_runtime[index] > in_mean - in_std and out_vocab_1000_password_runtime[index] < in_mean + in_std:
    #     print("it is in the area")

    # # # for i in range(len(out_vocab_1000_password_runtime)):
    # # #     if out_vocab_1000_password_runtime[i] > in_mean - in_std and out_vocab_1000_password_runtime[i] < in_mean + in_std:
    # # #         print(pws[i])


    # sys.exit()

    # means = []
    # stds = []
    # for i in range(len(in_vocab_test_runtime)):
    #     means.append(avg_mean_ner_in_vocab)
    # for i in range(len(in_vocab_test_runtime)):
    #     stds.append(avg_std_ner_in_vocab)

    # mean = np.array(means, dtype=np.float64)
    # std = np.array(stds, dtype=np.float64)

    # clrs = sns.color_palette("husl", 1)
    # with sns.axes_style("darkgrid"):
    #     plot1 = plt.figure(1)
    #     # fig, ax = plt.subplots()
    #     plt.plot(iteration, in_vocab_test2_runtime, 'o', iteration, out_vocab_password_runtime, 'v',
    #                 iteration, out_vocab_password_1_runtime, '<', iteration, out_vocab_password_2_runtime, '>', 
    #                 iteration, out_vocab_password_3_runtime, '<', iteration, out_vocab_password_4_runtime, '>', iteration, means, '-')
    #     plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
    #     plt.legend(['ner: 1000w in-vocab', 'ner: "62r?EM"', 'ner: "W&wy$&8"', 'ner: "CP5LK*.tW$?#k"', 
    #                  'ner: "5cUvl$Y"', 'ner: ")9yVE_m$"', 'average in-vocab mean +/- std'])
    #     plt.xlabel("word $i^{th}$")
    #     plt.ylabel('runtime (ms)')
    #     # ax = plt.gca()
    #     # ax.set_ylim(0, 9) 
    #     plt_dest = plt_folder + '5_out_pws_1000in_test_runtime_20210318_mean1.png'
    #     plt.savefig(plt_dest, dpi=300, bbox_inches='tight')

    # clrs = sns.color_palette("husl", 1)
    # with sns.axes_style("darkgrid"):
    #     plot1 = plt.figure(2)
    #     # fig, ax = plt.subplots()
    #     plt.plot(iteration, in_vocab_test2_runtime, 'o', iteration, out_vocab_1000_password_runtime, 'v', iteration, means, '-')
    #     plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
    #     plt.legend(['ner: 1000w in-vocab', 'ner: 1000 password out-vocab', 'average in-vocab mean +/- std'])
    #     plt.xlabel("word $i^{th}$")
    #     plt.ylabel('runtime (ms)')
    #     # ax = plt.gca()
    #     # ax.set_ylim(0, 9) 
    #     plt_dest = plt_folder + '1000out_pw_1000in_test_runtime_20210318_mean1.png'
    #     plt.savefig(plt_dest, dpi=300, bbox_inches='tight')


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


    # print(len(in_vocab_test2_runtime))
    # print(len(out_vocab_password_4_runtime))
    # print(len(out_vocab_password_3_runtime))
    # print(len(out_vocab_password_1_runtime))
    # print(len(out_vocab_password_2_runtime))
    # print(len(out_vocab_password_runtime))
    # print(len(means))
    
    # print(in_vocab_test2_runtime)
    # list_5_pw:['<EzPRIlq1', '6pB%3e', 'Q3u4%6rjD#I!e', 'FM&vJ3(rc-lr5T', 'QMrjxavQ$CZZ&?9']
    # list_5_pw:['eo2idT8qJ41>SZp', '9X=oYd&z9r7', 'y,n7ZZI', 'S,>!FTkGx6', 'Y*S8f(k,U?X9bc!Q']
    # list_5_pw:['Y1W4a4&kK%pdP', 'B*)D5xYNF*pA', 'eZWT6Ua#<9N^z', 'M<C1DOa', '2ejcNF9YK!U^24=']
    clrs = sns.color_palette("husl", 1)
    with sns.axes_style("darkgrid"):
        # plot1 = plt.figure(1)
        # plt.plot(iteration, in_vocab_test2_runtime, 'o', iteration, out_vocab_password_runtime, 'v',
        #             iteration, out_vocab_password_1_runtime, 'v', iteration, out_vocab_password_2_runtime, 'v', 
        #             iteration, out_vocab_password_3_runtime, 'v', iteration, out_vocab_password_4_runtime, 'v', iteration, means, '-')
        # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
        # plt.legend(['ner: 100w in-vocab', 'ner: "Y1W4a4&kK%pdP"', 'ner: "B*)D5xYNF*pA"', 'ner: "eZWT6Ua#<9N^z"', 
        #              'ner: "M<C1DOa"', 'ner: "2ejcNF9YK!U^24="', '1000w in-vocab mean +/- std'])

        # plt.plot(iteration, in_vocab_test2_runtime, 'o', iteration, avg_out_vocab_1_runtime, 'v',
        #             iteration, means, '-')
        # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
        # plt.legend(['ner: 100w in-vocab', 'ner: "4u5.9Df"', '1000w in-vocab mean +/- std'])
        # # plt.plot(iteration, in_vocab_test2_runtime, 'o', iteration, out_vocab_password_runtime, 'v', iteration, means, '-')
        # # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
        # # plt.legend(['ner: 1000w in-vocab', 'ner: "74QR+H?bQ)xf"', '1000w in-vocab mean +/- std'])
        # plt.xlabel("word $i^{th}$")
        # plt.ylabel('runtime (ms)')
        # ax = plt.gca()
        # ax.set_ylim(2.5, 3.8) 
        # plt.title("With reload model (avg of 5 runs)")
        # plt_dest = plt_folder + '5_out_pws_1000w-in_test_runtime_mean2_reload_model_avg_5runs.png'
        # plt.savefig(plt_dest, dpi=300, bbox_inches='tight')

        plot1 = plt.figure(1)
        # plt.plot(iteration, in_vocab_test2_runtime, 'o', iteration, out_vocab_password_NO_reload_runtime, 'v',
        #              iteration, out_vocab_password_NO_reload_runtime_2, 'v', 
        #             iteration, out_vocab_password_NO_reload_runtime_3, 'v', iteration, out_vocab_password_NO_reload_runtime_4, 'v', iteration, means, '-')
        
        plt.plot(iteration, in_vocab_test2_runtime, 'o', iteration, out_vocab_test_runtimes_list_no_reload_s, 'v',
                      iteration, means, '-')
        
        plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
        plt.legend(['ner: 1000w in-vocab', 'ner: 1000 out-vocab',  '1000w in-vocab mean +/- std'])
        
        plt.xlabel("word $i^{th}$")
        plt.ylabel('runtime (ms)')
        # ax = plt.gca()
        # ax.set_ylim(2.5, 3.8) 
        plt.title("Without reload model after each query")
        plt_dest = plt_folder + '1000_out-vocab_without_reload_model.png'
        plt.savefig(plt_dest, dpi=300, bbox_inches='tight')


        plot2 = plt.figure(2)
        # plt.plot(iteration, in_vocab_test2_runtime, 'o', iteration, out_vocab_100_password_runtime, 'v',
        #             iteration, out_vocab_100_password_runtime1, '<', iteration, out_vocab_100_password_runtime2, '>', 
        #             iteration, out_vocab_100_password_runtime3, '^', iteration, out_vocab_100_password_runtime4, 's', 
        #             iteration, means, '-')
        plt.plot(iteration, in_vocab_test2_runtime, 'o', iteration, out_vocab_test_runtimes_list_reload_s, 'v', iteration, means, '-')
        plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
        plt.legend(['ner: 1000w in-vocab', 'ner: 1000w out-vocab', 'mean +/- std (1000w in-vocab)'])
        plt.xlabel("word $i^{th}$")
        plt.ylabel('runtime (ms)')
        # ax = plt.gca()
        # ax.set_ylim(0, 9) 
        plt.title("Reload model after each query")
        plt_dest = plt_folder + '11000_out-vocab_reload_model.png'
        plt.savefig(plt_dest, dpi=300, bbox_inches='tight')


        # plot3 = plt.figure(3)
        # plt.plot(iteration, in_vocab_test2_runtime, 'o', iteration, out_vocab_100_password_runtime2, 'v', iteration, means, '-')
        # plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
        # plt.legend(['ner: 1000w in-vocab', 'ner: 1000w out-vocab (1 runs)', '1000w in-vocab mean +/- std'])
        # plt.xlabel("word $i^{th}$")
        # plt.ylabel('runtime (ms)')
        # # ax = plt.gca()
        # # ax.set_ylim(0, 9) 
        # plt_dest = plt_folder + '1000_out_pws_1run_1000in_test_runtime_mean2_perf_counter_timer.png'
        # plt.savefig(plt_dest, dpi=300, bbox_inches='tight')

    # clrs = sns.color_palette("husl", 1)
    # with sns.axes_style("darkgrid"):
    #     plot1 = plt.figure(4)
    #     # fig, ax = plt.subplots()
    #     plt.plot(iteration, in_vocab_test2_runtime, 'o', iteration, out_vocab_1000_password_runtime, 'v', iteration, means, '-')
    #     plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
    #     plt.legend(['ner: 1000w in-vocab', 'ner: 1000 password out-vocab', '1000w in-vocab mean +/- std'])
    #     plt.xlabel("word $i^{th}$")
    #     plt.ylabel('runtime (ms)')
    #     # ax = plt.gca()
    #     # ax.set_ylim(0, 9) 
    #     plt_dest = plt_folder + '1000out_pw_1000in_test_runtime_20210318_mean2.png'
        # plt.savefig(plt_dest, dpi=300, bbox_inches='tight')


    

        
    import numpy as np
    from sklearn import metrics
    vocab_in = np.zeros(len(in_vocab_test2_runtime)) 
    # print(vocab_out)
    vocab_out = np.ones(len(in_vocab_test2_runtime))
    # print(vocab_in)
    vocabs = [*vocab_in,*vocab_out]
    
    y = vocabs
    print(y)
    time = [*in_vocab_test2_runtime, *out_vocab_test_runtimes_list_no_reload_s]
    scores = np.array(time)
    print(scores)
    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
        
    print(thresholds)
    print(fpr)
    print(tpr)

    roc_auc = metrics.auc(fpr, tpr)
    print(roc_auc)
    
    random_guess = [0 for _ in range(len(y))]
    rg_fpr, rg_tpr, rg_thresholds = metrics.roc_curve(y, random_guess, pos_label=1)
    
    print(rg_thresholds)
    print(rg_fpr)
    print(rg_tpr)

    fig, ax = plt.subplots(figsize=(10,7))
    ax.plot(fpr, tpr, '-o', rg_fpr, rg_tpr, '--')
    # ax.plot(np.linspace(0, 1, 4),
    #         np.linspace(0, 1, 4),
    #         label='baseline',
    #         linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    # plt.legend(fontsize=12)
    plt_dest = plt_folder + 'roc_auc_1000_invocab_1000_out_vocab_without_reload.png'
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')    


    # 1 word, 1000 runs
    time2 = [*in_vocab_test2_runtime, *out_vocab_test_runtimes_list_reload_s]
    scores = np.array(time2)
    print(scores)
    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
        
    print(thresholds)
    print(fpr)
    print(tpr)

    roc_auc = metrics.auc(fpr, tpr)
    print(roc_auc)
    
    random_guess = [0 for _ in range(len(y))]
    rg_fpr, rg_tpr, rg_thresholds = metrics.roc_curve(y, random_guess, pos_label=1)
    
    # print(rg_thresholds)
    # print(rg_fpr)
    # print(rg_tpr)

    fig, ax = plt.subplots(figsize=(10,7))
    ax.plot(fpr, tpr, '-o', rg_fpr, rg_tpr, '--')
    # ax.plot(np.linspace(0, 1, 4),
    #         np.linspace(0, 1, 4),
    #         label='baseline',
    #         linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    # plt.legend(fontsize=12)
    plt_dest = plt_folder + 'roc_auc_1000_invocab_1000_out-vocab_reload.png'
    plt.savefig(plt_dest, dpi=300, bbox_inches='tight')   





    # ### 1000 words, 1 run
    # time3 = [*in_vocab_test2_runtime, *out_vocab_100_password_runtime]
    # scores = np.array(time3)
    # print(scores)
    # fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
        
    # print(thresholds[20])
    # print(fpr[20])
    # print(tpr[20])
    


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
    # plt_dest = plt_folder + 'roc_auc_1000_invocab_1000_password_1run.png'
    # plt.savefig(plt_dest, dpi=300, bbox_inches='tight')    




