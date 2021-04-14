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
    for file_name in files:
        br += 1
        print(file_name)
        if file_name == "20210318_timing_out_vocab_100_run_no_refresh_vocab_5pws_2.pickle3":
            file_path = os.path.join(res_folder, file_name)
            h = pickle.load(open(file_path, 'rb'))
            g.append(h)
    print('Read Disk')
    print('{} FILES FOUND'.format(len(g)))

    plt_folder = '{}_PLOTS/'.format(folder)

    mkdir_p(plt_folder)


    # secret_index = 3

    
    
    # print(g)
    # print(len(g))

    # print(g[:][:][0][0])
    # print(g[:][:][0][1])

    # in_vocab_runtime = g[:][:][0][0]
    # out_vocab_runtime = g[:][:][0][1]

    
    iterations = 100
    iteration = []
    for i in range(iterations):
        iteration.append(i)

    # print(iteration)


    # %matplotlib inline
    
    
    # plt.plot(iteration, in_vocab_runtime, 'o', iteration, out_vocab_runtime, 'v')
    # plt.legend(['in vocab', 'out vocab'])
    # plt.xlabel('Iteration i_th')
    # plt.ylabel('runtime (s)')
    # ax = plt.gca()
    # ax.set_ylim(0.00015, 0.00035)
    # file_name = "runtime_distribution.pdf"
    # plt.savefig(file_name, dpi=300, bbox_inches='tight')

    plot_names = []
    for plot_name in files:
        plot_name = plot_name.split('.')[0]
        plot_names.append(plot_name)
    print(plot_names)

    for i in range(len(g)):
        print(i)
        plot1 = plt.figure(i)

        runtimes = g[:][:][i][0][0]
        runtime_ms = [ner_runtime*1000 for ner_runtime in runtimes]

        runtimes_1 = g[:][:][i][0][1]
        runtime_ms_1 = [ner_runtime*1000 for ner_runtime in runtimes_1]

        runtimes_2 = g[:][:][i][0][2]
        runtime_ms_2 = [ner_runtime*1000 for ner_runtime in runtimes_2]

        runtimes_3 = g[:][:][i][0][3]
        runtime_ms_3 = [ner_runtime*1000 for ner_runtime in runtimes_3]

        runtimes_4 = g[:][:][i][0][4]
        runtime_ms_4 = [ner_runtime*1000 for ner_runtime in runtimes_4]

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

        # ner_runtimes = g[:][:][i][5]
        # ner_runtime = [ner_runtime*1000 for ner_runtime in ner_runtimes]

        # attr_runtimes = g[:][:][i][6]
        # attr_runtime = [attr_runtime*1000 for attr_runtime in attr_runtimes]

        # lemma_runtimes = g[:][:][i][7]
        # lemma_runtime = [lemma_runtime*1000 for lemma_runtime in lemma_runtimes]


        # plt.plot(iteration, whole_nlb_runtime, 'o', iteration, tokeniser_runtime, 'x', iteration, tok2vec_runtime, 'v', iteration, tagger_runtime, 's',
        #            iteration, parser_runtime, '>', iteration, ner_runtime, '<', iteration, attr_runtime, '*', iteration, lemma_runtime, '^')
        # # plt.plot(iteration, in_vocab_runtimes, 'o', iteration, out_vocab_runtimes, 'v')
        # plt.legend(['nlp', 'tokenizer', 'tok2vec', 'tagger', 'parser', 'ner', 'attr', 'lemma'])

        plt.plot(iteration, runtime_ms, 'o', iteration, runtime_ms_1, 'o', iteration, runtime_ms_2, 'o',
                    iteration, runtime_ms_3, 'o', iteration, runtime_ms_4, 'o')
        plt.title("Runtime of out-vocab without reload model")
        plt.legend(['ner: M^9fGV>nW4', 'ner: 4.cy<Sk-%4', 'ner: ThcPh$H?2h*', 'ner: B.tv1&b', 'ner: 7XDb#IJNnfj%x(3'])
        plt.xlabel("$i^{th}$")
        plt.ylabel('runtime (ms)')
        # ax = plt.gca()
        # ax.set_ylim(0, 9) 
        plt_dest = plt_folder + 'runtime_out_vocab_100runs_no_refresh_vocab_5pws_2.png'
        plt.savefig(plt_dest, dpi=300, bbox_inches='tight')

    










###############################################################################
    # for i in range(len(g)):
    #     print(i)
    #     plot1 = plt.figure(i)
    #     whole_nlp_runtimes = g[:][:][i][0]
    #     whole_nlb_runtime = [nlp_runtime*1000 for nlp_runtime in whole_nlp_runtimes]

    #     whole_nlp_runtimes_out = g[:][:][i][1]
    #     whole_nlb_runtime_out = [nlp_runtime*1000 for nlp_runtime in whole_nlp_runtimes_out]

    #     # tok2vec_runtimes = g[:][:][i][1]
    #     # tok2vec_runtime = [tok2vec_runtime*1000 for tok2vec_runtime in tok2vec_runtimes]

    #     # tagger_runtimes = g[:][:][i][2]
    #     # tagger_runtime = [tagger_runtime*1000 for tagger_runtime in tagger_runtimes]

    #     # parser_runtimes = g[:][:][i][3]
    #     # parser_runtime = [parser_runtime*1000 for parser_runtime in parser_runtimes]

    #     # ner_runtimes = g[:][:][i][8]
    #     # ner_runtime = [ner_runtime*1000 for ner_runtime in ner_runtimes]

    #     # ner_runtimes_out = g[:][:][i][9]
    #     # ner_runtime_out = [ner_runtime*1000 for ner_runtime in ner_runtimes_out]

    #     # attr_runtimes = g[:][:][i][5]
    #     # attr_runtime = [attr_runtime*1000 for attr_runtime in attr_runtimes]

    #     # lemma_runtimes = g[:][:][i][6]
    #     # lemma_runtime = [lemma_runtime*1000 for lemma_runtime in lemma_runtimes]


    #     plt.plot(iteration, whole_nlb_runtime, 'o',  iteration, whole_nlb_runtime_out, '<')
    #     # plt.plot(iteration, in_vocab_runtimes, 'o', iteration, out_vocab_runtimes, 'v')
    #     plt.legend(['nlp: in vocab', 'nlp: out vocab'])
    #     plt.xlabel("word $i^{th}$")
    #     plt.ylabel('runtime (ms)')
    #     ax = plt.gca()
    #     ax.set_ylim(5, 8) 
    #     plt_dest = plt_folder + 'compare_runtime_nlp_{}.png'.format(plot_names[i])
    #     plt.savefig(plt_dest, dpi=300, bbox_inches='tight')


    # for i in range(len(g)):
    #     print(i)
    #     plot2 = plt.figure(i+1)
    #     # whole_nlp_runtimes = g[:][:][i][0]
    #     # whole_nlb_runtime = [nlp_runtime*1000 for nlp_runtime in whole_nlp_runtimes]

    #     # whole_nlp_runtimes_out = g[:][:][i][1]
    #     # whole_nlb_runtime_out = [nlp_runtime*1000 for nlp_runtime in whole_nlp_runtimes_out]

    #     # tok2vec_runtimes = g[:][:][i][1]
    #     # tok2vec_runtime = [tok2vec_runtime*1000 for tok2vec_runtime in tok2vec_runtimes]

    #     # tagger_runtimes = g[:][:][i][2]
    #     # tagger_runtime = [tagger_runtime*1000 for tagger_runtime in tagger_runtimes]

    #     # parser_runtimes = g[:][:][i][3]
    #     # parser_runtime = [parser_runtime*1000 for parser_runtime in parser_runtimes]

    #     ner_runtimes = g[:][:][i][8]
    #     ner_runtime = [ner_runtime*1000 for ner_runtime in ner_runtimes]

    #     ner_runtimes_out = g[:][:][i][9]
    #     ner_runtime_out = [ner_runtime*1000 for ner_runtime in ner_runtimes_out]

    #     # attr_runtimes = g[:][:][i][5]
    #     # attr_runtime = [attr_runtime*1000 for attr_runtime in attr_runtimes]

    #     # lemma_runtimes = g[:][:][i][6]
    #     # lemma_runtime = [lemma_runtime*1000 for lemma_runtime in lemma_runtimes]


    #     plt.plot(iteration, ner_runtime, 's',  iteration, ner_runtime_out, '*')
    #     # plt.plot(iteration, in_vocab_runtimes, 'o', iteration, out_vocab_runtimes, 'v')
    #     plt.legend(['ner: in vocab', 'ner: out_vocab'])
    #     plt.xlabel("word $i^{th}$")
    #     plt.ylabel('runtime (ms)')
    #     ax = plt.gca()
    #     ax.set_ylim(2.5, 3.5) 
    #     plt_dest = plt_folder + 'comapre_runtime_ner_{}.png'.format(plot_names[i])
    #     plt.savefig(plt_dest, dpi=300, bbox_inches='tight')
    


