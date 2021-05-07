import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import math
import statistics as st
import heapq
import operator
import errno
from zxcvbn import zxcvbn
from itertools import islice
#from password_strength import PasswordStats
import argparse
from mpl_toolkits.mplot3d import Axes3D
import re
import itertools
from collections import OrderedDict
#from Levenshtein import distance as levenshtein_distance


def mkdir_p(path):
    """To make a directory given a path."""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def line_plot(plot_data, loss_data):

    fig, ax1 = plt.subplots(num=None, figsize=(6, 3.2), dpi=500, facecolor='w', edgecolor='k')

    linestyles_dict = OrderedDict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
     
    lines = ["-","--","-.",":", 'None']
    linestype = itertools.cycle(("dashed","densely dashed","dotted","dashdotdotted"))
    
    for data in plot_data:
        epochs = data[0]
        ranks = data[1]
        insertion = data[2]
        attack_type = data[3]
    
        ax1.plot(epochs, ranks, linestyle=linestyles_dict[next(linestype)], label='{}'.format(attack_type))

    ax1.set_ylabel("Ranks")
    ax1.set_xlabel("Epochs")

    ax2 = ax1.twinx()
    ax2.set_ylabel("NER Loss")

    for i in range(len(plot_data)):
        data = loss_data[i]
        attack_type = plot_data[i][3]
    
        ax2.plot(epochs, data, label='Loss - {}'.format(attack_type), alpha=0.5)

    

    file_name = 'Results/FINAL_PLOTS/RANK_PER_EPOCH_AND_INSERTION_AVERAGED_LINE_PLOT_MULTIPLE_EXPERIMENTS.pdf'
        
    ax1.legend(prop={'size': 6})
    ax2.legend(prop={'size': 6}, loc="center right")
    
    fig.tight_layout()
    plt_dest = file_name
    plt.savefig(plt_dest,
            bbox_inches="tight")



if __name__ == "__main__":

    attack_types = ['passwords','credit_card_numbers','phone_numbers','ip_addresses']
    plot_data = []
    loss_data = []
    for attack_type in attack_types:
        path = "Results/FINAL_PLOTS/SECRET_TYPE_RANK_PER_EPOCH_RESULTS/RANK_PER_EPOCH_AND_INSERTION_AVERAGED_LINE_PLOT_spacy3.0.3_{}.pickle".format(attack_type)
        print(path)
        file = open(path, 'rb')
        data = pickle.load(file)
        print(len(data))
        plot_data.extend(data[0])
        loss_data.append(data[1])
        print(len(loss_data))
    print(len(loss_data))
    line_plot(plot_data, loss_data)
        


#python spacy_mi_dictionary_attack_results_visualizer.py --attack_type passwords --loc results/20210302_spacy_3.0.3_10_passwords_dictionary_attack_5_insertions_50_epoch_2000_r_space_0_knowledge_strength_0.3-1.0_features_x