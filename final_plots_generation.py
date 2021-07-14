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


def line_plot(plot_data, loss_data, ner_score_data):

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
    
    label_dict = {
        "passwords": "Passwords",
        "credit_card_numbers": "Credit Card Numbers",
        "phone_numbers": "Phone Numbers",
        "ip_addresses": "IP Addresses"

    }

    for data in plot_data:
        epochs = data[0]
        ranks = data[1]
        insertion = data[2]
        attack_type = data[3]
    
        ax1.plot(epochs, ranks, linestyle=linestyles_dict[next(linestype)], label='{}'.format(label_dict[attack_type]))

    ax1.set_ylabel("Ranks/F1")
    ax1.set_xlabel("Epochs")


    ## FOR AVERAGE
    # ner_score_data = [np.mean(np.array(t)) for t in list(zip(*ner_score_data))]    
    # ax1.plot(epochs, ner_score_data, label='Averaged F1', alpha=0.5)


    ## FOR INDIVIDUAL
    for i in range(len(plot_data)):
        data = ner_score_data[i]
        attack_type = plot_data[i][3]
        ax1.plot(epochs, data, linestyle=linestyles_dict[next(linestype)], label='F1-{}'.format(label_dict[attack_type]))

    ax2 = ax1.twinx()
    ax2.set_ylabel("NER Loss")

    ## FOR AVERAGE
    # loss_data = [np.mean(np.array(t)) for t in list(zip(*loss_data))]    
    # ax2.plot(epochs, loss_data, label='Averaged Loss', alpha=0.5)

    ## FOR INDIVIDUAL
    for i in range(len(plot_data)):
        data = loss_data[i]
        attack_type = plot_data[i][3]
    
        ax2.plot(epochs, data, label='Loss - {}'.format(label_dict[attack_type]), alpha=0.5)

    

    file_name = 'Results/FINAL_PLOTS/RANK_PER_EPOCH_AND_INSERTION_AVERAGED_LINE_PLOT_MULTIPLE_EXPERIMENTS_APPENDIX.pdf'
        
    ax1.legend(prop={'size': 5}, loc="upper right", frameon=False,fancybox=None)
    ax2.legend(prop={'size': 5}, loc="center right", frameon=False,fancybox=None)
    
    fig.tight_layout()
    plt_dest = file_name
    plt.savefig(plt_dest,
            bbox_inches="tight")



if __name__ == "__main__":

    attack_types = ['passwords','credit_card_numbers','phone_numbers','ip_addresses']
    plot_data = []
    loss_data = []
    ner_score_data = []
    for attack_type in attack_types:
        path = "Results/Figure 5/RANK_PER_EPOCH_AND_INSERTION_AVERAGED_LINE_PLOT_spacy3.0.3_{}.pickle".format(attack_type)
        print(path)
        file = open(path, 'rb')
        data = pickle.load(file)
        print(len(data))
        plot_data.extend(data[0])
        loss_data.append(data[1])
        ner_score_data.append(data[2])
        print(len(loss_data))
    print(len(loss_data))
    line_plot(plot_data, loss_data, ner_score_data)
        


#python spacy_mi_dictionary_attack_results_visualizer.py --attack_type passwords --loc results/20210302_spacy_3.0.3_10_passwords_dictionary_attack_5_insertions_50_epoch_2000_r_space_0_knowledge_strength_0.3-1.0_features_x