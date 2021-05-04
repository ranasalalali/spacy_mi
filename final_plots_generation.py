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


def line_plot(plot_data):

    plt.figure(num=None, figsize=(6, 3.2), dpi=500, facecolor='w', edgecolor='k')

    for data in plot_data:
        epochs = data[0]
        ranks = data[1]
        insertion = data[2]
        attack_type = data[3]
    
        plt.plot(epochs, ranks, label='{}'.format(attack_type))

        plt.ylabel("Ranks")
        plt.xlabel("Epochs")

        file_name = 'Results/FINAL_PLOTS/RANK_PER_EPOCH_AND_INSERTION_AVERAGED_LINE_PLOT_MULTIPLE_EXPERIMENTS.pdf'
            
        plt.legend()

        plt.tight_layout()
        plt_dest = file_name
        plt.savefig(plt_dest,
                bbox_inches="tight")



if __name__ == "__main__":

    attack_types = ['passwords','credit_card_numbers','phone_numbers','ip_addresses']
    plot_data = []
    for attack_type in attack_types:
        path = "Results/FINAL_PLOTS/RANK_PER_EPOCH_AND_INSERTION_AVERAGED_LINE_PLOT_spacy3.0.3_{}.pickle".format(attack_type)
        print(path)
        file = open(path, 'rb')
        data = pickle.load(file)
        plot_data.extend(data)

    
    line_plot(plot_data)
        