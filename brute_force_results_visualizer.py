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
from Levenshtein import distance as levenshtein_distance
#import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from itertools import cycle
import itertools
from collections import OrderedDict
from sklearn.metrics import accuracy_score, confusion_matrix

def mkdir_p(path):
    """To make a directory given a path."""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
            
def unpack_data(res_folder=None):
    g = []
    br = False
    files = os.listdir(res_folder)
    for file_name in files:
        print(file_name)
        file_path = os.path.join(res_folder, file_name)
        h = pickle.load(open(file_path, 'rb'))
        g.append(h)
        if br:
            break
    
    print('Read Disk')
    print('{} TEST RUNS FOUND'.format(len(g)))

    return g




#res_folder = '2-3-4-missing-chars/'
res_folder = 'Brute_Force_Results/data/'

plot_folder = "Brute_Force_Results/PLOTS/"
mkdir_p(plot_folder)

g = unpack_data(res_folder)

for run in g:
    secrets = run[0]
    missing_char = run[1]
    extracted = run[2]
    ranks = run[3]
    
    secret_shapes = run[6]
    extracted_shapes = run[7]
    
    secret_norms = run[8]
    extracted_norms = run[9]
    
    secrets = run[0]
    
    accuracy = accuracy_score(secrets, extracted)
    shape_accuracy = accuracy_score(secret_shapes, extracted_shapes)
    norm_accuracy = accuracy_score(secret_norms, extracted_norms)
    
    avg_rank = np.mean(np.array(ranks))
    
    proportion = (avg_rank/62**missing_char)*100
    
    prob = 1/avg_rank/62**missing_char

    res = """|| {} Missing CHARS: Accuracy = {} || Norm Accuracy = {} || 
    Shape Accuracy = {} || Average Rank = {} || Proportion = {:.2f}% || Probability = {} 
    """.format(missing_char, accuracy, norm_accuracy, shape_accuracy, avg_rank, proportion, prob)

    print(res)

    file1 = open("{}Results.txt".format(plot_folder),"w")

    file1.write(res)