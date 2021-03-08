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
    files = os.listdir(res_folder)
    for file_name in files:
        print(file_name)
        file_path = os.path.join(res_folder, file_name)
        h = pickle.load(open(file_path, 'rb'))
        g.append(h)
    
    print('Read Disk')
    print('{} TEST RUNS FOUND'.format(len(g)))

    return g

def format_string(s):
    escaped = re.escape(s)
    return escaped

def word_shape(text=None):
    if len(text) >= 100:
        return "LONG"
    shape = []
    last = ""
    shape_char = ""
    seq = 0
    for char in text:
        if char.isalpha():
            if char.isupper():
                shape_char = "X"
            else:
                shape_char = "x"
        elif char.isdigit():
            shape_char = "d"
        else:
            shape_char = char
        if shape_char == last:
            seq += 1
        else:
            seq = 0
            last = shape_char
        if seq < 4:
            shape.append(shape_char)
    return "".join(shape)

def feature_distance(target=None, password=None):

    shape_t = word_shape(target)
    shape_p = word_shape(password)
    shape_distance = levenshtein_distance(shape_t, shape_p)

    prefix_t = target[1]
    prefix_p = password[1]
    prefix_distance = levenshtein_distance(prefix_t, prefix_p)

    suffix_t = target[-3]
    suffix_p = password[-3]
    suffix_distance = levenshtein_distance(suffix_t, suffix_p)

    norm_t = target.lower()
    norm_p = password.lower()
    norm_distance = levenshtein_distance(norm_p, norm_t)

    pref_suff = prefix_distance + suffix_distance
    pref_shape = prefix_distance + shape_distance
    suff_shape = suffix_distance + shape_distance
    pref_suff_shape = prefix_distance + suffix_distance + shape_distance
    pref_suff_shape_norm = shape_distance + prefix_distance + suffix_distance + norm_distance

    return (pref_suff, pref_shape, suff_shape, pref_suff_shape, pref_suff_shape_norm)

def get_feature_passwords(n_feature_passwords=0, feature=None, secret=None):
    feature_passwords = []

    if n_feature_passwords>0:
        feature_passwords_file = 'r_space_data/password_{}_features_{}_{}_passwords.pickle3'.format(secret, feature, n_feature_passwords)
        file = open(feature_passwords_file, 'rb')
        feature_passwords = pickle.load(file)
    else:
        feature_passwords = []

    return feature_passwords


def fig_feature_distance_rank_per_password(feature_distance_ranks_per_password=None, secret=None):
    std_error_per_dist = []
    for dist in feature_distance_ranks_per_password.keys():
        std_error_per_dist.append(np.std(np.array(feature_distance_ranks_per_password[dist])))
        feature_distance_ranks_per_password[dist] = np.mean(np.array(feature_distance_ranks_per_password[dist]))

    #FEATURE_DISTANCE_RANK_PER_PASSWORD
    fig = plt.figure(num=None, figsize=(8, 6), dpi=500, facecolor='w', edgecolor='k')
    plt.errorbar(feature_distance_ranks_per_password.keys(), feature_distance_ranks_per_password.values(), std_error_per_dist)
    plt.xlabel('DISTANCE')
    plt.ylabel('RANK')
    plt.title('FEATURE DISTANCE RANKS of target password {}'.format(secret))
    #plt.legend()
    plt.tight_layout()
    image_name = secret.replace('.','(dot)')
    plt_dest = plt_folder + 'FEATURE_DISTANCE_RANKS_{}'.format(image_name)
    plt.savefig(plt_dest,
            bbox_inches="tight")
    #FEATURE_DISTANCE_RANK_PER_PASSWORD END



def fig_cdf_per_password(all_passwords=None, radius=5, all_password_ranks=None, all_password_dist=None, all_password_shape=None, all_password_shape_dist=None):
    secret_rank_index = all_passwords.index(secret)

    if (len(all_passwords)-radius) >= secret_rank_index >= radius:
        secret_neighbour_index_left = secret_rank_index - radius
        secret_neighbour_rank_left = all_password_ranks[secret_neighbour_index_left]

        secret_neighbour_index_right = secret_rank_index + radius
        secret_neighbour_rank_right = all_password_ranks[secret_neighbour_index_right]
        
    else:
        secret_neighbour_index_left = 0
        secret_neighbour_rank_left = all_password_ranks[secret_neighbour_index_left]

        secret_neighbour_index_right = secret_rank_index + radius
        secret_neighbour_rank_right = all_password_ranks[secret_neighbour_index_right]
                
    fig = plt.figure(num=None, figsize=(8, 6), dpi=500, facecolor='w', edgecolor='k')
    yvals = np.zeros(len(all_passwords))
    total_passwords = len(all_passwords)
    for i in range(len(all_passwords)):
        yvals[i] = (i)/total_passwords
    x = all_password_ranks[secret_neighbour_index_left:secret_neighbour_index_right]
    y = yvals[secret_neighbour_index_left:secret_neighbour_index_right]
    plt.plot(x, y, 'k-', alpha=0.4, label='target_password = {} \n average rank = {} \n rank based of avg rank = {}'.format(secret, target_password_rank, secret_rank_index))
    for i in range(secret_neighbour_index_left, secret_neighbour_index_right):            
        if all_passwords[i] == secret:
            plt.annotate("{} - {} - {} - {}".format(all_password_dist[i], format_string(all_passwords[i]), all_password_shape[i], all_password_shape_dist[i]), (all_password_ranks[i], yvals[i]))
            plt.plot(all_password_ranks[i], yvals[i], 'x', color='green')
        elif all_passwords[i] in feature_passwords:
            plt.annotate("{} - {} - {} - {}".format(all_password_dist[i], format_string(all_passwords[i]), all_password_shape[i], all_password_shape_dist[i]), (all_password_ranks[i], yvals[i]))
            plt.plot(all_password_ranks[i], yvals[i], 'o', color='red', alpha=0.5)
        else:
            plt.annotate("{} - {} - {} - {}".format(all_password_dist[i], format_string(all_passwords[i]), all_password_shape[i], all_password_shape_dist[i]), (all_password_ranks[i], yvals[i]))
            plt.plot(all_password_ranks[i], yvals[i], 'o', color='black', alpha=0.5)

    plt.xlabel('Rank')
    plt.ylabel('Distribution')
    plt.xlim(secret_neighbour_rank_left, secret_neighbour_rank_right)
    plt.title('CDF of target password {} with a radius of {}'.format(secret, radius))
    plt.legend()
    plt.tight_layout()
    image_name = secret.replace('.','(dot)')
    plt_dest = plt_folder + 'CDF_{}'.format(image_name)
    plt.savefig(plt_dest,
            bbox_inches="tight")

    #CDF END
    
def fig_feature_passwords_avg_feature_distance_rank(avg_feature_passwords_feature_distance_ranks=None):
    #BLOCK FOR AVG FEATURE PASSWORDS FEATURE DISTANCE RANK
    avg_feature_passwords_feature_distance_ranks_stat = {}
    avg_feature_passwords_std_error_per_dist = []
    for dist in avg_feature_passwords_feature_distance_ranks.keys():
        avg_feature_passwords_feature_distance_ranks_stat[dist] = (np.mean(np.array(avg_feature_passwords_feature_distance_ranks[dist])), np.std(np.array(avg_feature_passwords_feature_distance_ranks[dist])), len(np.array(avg_feature_passwords_feature_distance_ranks[dist])))
    
    avg_feature_passwords_feature_distance_ranks_stat = dict(sorted(avg_feature_passwords_feature_distance_ranks_stat.items(), key=lambda i: i[0], reverse=False))
    
    feature_passwords_mean_dist = []
    feature_passwords_std_per_mean = []
    feature_passwords_passwords_per_distance = []
    for dist in avg_feature_passwords_feature_distance_ranks_stat.keys():
        feature_passwords_mean_dist.append(avg_feature_passwords_feature_distance_ranks_stat[dist][0])
        feature_passwords_std_per_mean.append(avg_feature_passwords_feature_distance_ranks_stat[dist][1])
        feature_passwords_passwords_per_distance.append(avg_feature_passwords_feature_distance_ranks_stat[dist][2])

    fig = plt.figure(num=None, figsize=(8, 6), dpi=500, facecolor='w', edgecolor='k')
    plt.errorbar(avg_feature_passwords_feature_distance_ranks_stat.keys(), feature_passwords_mean_dist, feature_passwords_std_per_mean, fmt='-o', ecolor='orange', capsize=2, label='Prefix + Suffix + Shape + Norm Distance')
    plt.bar(avg_feature_passwords_feature_distance_ranks_stat.keys(), feature_passwords_passwords_per_distance, color='grey', alpha=0.5)

    plt.xlabel('DISTANCE')
    plt.ylabel('RANK')
    plt.title('FEATURE PASSWORDS AVERAGE FEATURE DISTANCE RANKS {} PASSWORDS'.format(len(g)))
    plt.legend()
    plt.tight_layout()
    plt_dest = plt_folder + 'AVG_FEATURE_PASSWORDS_ALL_FEATURE_DISTANCE_RANKS_{}_PASSWORD'.format(len(g))
    plt.savefig(plt_dest,
            bbox_inches="tight")

    #BLOCK FOR AVG FEATURE PASSWORDS FEATURE DISTANCE RANK END

def fig_error_bar(x=None, y=None, e=None, bar=None, label=None, plot_name=None):

    fig = plt.figure(num=None, figsize=(8, 6), dpi=500, facecolor='w', edgecolor='k')
    plt.errorbar(x, y, e, fmt='-o', ecolor='orange', capsize=2, label=label)
    plt.bar(x, bar, color='grey', alpha=0.5)

    plt.xlabel('DISTANCE')
    plt.ylabel('RANK')
    plt.title(plot_name)
    plt.legend()
    plt.tight_layout()
    plt_dest = plt_folder + plot_name
    plt.savefig(plt_dest,
            bbox_inches="tight")

def fig_avg_all_feature_distance_rank(avg_feature_distance_ranks=None):
    avg_feature_distance_ranks_stat = {}
    avg_std_error_per_dist = []
    for dist in avg_feature_distance_ranks.keys():
        if len(np.array(avg_feature_distance_ranks[dist])) >= require_passwords:
            avg_feature_distance_ranks_stat[dist] = (np.mean(np.array(avg_feature_distance_ranks[dist])), np.std(np.array(avg_feature_distance_ranks[dist])), len(np.array(avg_feature_distance_ranks[dist])))
    
    avg_feature_distance_ranks_stat = dict(sorted(avg_feature_distance_ranks_stat.items(), key=lambda i: i[0], reverse=False))
    
    mean_dist = []
    std_per_mean = []
    passwords_per_distance = []
    for dist in avg_feature_distance_ranks_stat.keys():
        mean_dist.append(avg_feature_distance_ranks_stat[dist][0])
        std_per_mean.append(avg_feature_distance_ranks_stat[dist][1])
        passwords_per_distance.append(avg_feature_distance_ranks_stat[dist][2])

    x = avg_feature_distance_ranks_stat.keys()
    y = mean_dist
    e = std_per_mean
    bar = passwords_per_distance
    label = 'Prefix + Suffix + Shape + Norm Distance'
    plot_name = 'AVG_ALL_FEATURE_DISTANCE_RANKS_{}_PASSWORD'.format(number_of_experiments)
    fig_error_bar(x, y, e, bar, label, plot_name)
    
def fig_avg_pref_suff_feature_distance_rank(avg_feature_distance_ranks_pref_suff=None):

    avg_feature_distance_ranks_stat_pref_suff = {}
    avg_std_error_per_dist_pref_suff = []
    for dist in avg_feature_distance_ranks_pref_suff.keys():
        if len(np.array(avg_feature_distance_ranks_pref_suff[dist])) >= require_passwords:
            avg_feature_distance_ranks_stat_pref_suff[dist] = (np.mean(np.array(avg_feature_distance_ranks_pref_suff[dist])), np.std(np.array(avg_feature_distance_ranks_pref_suff[dist])), len(np.array(avg_feature_distance_ranks_pref_suff[dist])))
    
    avg_feature_distance_ranks_stat_pref_suff = dict(sorted(avg_feature_distance_ranks_stat_pref_suff.items(), key=lambda i: i[0], reverse=False))
    
    mean_dist_pref_suff = []
    std_per_mean_pref_suff = []
    passwords_per_distance_pref_suff = []
    for dist in avg_feature_distance_ranks_stat_pref_suff.keys():
        mean_dist_pref_suff.append(avg_feature_distance_ranks_stat_pref_suff[dist][0])
        std_per_mean_pref_suff.append(avg_feature_distance_ranks_stat_pref_suff[dist][1])  
        passwords_per_distance_pref_suff.append(avg_feature_distance_ranks_stat_pref_suff[dist][2])

    x = avg_feature_distance_ranks_stat_pref_suff.keys()
    y = mean_dist_pref_suff
    e = std_per_mean_pref_suff
    bar = passwords_per_distance_pref_suff
    label = 'Prefix + Suffix Distance'
    plot_name = 'AVG_PREF_SUFF_FEATURE_DISTANCE_RANKS_{}_PASSWORD'.format(number_of_experiments)
    fig_error_bar(x, y, e, bar, label, plot_name)

def fig_avg_pref_shape_feature_distance_rank(avg_feature_distance_ranks_pref_shape=None):
    avg_feature_distance_ranks_stat_pref_shape = {}
    avg_std_error_per_dist_pref_shape = []
    for dist in avg_feature_distance_ranks_pref_shape.keys():
        if len(np.array(avg_feature_distance_ranks_pref_shape[dist])) >= require_passwords:
            avg_feature_distance_ranks_pref_shape[dist] = (np.mean(np.array(avg_feature_distance_ranks_pref_shape[dist])), np.std(np.array(avg_feature_distance_ranks_pref_shape[dist])), len(np.array(avg_feature_distance_ranks_pref_shape[dist])))
    
    avg_feature_distance_ranks_stat_pref_shape = dict(sorted(avg_feature_distance_ranks_stat_pref_shape.items(), key=lambda i: i[0], reverse=False))
    
    mean_dist_pref_shape = []
    std_per_mean_pref_shape = []
    passwords_per_distance_pref_shape = []
    for dist in avg_feature_distance_ranks_stat_pref_shape.keys():
        mean_dist_pref_shape.append(avg_feature_distance_ranks_stat_pref_shape[dist][0])
        std_per_mean_pref_shape.append(avg_feature_distance_ranks_stat_pref_shape[dist][1])  
        passwords_per_distance_pref_shape.append(avg_feature_distance_ranks_stat_pref_shape[dist][2])  

    x = avg_feature_distance_ranks_stat_pref_shape.keys()
    y = mean_dist_pref_shape
    e = std_per_mean_pref_shape
    bar = passwords_per_distance_pref_shape
    label = 'Prefix + Shape Distance'
    plot_name = 'AVG_PREF_SHAPE_FEATURE_DISTANCE_RANKS_{}_PASSWORD'.format(number_of_experiments)
    fig_error_bar(x, y, e, bar, label, plot_name)

def fig_avg_suff_shape_feature_distance_rank(avg_feature_distance_ranks_suff_shape=None):

    avg_feature_distance_ranks_stat_suff_shape = {}
    avg_std_error_per_dist_suff_shape = []
    for dist in avg_feature_distance_ranks_suff_shape.keys():
        if len(np.array(avg_feature_distance_ranks_suff_shape[dist])) >= require_passwords:
            avg_feature_distance_ranks_suff_shape[dist] = (np.mean(np.array(avg_feature_distance_ranks_suff_shape[dist])), np.std(np.array(avg_feature_distance_ranks_suff_shape[dist])), len(np.array(avg_feature_distance_ranks_suff_shape[dist])))
    
    avg_feature_distance_ranks_stat_suff_shape = dict(sorted(avg_feature_distance_ranks_stat_suff_shape.items(), key=lambda i: i[0], reverse=False))
    
    mean_dist_suff_shape = []
    std_per_mean_suff_shape = []
    passwords_per_distance_suff_shape = []
    for dist in avg_feature_distance_ranks_stat_suff_shape.keys():
        mean_dist_suff_shape.append(avg_feature_distance_ranks_stat_suff_shape[dist][0])
        std_per_mean_suff_shape.append(avg_feature_distance_ranks_stat_suff_shape[dist][1])  
        passwords_per_distance_suff_shape.append(avg_feature_distance_ranks_stat_suff_shape[dist][2])
    
    x = avg_feature_distance_ranks_stat_suff_shape.keys()
    y = mean_dist_suff_shape
    e = std_per_mean_suff_shape
    bar = passwords_per_distance_suff_shape
    label = 'Suffix + Shape Distance'
    plot_name = 'AVG_SUFF_SHAPE_FEATURE_DISTANCE_RANKS_{}_PASSWORD'.format(number_of_experiments)
    fig_error_bar(x, y, e, bar, label, plot_name)

def fig_avg_pref_suff_shape_feature_distance_rank(avg_feature_distance_ranks_pref_suff_shape=None):
    avg_feature_distance_ranks_stat_pref_suff_shape = {}
    avg_std_error_per_dist_pref_suff_shape = []
    for dist in avg_feature_distance_ranks_pref_suff_shape.keys():
        if len(np.array(avg_feature_distance_ranks_pref_suff_shape[dist])) >= require_passwords:
            avg_feature_distance_ranks_pref_suff_shape[dist] = (np.mean(np.array(avg_feature_distance_ranks_pref_suff_shape[dist])), np.std(np.array(avg_feature_distance_ranks_pref_suff_shape[dist])), len(np.array(avg_feature_distance_ranks_pref_suff_shape[dist])))
    
    avg_feature_distance_ranks_stat_pref_suff_shape = dict(sorted(avg_feature_distance_ranks_stat_pref_suff_shape.items(), key=lambda i: i[0], reverse=False))
    
    mean_dist_pref_suff_shape = []
    std_per_mean_pref_suff_shape = []
    passwords_per_distance_pref_suff_shape = []
    for dist in avg_feature_distance_ranks_stat_pref_suff_shape.keys():
        mean_dist_pref_suff_shape.append(avg_feature_distance_ranks_stat_pref_suff_shape[dist][0])
        std_per_mean_pref_suff_shape.append(avg_feature_distance_ranks_stat_pref_suff_shape[dist][1])   
        passwords_per_distance_pref_suff_shape.append(avg_feature_distance_ranks_stat_pref_suff_shape[dist][2])
    
    x = avg_feature_distance_ranks_stat_pref_suff_shape.keys()
    y = mean_dist_pref_suff_shape
    e = std_per_mean_pref_suff_shape
    bar = passwords_per_distance_pref_suff_shape
    label = 'Prefix + Suffix + Shape Distance'
    plot_name = 'AVG_PREF_SUFF_SHAPE_FEATURE_DISTANCE_RANKS_{}_PASSWORD'.format(number_of_experiments)
    fig_error_bar(x, y, e, bar, label, plot_name)

def fig_epoch_vs_insertion_3d_plot(epoch_insertion_rank_per_password=None, zoomed=False):

    fig = plt.figure(num=None, figsize=(8, 6), dpi=500, facecolor='w', edgecolor='k')

    for i in epoch_insertion_rank_per_password:
        
        epochs = []
        insertions = []
        ranks = []
        for j in epoch_insertion_rank_per_password[i]:
            epochs.append(j[0])
            insertions.append(j[1])
            ranks.append(j[2])

        pr = fig.gca(projection='3d') 

        pr.scatter(insertions, epochs, ranks, label=i)
        
        pr.set_ylabel("Epochs")
        pr.set_xlabel("Insertions")
        pr.set_zlabel("Ranks")
        if zoomed:
            pr.set_zlim(0,500)
            file_name = 'RANK_PER_EPOCH_AND_INSERTION_ZOOMED'
        else:
            file_name = 'RANK_PER_EPOCH_AND_INSERTION'
        
    plt.legend(bbox_to_anchor=(1.20, 1), loc='upper left')
    plt.title('{} test with {} passwords'.format(version, number_of_experiments))
    plt.tight_layout()
    plt_dest = plt_folder + file_name
    plt.savefig(plt_dest,
            bbox_inches="tight")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--loc', type=str, help='Location of Results')

    args = parser.parse_args()

    loc = args.loc

    require_passwords = 10

    folder = loc

    global features
    features = folder.split("_")[-1]

    global version
    version = str(folder.split("_")[1]) + str(folder.split("_")[2])

    # Load results for plotting
    #res_folder = 'Results/results_{}_len/'.format(secret_len)
    res_folder = '{}/'.format(folder)

    g = unpack_data(res_folder)
    
    global plt_folder

    plt_folder = '{}_PLOTS/'.format(folder)

    mkdir_p(plt_folder)

    global number_of_experiments
    number_of_experiments = len(g)

    secret_index = 3

    avg_epoch_exposure_per_password = {g[i][1].split()[secret_index]:None for i in range(number_of_experiments)}
    avg_epoch_rank_per_password = {g[i][1].split()[secret_index]:None for i in range(number_of_experiments)}

    agg_exposures = {}
    avg_exposure_rank_per_secret = {g[i][1].split()[secret_index]:[] for i in range(number_of_experiments)}
    avg_exposure_per_secret = {g[i][1].split()[secret_index]:[] for i in range(number_of_experiments)}  
        
        
    agg_scores = {}
    ranks_per_code = {}
    avg_rank_per_secret = {g[i][1].split()[secret_index]:[] for i in range(number_of_experiments)}
    avg_score_per_secret = {g[i][1].split()[secret_index]:[] for i in range(number_of_experiments)}

    password_Stat = {}

    n_feature_passwords = 7000

    avg_feature_distance_ranks = {}
    avg_feature_distance_ranks_pref_suff = {}
    avg_feature_distance_ranks_pref_shape = {}
    avg_feature_distance_ranks_suff_shape = {}
    avg_feature_distance_ranks_pref_suff_shape = {}
    avg_feature_distance_ranks_pref_suff_shape_norm = {}

    avg_feature_passwords_feature_distance_ranks = {}

    for i in range(number_of_experiments):
        avg_epoch_exposure = {key:[] for key in g[i][5][0]}
        avg_epoch_rank = {key:[] for key in g[i][5][0]}
        
        agg_scores = {}
        agg_exposures = {}
        
        ranks_per_code = {}
        exposure_rank_per_code = {}
        
        scores = g[i][0]
        exposures = g[i][4]
        epoch_scores = g[i][5]
        
        secret = g[i][1].split()[secret_index]

        secret_shape = word_shape(secret)

        feature_passwords = get_feature_passwords(n_feature_passwords, features, secret)
        
        for score in scores:
            sorted_score = dict(sorted(score.items(), key=operator.itemgetter(1), reverse=True))
            rank = 1
            for code in sorted_score.items():
                if code[0] not in agg_scores.keys():
                    agg_scores[code[0]] = []
                    ranks_per_code[code[0]] = []
                    ranks_per_code[code[0]].append(rank)
                    agg_scores[code[0]].append(code[1])
                    rank+=1
                else:    
                    ranks_per_code[code[0]].append(rank)
                    agg_scores[code[0]].append(code[1])
                    rank+=1

        avg_rank_per_secret[secret] = np.mean(np.array(ranks_per_code[secret]))
        avg_score_per_secret[secret] = np.mean(np.array(agg_scores[secret]))

        for exposure in exposures:
            sorted_exposure = dict(sorted(exposure.items(), key=operator.itemgetter(1), reverse=True))
            rank = 1
            for code in sorted_exposure.items():
                if code[0] not in agg_exposures.keys():
                    agg_exposures[code[0]] = []
                    exposure_rank_per_code[code[0]] = []
                    exposure_rank_per_code[code[0]].append(rank)
                    agg_exposures[code[0]].append(code[1])
                    rank+=1
                else:    
                    exposure_rank_per_code[code[0]].append(rank)
                    agg_exposures[code[0]].append(code[1])
                    rank+=1

        avg_exposure_rank_per_secret[secret] = np.mean(np.array(exposure_rank_per_code[secret]))
        avg_exposure_per_secret[secret] = np.mean(np.array(agg_exposures[secret]))
        
        target_password_rank = np.mean(np.array(exposure_rank_per_code[secret]))
        all_password_ranks = [np.mean(np.array(exposure_rank_per_code[code])) for code in exposure_rank_per_code]

        for subrun in epoch_scores:
            for key in subrun:
                sorted_epoch_exposure = dict(sorted(subrun[key].items(), key=operator.itemgetter(1), reverse=True))
                avg_epoch_exposure[key].append(subrun[key][secret])
                avg_epoch_rank[key].append(list(sorted_epoch_exposure.keys()).index(secret))
                
        for key in avg_epoch_exposure:
            avg_epoch_exposure[key] = np.mean(np.array(avg_epoch_exposure[key]))
            
        for key in avg_epoch_rank:
            avg_epoch_rank[key] = np.mean(np.array(avg_epoch_rank[key]))

        avg_epoch_exposure_per_password[secret] = avg_epoch_exposure
        avg_epoch_rank_per_password[secret] = avg_epoch_rank

        all_password_stat = {code:(np.mean(np.array(exposure_rank_per_code[code])), levenshtein_distance(code, secret), word_shape(code), levenshtein_distance(secret_shape, word_shape(code)), feature_distance(code, secret)) for code in exposure_rank_per_code}

        all_password_stat_sorted = dict(sorted(all_password_stat.items(), key=lambda i: i[1][0], reverse=False))

        all_passwords = [code for code in all_password_stat_sorted]
        all_password_ranks = [all_password_stat_sorted[code][0] for code in all_password_stat_sorted]
        all_password_dist = [all_password_stat_sorted[code][1] for code in all_password_stat_sorted]
        all_password_shape = [all_password_stat_sorted[code][2] for code in all_password_stat_sorted]
        all_password_shape_dist = [all_password_stat_sorted[code][3] for code in all_password_stat_sorted]

        all_password_feature_dist = [all_password_stat_sorted[code][4][4] for code in all_password_stat_sorted]
        all_password_feature_dist_pref_suff = [all_password_stat_sorted[code][4][0] for code in all_password_stat_sorted]
        all_password_feature_dist_pref_shape = [all_password_stat_sorted[code][4][1] for code in all_password_stat_sorted]
        all_password_feature_dist_suff_shape = [all_password_stat_sorted[code][4][2] for code in all_password_stat_sorted]
        all_password_feature_dist_pref_suff_shape = [all_password_stat_sorted[code][4][3] for code in all_password_stat_sorted]
        all_password_feature_dist_pref_suff_shape_norm = [all_password_stat_sorted[code][4][4] for code in all_password_stat_sorted]

        all_dists = set(all_password_feature_dist)

        feature_distance_ranks_per_password = {dist:[] for dist in all_dists}

        for index in range(len(all_password_ranks)):
            #feature_distance_ranks_per_password[all_password_feature_dist[index]].append(all_password_ranks[index])
            feature_distance_ranks_per_password[all_password_feature_dist[index]].append(index)

            if all_password_feature_dist[index] in avg_feature_distance_ranks:
                avg_feature_distance_ranks[all_password_feature_dist[index]].append(all_password_ranks[index])

            if all_password_feature_dist_pref_suff[index] in avg_feature_distance_ranks_pref_suff:
                avg_feature_distance_ranks_pref_suff[all_password_feature_dist_pref_suff[index]].append(all_password_ranks[index])

            if all_password_feature_dist_pref_shape[index] in avg_feature_distance_ranks_pref_shape:
                avg_feature_distance_ranks_pref_shape[all_password_feature_dist_pref_shape[index]].append(all_password_ranks[index])

            if all_password_feature_dist_suff_shape[index] in avg_feature_distance_ranks_suff_shape:
                avg_feature_distance_ranks_suff_shape[all_password_feature_dist_suff_shape[index]].append(all_password_ranks[index])

            if all_password_feature_dist_pref_suff_shape[index] in avg_feature_distance_ranks_pref_suff_shape:
                avg_feature_distance_ranks_pref_suff_shape[all_password_feature_dist_pref_suff_shape[index]].append(all_password_ranks[index])

            if all_password_feature_dist_pref_suff_shape_norm[index] in avg_feature_distance_ranks_pref_suff_shape_norm:
                avg_feature_distance_ranks_pref_suff_shape_norm[all_password_feature_dist_pref_suff_shape_norm[index]].append(all_password_ranks[index])

            # NOT IN

            if all_password_feature_dist[index] not in avg_feature_distance_ranks:
                avg_feature_distance_ranks[all_password_feature_dist[index]] = []
                avg_feature_distance_ranks[all_password_feature_dist[index]].append(all_password_ranks[index])

            if all_password_feature_dist_pref_suff[index] not in avg_feature_distance_ranks_pref_suff:
                avg_feature_distance_ranks_pref_suff[all_password_feature_dist_pref_suff[index]] = []
                avg_feature_distance_ranks_pref_suff[all_password_feature_dist_pref_suff[index]].append(all_password_ranks[index])

            if all_password_feature_dist_pref_shape[index] not in avg_feature_distance_ranks_pref_shape:
                avg_feature_distance_ranks_pref_shape[all_password_feature_dist_pref_shape[index]] = []
                avg_feature_distance_ranks_pref_shape[all_password_feature_dist_pref_shape[index]].append(all_password_ranks[index])

            if all_password_feature_dist_suff_shape[index] not in avg_feature_distance_ranks_suff_shape:
                avg_feature_distance_ranks_suff_shape[all_password_feature_dist_suff_shape[index]] = []
                avg_feature_distance_ranks_suff_shape[all_password_feature_dist_suff_shape[index]].append(all_password_ranks[index])

            if all_password_feature_dist_pref_suff_shape[index] not in avg_feature_distance_ranks_pref_suff_shape:
                avg_feature_distance_ranks_pref_suff_shape[all_password_feature_dist_pref_suff_shape[index]] = []
                avg_feature_distance_ranks_pref_suff_shape[all_password_feature_dist_pref_suff_shape[index]].append(all_password_ranks[index])

            if all_password_feature_dist_pref_suff_shape_norm[index] not in avg_feature_distance_ranks_pref_suff_shape_norm:
                avg_feature_distance_ranks_pref_suff_shape_norm[all_password_feature_dist_pref_suff_shape_norm[index]] = []
                avg_feature_distance_ranks_pref_suff_shape_norm[all_password_feature_dist_pref_suff_shape_norm[index]].append(all_password_ranks[index])

        #CHECK GENERATED FEATURE PASSWORDS

        feature_passwords_stat_sorted = {code:all_password_stat_sorted[code] for code in feature_passwords}

        feature_passwords = [code for code in feature_passwords_stat_sorted]
        feature_passwords_ranks = [feature_passwords_stat_sorted[code][0] for code in feature_passwords_stat_sorted]
        feature_passwords_feature_dist = [feature_passwords_stat_sorted[code][4][4] for code in feature_passwords_stat_sorted]

        for index in range(len(feature_passwords)):
            if feature_passwords_feature_dist[index] in avg_feature_passwords_feature_distance_ranks:
                avg_feature_passwords_feature_distance_ranks[feature_passwords_feature_dist[index]].append(feature_passwords_ranks[index])
            elif feature_passwords_feature_dist[index] not in avg_feature_passwords_feature_distance_ranks:
                avg_feature_passwords_feature_distance_ranks[feature_passwords_feature_dist[index]] = []
                avg_feature_passwords_feature_distance_ranks[feature_passwords_feature_dist[index]].append(feature_passwords_ranks[index])

        #CHECK GENERATED FEATURE PASSWORDS END


        #FIGURE FUNCTIONS
        fig_cdf_per_password(all_passwords, 5, all_password_ranks, all_password_dist, all_password_shape, all_password_shape_dist)
        
        fig_feature_distance_rank_per_password(feature_distance_ranks_per_password, secret)
        
        password_Stat[secret] = PasswordStats(secret)



    fig_feature_passwords_avg_feature_distance_rank(avg_feature_passwords_feature_distance_ranks)

    #BLOCK FOR AVG FEATURE DISTANCE RANK

    ##ALL DISTANCES
    fig_avg_all_feature_distance_rank(avg_feature_distance_ranks)

    ##PREF SUFF DISTANCE
    fig_avg_pref_suff_feature_distance_rank(avg_feature_distance_ranks_pref_suff)

    ##PREF SHAPE DISTANCE
    fig_avg_pref_shape_feature_distance_rank(avg_feature_distance_ranks_pref_shape)

    ##SUFFIX SHAPE DISTANCE
    fig_avg_suff_shape_feature_distance_rank(avg_feature_distance_ranks_suff_shape)

    ##PREFIX SUFFIX SHAPE DISTANCE    
    fig_avg_pref_suff_shape_feature_distance_rank(avg_feature_distance_ranks_pref_suff_shape)

    #BLOCK FOR AVG FEATURE DISTANCE RANK END



    #FIGURE 0 EPOCH VS INSERTIONS VS RANKS

    epoch_insertion_rank_per_password = {g[i][1].split()[secret_index]:[] for i in range(number_of_experiments)}

    for secret in avg_epoch_rank_per_password:
        for j in avg_epoch_rank_per_password[secret].keys():
            epoch_insertion_rank_per_password[secret].append((j[0],j[1],avg_epoch_rank_per_password[secret][j]))

    fig_epoch_vs_insertion_3d_plot(epoch_insertion_rank_per_password, False)
    fig_epoch_vs_insertion_3d_plot(epoch_insertion_rank_per_password, True)


    
    # #FIGURE 2 - DIGITS vs LETTERS EXPOSURE RANK

    # plt.figure(num=None, figsize=(8, 6), dpi=500, facecolor='w', edgecolor='k')

    # index = 1
    # label1 = True
    # label2 = True
    # for password, rank in avg_exposure_rank_per_secret.items():
    #     if password[0].isdigit():
    #         plt.plot(index, rank, 'ok', label="first char is digit" if label1 else "")
    #         label1 = False
    #     else:
    #         plt.plot(index, rank, 'xk', label="first char is letter" if label2 else "")
    #         label2 = False
    #     index+=1
    # plt.xlabel("Codes")
    # plt.ylabel("Exposure Rank")
    # plt.yscale("log")
    # #plt.ylim(0,200)
    # plt.legend()
    # plt.tight_layout()
    # plt_dest = plt_folder + 'DIGITS_VS_LETTERS_EXPOSURE_RANKS'
    # plt.savefig(plt_dest,
    #         bbox_inches="tight")


    # #FIGURE 3 - DIGITS vs LETTERS SCORE RANK

    # plt.figure(num=None, figsize=(8, 6), dpi=500, facecolor='w', edgecolor='k')
    
    # index = 1
    # label1 = True
    # label2 = True
    # for password, rank in avg_rank_per_secret.items():
    #     if password[0].isdigit():
    #         plt.plot(index, rank, 'ok', label="first char is digit" if label1 else "")
    #         label1 = False
    #     else:
    #         plt.plot(index, rank, 'xk', label="first char is letter" if label2 else "")
    #         label2 = False
    #     index+=1
    # plt.xlabel("Codes")
    # plt.ylabel("Confidence Rank")
    # plt.yscale("log")
    # #plt.ylim(0,200)
    # plt.legend()
    # plt.tight_layout()
    # plt_dest = plt_folder + 'DIGITS_VS_LETTERS_SCORE_RANKS'
    # plt.savefig(plt_dest,
    #         bbox_inches="tight")


    # #FIGURE 4 - DIGITS vs LETTERS EXPOSURES

    # plt.figure(num=None, figsize=(8, 6), dpi=500, facecolor='w', edgecolor='k')

    # index = 1
    # label1 = True
    # label2 = True
    # for password, rank in avg_exposure_per_secret.items():
    #     if password[0].isdigit():
    #         plt.plot(index, rank, 'ok', label="first char is digit" if label1 else "")
    #         label1 = False
    #     else:
    #         plt.plot(index, rank, 'xk', label="first char is letter" if label2 else "")
    #         label2 = False
    #     index+=1
    # plt.xlabel("Codes")
    # plt.ylabel("Exposure")
    # #plt.yscale("log")
    # #plt.ylim(0,200)
    # plt.legend()
    # plt.tight_layout()
    # plt_dest = plt_folder + 'DIGITS_VS_LETTERS_EXPOSURE'
    # plt.savefig(plt_dest,
    #         bbox_inches="tight")


    # #FIGURE 5 - AVG RANKING PER SECRET

    # plt.figure(num=None, figsize=(8, 6), dpi=500, facecolor='w', edgecolor='k')

    # newA = avg_rank_per_secret
    # digits_per_password = {password:stat.numbers for password, stat in password_Stat.items()}
    # letters_per_password = {password:stat.letters for password, stat in password_Stat.items()}
    # special_char_per_password = {password:stat.special_characters for password, stat in password_Stat.items()}


    # lists = newA.items()

    # x, y = zip(*lists)

    # x = range(len(x))

    # fig = plt.figure()
    # ax = fig.add_axes([0,0,1,1])

    # index = 0
    # label1 = True
    # label2 = True
    # for password, rank in avg_rank_per_secret.items():
    #     if password[0].isdigit():
    #         plt.plot(index, rank, 'ok', label="first char is digit" if label1 else "")
    #         label1 = False
    #     else:
    #         plt.plot(index, rank, 'xk', label="first char is letter" if label2 else "")
    #         label2 = False
    #     index+=1


    # #plt.plot(x, y, 'xk')
    # plt.yscale("log")
    # #plt.ylim(0,10)
    # ax.bar(x, digits_per_password.values(), alpha = 0.6, color = 'cyan', width = 0.25, label='digits')
    # ax.bar(x, letters_per_password.values(), alpha = 0.6, color = 'orange', width = 0.25, label='letters')
    # ax.bar(x, special_char_per_password.values(), alpha = 0.6, color = 'r', width = 0.25, label='special characters')

    # plt.xlabel('Codes')
    # plt.ylabel('Avg Rank')
    # plt.title('Avg Rank over {} Runs'.format(len(g[0][0])))
    # plt.legend(loc='upper right')
    # #plt.tight_layout()
    # plt_dest = plt_folder + 'PASSWORDS_AVERAGE_RANK_WRT_DIGIT_LETTER_COUNT'
    # plt.savefig(plt_dest,
    #         bbox_inches="tight")
    # plt.show()



    # #FIGURE 6 - AVG RANKING PER STRENGTH

    # plt.figure(num=None, figsize=(8, 6), dpi=500, facecolor='w', edgecolor='k')

    # info_per_password = {password:[] for password, stat in password_Stat.items()}

    # strength_per_password = {password:stat.strength() for password, stat in password_Stat.items()}
    # digits_per_password = {password:stat.numbers for password, stat in password_Stat.items()}
    # letters_per_password = {password:stat.letters for password, stat in password_Stat.items()}
    # special_char_per_password = {password:stat.special_characters for password, stat in password_Stat.items()}


    # for key in avg_rank_per_secret.keys():
    #     info_per_password[key].extend([strength_per_password[key], avg_rank_per_secret[key], digits_per_password[key], letters_per_password[key], special_char_per_password[key]])


    # strength = [value[0] for key, value in info_per_password.items()]
    # rank = [value[1] for key, value in info_per_password.items()]
    # digits = [value[2] for key, value in info_per_password.items()]
    # letters = [value[3] for key, value in info_per_password.items()]
    # special_chars = [value[4] for key, value in info_per_password.items()]

    # fig = plt.figure()
    # ax = fig.add_axes([0,0,1,1])

    # label1 = True
    # label2 = True
    # for password, stats in info_per_password.items():
    #     if password[0].isdigit():
    #         plt.plot(stats[0], stats[1], 'ok', label="first char is digit" if label1 else "")
    #         label1 = False
    #     else:
    #         plt.plot(stats[0], stats[1], 'xk', label="first char is letter" if label2 else "")
    #         label2 = False
        
    # #plt.plot(x, y, 'xk')
    # plt.yscale("log")
    # #plt.ylim(0,10)
    # ax.bar(strength, digits_per_password.values(), alpha = 0.2, color = 'cyan', width = 0.25, label='digits')
    # ax.bar(strength, letters_per_password.values(), alpha = 0.2, color = 'orange', width = 0.25, label='letters')
    # ax.bar(strength, special_char_per_password.values(), alpha = 0.2, color = 'r', width = 0.25, label='special characters')

    # #plt.plot(x, y, 'xk')
    # plt.xlabel('Strength of Password')
    # plt.ylabel('Avg Rank')
    # plt.title('Avg Rank over {} Runs'.format(len(g[0][0])))
    # plt.legend(loc='upper right')
    # #plt.tight_layout()
    # plt_dest = plt_folder + 'AVERAGE_RANK_PER_STRENGTH'
    # plt.savefig(plt_dest,
    #         bbox_inches="tight")
    # plt.show()


    # #FIGURE 7 - AVG RANKING PER ENTROPY BITS

    # plt.figure(num=None, figsize=(8, 6), dpi=500, facecolor='w', edgecolor='k')

    # info_per_password = {password:[] for password, stat in password_Stat.items()}

    # entropy_bits_per_password = {password:stat.entropy_bits for password, stat in password_Stat.items()}
    # digits_per_password = {password:stat.numbers for password, stat in password_Stat.items()}
    # letters_per_password = {password:stat.letters for password, stat in password_Stat.items()}
    # special_char_per_password = {password:stat.special_characters for password, stat in password_Stat.items()}


    # for key in avg_rank_per_secret.keys():
    #     info_per_password[key].extend([entropy_bits_per_password[key], avg_rank_per_secret[key], digits_per_password[key], letters_per_password[key], special_char_per_password[key]])


    # strength = [value[0] for key, value in info_per_password.items()]
    # rank = [value[1] for key, value in info_per_password.items()]
    # digits = [value[2] for key, value in info_per_password.items()]
    # letters = [value[3] for key, value in info_per_password.items()]
    # special_chars = [value[4] for key, value in info_per_password.items()]

    # fig = plt.figure()
    # ax = fig.add_axes([0,0,1,1])

    # label1 = True
    # label2 = True
    # for password, stats in info_per_password.items():
    #     if password[0].isdigit():
    #         plt.plot(stats[0], stats[1], 'ok', label="first char is digit" if label1 else "")
    #         label1 = False
    #     else:
    #         plt.plot(stats[0], stats[1], 'xk', label="first char is letter" if label2 else "")
    #         label2 = False
        
    # #plt.plot(x, y, 'xk')
    # plt.yscale("log")
    # #plt.ylim(0,10)
    # ax.bar(strength, digits_per_password.values(), alpha = 0.2, color = 'cyan', width = 0.25, label='digits')
    # ax.bar(strength, letters_per_password.values(), alpha = 0.2, color = 'orange', width = 0.25, label='letters')
    # ax.bar(strength, special_char_per_password.values(), alpha = 0.2, color = 'r', width = 0.25, label='special characters')

    # #plt.plot(x, y, 'xk')
    # plt.xlabel("ENTROPY BITS (log2 of the number of possible passwords)")
    # plt.ylabel('Avg Rank')
    # plt.title('Avg Rank over {} Runs'.format(len(g[0][0])))
    # plt.legend(loc='upper right')
    # #plt.tight_layout()
    # plt_dest = plt_folder + 'AVERAGE_RANK_PER_ENTROPY_BITS'
    # plt.savefig(plt_dest,
    #         bbox_inches="tight")
    # plt.show()


    # #FIGURE 8 - AVG RANKING PER ENTROPY DENSITY

    # plt.figure(num=None, figsize=(8, 6), dpi=500, facecolor='w', edgecolor='k')

    # info_per_password = {password:[] for password, stat in password_Stat.items()}

    # entropy_density_per_password = {password:stat.entropy_density for password, stat in password_Stat.items()}
    # digits_per_password = {password:stat.numbers for password, stat in password_Stat.items()}
    # letters_per_password = {password:stat.letters for password, stat in password_Stat.items()}
    # special_char_per_password = {password:stat.special_characters for password, stat in password_Stat.items()}


    # for key in avg_rank_per_secret.keys():
    #     info_per_password[key].extend([entropy_density_per_password[key], avg_rank_per_secret[key], digits_per_password[key], letters_per_password[key], special_char_per_password[key]])


    # strength = [value[0] for key, value in info_per_password.items()]
    # rank = [value[1] for key, value in info_per_password.items()]
    # digits = [value[2] for key, value in info_per_password.items()]
    # letters = [value[3] for key, value in info_per_password.items()]
    # special_chars = [value[4] for key, value in info_per_password.items()]

    # fig = plt.figure()
    # ax = fig.add_axes([0,0,1,1])

    # label1 = True
    # label2 = True
    # for password, stats in info_per_password.items():
    #     if password[0].isdigit():
    #         plt.plot(stats[0], stats[1], 'ok', label="first char is digit" if label1 else "")
    #         label1 = False
    #     else:
    #         plt.plot(stats[0], stats[1], 'xk', label="first char is letter" if label2 else "")
    #         label2 = False
        
    # #plt.plot(x, y, 'xk')
    # plt.yscale("log")
    # #plt.ylim(0,10)
    # ax.bar(strength, digits_per_password.values(), alpha = 0.2, color = 'cyan', width = 0.25, label='digits')
    # ax.bar(strength, letters_per_password.values(), alpha = 0.2, color = 'orange', width = 0.25, label='letters')
    # ax.bar(strength, special_char_per_password.values(), alpha = 0.2, color = 'r', width = 0.25, label='special characters')

    # #plt.plot(x, y, 'xk')
    # plt.xlabel("ENTROPY DENSITY")
    # plt.ylabel('Avg Rank')
    # plt.title('Avg Rank over {} Runs'.format(len(g[0][0])))
    # plt.legend(loc='upper right')
    # #plt.tight_layout()
    # plt_dest = plt_folder + 'AVERAGE_RANK_PER_ENTROPY_DENSITY'
    # plt.savefig(plt_dest,
    #         bbox_inches="tight")
    # plt.show()