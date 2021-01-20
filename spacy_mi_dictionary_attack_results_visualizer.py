import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import os
import math
import sklearn.metrics as metrics
import scikitplot as skplt
import statistics as st
import heapq
import operator
import errno
from itertools import islice
from password_strength import PasswordStats



def mkdir_p(path):
    """To make a directory given a path."""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


if __name__ == "__main__":
    folder = '20210120_10_passwords_dictionary_attack_1_insertions_500_epoch_10000_r_space_0_knowledge'

    # Load results for plotting
    #res_folder = 'Results/results_{}_len/'.format(secret_len)
    res_folder = 'Results/{}/results/'.format(folder)

    files = os.listdir(res_folder)

    g = []
    br = 0
    for file_name in files:
        br += 1
        print(file_name)
        file_path = os.path.join(res_folder, file_name)
        h = pickle.load(open(file_path, 'rb'))
        g.append(h)
    print('Read Disk')
    print('{} TEST RUNS FOUND'.format(len(g)))

    plt_folder = 'Results/{}_PLOTS/'.format(folder)

    mkdir_p(plt_folder)


    avg_epoch_exposure_per_password = {g[i][1].split()[-1]:None for i in range(len(g))}
    avg_epoch_rank_per_password = {g[i][1].split()[-1]:None for i in range(len(g))}

    agg_exposures = {}
    exposure_rank_per_code = {}
    avg_exposure_rank_per_secret = {g[i][1].split()[-1]:[] for i in range(len(g))}
    avg_exposure_per_secret = {g[i][1].split()[-1]:[] for i in range(len(g))}  
        
        
    agg_scores = {}
    ranks_per_code = {}
    avg_rank_per_secret = {g[i][1].split()[-1]:[] for i in range(len(g))}
    avg_score_per_secret = {g[i][1].split()[-1]:[] for i in range(len(g))}

    password_Stat = {}


    for i in range(len(g)):
        avg_epoch_exposure = {key:[] for key in g[i][5][0]}
        avg_epoch_rank = {key:[] for key in g[i][5][0]}
        
        agg_scores = {}
        agg_exposures = {}
        
        ranks_per_code = {}
        exposure_rank_per_code = {}
        
        scores = g[i][0]
        exposures = g[i][4]
        epoch_scores = g[i][5]
        
        secret = g[i][1].split()[-1]
        print(secret)
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
        
        for subrun in epoch_scores:
            for key in subrun:
                sorted_exposure = dict(sorted(subrun[key].items(), key=operator.itemgetter(1), reverse=True))
                avg_epoch_exposure[key].append(subrun[key][secret])
                avg_epoch_rank[key].append(list(sorted_exposure.keys()).index(secret))

                
        for key in avg_epoch_exposure:
            avg_epoch_exposure[key] = np.mean(np.array(avg_epoch_exposure[key]))
            
        for key in avg_epoch_rank:
            avg_epoch_rank[key] = np.mean(np.array(avg_epoch_rank[key]))

        avg_epoch_exposure_per_password[secret] = avg_epoch_exposure
        avg_epoch_rank_per_password[secret] = avg_epoch_rank
        
        avg_exposure_rank_per_secret[secret] = np.mean(np.array(exposure_rank_per_code[secret]))
        avg_exposure_per_secret[secret] = np.mean(np.array(agg_exposures[secret]))
        
        avg_rank_per_secret[secret] = np.mean(np.array(ranks_per_code[secret]))
        avg_score_per_secret[secret] = np.mean(np.array(agg_scores[secret]))
        
        password_Stat[secret] = PasswordStats(secret)



    #FIGURE 1 - RANK PER EPOCH/INSERTIONS

    plt.figure()

    for i in avg_epoch_rank_per_password:
        epochs = avg_epoch_rank_per_password[i].keys()
        ranks = avg_epoch_rank_per_password[i].values()
        
        plt.xlabel('Epochs')
        plt.ylabel('Ranks')
        plt.plot(epochs, ranks, label = i)
    plt.legend()
    lt.tight_layout()
    plt_dest = plt_folder + 'RANK_PER_EPOCH_OR_INSERTION'
    plt.savefig(plt_dest,
            bbox_inches="tight")


    #FIGURE 2 - DIGITS vs LETTERS EXPOSURE RANK

    index = 1
    label1 = True
    label2 = True
    for password, rank in avg_exposure_rank_per_secret.items():
        if password[0].isdigit():
            plt.plot(index, rank, 'ok', label="first char is digit" if label1 else "")
            label1 = False
        else:
            plt.plot(index, rank, 'xk', label="first char is letter" if label2 else "")
            label2 = False
        index+=1
    plt.xlabel("Codes")
    plt.ylabel("Exposure Rank")
    plt.yscale("log")
    #plt.ylim(0,200)
    plt.legend()
    plt.tight_layout()
    plt_dest = plt_folder + 'DIGITS_VS_LETTERS_EXPOSURE_RANKS'
    plt.savefig(plt_dest,
            bbox_inches="tight")


    #FIGURE 3 - DIGITS vs LETTERS SCORE RANK

    index = 1
    label1 = True
    label2 = True
    for password, rank in avg_rank_per_secret.items():
        if password[0].isdigit():
            plt.plot(index, rank, 'ok', label="first char is digit" if label1 else "")
            label1 = False
        else:
            plt.plot(index, rank, 'xk', label="first char is letter" if label2 else "")
            label2 = False
        index+=1
    plt.xlabel("Codes")
    plt.ylabel("Confidence Rank")
    plt.yscale("log")
    #plt.ylim(0,200)
    plt.legend()
    plt.tight_layout()
    plt_dest = plt_folder + 'DIGITS_VS_LETTERS_SCORE_RANKS'
    plt.savefig(plt_dest,
            bbox_inches="tight")


    #FIGURE 4 - DIGITS vs LETTERS EXPOSURES

    index = 1
    label1 = True
    label2 = True
    for password, rank in avg_exposure_per_secret.items():
        if password[0].isdigit():
            plt.plot(index, rank, 'ok', label="first char is digit" if label1 else "")
            label1 = False
        else:
            plt.plot(index, rank, 'xk', label="first char is letter" if label2 else "")
            label2 = False
        index+=1
    plt.xlabel("Codes")
    plt.ylabel("Exposure")
    #plt.yscale("log")
    #plt.ylim(0,200)
    plt.legend()
    plt.tight_layout()
    plt_dest = plt_folder + 'DIGITS_VS_LETTERS_EXPOSURE'
    plt.savefig(plt_dest,
            bbox_inches="tight")


    #FIGURE 5 - AVG RANKING PER SECRET

    plt.figure()

    #newA = dict(sorted(ranks_per_secret.items(), key=operator.itemgetter(1), reverse=False))
    newA = avg_rank_per_secret
    digits_per_password = {password:stat.numbers for password, stat in password_Stat.items()}
    letters_per_password = {password:stat.letters for password, stat in password_Stat.items()}
    special_char_per_password = {password:stat.special_characters for password, stat in password_Stat.items()}


    lists = newA.items()

    x, y = zip(*lists)

    x = range(len(x))

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])

    index = 0
    label1 = True
    label2 = True
    for password, rank in avg_rank_per_secret.items():
        if password[0].isdigit():
            plt.plot(index, rank, 'ok', label="first char is digit" if label1 else "")
            label1 = False
        else:
            plt.plot(index, rank, 'xk', label="first char is letter" if label2 else "")
            label2 = False
        index+=1


    #plt.plot(x, y, 'xk')
    plt.yscale("log")
    #plt.ylim(0,10)
    ax.bar(x, digits_per_password.values(), alpha = 0.6, color = 'cyan', width = 0.25, label='digits')
    ax.bar(x, letters_per_password.values(), alpha = 0.6, color = 'orange', width = 0.25, label='letters')
    ax.bar(x, special_char_per_password.values(), alpha = 0.6, color = 'r', width = 0.25, label='special characters')

    plt.xlabel('Codes')
    plt.ylabel('Avg Rank')
    plt.title('Avg Rank over {} Runs'.format(len(g[0][0])))
    plt.legend(loc='upper right')
    #plt.tight_layout()
    plt_dest = plt_folder + 'PASSWORDS_AVERAGE_RANK_WRT_DIGIT_LETTER_COUNT'
    plt.savefig(plt_dest,
            bbox_inches="tight")
    plt.show()



    #FIGURE 6 - AVG RANKING PER STRENGTH

    plt.figure()

    info_per_password = {password:[] for password, stat in password_Stat.items()}

    strength_per_password = {password:stat.strength() for password, stat in password_Stat.items()}
    digits_per_password = {password:stat.numbers for password, stat in password_Stat.items()}
    letters_per_password = {password:stat.letters for password, stat in password_Stat.items()}
    special_char_per_password = {password:stat.special_characters for password, stat in password_Stat.items()}


    for key in avg_rank_per_secret.keys():
        info_per_password[key].extend([strength_per_password[key], avg_rank_per_secret[key], digits_per_password[key], letters_per_password[key], special_char_per_password[key]])


    strength = [value[0] for key, value in info_per_password.items()]
    rank = [value[1] for key, value in info_per_password.items()]
    digits = [value[2] for key, value in info_per_password.items()]
    letters = [value[3] for key, value in info_per_password.items()]
    special_chars = [value[4] for key, value in info_per_password.items()]

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])

    label1 = True
    label2 = True
    for password, stats in info_per_password.items():
        if password[0].isdigit():
            plt.plot(stats[0], stats[1], 'ok', label="first char is digit" if label1 else "")
            label1 = False
        else:
            plt.plot(stats[0], stats[1], 'xk', label="first char is letter" if label2 else "")
            label2 = False
        
    #plt.plot(x, y, 'xk')
    plt.yscale("log")
    #plt.ylim(0,10)
    ax.bar(strength, digits_per_password.values(), alpha = 0.2, color = 'cyan', width = 0.25, label='digits')
    ax.bar(strength, letters_per_password.values(), alpha = 0.2, color = 'orange', width = 0.25, label='letters')
    ax.bar(strength, special_char_per_password.values(), alpha = 0.2, color = 'r', width = 0.25, label='special characters')

    #plt.plot(x, y, 'xk')
    plt.xlabel('Strength of Password')
    plt.ylabel('Avg Rank')
    plt.title('Avg Rank over {} Runs'.format(len(g[0][0])))
    plt.legend(loc='upper right')
    #plt.tight_layout()
    plt_dest = plt_folder + 'AVERAGE_RANK_PER_STRENGTH'
    plt.savefig(plt_dest,
            bbox_inches="tight")
    plt.show()


    #FIGURE 7 - AVG RANKING PER ENTROPY BITS

    plt.figure()

    info_per_password = {password:[] for password, stat in password_Stat.items()}

    entropy_bits_per_password = {password:stat.entropy_bits for password, stat in password_Stat.items()}
    digits_per_password = {password:stat.numbers for password, stat in password_Stat.items()}
    letters_per_password = {password:stat.letters for password, stat in password_Stat.items()}
    special_char_per_password = {password:stat.special_characters for password, stat in password_Stat.items()}


    for key in avg_rank_per_secret.keys():
        info_per_password[key].extend([entropy_bits_per_password[key], avg_rank_per_secret[key], digits_per_password[key], letters_per_password[key], special_char_per_password[key]])


    strength = [value[0] for key, value in info_per_password.items()]
    rank = [value[1] for key, value in info_per_password.items()]
    digits = [value[2] for key, value in info_per_password.items()]
    letters = [value[3] for key, value in info_per_password.items()]
    special_chars = [value[4] for key, value in info_per_password.items()]

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])

    label1 = True
    label2 = True
    for password, stats in info_per_password.items():
        if password[0].isdigit():
            plt.plot(stats[0], stats[1], 'ok', label="first char is digit" if label1 else "")
            label1 = False
        else:
            plt.plot(stats[0], stats[1], 'xk', label="first char is letter" if label2 else "")
            label2 = False
        
    #plt.plot(x, y, 'xk')
    plt.yscale("log")
    #plt.ylim(0,10)
    ax.bar(strength, digits_per_password.values(), alpha = 0.2, color = 'cyan', width = 0.25, label='digits')
    ax.bar(strength, letters_per_password.values(), alpha = 0.2, color = 'orange', width = 0.25, label='letters')
    ax.bar(strength, special_char_per_password.values(), alpha = 0.2, color = 'r', width = 0.25, label='special characters')

    #plt.plot(x, y, 'xk')
    plt.xlabel("ENTROPY BITS (log2 of the number of possible passwords)")
    plt.ylabel('Avg Rank')
    plt.title('Avg Rank over {} Runs'.format(len(g[0][0])))
    plt.legend(loc='upper right')
    #plt.tight_layout()
    plt_dest = plt_folder + 'AVERAGE_RANK_PER_ENTROPY_BITS'
    plt.savefig(plt_dest,
            bbox_inches="tight")
    plt.show()


    #FIGURE 8 - AVG RANKING PER ENTROPY DENSITY

    plt.figure()

    info_per_password = {password:[] for password, stat in password_Stat.items()}

    entropy_density_per_password = {password:stat.entropy_density for password, stat in password_Stat.items()}
    digits_per_password = {password:stat.numbers for password, stat in password_Stat.items()}
    letters_per_password = {password:stat.letters for password, stat in password_Stat.items()}
    special_char_per_password = {password:stat.special_characters for password, stat in password_Stat.items()}


    for key in avg_rank_per_secret.keys():
        info_per_password[key].extend([entropy_density_per_password[key], avg_rank_per_secret[key], digits_per_password[key], letters_per_password[key], special_char_per_password[key]])


    strength = [value[0] for key, value in info_per_password.items()]
    rank = [value[1] for key, value in info_per_password.items()]
    digits = [value[2] for key, value in info_per_password.items()]
    letters = [value[3] for key, value in info_per_password.items()]
    special_chars = [value[4] for key, value in info_per_password.items()]

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])

    label1 = True
    label2 = True
    for password, stats in info_per_password.items():
        if password[0].isdigit():
            plt.plot(stats[0], stats[1], 'ok', label="first char is digit" if label1 else "")
            label1 = False
        else:
            plt.plot(stats[0], stats[1], 'xk', label="first char is letter" if label2 else "")
            label2 = False
        
    #plt.plot(x, y, 'xk')
    plt.yscale("log")
    #plt.ylim(0,10)
    ax.bar(strength, digits_per_password.values(), alpha = 0.2, color = 'cyan', width = 0.25, label='digits')
    ax.bar(strength, letters_per_password.values(), alpha = 0.2, color = 'orange', width = 0.25, label='letters')
    ax.bar(strength, special_char_per_password.values(), alpha = 0.2, color = 'r', width = 0.25, label='special characters')

    #plt.plot(x, y, 'xk')
    plt.xlabel("ENTROPY DENSITY")
    plt.ylabel('Avg Rank')
    plt.title('Avg Rank over {} Runs'.format(len(g[0][0])))
    plt.legend(loc='upper right')
    #plt.tight_layout()
    plt_dest = plt_folder + 'AVERAGE_RANK_PER_ENTROPY_DENSITY'
    plt.savefig(plt_dest,
            bbox_inches="tight")
    plt.show()