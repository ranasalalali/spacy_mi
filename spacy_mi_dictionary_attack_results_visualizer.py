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

    parser = argparse.ArgumentParser()
    parser.add_argument('--loc', type=str, help='Location of Results')

    args = parser.parse_args()

    loc = args.loc

    folder = loc

    # Load results for plotting
    #res_folder = 'Results/results_{}_len/'.format(secret_len)
    res_folder = '{}/'.format(folder)

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

    plt_folder = '{}_PLOTS/'.format(folder)

    mkdir_p(plt_folder)


    secret_index = 3

    avg_epoch_exposure_per_password = {g[i][1].split()[secret_index]:None for i in range(len(g))}
    avg_epoch_rank_per_password = {g[i][1].split()[secret_index]:None for i in range(len(g))}

    agg_exposures = {}
    exposure_rank_per_code = {}
    avg_exposure_rank_per_secret = {g[i][1].split()[secret_index]:[] for i in range(len(g))}
    avg_exposure_per_secret = {g[i][1].split()[secret_index]:[] for i in range(len(g))}  
        
        
    agg_scores = {}
    ranks_per_code = {}
    avg_rank_per_secret = {g[i][1].split()[secret_index]:[] for i in range(len(g))}
    avg_score_per_secret = {g[i][1].split()[secret_index]:[] for i in range(len(g))}

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
        
        secret = g[i][1].split()[secret_index]
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
                sorted_epoch_exposure = dict(sorted(subrun[key].items(), key=operator.itemgetter(1), reverse=True))
                avg_epoch_exposure[key].append(subrun[key][secret])
                avg_epoch_rank[key].append(list(sorted_epoch_exposure.keys()).index(secret))

                
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

        target_password_rank = np.mean(np.array(exposure_rank_per_code[secret]))
        all_password_ranks = [np.mean(np.array(exposure_rank_per_code[code])) for code in exposure_rank_per_code]

        all_password_ranks = np.sort(np.array(all_password_ranks), axis=None)

        #CDF PER TARGET_PASSWORD
        fig = plt.figure(num=None, figsize=(8, 6), dpi=500, facecolor='w', edgecolor='k')
        yvals = np.zeros(len(all_password_ranks))
        for i in range(len(all_password_ranks)):
            yvals[i] = (i+1)/len(yvals)
        plt.plot(all_password_ranks, yvals, 'k-', label='target_password = {} \n average rank = {}'.format(secret, target_password_rank))
        plt.legend()
        plt.tight_layout()
        plt_dest = plt_folder + 'CDF_{}'.format(secret)
        plt.savefig(plt_dest,
                bbox_inches="tight")
        #CDF END


        
        password_Stat[secret] = PasswordStats(secret)


    epoch_insertion_rank_per_password = {g[i][1].split()[secret_index]:[] for i in range(len(g))}
    for secret in avg_epoch_rank_per_password:
        for j in avg_epoch_rank_per_password[secret].keys():
            epoch_insertion_rank_per_password[secret].append((j[0],j[1],avg_epoch_rank_per_password[secret][j]))

    #FIGURE 0 EPOCH VS INSERTIONS VS RANKS

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
        #pr.set_zlim(0,500)
        
    plt.legend(bbox_to_anchor=(1.20, 1), loc='upper left')
    plt.tight_layout()
    plt_dest = plt_folder + 'RANK_PER_EPOCH_AND_INSERTION'
    plt.savefig(plt_dest,
            bbox_inches="tight")


    #FIGURE 0.1 EPOCH VS INSERTIONS VS RANKS 2X

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
        pr.set_zlim(0,500)
        
    plt.legend(bbox_to_anchor=(1.20, 1), loc='upper left')
    plt.tight_layout()
    plt_dest = plt_folder + 'RANK_PER_EPOCH_AND_INSERTION_ZOOMED'
    plt.savefig(plt_dest,
            bbox_inches="tight")

    #FIGURE 1 - RANK PER EPOCH/INSERTIONS

    # plt.figure(num=None, figsize=(8, 6), dpi=500, facecolor='w', edgecolor='k')

    # for i in avg_epoch_rank_per_password:
    #     epochs = avg_epoch_rank_per_password[i].keys()
    #     ranks = avg_epoch_rank_per_password[i].values()
        
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Ranks')
    #     plt.plot(epochs, ranks, label = i)
    # plt.ylim(0,1000)
    # plt.legend(bbox_to_anchor=(1.05, 1))
    # plt.tight_layout()
    # plt_dest = plt_folder + 'RANK_PER_EPOCH_OR_INSERTION'
    # plt.savefig(plt_dest,
    #         bbox_inches="tight")

    # #FIGURE 0 - AVG RANK PER EPOCH/INSERTIONS

    # plt.figure()

    # first_key = list(avg_epoch_rank_per_password.keys())[0]
    # overall_avg_epoch_rank = {epoch:[] for epoch in epochs}

    # for i in avg_epoch_rank_per_password:
    #     for epoch, rank in avg_epoch_rank_per_password.items():
    #         overall_avg_epoch_rank[epoch].append(rank)
        
    # for key in overall_avg_epoch_rank:
    #     overall_avg_epoch_rank[key] = np.mean(np.array(overall_avg_epoch_rank[key]))

    
    # plt.xlabel('Epochs')
    # plt.ylabel('Ranks')
    # plt.plot(overall_avg_epoch_rank.keys(), overall_avg_epoch_rank.values())
    # plt.ylim(0,1000)
    # plt.legend()
    # plt.tight_layout()
    # plt_dest = plt_folder + 'OVERALL_AVERAGE_RANK_PER_EPOCH_OR_INSERTION'
    # plt.savefig(plt_dest,
    #         bbox_inches="tight")


    #FIGURE 2 - DIGITS vs LETTERS EXPOSURE RANK

    plt.figure(num=None, figsize=(8, 6), dpi=500, facecolor='w', edgecolor='k')

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

    plt.figure(num=None, figsize=(8, 6), dpi=500, facecolor='w', edgecolor='k')
    
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

    plt.figure(num=None, figsize=(8, 6), dpi=500, facecolor='w', edgecolor='k')

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

    plt.figure(num=None, figsize=(8, 6), dpi=500, facecolor='w', edgecolor='k')

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

    plt.figure(num=None, figsize=(8, 6), dpi=500, facecolor='w', edgecolor='k')

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

    plt.figure(num=None, figsize=(8, 6), dpi=500, facecolor='w', edgecolor='k')

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

    plt.figure(num=None, figsize=(8, 6), dpi=500, facecolor='w', edgecolor='k')

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