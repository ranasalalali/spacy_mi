


from __future__ import unicode_literals, print_function
import spacy
print(spacy.__version__)
from collections import defaultdict
from thinc.api import set_gpu_allocator,require_gpu


from spacy.training import Example
import random
import warnings
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from collections import defaultdict
import operator
import string
import numpy as np
from itertools import permutations, islice
import os
import errno
import multiprocessing as mp
from spacy.vectors import Vectors
import time
import sys
import pickle

import random
import warnings
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from collections import defaultdict
import operator
import pickle
import argparse, sys
from datetime import datetime, date
import os
import errno
import itertools
import multiprocessing as mp
import shutil
import numpy as np
import math
from spacy.training import Example
from thinc.api import set_gpu_allocator, require_gpu
# from password_generator import PasswordGenerator
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score



# file_name = 'password_11116_updated_ner_ROC_20210502/20210502_time_diff_orig_ner_2000_words_three_runs_VM.pickle3'

num_test = 2000

# file_name_time_diff_ner = 'time_difference_between_two_runs_ner_task_{}_words_VM_in_all_out_all.png'.format(num_test)
# file_name_time_diff_tok = 'time_difference_between_two_runs_tok_task_{}_words_VM_in_all_out_all.png'.format(num_test)
# file_name_time_diff_both = 'time_difference_between_two_runs_both_task_{}_words_VM_in_all_out_all.png'.format(num_test)
# file_name_first_run_ner = 'absolute_time_first_run_ner_task_{}_words_VM_in_all_out_all.png'.format(num_test)
# file_name_second_run_ner = 'absolute_time_second_run_ner_task_{}_words_VM_in_all_out_all.png'.format(num_test)
# file_name_first_run_tok = 'absolute_time_first_run_tok_task_{}_words_VM_in_all_out_all.png'.format(num_test)
# file_name_second_run_tok = 'absolute_time_second_run_tok_task_{}_words_VM_in_all_out_all.png'.format(num_test)
# file_name_first_run_both = 'absolute_time_first_run_both_task_{}_words_VM_in_all_out_all.png'.format(num_test)
# file_name_second_run_both = 'absolute_time_second_run_both_task_{}_words_VM_in_all_out_all.png'.format(num_test)

# all_roc_graph_name = 'roc_time_abs_both_task_tok_ner_{}_words_PC.png'.format(num_test)
# all_roc_graph_name_pdf = 'roc_time_abs_both_task_tok_ner_{}_words_PC.pdf'.format(num_test)

# file_name_time_diff_2_3_ner = 'time_difference_between_runs2_3_ner_task_{}_words_VM_in_all_out_all.png'.format(num_test)
# file_name_time_diff_2_3_tok = 'time_difference_between_runs2_3_tok_task_{}_words_VM_in_all_out_all.png'.format(num_test)
# file_name_time_diff_2_3_both = 'time_difference_between_runs2_3_both_task_{}_words_VM_in_all_out_all.png'.format(num_test)

plt_folder = 'new_plot_ner_PC_roc_3.0.3/'


# file_name = 'password_11116_updated_ner_ROC_20210503/20210503_time_diff_updated_ner_100_words_three_runs_PC_reload_model.pickle3'
# file_name = '/home/tham/spacy_mi/password_11116_updated_ner_ROC_20210505/20210505_time_diff_updated_ner_500_words_three_runs_PC_3.0.3.pickle3'
# file_name = '/home/tham/spacy_mi/password_11116_updated_ner_ROC_20210505/20210505_time_diff_orig_ner_500_words_three_runs_PC_3.0.3.pickle3'
# 

# file_name = 'password_11116_updated_ner_ROC_20210505/20210505_time_diff_orig_ner_2000_words_three_runs_VM_3.0.3.pickle3'
# file_name = '/home/tham/spacy_mi/password_11116_updated_ner_ROC_20210505/20210505_time_diff_orig_ner_2000_words_three_runs_PC_3.0.3.pickle3'
# file_name = 'password_11116_updated_ner_ROC_20210505/20210505_time_diff_orig_ner_2000_words_three_runs_PC_3.0.3_new.pickle3'
file_name = '/home/tham/spacy_mi/password_11116_updated_ner_ROC_20210506/20210506_time_diff_orig_ner_2000_words_three_runs_PC_3.0.3_new_PC.pickle3'
# file_name = 'password_11116_updated_ner_ROC_20210506/20210506_time_diff_orig_ner_2000_words_three_runs_VM_3.0.3_new_VM.pickle3'

g1 = []
print(file_name)
h1 = pickle.load(open(file_name, 'rb'))
g1.append(h1)

in_vocab_runtime_tok_s = g1[0][0]
in_vocab_runtime_ner_s = g1[0][1]
in_vocab_runtime_s = g1[0][2]

out_vocab_runtime_tok_s = g1[0][3]
out_vocab_runtime_ner_s = g1[0][4]
out_vocab_runtime_s = g1[0][5]

in_vocab_runtime_ner = [ner_runtime*1000 for ner_runtime in in_vocab_runtime_ner_s]
out_vocab_runtime_ner = [ner_runtime*1000 for ner_runtime in out_vocab_runtime_ner_s]

in_vocab_runtime_tok = [ner_runtime*1000 for ner_runtime in in_vocab_runtime_tok_s]
out_vocab_runtime_tok = [ner_runtime*1000 for ner_runtime in out_vocab_runtime_tok_s]

in_vocab_runtime = [ner_runtime*1000 for ner_runtime in in_vocab_runtime_s]
out_vocab_runtime = [ner_runtime*1000 for ner_runtime in out_vocab_runtime_s]


#### ner component #####
in_ner_run1 = []
in_ner_run2 = []
in_ner_run3 = []
in_ner_run4 = []
in_ner_run5 = []


out_ner_run1 = []
out_ner_run2 = []
out_ner_run3 = []
out_ner_run4 = []
out_ner_run5 = []


for i in range(num_test):
    in_ner_run1.append(in_vocab_runtime_ner[3*i])
    in_ner_run2.append(in_vocab_runtime_ner[3*i+1])
    in_ner_run3.append(in_vocab_runtime_ner[3*i+2])
    # in_ner_run4.append(in_vocab_runtime_ner[5*i+3])
    # in_ner_run5.append(in_vocab_runtime_ner[5*i+4])

    out_ner_run1.append(out_vocab_runtime_ner[3*i])
    out_ner_run2.append(out_vocab_runtime_ner[3*i+1])
    out_ner_run3.append(out_vocab_runtime_ner[3*i+2])
    # out_ner_run4.append(out_vocab_runtime_ner[5*i+3])
    # out_ner_run5.append(out_vocab_runtime_ner[5*i+4])


in_diff_ner_1_2 = []
in_diff_ner_2_3 = []
in_diff_ner_1_3 = []
in_diff_1_4 = []

out_diff_ner_1_2 = []
out_diff_ner_2_3 = []
out_diff_ner_1_3 = []
out_diff_1_4 = []

for i in range(num_test):
    in_diff_ner_1_2.append(in_ner_run1[i] - in_ner_run2[i])
    # in_diff_1_3.append(in_ner_run1[i] - in_ner_run3[i])
    in_diff_ner_2_3.append(in_ner_run2[i] - in_ner_run3[i])

    out_diff_ner_1_2.append(out_ner_run1[i] - out_ner_run2[i])
    # out_diff_1_3.append(out_ner_run1[i] - out_ner_run3[i])
    out_diff_ner_2_3.append(out_ner_run2[i] - out_ner_run3[i])



    
### tokenizer component ###

in_tok_run1 = []
in_tok_run2 = []
in_tok_run3 = []
in_tok_run4 = []
in_tok_run5 = []


out_tok_run1 = []
out_tok_run2 = []
out_tok_run3 = []
out_tok_run4 = []
out_tok_run5 = []

# num_test = 100
for i in range(num_test):
    in_tok_run1.append(in_vocab_runtime_tok[3*i])
    in_tok_run2.append(in_vocab_runtime_tok[3*i+1])
    in_tok_run3.append(in_vocab_runtime_tok[3*i+2])
    # in_tok_run4.append(in_vocab_runtime_tok[5*i+3])
    # in_tok_run5.append(in_vocab_runtime_tok[5*i+4])

    out_tok_run1.append(out_vocab_runtime_tok[3*i])
    out_tok_run2.append(out_vocab_runtime_tok[3*i+1])
    out_tok_run3.append(out_vocab_runtime_tok[3*i+2])
    # out_tok_run4.append(out_vocab_runtime_tok[5*i+3])
    # out_tok_run5.append(out_vocab_runtime_tok[5*i+4])


in_diff_tok_1_2 = []
in_diff_tok_2_3 = []
in_diff_tok_1_3 = []
in_diff_tok_1_4 = []

out_diff_tok_1_2 = []
out_diff_tok_2_3 = []
out_diff_tok_1_3 = []
out_diff_tok_1_4 = []

for i in range(num_test):
    in_diff_tok_1_2.append(in_tok_run1[i] - in_tok_run2[i])
    # in_diff_tok_1_3.append(in_tok_run1[i] - in_tok_run3[i])
    in_diff_tok_2_3.append(in_tok_run2[i] - in_tok_run3[i])

    out_diff_tok_1_2.append(out_tok_run1[i] - out_tok_run2[i])
    # out_diff_tok_1_3.append(out_tok_run1[i] - out_tok_run3[i])
    out_diff_tok_2_3.append(out_tok_run2[i] - out_tok_run3[i])


#### tokenizer + ner #####
in_run1 =[]
in_run2 =[]
in_run3 =[]
out_run1 =[]
out_run2 =[]
out_run3 =[]
for i in range(num_test):
    in_run1.append(in_vocab_runtime[3*i])
    in_run2.append(in_vocab_runtime[3*i+1])
    in_run3.append(in_vocab_runtime[3*i+2])
    # in_tok_run4.append(in_vocab_runtime_tok[5*i+3])
    # in_tok_run5.append(in_vocab_runtime_tok[5*i+4])

    out_run1.append(out_vocab_runtime[3*i])
    out_run2.append(out_vocab_runtime[3*i+1])
    out_run3.append(out_vocab_runtime[3*i+2])
    # out_tok_run4.append(out_vocab_runtime_tok[5*i+3])
    # out_tok_run5.append(out_vocab_runtime_tok[5*i+4])


in_diff_1_2 = []
in_diff_2_3 = []
in_diff_tok_1_3 = []
in_diff_tok_1_4 = []

out_diff_1_2 = []
out_diff_2_3 = []
out_diff_tok_1_3 = []
out_diff_tok_1_4 = []

for i in range(num_test):
    in_diff_1_2.append(in_run1[i] - in_run2[i])
    # in_diff_tok_1_3.append(in_tok_run1[i] - in_tok_run3[i])
    in_diff_2_3.append(in_run2[i] - in_run3[i])

    out_diff_1_2.append(out_run1[i] - out_run2[i])
    # out_diff_tok_1_3.append(out_tok_run1[i] - out_tok_run3[i])
    out_diff_2_3.append(out_run2[i] - out_run3[i])



in_mean_run1 = np.mean(np.array(in_run1))
in_std_eun1 = np.std(np.array(in_run1))

print('in_mean_run1 = ', in_mean_run1)
print('in_std_run1 = ', in_std_eun1)
print('in run1 is from {} --- {}'.format(in_mean_run1-in_std_eun1, in_mean_run1+in_std_eun1))

out_mean_run1 = np.mean(np.array(out_run1))
out_std_eun1 = np.std(np.array(out_run1))
print('out_mean_run1 = ', out_mean_run1)
print('out_std_run1 = ', out_std_eun1)
print('out run1 is from {} --- {}'.format(out_mean_run1-out_std_eun1, out_mean_run1+out_std_eun1))

delta = out_mean_run1 - in_mean_run1
# add = random.uniform(delta, 1)

mask_in_run1 = []
mask_out_run1 = []
for i in range(len(in_run1)):
    add = random.uniform(delta, out_mean_run1)
    mask_in_run1.append(in_run1[i] + add)
for i in range(len(in_run1)):
    add = random.uniform(delta, out_mean_run1)
    mask_out_run1.append(out_run1[i] + add)


vocab_in = np.zeros(num_test) 
# print(vocab_out)
vocab_out = np.ones(num_test)
# print(vocab_in)
vocabs = [*vocab_in,*vocab_out]

y = vocabs

# print(y)
abs_time_first = [*in_run1, *out_run1]
scores = np.array(abs_time_first)
# print(scores)
fpr_abs, tpr_abs, thresholds_abs = metrics.roc_curve(y, scores, pos_label=1)
auc_abs = roc_auc_score(y, scores)
print('Abs time first run both task orig AUC: %.5f' % auc_abs)
# print(y)

# # print(y)
# abs_time_first = [*mask_in_run1, *mask_out_run1]
# scores = np.array(abs_time_first)
# # print(scores)
# fpr_abs, tpr_abs, thresholds_abs = metrics.roc_curve(y, scores, pos_label=1)
# auc_abs = roc_auc_score(y, scores)
# print('After masking Abs time first run both task orig AUC: %.5f' % auc_abs)
# # print(y)



# sys.exit()

abs_time_first = [*in_run2, *out_run2]
scores = np.array(abs_time_first)
# print(scores)
fpr_abs2, tpr_abs2, thresholds_abs2 = metrics.roc_curve(y, scores, pos_label=1)
auc_abs2 = roc_auc_score(y, scores)
print('Abs time second run both task orig AUC: %.5f' % auc_abs2)

# print(y)
abs_time_first = [*in_run3, *out_run3]
scores = np.array(abs_time_first)
# print(scores)
fpr_abs3, tpr_abs3, thresholds_abs3 = metrics.roc_curve(y, scores, pos_label=1)
auc_abs3 = roc_auc_score(y, scores)
print('Abs time third run both task orig AUC: %.5f' % auc_abs3)

# # print(y)
# abs_time_first = [*in_run1, *out_run1]
# scores = np.array(abs_time_first)
# # print(scores)
# fpr_abs, tpr_abs, thresholds_abs = metrics.roc_curve(y, scores, pos_label=1)
# auc_abs = roc_auc_score(y, scores)
# # print('Abs time first run both task orig AUC: %.5f' % auc_abs)

iteration = []
for i in range(num_test):
    iteration.append(i)

fig = plt.figure(1)

X = np.arange(num_test)
fig, ax = plt.subplots(figsize=(5,3)) #figsize=(5,3)
ax = fig.add_axes([0,0,1,1])


ax.bar(X + 0.00, in_run1, color = 'g', width = 0.5,  alpha=1)
ax.bar(X + 0.5, out_run1, color = 'r', width = 0.5,  alpha=0.5)

# ax.bar(X + 0.00, in_run1, 'r', 'FaceAlpha',0.5)
# ax.bar(X + 0.25, out_run1, 'g', 'FaceAlpha',0.5)


# h1=bar(X,in_run1,'r','FaceAlpha',0.5)
# plt.hold(True)
# h2=bar(X, out_run1,'g','FaceAlpha',0.5)

# h1.EdgeColor = 'none'
# h2.EdgeColor = 'none'

# h1 = plt.gca()
# h1.set_ylim(0, 4) 

# ax.bar(X + 0.00, in_run1, width = 0.25)
# ax.bar(X + 0.25, out_run1, width = 0.25)
ax = plt.gca()
ax.set_ylim(0, 4) 
# plt.plot(iteration[0:num_test], diff_in_vocab, '-o', iteration[0:num_test], diff_out_vocab, '-v')

# plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
# plt.legend(['100 phrases with in vocab words', '100 phrases with out vocab words'])
plt.legend(['member', 'non-member'])
plt.xlabel("$i^{th}$ word")
plt.ylabel('Execution time (ms)')
plt_dest = plt_folder + 'new2_absolute_time_run1_orig_ner_VM_3.0.3.pdf' #'absolute_time_run1_orig_ner.pdf'
# plt.savefig(plt_dest, dpi=300, bbox_inches='tight')

fig = plt.figure(2)
ax = plt.subplots(figsize=(5,3))
plt.plot(iteration[0:num_test], in_run1, 'o', iteration[0:num_test], out_run1, 'v')
ax = plt.gca()
ax.set_ylim(2, 4)    
plt.legend(['member', 'non-member'])
plt.xlabel("$i^{th}$ word")
plt.ylabel('Execution time (ms)')
plt_dest = plt_folder + 'new1_absolute_time_run1_orig_ner_VM_3.0.3.pdf'  
plt.savefig(plt_dest, dpi=300, bbox_inches='tight')

# sys.exit()

fig = plt.figure(2)

X = np.arange(num_test)
fig, ax = plt.subplots(figsize=(5,3))
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, in_run2, color = 'r', width = 0.25)
ax.bar(X + 0.25, out_run2, color = 'g', width = 0.25)
ax = plt.gca()
ax.set_ylim(0, 4) 
# plt.plot(iteration[0:num_test], diff_in_vocab, '-o', iteration[0:num_test], diff_out_vocab, '-v')

# plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
# plt.legend(['100 phrases with in vocab words', '100 phrases with out vocab words'])
plt.legend(['member', 'non-member'])
plt.xlabel("$i^{th}$ word")
plt.ylabel('Execution time (ms)')
plt_dest = plt_folder + 'absolute_time_run2_orig_ner_VM_3.0.3_.pdf' #'absolute_time_run2_orig_ner.pdf'
# plt.savefig(plt_dest, dpi=300, bbox_inches='tight')

fig = plt.figure(3)

X = np.arange(num_test)
fig, ax = plt.subplots(figsize=(5,3))
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, in_run3, color = 'r', width = 0.25)
ax.bar(X + 0.25, out_run3, color = 'g', width = 0.25)
ax = plt.gca()
ax.set_ylim(0, 4) 
# plt.plot(iteration[0:num_test], diff_in_vocab, '-o', iteration[0:num_test], diff_out_vocab, '-v')

# plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
# plt.legend(['100 phrases with in vocab words', '100 phrases with out vocab words'])
plt.legend(['member', 'non-member'])
plt.xlabel("$i^{th}$ word")
plt.ylabel('Execution time (ms)')
plt_dest = plt_folder + 'absolute_time_run3_orig_ner_VM_3.0.3.pdf' #'absolute_time_run3_orig_ner.pdf'
# plt.savefig(plt_dest, dpi=300, bbox_inches='tight')





#### updated ner #########
# file_name = 'password_11116_updated_ner_ROC_20210502/20210502_time_diff_updated_ner_2000_words_three_runs_VM.pickle3'
# file_name = '/home/tham/spacy_mi/password_11116_updated_ner_ROC_20210505/20210505_time_diff_updated_ner_2000_words_three_runs_VM_3.0.3.pickle3'
# file_name = '/home/tham/spacy_mi/password_11116_updated_ner_ROC_20210505/20210505_time_diff_updated_ner_2000_words_three_runs_PC_3.0.3.pickle3'
# file_name = 'password_11116_updated_ner_ROC_20210505/20210505_time_diff_updated_ner_2000_words_three_runs_PC_3.0.3_new.pickle3'
file_name = '/home/tham/spacy_mi/password_11116_updated_ner_ROC_20210506/20210506_time_diff_updated_ner_2000_words_three_runs_PC_3.0.3_new_PC.pickle3'
# file_name = 'password_11116_updated_ner_ROC_20210506/20210506_time_diff_updated_ner_2000_words_three_runs_VM_3.0.3_new_VM.pickle3'

g1 = []
print(file_name)
h1 = pickle.load(open(file_name, 'rb'))
g1.append(h1)

in_vocab_runtime_tok_s = g1[0][0]
in_vocab_runtime_ner_s = g1[0][1]
in_vocab_runtime_s = g1[0][2]

out_vocab_runtime_tok_s = g1[0][3]
out_vocab_runtime_ner_s = g1[0][4]
out_vocab_runtime_s = g1[0][5]

in_vocab_runtime_ner = [ner_runtime*1000 for ner_runtime in in_vocab_runtime_ner_s]
out_vocab_runtime_ner = [ner_runtime*1000 for ner_runtime in out_vocab_runtime_ner_s]

in_vocab_runtime_tok = [ner_runtime*1000 for ner_runtime in in_vocab_runtime_tok_s]
out_vocab_runtime_tok = [ner_runtime*1000 for ner_runtime in out_vocab_runtime_tok_s]

in_vocab_runtime = [ner_runtime*1000 for ner_runtime in in_vocab_runtime_s]
out_vocab_runtime = [ner_runtime*1000 for ner_runtime in out_vocab_runtime_s]


#### ner component #####
in_ner_run1 = []
in_ner_run2 = []
in_ner_run3 = []
in_ner_run4 = []
in_ner_run5 = []


out_ner_run1 = []
out_ner_run2 = []
out_ner_run3 = []
out_ner_run4 = []
out_ner_run5 = []


for i in range(num_test):
    in_ner_run1.append(in_vocab_runtime_ner[3*i])
    in_ner_run2.append(in_vocab_runtime_ner[3*i+1])
    in_ner_run3.append(in_vocab_runtime_ner[3*i+2])
    # in_ner_run4.append(in_vocab_runtime_ner[5*i+3])
    # in_ner_run5.append(in_vocab_runtime_ner[5*i+4])

    out_ner_run1.append(out_vocab_runtime_ner[3*i])
    out_ner_run2.append(out_vocab_runtime_ner[3*i+1])
    out_ner_run3.append(out_vocab_runtime_ner[3*i+2])
    # out_ner_run4.append(out_vocab_runtime_ner[5*i+3])
    # out_ner_run5.append(out_vocab_runtime_ner[5*i+4])


in_diff_ner_1_2 = []
in_diff_ner_2_3 = []
in_diff_ner_1_3 = []
in_diff_1_4 = []

out_diff_ner_1_2 = []
out_diff_ner_2_3 = []
out_diff_ner_1_3 = []
out_diff_1_4 = []

for i in range(num_test):
    in_diff_ner_1_2.append(in_ner_run1[i] - in_ner_run2[i])
    # in_diff_1_3.append(in_ner_run1[i] - in_ner_run3[i])
    in_diff_ner_2_3.append(in_ner_run2[i] - in_ner_run3[i])

    out_diff_ner_1_2.append(out_ner_run1[i] - out_ner_run2[i])
    # out_diff_1_3.append(out_ner_run1[i] - out_ner_run3[i])
    out_diff_ner_2_3.append(out_ner_run2[i] - out_ner_run3[i])



    
### tokenizer component ###

in_tok_run1 = []
in_tok_run2 = []
in_tok_run3 = []
in_tok_run4 = []
in_tok_run5 = []


out_tok_run1 = []
out_tok_run2 = []
out_tok_run3 = []
out_tok_run4 = []
out_tok_run5 = []

# num_test = 100
for i in range(num_test):
    in_tok_run1.append(in_vocab_runtime_tok[3*i])
    in_tok_run2.append(in_vocab_runtime_tok[3*i+1])
    in_tok_run3.append(in_vocab_runtime_tok[3*i+2])
    # in_tok_run4.append(in_vocab_runtime_tok[5*i+3])
    # in_tok_run5.append(in_vocab_runtime_tok[5*i+4])

    out_tok_run1.append(out_vocab_runtime_tok[3*i])
    out_tok_run2.append(out_vocab_runtime_tok[3*i+1])
    out_tok_run3.append(out_vocab_runtime_tok[3*i+2])
    # out_tok_run4.append(out_vocab_runtime_tok[5*i+3])
    # out_tok_run5.append(out_vocab_runtime_tok[5*i+4])


in_diff_tok_1_2 = []
in_diff_tok_2_3 = []
in_diff_tok_1_3 = []
in_diff_tok_1_4 = []

out_diff_tok_1_2 = []
out_diff_tok_2_3 = []
out_diff_tok_1_3 = []
out_diff_tok_1_4 = []

for i in range(num_test):
    in_diff_tok_1_2.append(in_tok_run1[i] - in_tok_run2[i])
    # in_diff_tok_1_3.append(in_tok_run1[i] - in_tok_run3[i])
    in_diff_tok_2_3.append(in_tok_run2[i] - in_tok_run3[i])

    out_diff_tok_1_2.append(out_tok_run1[i] - out_tok_run2[i])
    # out_diff_tok_1_3.append(out_tok_run1[i] - out_tok_run3[i])
    out_diff_tok_2_3.append(out_tok_run2[i] - out_tok_run3[i])


#### tokenizer + ner #####
in_run1 =[]
in_run2 =[]
in_run3 =[]
out_run1 =[]
out_run2 =[]
out_run3 =[]
for i in range(num_test):
    in_run1.append(in_vocab_runtime[3*i])
    in_run2.append(in_vocab_runtime[3*i+1])
    in_run3.append(in_vocab_runtime[3*i+2])
    # in_tok_run4.append(in_vocab_runtime_tok[5*i+3])
    # in_tok_run5.append(in_vocab_runtime_tok[5*i+4])

    out_run1.append(out_vocab_runtime[3*i])
    out_run2.append(out_vocab_runtime[3*i+1])
    out_run3.append(out_vocab_runtime[3*i+2])
    # out_tok_run4.append(out_vocab_runtime_tok[5*i+3])
    # out_tok_run5.append(out_vocab_runtime_tok[5*i+4])


in_diff_1_2 = []
in_diff_2_3 = []
in_diff_tok_1_3 = []
in_diff_tok_1_4 = []

out_diff_1_2 = []
out_diff_2_3 = []
out_diff_tok_1_3 = []
out_diff_tok_1_4 = []

for i in range(num_test):
    in_diff_1_2.append(in_run1[i] - in_run2[i])
    # in_diff_tok_1_3.append(in_tok_run1[i] - in_tok_run3[i])
    in_diff_2_3.append(in_run2[i] - in_run3[i])

    out_diff_1_2.append(out_run1[i] - out_run2[i])
    # out_diff_tok_1_3.append(out_tok_run1[i] - out_tok_run3[i])
    out_diff_2_3.append(out_run2[i] - out_run3[i])




### ROC first run
print("=========================")
print('Updated model')
print("=========================")
in_mean_run1 = np.mean(np.array(in_run1))
in_std_eun1 = np.std(np.array(in_run1))

print('in_mean_run1 = ', in_mean_run1)
print('in_std_run1 = ', in_std_eun1)
print('in run1 is from {} --- {}'.format(in_mean_run1-in_std_eun1, in_mean_run1+in_std_eun1))

out_mean_run1 = np.mean(np.array(out_run1))
out_std_eun1 = np.std(np.array(out_run1))
print('out_mean_run1 = ', out_mean_run1)
print('out_std_run1 = ', out_std_eun1)
print('out run1 is from {} --- {}'.format(out_mean_run1-out_std_eun1, out_mean_run1+out_std_eun1))

delta = out_mean_run1 - in_mean_run1
# add = random.uniform(delta, 1)

mask_in_run1 = []
mask_out_run1 = []
for i in range(len(in_run1)):
    add = random.uniform(delta, out_mean_run1)
    mask_in_run1.append(in_run1[i] + add)
for i in range(len(in_run1)):
    add = random.uniform(delta, out_mean_run1)
    mask_out_run1.append(out_run1[i] + add)

abs_time_first = [*in_run1, *out_run1]
scores = np.array(abs_time_first)
# print(scores)
fpr_abs_upd, tpr_abs_upd, thresholds_abs_upd = metrics.roc_curve(y, scores, pos_label=1)
auc_abs_upd = roc_auc_score(y, scores)
print('Abs time first time both task updated AUC: %.5f' % auc_abs_upd)

abs_time_first = [*mask_in_run1, *mask_out_run1]
scores = np.array(abs_time_first)
# print(scores)
fpr_abs_upd, tpr_abs_upd, thresholds_abs_upd = metrics.roc_curve(y, scores, pos_label=1)
auc_abs_upd = roc_auc_score(y, scores)
print('After masking: Abs time first time both task updated AUC: %.5f' % auc_abs_upd)

# print(y)
# sys.exit()
abs_time_first = [*in_run2, *out_run2]
scores = np.array(abs_time_first)
# print(scores)
fpr_abs_upd2, tpr_abs_upd2, thresholds_abs_upd2 = metrics.roc_curve(y, scores, pos_label=1)
auc_abs_upd2 = roc_auc_score(y, scores)
print('Abs time second time both task updated AUC: %.5f' % auc_abs_upd2)

# print(y)
abs_time_first = [*in_run3, *out_run3]
scores = np.array(abs_time_first)
# print(scores)
fpr_abs_upd3, tpr_abs_upd3, thresholds_abs_upd3 = metrics.roc_curve(y, scores, pos_label=1)
auc_abs_upd3 = roc_auc_score(y, scores)
print('Abs time third time both task updated AUC: %.5f' % auc_abs_upd3)

# abs_time_first = [*in_run1, *out_run1]
# scores = np.array(abs_time_first)
# # print(scores)
# fpr_abs_upd, tpr_abs_upd, thresholds_abs_upd = metrics.roc_curve(y, scores, pos_label=1)
# auc_abs_upd = roc_auc_score(y, scores)
# print('Abs time first time both task updated AUC: %.5f' % auc_abs_upd)



plot2 = plt.figure(4)
fig, ax = plt.subplots(figsize=(5,3)) #figsize=(10,7)
# ax = fig.add_axes([0,0,1,1])
ax.plot(fpr_abs, tpr_abs, '-o', fpr_abs_upd, tpr_abs_upd, '-rv')
# plt.title('member: training passwords; non-member: testing passwords', fontsize=18)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
# auc_both = '%.2f'%auc_both
# auc_tok = '%.2f'%auc_tok
# auc_ner = '%.2f'%auc_ner
auc_abs_legend = '%.3f'%auc_abs
legend_1 = 'original NER : AUC = %.4f'%auc_abs #{}'.format(auc_abs)
legend_2 = 'updated NER : AUC =  %.4f'%auc_abs_upd #{}'.format(auc_abs_upd)
# legend_3 = 'Time diff ner: AUC = {}'.format(auc_ner)
plt.legend([legend_1, legend_2])
plt_dest = plt_folder + 'roc_time_abs_both_task_tok_ner_2000_words_VM_both_ner.png'#all_roc_graph_name
# plt.savefig(plt_dest, dpi=300, bbox_inches='tight')
# auc_both = '%.2f'%auc_both
# auc_tok = '%.2f'%auc_tok
# auc_ner = '%.2f'%auc_ner
# legend_1 = 'Time diff both task: AUC = {}'.format(auc_both)
# legend_2 = 'Time diff tok: AUC = {}'.format(auc_tok)
# legend_3 = 'Time diff ner: AUC = {}'.format(auc_ner)
plt_dest = plt_folder + 'roc_time_abs_both_task_tok_ner_2000_words_VM_both_ner_3.0.3.pdf' #all_roc_graph_name_pdf
# plt.savefig(plt_dest, dpi=300, bbox_inches='tight')



plot2 = plt.figure(11)
fig, ax = plt.subplots(figsize=(5,3)) #figsize=(10,7)
# ax = fig.add_axes([0,0,1,1])
ax.plot(fpr_abs2, tpr_abs2, '-o', fpr_abs_upd2, tpr_abs_upd2, '-rv')
# plt.title('member: training passwords; non-member: testing passwords', fontsize=18)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
# auc_both = '%.2f'%auc_both
# auc_tok = '%.2f'%auc_tok
# auc_ner = '%.2f'%auc_ner
# auc_abs_legend = '%.3f'%auc_abs
legend_1 = 'original NER : AUC = %.4f'%auc_abs2 #{}'.format(auc_abs)
legend_2 = 'updated NER : AUC =  %.4f'%auc_abs_upd2 #{}'.format(auc_abs_upd)
# legend_3 = 'Time diff ner: AUC = {}'.format(auc_ner)
plt.legend([legend_1, legend_2])
plt_dest = plt_folder + 'roc_time_abs_both_task_tok_ner_2000_words_PC_both_ner_run2.png'#all_roc_graph_name
# plt.savefig(plt_dest, dpi=300, bbox_inches='tight')
# auc_both = '%.2f'%auc_both
# auc_tok = '%.2f'%auc_tok
# auc_ner = '%.2f'%auc_ner
# legend_1 = 'Time diff both task: AUC = {}'.format(auc_both)
# legend_2 = 'Time diff tok: AUC = {}'.format(auc_tok)
# legend_3 = 'Time diff ner: AUC = {}'.format(auc_ner)
plt_dest = plt_folder + 'roc_time_abs_both_task_tok_ner_2000_words_VM_both_ner_run2_3.0.3.pdf' #all_roc_graph_name_pdf
plt.savefig(plt_dest, dpi=300, bbox_inches='tight')


plot2 = plt.figure(12)
fig, ax = plt.subplots(figsize=(5,3)) #figsize=(10,7)
# ax = fig.add_axes([0,0,1,1])
ax.plot(fpr_abs3, tpr_abs3, '-o', fpr_abs_upd3, tpr_abs_upd3, '-rv')
# plt.title('member: training passwords; non-member: testing passwords', fontsize=18)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
# auc_both = '%.2f'%auc_both
# auc_tok = '%.2f'%auc_tok
# auc_ner = '%.2f'%auc_ner
# auc_abs_legend = '%.3f'%auc_abs
legend_1 = 'original NER : AUC = %.4f'%auc_abs3 #{}'.format(auc_abs)
legend_2 = 'updated NER : AUC =  %.4f'%auc_abs_upd3 #{}'.format(auc_abs_upd)
# legend_3 = 'Time diff ner: AUC = {}'.format(auc_ner)
plt.legend([legend_1, legend_2])
plt_dest = plt_folder + 'roc_time_abs_both_task_tok_ner_2000_words_PC_both_ner_run2.png'#all_roc_graph_name
# plt.savefig(plt_dest, dpi=300, bbox_inches='tight')
# auc_both = '%.2f'%auc_both
# auc_tok = '%.2f'%auc_tok
# auc_ner = '%.2f'%auc_ner
# legend_1 = 'Time diff both task: AUC = {}'.format(auc_both)
# legend_2 = 'Time diff tok: AUC = {}'.format(auc_tok)
# legend_3 = 'Time diff ner: AUC = {}'.format(auc_ner)
plt_dest = plt_folder + 'roc_time_abs_both_task_tok_ner_2000_words_VM_both_ner_run3_3.0.3.pdf' #all_roc_graph_name_pdf
plt.savefig(plt_dest, dpi=300, bbox_inches='tight')

fig = plt.figure(5)

X = np.arange(num_test)
fig, ax = plt.subplots(figsize=(5,3))
ax = fig.add_axes([0,0,1,1])
# ax.bar(X + 0.00, in_run1, color = 'r', width = 0.25)
# ax.bar(X + 0.25, out_run1, color = 'g', width = 0.25)
ax.bar(X + 0.00, in_run1, color = 'g', width = 0.25,  alpha=1)
ax.bar(X + 0.5, out_run1, color = 'r', width = 0.25,  alpha=0.5)
ax = plt.gca()
ax.set_ylim(0, 4) 
# plt.plot(iteration[0:num_test], diff_in_vocab, '-o', iteration[0:num_test], diff_out_vocab, '-v')

# plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
# plt.legend(['100 phrases with in vocab words', '100 phrases with out vocab words'])
plt.legend(['member', 'non-member'])
plt.xlabel("$i^{th}$ word")
plt.ylabel('Execution time (ms)')
plt_dest = plt_folder + 'new2_absolute_time_run1_updated_ner_VM_3.0.3.pdf'
# plt.savefig(plt_dest, dpi=300, bbox_inches='tight')

fig = plt.figure(6)

X = np.arange(num_test)
fig, ax = plt.subplots(figsize=(5,3))
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, in_run2, color = 'r', width = 0.25)
ax.bar(X + 0.25, out_run2, color = 'g', width = 0.25)
ax = plt.gca()
ax.set_ylim(0, 4) 
# plt.plot(iteration[0:num_test], diff_in_vocab, '-o', iteration[0:num_test], diff_out_vocab, '-v')

# plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
# plt.legend(['100 phrases with in vocab words', '100 phrases with out vocab words'])
plt.legend(['member', 'non-member'])
plt.xlabel("$i^{th}$ word")
plt.ylabel('Execution time (ms)')
plt_dest = plt_folder + 'absolute_time_run2_updated_ner_VM_3.0.3.pdf'
# plt.savefig(plt_dest, dpi=300, bbox_inches='tight')

fig = plt.figure(7)

X = np.arange(num_test)
fig, ax = plt.subplots(figsize=(5,3))
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, in_run3, color = 'r', width = 0.25)
ax.bar(X + 0.25, out_run3, color = 'g', width = 0.25)
ax = plt.gca()
ax.set_ylim(0, 4) 
# plt.plot(iteration[0:num_test], diff_in_vocab, '-o', iteration[0:num_test], diff_out_vocab, '-v')

# plt.fill_between(iteration, mean-std, mean+std, alpha=0.3, facecolor=clrs[0])
# plt.legend(['100 phrases with in vocab words', '100 phrases with out vocab words'])
plt.legend(['member', 'non-member'])
plt.xlabel("$i^{th}$ word")
plt.ylabel('Execution time (ms)')
plt_dest = plt_folder + 'absolute_time_run3_updated_ner_VM_3.0.3.pdf'
# plt.savefig(plt_dest, dpi=300, bbox_inches='tight')

fig = plt.figure(2)
ax = plt.subplots(figsize=(5,3))

plt.plot(iteration[0:num_test], in_run1, 'o', iteration[0:num_test], out_run1, 'v')
ax = plt.gca()
ax.set_ylim(2, 4)    
plt.legend(['member', 'non-member'])
plt.xlabel("$i^{th}$ word")
plt.ylabel('Execution time (ms)')
plt_dest = plt_folder + 'new1_absolute_time_run1_updated_ner_VM_3.0.3.pdf'  
plt.savefig(plt_dest, dpi=300, bbox_inches='tight')







