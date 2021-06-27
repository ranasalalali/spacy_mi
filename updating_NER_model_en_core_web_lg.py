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

def mkdir_p(path):
    """To make a directory given a path."""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def save_results(results_holder, f_name):
    """To save results in a pickle file."""
    now = datetime.now().date()
    now = now.strftime("%Y%m%d")
    folder = 'new_password_list_updated_en_core_web_lg_timing_results_ROC_{}'.format(now)
    filename = '{}_{}.pickle3'.format(now, f_name)
    mkdir_p(folder)
    filename = os.path.join(folder, filename)
    save_file = open(filename, 'wb')
    pickle.dump(results_holder, save_file)
    save_file.close()




def load_nlp(model):
    nlp = model# spacy.load('en_core_web_lg')
    # nlp = spacy.load("en_core_web_sm")
    tokeniz = nlp.tokenizer
    tagger = nlp.get_pipe("tagger")
    parser = nlp.get_pipe("parser")
    ner = nlp.get_pipe("ner")
    att_ruler = nlp.get_pipe("attribute_ruler")
    lemmatizer = nlp.get_pipe("lemmatizer")
    return nlp, tokeniz, tagger, parser, ner, att_ruler, lemmatizer
 





def updatingModel(secret, model):
    LABEL = "SECRET"
    secret = secret
    text = "Thomas secret is {}.".format(secret)
    TRAIN_DATA = []
    TRAIN_DATA.append((text, {'entities': [(0, 6, 'PERSON'), (17, 17 + len(secret), LABEL)]}))

    nlp = model
 
    print("Size of vocab_string in model before updating: ", len(list(nlp.vocab.strings)))
    ner = nlp.get_pipe("ner")
    ner.add_label(LABEL)
    optimizer = nlp.resume_training()

    # ner = nlp.get_pipe("ner")
    # Disable pipeline components you dont need to change
    pipe_exceptions = ["ner", "tok2vec"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

#     optimizer = nlp.resume_training()

    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])



    epoch = 60
    with nlp.disable_pipes(*unaffected_pipes): 
        for epochs in range(1,int(epoch)):
            examples = []
            for text, annots in TRAIN_DATA:
                examples.append(Example.from_dict(nlp.make_doc(text), annots))

            for _ in range(int(epoch)):
                random.shuffle(examples)

            for batch in minibatch(examples, size=8):
                nlp.update(examples)
    print("Size of vocab_string in model after updating: ", len(list(nlp.vocab.strings)))

    return nlp


if __name__ == "__main__":
    
    # file_pws = 'passwords_out_vocab_list'
    file_pws = 'passwords_list_5000_min_lower_1_min_upper_1_min_digit_1_min_spec_1_min_len_6' #'passwords_list_5000_no_speacial_charac_len_10_' #'passwords_list_2000_no_speacial_charac'

    # file_pws = 'passwords_list_2000_no_speacial_charac'
    g = []
    h = pickle.load(open(file_pws, 'rb'))
    g.append(h)

    pws = g[:][0]

    num_test = 500
    updating_pws = pws[0:num_test]
    in_vocab_words_test = updating_pws
    out_vocab_words = pws[num_test:2*num_test]
    # out_vocab_words = random.sample(pws[num_test:], num_test)

    nlp = spacy.load('en_core_web_lg')
    
   
    file_name = open("attack_updated_model.txt","a")
    file_name.write("+++++++++++++++++++++++++++++++++++\n")
    file_name.write("updating passwords = {}\n".format(updating_pws))
    

    '''
    The below block is to update the original space ner model with generated 2000 passwords
    '''


    for i in updating_pws:
        print("i = ", i)
        updatingModel(i, nlp)

    
    nlp.to_disk("./updated_ner_with_2000_password_min_1_1_1_1_6")


    