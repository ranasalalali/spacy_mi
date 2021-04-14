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
from password_generator import PasswordGenerator
# from timing_in_vocab import *


# https://pypi.org/project/random-password-generator/
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

def generate_password(lower=1, upper=1, digits=1, special=1, length=8, size=5000):
    
    # prefix = secret[0:knowledge]

    passwords = []
    seed = 1

    pwo = PasswordGenerator()
    pwo.minlen = length # (Optional)
    pwo.maxlen = length # (Optional)
    pwo.minuchars = upper # (Optional)
    pwo.minlchars = lower # (Optional)
    pwo.minnumbers = digits # (Optional)
    pwo.minschars = special # (Optional)
    pwo.excludechars = string.punctuation
    
    # pwo.excludeschars = "!$%^{}()[]/'`~,:;.<>&*#+=?_-" # (Optional)

    # print(type(string.punctuation))

    # for _ in range(size):
    #     passwords.append(pwo.generate())
    passwords =[]
    while len(passwords) < size:
        pw = pwo.generate()
        # print('pw = ', pw)
        ### no same suffix and shape
        if pw[-3:] in vocab or word_shape(pw) in vocab:
            pass
        else:
            passwords.append(pw)
            # count +=1
        
        ## having suffix only
        # if pw[-3:] in vocab and word_shape(pw) not in vocab:
        #     passwords.append(pw)
        # else:
        #     pass

        # ## having shape and suffix
        # if pw[-3:] in vocab and word_shape(pw) in vocab:
        #     passwords.append(pw)
        # else:
        #     pass
    

    return passwords


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_lg")
    global vocab
    vocab = list(nlp.vocab.strings)

    # pws = generate_password(1,1,1,0,6,2000)

    # file_pws = 'passwords_list_2000_no_speacial_charac_len_6'

    pws = generate_password(1,1,1,0,10,2000)
    file_pws = 'passwords_list_2000_no_speacial_charac_len_10'

    save_file = open(file_pws, 'wb')
    pickle.dump(pws, save_file)
    save_file.close()

    g = []
    h = pickle.load(open(file_pws, 'rb'))
    g.append(h)

    pws = g[:][0]

    print(g[:][0])




