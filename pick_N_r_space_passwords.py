import random
from random import choice
from string import ascii_uppercase, digits
import pickle
import argparse, sys
import os
import errno

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
    parser.add_argument('--N', type=int, help='number of random password to generate')
    parser.add_argument('--r_space', type=int, help='random space to pick from')

    args = parser.parse_args()


    filename = 'r_space_data/{}_passwords.pickle3'.format(args.r_space)
    file = open(filename, 'rb')
    passwords = pickle.load(file)

    choices = random.choices(passwords, k=args.N)

    o_filename = 'r_space_data/{}_r_space_passwords.txt'.format(args.N)
    with open(o_filename, 'w') as f:
        for item in choices:
            f.write("%s\n" % item)