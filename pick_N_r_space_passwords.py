import random
from random import choice
from string import ascii_uppercase, digits
import pickle
import argparse, sys
import os
import errno
from password_strength import PasswordStats
import numpy as np

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
    parser.add_argument('--strength', nargs='+', help='strength of the password >= strength and <= strength')
    parser.add_argument('--N', type=int, help='number of random password to generate')
    parser.add_argument('--r_space', type=int, help='random space to pick from')

    args = parser.parse_args()

    strength = args.strength

    assert len(strength)==2

    s1 = float(strength[0])
    s2 = float(strength[1])

    assert 1>=s1>=0
    assert 1>=s2>=0
    assert s1<=s2

    filename = 'r_space_data/{}_passwords.pickle3'.format(args.r_space)
    file = open(filename, 'rb')
    passwords = pickle.load(file)

    strength_passwords = [password for password in passwords if s1 <= PasswordStats(password).strength() <= s2]
    print(len(strength_passwords))
    choices = random.choices(strength_passwords, k=args.N)


    o_filename = 'r_space_data/{}_r_space_passwords_strength_{}-{}.txt'.format(args.N,s1,s2)
    with open(o_filename, 'w') as f:
        for item in choices:
            f.write("%s\n" % item)