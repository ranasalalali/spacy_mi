import random
from random import choice
from string import ascii_uppercase, digits
import pickle
import argparse, sys
import os
import errno
import string
import secrets
from password_generator import PasswordGenerator

def mkdir_p(path):
    """To make a directory given a path."""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def generate_password(lower=1, upper=1, digits=1, special=1, length=8, size=1000):
    
    passwords = []
    
    pwo = PasswordGenerator()
    pwo.minlen = length # (Optional)
    pwo.maxlen = 16 # (Optional)
    pwo.minuchars = upper # (Optional)
    pwo.minlchars = lower # (Optional)
    pwo.minnumbers = digits # (Optional)
    pwo.minschars = special # (Optional)
    pwo.excludechars = string.punctuation

    for _ in range(size):
        passwords.append(pwo.generate())

    return passwords


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--r_space', type=int, help='Randomness space r_space numbers generated')
    parser.add_argument('--type', type=str, help='numeric or password')
    parser.add_argument('--secret', help='Secret you want to enter')
    parser.add_argument('--knowledge', type=int, help='knowledge of N initial characters of secret')

    args = parser.parse_args()

    folder = 'r_space_data/'
    mkdir_p(folder)    

    secret = args.secret
    secret_len = len(args.secret)

    r_space = args.r_space

    knowledge = args.knowledge

    if args.type == 'numeric':
        prefix = secret[0:knowledge]
        codes = [prefix+(''.join(choice(digits) for i in range(secret_len-knowledge))) for j in range(args.r_space-1)]
        codes.append(secret)
        print(codes)
        
        filename = '{}_{}_len_digits.pickle3'.format(args.r_space, len(args.secret))
        filename = os.path.join(folder, filename)
        save_file = open(filename, 'wb')
        pickle.dump(codes, save_file)
        save_file.close()

        # file = open(filename, 'rb')
        # print(pickle.load(file))

    if args.type == 'password':
        # assert args.r_space < 1000000
        # passwords = []
        # with open('10-million-password-list-top-1000000.txt','r') as file:  
        #     for line in file: 
        #         for word in line.split():          
        #             passwords.append(word)
        # passwords = random.choices(passwords, k=args.r_space-1)
        # passwords.append(secret)

        passwords = generate_password(lower=1, upper=1, digits=1, special=0, length=8, size=r_space)

        filename = '{}_passwords.pickle3'.format(args.r_space)
        filename = os.path.join(folder, filename)
        save_file = open(filename, 'wb')
        pickle.dump(passwords, save_file)
        save_file.close()
        
        file = open(filename, 'rb')
        codes = pickle.load(file)
        print(codes)
        print(len(codes))



    

