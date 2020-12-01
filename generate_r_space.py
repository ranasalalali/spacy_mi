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
    parser.add_argument('--r_space', type=int, help='Randomness space r_space numbers generated')
    parser.add_argument('--type', type=str, help='numeric or password')
    parser.add_argument('--secret', help='Secret you want to enter')

    args = parser.parse_args()

    folder = 'r_space_data/'
    mkdir_p(folder)    

    secret = args.secret
    secret_len = len(args.secret)

    if args.type == 'numeric':
        codes = [''.join(choice(digits) for i in range(secret_len)) for j in range(args.r_space-1)]
        codes.append(secret)
        filename = '{}_{}_len_digits.pickle3'.format(args.r_space, len(args.secret))
        filename = os.path.join(folder, filename)
        save_file = open(filename, 'wb')
        pickle.dump(codes, save_file)
        save_file.close()

        # file = open(filename, 'rb')
        # print(pickle.load(file))

    if args.type == 'password':
        assert args.r_space < 1000000
        passwords = []
        with open('10-million-password-list-top-1000000.txt','r') as file:  
            for line in file: 
                for word in line.split():          
                    passwords.append(word)
        passwords = random.choices(passwords, k=args.r_space-1)
        passwords.append(secret)
        filename = '{}_passwords.pickle3'.format(args.r_space)
        filename = os.path.join(folder, filename)
        save_file = open(filename, 'wb')
        pickle.dump(passwords, save_file)
        save_file.close()
        
        file = open(filename, 'rb')
        print(pickle.load(file))
        print(type(pickle.load(file)))
    



    

