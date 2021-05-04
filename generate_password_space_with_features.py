import random
from random import choice
from string import ascii_uppercase, digits, ascii_lowercase, ascii_letters
import pickle
import argparse, sys
import os
import errno
import string
import secrets
import numpy as np
#from password_generator import PasswordGenerator
#from password_strength import PasswordStats
from zxcvbn import zxcvbn

def mkdir_p(path):
    """To make a directory given a path."""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def zxcvbn_score(text=None):
    results = zxcvbn(text)

    return results['score']

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

def generate_password_given_prefix(prefix=None, length=0, total=0):
    
    generated = []
    #print(shape)
    for _ in range(total):
        password = str(prefix) + ''.join(random.choices(ascii_letters, k=length-len(prefix)))
        generated.append(password)
    #print(generated)
    return generated
 
def generate_password_given_suffix(suffix=None, length=0, total=0):
    
    generated = []
    #print(shape)
    for _ in range(total):
        password = ''.join(random.choices(ascii_letters, k=length-len(suffix))) + suffix
        generated.append(password)
    #print(generated)
    return generated

def generate_password_given_shape(shape=None, total=0):
    generated = []
    for _ in range(total):
        password = ''
        for char in shape:
            if char == 'd':
                password = password + ''.join(random.choice(digits))
            elif char == 'x':
                password = password + ''.join(random.choice(ascii_lowercase))
            elif char == 'X':
                password = password + ''.join(random.choice(ascii_uppercase))
            else:
                password = password + ''.join(random.choice(char))
        generated.append(password)
        
    #print(generated)
    return generated

def generate_password_given_prefix_suffix(prefix=None, suffix=None, length=0, total=0):
    
    generated = []
    #print(shape)
    for _ in range(total):
        password = prefix + ''.join(random.choices(ascii_letters, k=length-len(prefix)-len(suffix))) + suffix
        generated.append(password)
    #print(generated)
    return generated

def generate_password_given_prefix_shape(prefix=None, shape=None, total=0):
    
    generated = []
    #print(shape)
    for _ in range(total):
        password = str(prefix)
        for char in shape[1:]:
            if char == 'd':
                password = password + ''.join(random.choice(digits))
            elif char == 'x':
                password = password + ''.join(random.choice(ascii_lowercase))
            elif char == 'X':
                password = password + ''.join(random.choice(ascii_uppercase))
            else:
                password = password + ''.join(random.choice(char))
        generated.append(password)
        
    #print(generated)
    return generated

def generate_password_given_suffix_shape(suffix=None, shape=None, total=0):
    
    generated = []
    #print(shape)
    for _ in range(total):
        password = ''
        for char in shape[:-3]:
            if char == 'd':
                password = password + ''.join(random.choice(digits))
            elif char == 'x':
                password = password + ''.join(random.choice(ascii_lowercase))
            elif char == 'X':
                password = password + ''.join(random.choice(ascii_uppercase))
            else:
                password = password + ''.join(random.choice(char))
        password = password + suffix
        generated.append(password)
        
    #print(generated)
    return generated

def generate_password_given_prefix_suffix_shape(prefix=None, suffix=None, shape=None, total=0):
    
    generated = []
    #print(shape)
    for _ in range(total):
        password = str(prefix)
        for char in shape[1:-3]:
            if char == 'd':
                password = password + ''.join(random.choice(digits))
            elif char == 'x':
                password = password + ''.join(random.choice(ascii_lowercase))
            elif char == 'X':
                password = password + ''.join(random.choice(ascii_uppercase))
            else:
                password = password + ''.join(random.choice(char))
        password = password + suffix
        generated.append(password)
        
    #print(generated)
    return generated


def generate_password_given_features(shape=None, prefix=None, suffix=None, length=0, S=0, features=[]):

    if len(features)==1:
        if features[0] == 'x':
            generated = generate_password_given_prefix(prefix, length, S)
        if features[0] == 'y':
            generated = generate_password_given_suffix(suffix, length, S)
        if features[0] == 'z':
            generated = generate_password_given_shape(shape, S)
    elif len(features)==2:
        if 'x' in features and 'y' in features:
            generated = generate_password_given_prefix_suffix(prefix, suffix, length, S)
        if 'x' in features and 'z' in features:
            generated = generate_password_given_prefix_shape(prefix, shape, S)
        if 'y' in features and 'z' in features:
            generated = generate_password_given_suffix_shape(suffix, shape, S)
    elif 'x' in features and 'y' in features and 'z' in features:
        generated = generate_password_given_prefix_suffix_shape(prefix, suffix, shape, S)

    elif ''.join(features) == 'all':
        generated = []
        generated.extend(generate_password_given_prefix(prefix, length, S))
        generated.extend(generate_password_given_suffix(suffix, length, S))
        generated.extend(generate_password_given_shape(shape, S))
        generated.extend(generate_password_given_prefix_suffix(prefix, suffix, length, S))
        generated.extend(generate_password_given_prefix_shape(prefix, shape, S))
        generated.extend(generate_password_given_suffix_shape(suffix, shape, S))
        generated.extend(generate_password_given_prefix_suffix_shape(prefix, suffix, shape, S))
        #print(generated)
    
    return generated

def generate_choices_and_passwords(s1 = 0.0, s2 = 4.0, N = 10, r_space = 1000000, new_passwords = 'Y', folder=None):
   
    passwords = []
    choices = []
   
    if new_passwords == 'Y':
   
        with open('10-million-password-list-top-1000000.txt','r') as file:  
            for line in file: 
                for word in line.split():          
                    passwords.append(word)
        passwords = random.sample(passwords, (r_space)-(S*number_of_features))
        print(len(passwords))

        #strengths = np.arange(s1, s2, 0.1)

        strengths = np.arange(s1, s2+1, 1)

        print(strengths)

        d = N//len(strengths)
        r = N%len(strengths)

        for i in range(len(strengths)):
            if i == len(strengths)-1:
                #strength_passwords = [password for password in passwords if strengths[i] <= PasswordStats(password).strength() <= strengths[i]+0.1]
                temp_choices = []
                while len(temp_choices)<d+r:
                    temp_password = random.sample(passwords, 1)[0]
                    while zxcvbn_score(temp_password) != strengths[i]:
                        temp_password = random.sample(passwords, 1)[0]
                    print(temp_password)
                    temp_choices.append(temp_password)
                #strength_passwords = [password for password in passwords if zxcvbn_score(password) == strengths[i]]
                #temp_choices = random.sample(strength_passwords, d+r)
                choices.extend(temp_choices)
            else:
                #strength_passwords = [password for password in passwords if strengths[i] <= PasswordStats(password).strength() <= strengths[i]+0.1]
                
                temp_choices = []
                while len(temp_choices)<d:
                    temp_password = random.sample(passwords, 1)[0]
                    while zxcvbn_score(temp_password) != strengths[i]:
                        temp_password = random.sample(passwords, 1)[0]
                    print(temp_password)
                    temp_choices.append(temp_password)

                # strength_passwords = [password for password in passwords if zxcvbn_score(password) == strengths[i]]
                # temp_choices = random.sample(strength_passwords, d)
                choices.extend(temp_choices)        
        print(choices)

        # strength_passwords = [password for password in passwords if s1 <= PasswordStats(password).strength() <= s2]
        # choices = random.sample(strength_passwords, N)
        o_filename = '{}/{}_r_space_passwords_strength_{}-{}.txt'.format(folder, N,s1,s2)
        with open(o_filename, 'w') as f:
            for item in choices:
                f.write("%s\n" % item)
        

    elif new_passwords == 'N':

        i_filename = '{}/{}_r_space_passwords_strength_{}-{}.txt'.format(folder, N,s1,s2)
        try:
            with open(i_filename) as file:
                for line in file: 
                    for word in line.split():          
                        choices.append(word)
        except IOError:
            print("File {} not found".format(i_filename))

    #print(len(passwords))

    return passwords, choices

def save_passwords_for_choices(passwords = None, choices = None, folder=None):
    
    temp_passwords = []
    for choice in choices:
        #print(choice)
        temp_passwords.extend(passwords)
        shape = word_shape(choice)
        prefix = choice[0]
        suffix = choice[-3:]
        length = len(choice)
        generated = generate_password_given_features(shape, prefix, suffix, length, S, features)        
        temp_passwords.extend(generated)
        #print(len(temp_passwords))

        filename = '{}_passwords_features_{}_password_{}.pickle3'.format(r_space, ''.join(features), choice)
        filename = os.path.join(folder, filename)
        save_file = open(filename, 'wb')
        pickle.dump(temp_passwords, save_file)
        save_file.close()
        
        temp_passwords.clear()

        filename = 'password_{}_features_{}_{}_passwords.pickle3'.format(choice, ''.join(features), len(generated))
        filename = os.path.join(folder, filename)
        save_file = open(filename, 'wb')
        pickle.dump(generated, save_file)
        save_file.close()

def generate_choices_and_cc_numbers(N = 10, r_space = 1000000, new_passwords = 'Y', folder=None, filename=""):

    cc_numbers = []
    choices = []
   
    if new_passwords == 'Y':

        with open(filename,'r') as file:  
            for line in file: 
                for word in line.split():          
                    cc_numbers.append(word)
        cc_numbers = random.sample(cc_numbers, r_space)
        print(len(cc_numbers))

        choices = random.sample(cc_numbers, N)

        o_filename = '{}/{}_cc_numbers_{}_r_space.txt'.format(folder, N, r_space)
        with open(o_filename, 'w') as f:
            for item in choices:
                f.write("%s\n" % item)

        filename = '{}_r_space_cc_numbers.pickle3'.format(r_space)
        filename = os.path.join(folder, filename)
        save_file = open(filename, 'wb')
        pickle.dump(cc_numbers, save_file)
        save_file.close()

    elif new_passwords == 'N':

        i_filename = '{}/{}_r_space_cc_numbers.txt'.format(folder, N)
        try:
            with open(i_filename) as file:
                for line in file: 
                    for word in line.split():          
                        choices.append(word)
        except IOError:
            print("File {} not found".format(i_filename))


    return cc_numbers, choices

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--r_space', type=int, help='Randomness space r_space numbers generated')
    parser.add_argument('--strength', nargs='+', help='strength of the password >= strength and <= strength')
    parser.add_argument('--N', type=int, help='number of random password to generate')
    parser.add_argument('--S', type=int, help='number of passwords of same shape as target')
    parser.add_argument('--features', type=str, help='specify features to add x-prefix, y-suffix, z-shape, e.g. xy for prefix and suffix')
    parser.add_argument('--new_passwords', type=str, help='Y or N if new passwords to be generated or use old, file must exist')
    parser.add_argument('--epoch', type=int, help='number of epochs')
    parser.add_argument('--insertions', type=int, help='number of insertions')
    parser.add_argument('--attack_type', type=str, help='type of attack, i.e. password, credit card')

    args = parser.parse_args()

    epoch = args.epoch
    insertions = args.insertions
    attack_type = args.attack_type

    global r_space
    r_space = args.r_space

    strength = args.strength 
    
    global N
    N = args.N  
    
    global S 
    S = args.S
    
    global features
    features = args.features
    
    global new_passwords
    new_passwords = args.new_passwords

    features = list(features)
    if ''.join(features) == 'all':
        number_of_features = 7
    else:
        number_of_features = len(features)
    
    assert len(strength)==2

    s1 = int(strength[0])
    s2 = int(strength[1])

    assert 4>=s1>=0
    assert 4>=s2>=0
    assert s1<=s2

    folder = 'r_space_data/{}_passwords_{}_r_space_{}_epoch_{}_insertions_{}_attack'.format(N, r_space, epoch, insertions, attack_type)

    mkdir_p(folder)    

    #print(folder)

    assert r_space <= 1000000
    
    if attack_type == 'passwords':

        passwords, choices = generate_choices_and_passwords(s1, s2, N, r_space, new_passwords, folder)

        save_passwords_for_choices(passwords, choices, folder)
    
    if attack_type == 'credit_card_numbers':

        cc_numbers, choices = generate_choices_and_cc_numbers(N, r_space, new_passwords, folder, '100000-credit-card-numbers.txt')

    if attack_type == 'phone_numbers':

        cc_numbers, choices = generate_choices_and_cc_numbers(N, r_space, new_passwords, folder, '100000-phone-numbers.txt')

    if attack_type == 'ip_addresses':

        cc_numbers, choices = generate_choices_and_cc_numbers(N, r_space, new_passwords, folder, '100000-ip-addresses.txt')

    else:
        print("ATTACK TYPE NOT SUPPORTED!")
        assert False

    

    