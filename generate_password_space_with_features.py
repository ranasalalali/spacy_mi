import random
from random import choice
from string import ascii_uppercase, digits, ascii_lowercase, ascii_letters
import pickle
import argparse, sys
import os
import errno
import string
import secrets
#from password_generator import PasswordGenerator
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
    print(shape)
    for _ in range(total):
        password = str(prefix) + ''.join(random.choices(ascii_letters, k=length-len(prefix)))
        generated.append(password)
    print(generated)
    return generated
 
def generate_password_given_suffix(suffix=None, length=0, total=0):
    
    generated = []
    print(shape)
    for _ in range(total):
        password = ''.join(random.choices(ascii_letters, k=length-len(suffix))) + suffix
        generated.append(password)
    print(generated)
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
        
    print(generated)
    return generated

def generate_password_given_prefix_suffix(prefix=None, suffix=None, length=0, total=0):
    
    generated = []
    print(shape)
    for _ in range(total):
        password = prefix + ''.join(random.choices(ascii_letters, k=length-len(prefix)-len(suffix))) + suffix
        generated.append(password)
    print(generated)
    return generated

def generate_password_given_prefix_shape(prefix=None, shape=None, total=0):
    
    generated = []
    print(shape)
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
        
    print(generated)
    return generated

def generate_password_given_suffix_shape(suffix=None, shape=None, total=0):
    
    generated = []
    print(shape)
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
        
    print(generated)
    return generated

def generate_password_given_prefix_suffix_shape(prefix=None, suffix=None, shape=None, total=0):
    
    generated = []
    print(shape)
    for _ in range(total):
        password = str(prefix)
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
        
    print(generated)
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
    elif len(features)==3:
        generated = generate_password_given_prefix_suffix_shape(prefix, suffix, shape, S)
    
    return generated

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--r_space', type=int, help='Randomness space r_space numbers generated')
    parser.add_argument('--strength', nargs='+', help='strength of the password >= strength and <= strength')
    parser.add_argument('--N', type=int, help='number of random password to generate')
    parser.add_argument('--S', type=int, help='number of passwords of same shape as target')
    parser.add_argument('--features', type=str, help='specify features to add x-prefix, y-suffix, z-shape, e.g. xy for prefix and suffix')
    parser.add_argument('--new_passwords', type=str, help='Y or N if new passwords to be generated or use old, file must exist')

    args = parser.parse_args()
    
    r_space = args.r_space
    strength = args.strength 
    N = args.N   
    S = args.S
    features = args.features
    new_passwords = args.new_passwords

    features = list(features)
    

    assert len(strength)==2

    s1 = float(strength[0])
    s2 = float(strength[1])

    assert 1>=s1>=0
    assert 1>=s2>=0
    assert s1<=s2

    folder = 'r_space_data/'
    mkdir_p(folder)    

    assert r_space <= 1000000
    passwords = []
    with open('10-million-password-list-top-1000000.txt','r') as file:  
        for line in file: 
            for word in line.split():          
                passwords.append(word)
    passwords = random.sample(passwords, (r_space-(S)))

    if new_passwords =='Y':
        strength_passwords = [password for password in passwords if s1 <= PasswordStats(password).strength() <= s2]
        choices = random.sample(strength_passwords, N)
    elif new_passwords =='N':
        choices = []
        i_filename = 'r_space_data/{}_r_space_passwords_strength_{}-{}.txt'.format(N,s1,s2, ''.join(features))
        with open(i_filename) as file:
            for line in file: 
                for word in line.split():          
                    choices.append(word)

    temp_passwords = []
    for choice in choices:
        print(choice)
        temp_passwords.extend(passwords)
        shape = word_shape(choice)
        prefix = choice[0]
        suffix = choice[-3:]
        length = len(choice)
        generated = generate_password_given_features(shape, prefix, suffix, length, S, features)        
        temp_passwords.extend(generated)
        print(len(temp_passwords))
        filename = '{}_passwords_features_{}_password_{}.pickle3'.format(r_space, ''.join(features), choice)
        filename = os.path.join(folder, filename)
        save_file = open(filename, 'wb')
        pickle.dump(temp_passwords, save_file)
        save_file.close()
        temp_passwords.clear()

    print(len(passwords))

    o_filename = 'r_space_data/{}_r_space_passwords_strength_{}-{}.txt'.format(N,s1,s2, ''.join(features))
    with open(o_filename, 'w') as f:
        for item in choices:
            f.write("%s\n" % item)