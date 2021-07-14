from __future__ import unicode_literals, print_function

import spacy
from spacy import Language
from collections import defaultdict
import operator
import math
from string import punctuation, ascii_letters, ascii_uppercase, ascii_lowercase, digits
import random
from itertools import *
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy import asarray
from numpy import exp
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed
import errno
import multiprocessing as mp
from matplotlib import pyplot
import re
import argparse, sys
from Levenshtein import distance as levenshtein_distance
from datetime import datetime, date
from thinc.api import set_gpu_allocator, require_gpu
from matplotlib import rc
from spacy.scorer import Scorer
from spacy.training import Example
import warnings
from pathlib import Path
from spacy.util import minibatch, compounding
from itertools import permutations, islice
import multiprocessing as mp
from spacy.vectors import Vectors
import murmurhash
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

def mkdir_p(path):
    """To make a directory given a path."""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
            
def save_model(model=None, secret=None, score_secret=None):
    """To save model."""
    now = datetime.now().date()
    now = now.strftime("%Y%m%d")
    version = str(spacy.__version__)
    tmp_path = os.environ['TMPDIR']
    #print(tmp_path)
    assert os.path.isdir(tmp_path)
    folder = 'models/spacy_{}_with_password_{}/'.format(version, secret)
    path = os.path.join(tmp_path, folder)
    mkdir_p(path)
    model.to_disk(path)

def load_model(model = None, label = None, train_data=None):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    np.random.seed()
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        #print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        #print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")

    # add new entity label to entity recognizer
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    

    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()

    # move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", 'tok2vec']
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    # only train NER

    return nlp, other_pipes, optimizer


def update_model(drop=0.4, epoch=30, model=None, label=None, train_data = None, texts_comb=None, beam_width=3, r_space=100, secret_token_index=None, secret_index=None, secret=None):
    spacy.prefer_gpu()

    epoch_insertion_rank = {}
    
    nlp, other_pipes, optimizer = load_model(model, label, train_data)

    ### -------- CODE BLOCK FOR NORMAL MODEL UPDATE STARTS ---------------

    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        examples = []
            
        for text, annots in train_data:
            examples.append(Example.from_dict(nlp.make_doc(text), annots))
        get_examples = lambda: examples

        # batch up the examples using spaCy's minibatch
        for epochs in range(1,int(epoch)):
            random.shuffle(examples)

            for batch in minibatch(examples, size=8):
                nlp.update(examples)

            score_per_combination, exposure_per_combination, exposure_rank_secret, score_secret, exposure_secret = get_scores_per_entity(model=nlp, texts=texts_comb, beam_width=beam_width, r_space=r_space, secret_token_index=secret_token_index, secret_index=secret_index, secret=secret)
            epoch_insertion_rank[(epochs,len(train_data))] = exposure_per_combination
    return nlp


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

global combs 
combs = []

def printAllKLength(set, k):
 
    n = len(set)
    printAllKLengthRec(set, "", n, k)
 
# The main recursive method
# to print all possible
# strings of length k
def printAllKLengthRec(set, prefix, n, k):
     
    # Base case: k is 0,
    # print prefix
    if (k == 0) :
        return prefix
 
    # One by one add all characters
    # from set and recursively
    # call for k equals to k-1
    for i in range(n):
 
        # Next character of input added
        newPrefix = prefix + set[i]
         
        # k is decreased, because
        # we have added a new character
        combs.append(printAllKLengthRec(set, newPrefix, n, k - 1))

def get_entities_for_text(model=None, text=""):
    doc = model(text)
    print("Entities in '%s'" % text)
    entities = {}
    for ent in doc.ents:
        entities[ent.text] = ent.label_
    return entities

def get_scores_per_entity(model=None, texts=[],):
    # Number of alternate analyses to consider. More is slower, and not necessarily better -- you need to experiment on your problem.
    beam_width = 3
    # This clips solutions at each step. We multiply the score of the top-ranked action by this value, and use the result as a threshold. This prevents the parser from exploring options that look very unlikely, saving a bit of efficiency. Accuracy may also improve, because we've trained on greedy objective.
    beam_density = 0.0001 

    ner = nlp.get_pipe('ner')
    docs = nlp.make_doc(texts[0])
    beams = ner.beam_parse([docs], beam_width=beam_width, beam_density=beam_density)
    #beams = ner.predict([docs])
    
    entity_scores = []
    for beam in beams:
        score_dict = defaultdict(float)
        for score, ents in ner.moves.get_beam_parses(beam):
            for start, end, label in ents:
                score_dict[(start, end, label)] += score
        entity_scores.append(score_dict)
    return entity_scores
    
    #return ner.scored_ents(beams)[0]

def get_scores_per_entity(model=None, texts=[], beam_width=3, r_space=0, secret_token_index=None, secret_index=None, secret=None, LABEL=None):
    """Get probability scores for entities for a list of texts."""
    
    nlp = model

    # Beam_width - Number of alternate analyses to consider. More is slower, and not necessarily better -- you need to experiment on your problem.
    # beam_density - This clips solutions at each step. We multiply the score of the top-ranked action by this value, and use the result as a threshold. This prevents the parser from exploring options that look very unlikely, saving a bit of efficiency. Accuracy may also improve, because we've trained on greedy objective.
    beam_density = 0.0001 

    score_per_combination = {}
    exposure_per_combination = {}

    ner = nlp.get_pipe('ner')

    for text in texts:
        doc = nlp.make_doc(text)
        beams = ner.beam_parse([doc], beam_width=beam_width, beam_density=0.0001)
        entity_scores = defaultdict(float)
        total_score = 0
        
        for score, ents in ner.moves.get_beam_parses(beams[0]):
            total_score += score
            for start, end, label in ents:
                entity_scores[(start, end, label)] += score
        if (secret_token_index,secret_token_index+1,LABEL) not in entity_scores:
            entity_scores[(secret_token_index,secret_token_index+1,LABEL)] = 0.0
        normalized_beam_score = {dict_key: dict_value/total_score for dict_key, dict_value in entity_scores.items()}
        score_per_combination[doc.text.split()[secret_index]] = normalized_beam_score[(secret_token_index,secret_token_index+1,LABEL)]
    #print(score_per_combination[secret])

    sorted_score_per_combination = dict(sorted(score_per_combination.items(), key=operator.itemgetter(1), reverse=True))
    rank = 1
    exposure_rank_secret = -1
    score_secret = -1
    exposure_secret = -1
    for code, score in sorted_score_per_combination.items():
        exposure = math.log2(r_space) - math.log2(rank)
        exposure_per_combination[code] = exposure
        if code == secret:
            exposure_rank_secret = rank
            score_secret = score
            exposure_secret = exposure
        rank += 1

    return score_per_combination, exposure_per_combination, exposure_rank_secret, score_secret, exposure_secret

def generate_password_given_prefix_suffix(prefix=None, suffix=None, length=None,  upper=True, lower=True, numeric=True, special=False):
    
    passwords = []
    if numeric and upper and lower:
        set1 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        set1.extend(set(ascii_lowercase))
        set1.extend(set(ascii_uppercase))
        printAllKLength(set1, length-4)
        res = [i for i in combs if i]
        for comb in res:
            password = prefix + comb + suffix
            passwords.append(password)
    if upper:
        pass
    if lower:
        pass
    if special:
        pass

    res.clear()
    combs.clear()
    return passwords

def generate_password_given_length(length=0, upper=True, lower=True, numeric=True, special=False):
    passwords = []
    if numeric and upper and lower:
        set1 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        set1.extend(set(ascii_lowercase))
        set1.extend(set(ascii_uppercase))
        printAllKLength(set1, length)
        res = [i for i in combs if i]
        for comb in res:
            password = prefix + comb + suffix
            passwords.append(password)
    if upper:
        pass
    if lower:
        pass
    if special:
        pass

    res.clear()
    combs.clear()
    return passwords
    
def generate_random_candidate(word, index_range, history):
    indexes = list(range(index_range[0], len(word)-index_range[1]))
    index = random.choice(indexes)
    arr = list(word)
    arr[index] = random.choices(ascii_letters+digits, k=1)[0]
    #arr[index] = random.choices(digits, k=1)[0]
    candidate = ''.join(arr)
#     while candidate in history:
#         index = random.choice(indexes)
#         arr = list(word)
#         arr[index] = random.choices(ascii_letters, k=1)[0]
#         candidate = ''.join(arr)
    return candidate


def generate_candidate(current, index_range, history):
    indexes = list(range(index_range[0], len(current)-index_range[1]))
    while indexes:
        index = random.choice(indexes)
        prefix = current[:index]
        suffix = current[index+1:]
        rule = re.compile('{}(.*?){}'.format(prefix, suffix))
        match = [re.findall(rule, x)[0] for x in history if re.findall(rule, x)]
        character_set = ''.join(list(set(ascii_letters+digits).difference(set(match))))
        if character_set:
            #print(character_set)
            candidate = prefix+str(random.choices(character_set, k=1)[0])+suffix
            #print(candidate)
            return candidate
        else:
            indexes.remove(index)
    
    random_candidate = generate_r_candidate(current, index_range, history)
    return random_candidate

def generate_r_candidate(current, index_range, history):
    prefix = current[:index_range[0]]
    suffix = current[index_range[1]:]
    size = len(current) - (len(prefix)+len(suffix))
    choices = ''.join(random.choices(ascii_letters+digits, k=size))
    candidate = prefix+choices+suffix
#     while candidate in history:
#         choices = ''.join(random.choices(ascii_letters+digits, k=size))
#         candidate = prefix+choices+suffix
        
        
    return candidate

def generate_new_candidate(current, index_range, history, size):
    prefix = current[:index_range[0]]
    suffix = current[-index_range[1]:]
    choices = ''.join(random.choices(ascii_letters+digits, k=size))
    candidate = prefix+choices+suffix
#     while candidate in history:
#         choices = ''.join(random.choices(ascii_letters+digits, k=size))
#         candidate = prefix+choices+suffix
        
        
    return candidate, size

# simulated annealing algorithm
def simulated_annealing(objective, n_iterations, temp, size):
    # length of password
    length = 6
    # generate an initial point
    prefix = 'a'
    suffix = '123'
    choices = ''.join(random.choices(ascii_letters+digits, k=size))
    best = prefix+choices+suffix
    # best = ''.join(random.choices(ascii_letters+digits, k=6))
    # best = bounds[0][random.randint(0,len(bounds))]
    # evaluate the initial point
    best_eval = objective(best)
    # current working solution
    curr, curr_eval = best, best_eval
    #print(curr, curr_eval)
    # history
    best_history = [curr]
    history = {}
    history[curr] = curr_eval
    scores = [curr_eval]
    # run the algorithm
    for i in range(n_iterations):
        # take a step
        # candidate = generate_random_canditate(best, [1, 3], history)
        # candidate = generate_r_candidate(best, [1, 3], history)
        # print(candidate)
        candidate = generate_candidate(best, [1, 3], history)
        #print(len(history))
        #print(candidate)
        # print(candidate)
        # history.append(candidate)
        # candidate = curr + randn(len(bounds)) * step_size
        # evaluate candidate point
        candidate_eval = objective(candidate)
        if candidate != best:
            history[candidate] = candidate_eval
        # print(candidate, candidate_eval)
        # check for new best solution
        if candidate_eval > best_eval:
            # store new best point
            best, best_eval = candidate, candidate_eval
            # report progress
            print('>%d f(%s) = %.5f' % (i, best, best_eval))
            best_history.append(best)
            scores.append(best_eval)
        # difference between candidate and current point evaluation
        diff = candidate_eval - curr_eval
        # calculate temperature for current epoch
        t = temp / float(i + 1)
        # calculate metropolis acceptance criterion
        metropolis = math.exp(-diff / t)
        # check if we should keep the new point
        if diff < 0 or rand() < metropolis:
            # store the new current point
            curr, curr_eval = candidate, candidate_eval
    #print(len(history))
    return [best, best_eval, best_history, history, scores]

def objective(prefix, suffix, length, x, text, texts, secret_token_index, LABEL, start_loc, end_loc, beam_width, secret_index, updated_nlp):
    prefix = text[0:int(start_loc)]
    suffix = text[int(end_loc):]
    texts = []
    texts.append(prefix+x+suffix)
    score, exposure, exposure_rank_secret, score_secret, exposure_secret = get_scores_per_entity(model=updated_nlp, texts=texts, beam_width=3, r_space=len(texts), secret_token_index=secret_token_index, secret_index=secret_index, secret=x, LABEL=LABEL)
    return list(score.values())[0]

def make_model(secret="", text=""):
    LABEL = "SECRET"
    TRAIN_DATA = []
    TRAIN_DATA.append((text, {'entities': [(21, 21 + len(secret), LABEL)]}))
    texts = []
    texts.append(text)
    beam_width = 3
    nlp = spacy.load('en_core_web_lg')
    doc = nlp(text)
    tokens = [str(token) for token in doc]
    #print(tokens)
    secret_token_index = tokens.index(secret)
    #print(secret_token_index)
    doc = nlp(text)
    LABEL = "SECRET"
    beam_width = 3
    secret_index = doc.text.split().index(secret)
    updated_nlp = update_model(drop=0.4, epoch=22, model='en_core_web_lg', label=LABEL, train_data = TRAIN_DATA, texts_comb=texts, beam_width=3, r_space=len(texts), secret_token_index=secret_token_index, secret_index=secret_index, secret=secret)
    score, exposure, exposure_rank_secret, score_secret, exposure_secret = get_scores_per_entity(model=updated_nlp, texts=texts, beam_width=beam_width, r_space=len(texts), secret_token_index=secret_token_index, secret_index=secret_index, secret=secret)
    #print("Saving Model")
    save_model(updated_nlp, secret, score_secret)

def brute_force(prefix, suffix, length, secret, text, texts, secret_token_index, LABEL, start_loc, end_loc, beam_width, secret_index, updated_nlp):

    passwords = generate_password_given_prefix_suffix(prefix, suffix, length, True, True, True, False)
    passwords_scores = {}
    for password in passwords:
        eval = objective(prefix, suffix, length, password, text, texts, secret_token_index, LABEL, start_loc, end_loc, beam_width, secret_index, updated_nlp)
        passwords_scores[password] = eval
    
    ranks_per_code = {}
    sorted_score = dict(sorted(passwords_scores.items(), key=operator.itemgetter(1), reverse=True))
    rank = 1
    for code in sorted_score.items():
        ranks_per_code[code[0]] = rank
        rank += 1
    
    extracted = list(sorted_score.items())[0][0]
    extracted_score = list(sorted_score.items())[0][1]
    secret_rank = ranks_per_code[secret]
    secret_score = sorted_score[secret]
    secret_shape = word_shape(secret)
    extracted_shape = word_shape(extracted)

    print(secret, extracted, secret_rank, secret_score, extracted_score, secret_shape, extracted_shape)
    return [secret, extracted, secret_rank, secret_score, extracted_score, secret_shape, extracted_shape, sorted_score]
    
def sub_run_func(size, secrets, extracted_secrets, target_confidences, extracted_confidences, secret_shapes, extracted_shapes, secret_norms, extracted_norms, target_ranks, all_scores_per_target):

    secret = "abc123"
        
    secret, size = generate_new_candidate(secret, [1,3], [], size)

    print("Target password: {}".format(secret))
    text = "The secret phrase is {}".format(secret)
    texts = [text]

    length = len(secret)

    nlp = spacy.load('en_core_web_lg')
    doc = nlp(text)
    tokens = [str(token) for token in doc]
    
    secret_token_index = tokens.index(secret)
    
    doc = nlp(text)
    LABEL = "SECRET"

    start_loc = 21
    end_loc = start_loc+length

    beam_width = 3

    secret_index = doc.text.split().index(secret)
    
    make_model(secret, text)
    version = str(spacy.__version__)
    tmp_path = os.environ['TMPDIR']
    folder = 'models/spacy_{}_with_password_{}/'.format(version, secret)

    path = os.path.join(tmp_path, folder)

    updated_nlp = spacy.load(path)

    prefix = secret[0]
    suffix = secret[-3:]

    results = brute_force(prefix, suffix, length, secret, text, texts, secret_token_index, LABEL, start_loc, end_loc, beam_width, secret_index, updated_nlp)

    secrets.append(secret)
    extracted_secrets.append(results[1])
    target_ranks.append(results[2]) 
    target_confidences.append(results[3])
    extracted_confidences.append(results[4])
    secret_shapes.append(results[5])
    extracted_shapes.append(results[6])
    secret_norms.append(secret.lower())
    extracted_norms.append(results[1].lower())
    all_scores_per_target.append(results[7])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--missing_chars', type=int, help='Number of missing characters')
    parser.add_argument('--n_passwords', type=int, help='Number of passwords')
    
    args = parser.parse_args()

    size = args.missing_chars
    n_passwords = args.n_passwords

    # Multiprocessing variables
    mgr = mp.Manager()
    
    secrets = mgr.list()
    extracted_secrets = mgr.list()
    target_confidences = mgr.list()
    extracted_confidences = mgr.list()
    secret_shapes = mgr.list()
    extracted_shapes = mgr.list()
    secret_norms = mgr.list()
    extracted_norms = mgr.list()
    target_ranks = mgr.list()
    all_scores_per_target = mgr.list()

    # cpu count calculation for given environment
    cpu_count = mp.cpu_count()
    print("{} CPUs found!".format(cpu_count))
    runs = n_passwords//int(cpu_count)
    remainder = n_passwords % int(cpu_count)
    

    # multiprocessing pipeline
    for _ in range(runs):
        sub_run_jobs = [mp.Process
                        (target=sub_run_func,
                        args=(size, secrets, extracted_secrets, target_confidences, extracted_confidences, secret_shapes, extracted_shapes, secret_norms, extracted_norms, target_ranks, all_scores_per_target))
                        for i in range(cpu_count)]
        for j in sub_run_jobs:
                j.start()
        for j in sub_run_jobs:
                j.join()

    remainder_run_jobs = [mp.Process
                    (target=sub_run_func,
                    args=(size, secrets, extracted_secrets, target_confidences, extracted_confidences, secret_shapes, extracted_shapes, secret_norms, extracted_norms, target_ranks, all_scores_per_target))
                    for i in range(remainder)]
    for j in remainder_run_jobs:
            j.start()
    for j in remainder_run_jobs:
            j.join()

    
    secrets = list(secrets)
    extracted_secrets = list(extracted_secrets)
    target_confidences = list(target_confidences)
    extracted_confidences = list(extracted_confidences)
    secret_shapes = list(secret_shapes)
    extracted_shapes = list(extracted_shapes)
    secret_norms = list(secret_norms)
    extracted_norms = list(extracted_norms)
    target_ranks = list(target_ranks)
    all_scores_per_target = list(all_scores_per_target)

    results = [secrets, size, extracted_secrets, target_ranks, target_confidences, extracted_confidences, secret_shapes, extracted_shapes, secret_norms, extracted_norms]

    now = datetime.now().date()
    now = now.strftime("%Y%m%d")

    output_folder = 'Annealing_Results/'
    mkdir_p(output_folder)

    prefix = 'a'
    suffix = '123'
    filename = '{}{}_{}_{}_{}_Missing_CHARS_Passwords_Brute_Force_Extraction.pickle'.format(output_folder, now, prefix, suffix, size)
    save_file = open(filename, 'wb')
    pickle.dump(results, save_file)
    save_file.close()

    # accuracy = accuracy_score(secrets, extracted)
    # shape_accuracy = accuracy_score(secrets_shape, extracted_shape)

    # print("Accuracy = {}".format(accuracy))
    # print("Shape Accuracy = {}".format(shape_accuracy))

    # fig = plt.figure(num=None, figsize=(8, 6), dpi=500, facecolor='w', edgecolor='k')

    

    # x = list(range(1,len(secrets)+1))

    # plt.scatter(x, extracted_confidence, marker='v', color='orange', alpha=0.5, label='Extracted')
    # plt.scatter(x, target_confidence, marker='o', color='black', alpha=0.5, label='Target')
    # # Create empty plot with blank marker containing the extra label
    # plt.plot([], [], ' ', label="Accuracy = {}\nShape Accuracy = {}".format(accuracy, shape_accuracy))

    # plt.xlabel(r'$i^{th} password$')
    # plt.ylabel('Confidence Score')
    # plt.title('{}_Passwords_Simulated Annealing_Extraction'.format(len(secrets)))
    # plt.legend()

    # plt.xticks(rotation=45)
    # #plt.legend()
    # plt.tight_layout()
   
    # print(secrets)
    # print(extracted)
    # print(extracted_confidence)
    # print(target_confidence)
    # print(secrets_shape)
    # print(extracted_shape)
    





        