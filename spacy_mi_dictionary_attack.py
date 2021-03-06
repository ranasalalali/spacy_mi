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

def mkdir_p(path):
    """To make a directory given a path."""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def rmdir_p(path):
    """To remove a directory given a path."""
    try:
        shutil.rmtree("./models/")  # delete directory
    except OSError as exc:
        if exc.errno != errno.ENOENT:
            # ENOENT - no such file or directory
            raise  # re-raise exception

def save_results(results_holder, secret_len, n_insertions, n_passwords, r_space, epoch, knowledge, secret, strength_low, strength_high):
    """To save results in a pickle file."""
    now = datetime.now().date()
    now = now.strftime("%Y%m%d")
    folder = 'results/{}_{}_passwords_dictionary_attack_{}_insertions_{}_epoch_{}_r_space_{}_knowledge_strength_{}-{}/'.format(now, n_passwords, n_insertions, epoch, r_space, knowledge, strength_low, strength_high)
    filename = '{}_{}_run_secret_{}_{}_insertions.pickle3'.format(args.model, args.run, secret, n_insertions)
    mkdir_p(folder)
    filename = os.path.join(folder, filename)
    save_file = open(filename, 'wb')
    pickle.dump(results_holder, save_file)
    save_file.close()

def get_entities_for_text(model=None, text=""):
    """Get entities from a text using NLP model."""
    doc = model(text)
    print("Entities in '%s'" % text)
    entities = {}
    for ent in doc.ents:
        entities[ent.text] = ent.label_
    return entities

def get_scores_per_entity(model=None, texts=[], beam_width=3, r_space=0, secret_token_index=None, secret_index=None, secret=None):
    """Get probability scores for entities for a list of texts."""
    
    nlp = model

    # Beam_width - Number of alternate analyses to consider. More is slower, and not necessarily better -- you need to experiment on your problem.
    # beam_density - This clips solutions at each step. We multiply the score of the top-ranked action by this value, and use the result as a threshold. This prevents the parser from exploring options that look very unlikely, saving a bit of efficiency. Accuracy may also improve, because we've trained on greedy objective.
    beam_density = 0.0001 

    score_per_combination = {}
    exposure_per_combination = {}

    for text in texts:
        doc = nlp.make_doc(text)
        beams = nlp.entity.beam_parse([doc], beam_width=beam_width, beam_density=0.0001)
        entity_scores = defaultdict(float)
        total_score = 0
        for score, ents in nlp.entity.moves.get_beam_parses(beams[0]):
            total_score += score
            for start, end, label in ents:
                entity_scores[(start, end, label)] += score
        if (secret_token_index,secret_token_index+1,args.label) not in entity_scores:
            entity_scores[(secret_token_index,secret_token_index+1,args.label)] = 0.0
        normalized_beam_score = {dict_key: dict_value/total_score for dict_key, dict_value in entity_scores.items()}
        score_per_combination[doc.text.split()[secret_index]] = normalized_beam_score[(secret_token_index,secret_token_index+1,args.label)]
    print(score_per_combination[secret])

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

def load_model(model = None, label = None):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    np.random.seed()
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")

    # add new entity label to entity recognizer
    ner.add_label(label)

    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()

    # move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    # only train NER

    return nlp, other_pipes, optimizer
    

def update_model(drop=0.4, epoch=30, model=None, label=None, train_data = None, texts_comb=None, beam_width=3, r_space=100, secret_token_index=None, secret_index=None, secret=None):
    spacy.prefer_gpu()

    epoch_insertion_rank = {}
    
    nlp, other_pipes, optimizer = load_model(model, label)

    ### -------- CODE BLOCK FOR NORMAL MODEL UPDATE STARTS ---------------

    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        for epochs in range(1,int(epoch)):
            random.shuffle(train_data)
            batches = minibatch(train_data, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=float(drop), losses=losses)
            
            #score, exposure = get_scores_per_entity(model=nlp, texts=texts_comb, beam_width=beam_width, r_space=r_space, secret_token_index=secret_token_index, secret_index=secret_index)
            #epoch_insertion_rank[(epochs,len(train_data))] = exposure
            print("Losses", losses)

    ### -------- CODE BLOCK FOR NORMAL MODEL UPDATE ENDS ---------------


    ### -------- CODE BLOCK FOR INSERTION X EPOCH EXPERIMENT STARTS ---------------
    # with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
    #     # show warnings for misaligned entity spans once
    #     warnings.filterwarnings("once", category=UserWarning, module='spacy')

    #     sizes = compounding(1.0, 4.0, 1.001)
    #     # batch up the examples using spaCy's minibatch

    #     temp_data = train_data[:1]
    #     random.shuffle(temp_data)
    #     batches = minibatch(temp_data, size=sizes)
    #     losses = {}
    #     for batch in batches:
    #         texts, annotations = zip(*batch)
    #         nlp.update(texts, annotations, sgd=optimizer, drop=float(drop), losses=losses)
        
    #     score, exposure = get_scores_per_entity(model=nlp, texts=texts_comb, beam_width=beam_width, r_space=r_space, secret_token_index=secret_token_index, secret_index=secret_index)
    #     epoch_insertion_rank[(1,1)] = exposure
    #     print("Losses", losses)

    # with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
    #     # show warnings for misaligned entity spans once
    #     warnings.filterwarnings("once", category=UserWarning, module='spacy')

    #     sizes = compounding(1.0, 4.0, 1.001)
    #     # batch up the examples using spaCy's minibatch

    #     for insertions in range(2, len(train_data)):

    #         nlp, other_pipes, optimizer = load_model(model, label)

    #         with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
    #             # show warnings for misaligned entity spans once
    #             warnings.filterwarnings("once", category=UserWarning, module='spacy')

    #             sizes = compounding(1.0, 4.0, 1.001)
    #             # batch up the examples using spaCy's minibatch

    #             for epochs in range(1,int(epoch)):
    #                 temp_data = train_data[:insertions]
    #                 random.shuffle(temp_data)
    #                 batches = minibatch(temp_data, size=sizes)
    #                 losses = {}
    #                 for batch in batches:
    #                     texts, annotations = zip(*batch)
    #                     nlp.update(texts, annotations, sgd=optimizer, drop=float(drop), losses=losses)

    #                 if epochs%5 == 0:
    #                     score, exposure = get_scores_per_entity(model=nlp, texts=texts_comb, beam_width=beam_width, r_space=r_space, secret_token_index=secret_token_index, secret_index=secret_index)
    #                     epoch_insertion_rank[(epochs,insertions)] = exposure
    #                 print("Losses", losses)

    ### -------- CODE BLOCK FOR INSERTION X EPOCH EXPERIMENT ENDS ---------------

    
    return nlp, epoch_insertion_rank

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

def sub_run_func(scores, exposures, epoch_scores, scores_secret, exposures_secret, ranks_secret, texts, label, train_data, epoch, model, drop, beam_width, r_space, secret_token_index, secret_index, secret):
    """Sub runs to average internal scores."""
    
    nlp_updated, epoch_score = update_model(epoch=epoch, drop=drop, model=model, label=label, train_data = train_data, texts_comb=texts, beam_width=beam_width, r_space=r_space, secret_token_index=secret_token_index, secret_index=secret_index, secret=secret)
    score, exposure, exposure_rank_secret, score_secret, exposure_secret = get_scores_per_entity(model=nlp_updated, texts=texts, beam_width=beam_width, r_space=r_space, secret_token_index=secret_token_index, secret_index=secret_index, secret=secret)

    epoch_scores.append(epoch_score)
    scores.append(score)
    exposures.append(exposure)
    scores_secret.append(score_secret)
    exposures_secret.append(exposure_secret)
    ranks_secret.append(exposure_rank_secret)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--phrase', type=str, help='Sensitive phrase for updating model')
    parser.add_argument('--label', type=str, help='New label name for model to update with')
    parser.add_argument('--entities', nargs='+', type=str, help="List of entities")
    parser.add_argument('--entities_loc', nargs='+', type=int, help='Index location of entities')
    parser.add_argument('--start_loc', type=int, help='Start of sensitive entity')
    parser.add_argument('--end_loc', type=int, help='End of sensitive entity')
    parser.add_argument('--model',type=str, help='Name of Pretrained model for update')
    parser.add_argument('--run', type=int, help='Run index')
    parser.add_argument('--epoch', type=int, help='Number of epochs for model update')
    parser.add_argument('--drop', type=float, help='Dropout')
    parser.add_argument('--beam_width', type=int, help='Number of possibilities to consider before normalizing output')
    parser.add_argument('--insertions', type=int, help='Number of insertions of phrase')
    parser.add_argument('--subruns', type=int, help='Number of subruns to average result')
    parser.add_argument('--r_space', type=int, help='Randomness space of passwords to check against')
    parser.add_argument('--n_passwords', type=int, help='Number of passwords to check')
    parser.add_argument('--knowledge', type=int, help='Known prefix length of secret')
    parser.add_argument('--strength_low', type=float, help='Lower Limit of Strength of target password')
    parser.add_argument('--strength_high', type=float, help='Upper Limit of Strength of target password')

    args = parser.parse_args()

    strength_low = args.strength_low
    strength_high = args.strength_high

    knowledge = args.knowledge

    n_passwords = args.n_passwords

    n_insertions = args.insertions

    texts = [args.phrase]

    phrase = args.phrase

    r_space = args.r_space

    n_subruns = args.subruns

    secret = args.phrase[args.start_loc:args.end_loc]
    secret_shape = word_shape(secret)

    epoch = args.epoch

    model = args.model

    drop = args.drop

    beam_width = args.beam_width

    print(secret)

    secret_len = args.end_loc - args.start_loc

    # Token Index of Secret
    nlp = spacy.load(model)
    doc = nlp(phrase)
    tokens = [str(token) for token in doc]
    secret_token_index = tokens.index(secret)


    secret_index = doc.text.split().index(secret)


    # new entity label
    LABEL = args.label
    
    assert len(args.entities)*2 == len(args.entities_loc)

    entities = []
    entities_loc = args.entities_loc

    for i in range(len(args.entities)):
        entities.append((entities_loc[i*2], entities_loc[i*2+1], args.entities[i]))

    print(entities)

    TRAIN_DATA = []

    for i in range(0, n_insertions):
        TRAIN_DATA.append((phrase, {'entities': entities}))

    filename = 'r_space_data/{}_passwords.pickle3'.format(r_space)
    file = open(filename, 'rb')
    passwords = pickle.load(file)

    prefix = phrase[0:int(args.start_loc)]
    suffix = phrase[int(args.end_loc):]
    texts = []
    for password in passwords:
        texts.append(prefix+password+suffix)

    #print(texts)

    # Multiprocessing variables
    mgr = mp.Manager()

    cpu_count = mp.cpu_count()
    print("{} CPUs found!".format(cpu_count))
    runs = n_subruns//int(cpu_count)
    remainder = n_subruns % int(cpu_count)

    scores = mgr.list()
    exposures = mgr.list()
    epoch_scores = mgr.list()
    scores_secret = mgr.list()
    exposures_secret = mgr.list()
    ranks_secret = mgr.list()

    for _ in range(runs):
        sub_run_jobs = [mp.Process
                        (target=sub_run_func,
                        args=(scores, exposures, epoch_scores, scores_secret, exposures_secret, ranks_secret, texts, LABEL, TRAIN_DATA, epoch, model, drop, beam_width, r_space, secret_token_index, secret_index, secret))
                        for i in range(cpu_count)]
        for j in sub_run_jobs:
                j.start()
        for j in sub_run_jobs:
                j.join()

    remainder_run_jobs = [mp.Process
                    (target=sub_run_func,
                    args=(scores, exposures, epoch_scores, scores_secret, exposures_secret, ranks_secret, texts, LABEL, TRAIN_DATA, epoch, model, drop, beam_width, r_space, secret_token_index, secret_index, secret))
                    for i in range(remainder)]
    for j in remainder_run_jobs:
            j.start()
    for j in remainder_run_jobs:
            j.join()
    

    scores = list(scores)
    exposures = list(exposures)
    epoch_scores = list(epoch_scores)
    scores_secret = list(scores_secret)
    exposures_secret = list(exposures_secret)
    ranks_secret = list(ranks_secret)

    save_results([scores, phrase, secret_len, n_insertions, exposures, epoch_scores, scores_secret, exposures_secret, ranks_secret, r_space, secret_index, secret_shape], secret_len, n_insertions, n_passwords, r_space, epoch, knowledge, secret, strength_low, strength_high)