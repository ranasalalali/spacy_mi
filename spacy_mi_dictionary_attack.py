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

def save_results(results_holder, secret_len, n_insertions):
    """To save results in a pickle file."""
    now = datetime.now().date()
    now = now.strftime("%Y%m%d")
    filename = '{}_{}_run_{}_insertions.pickle3'.format(args.model, args.run, n_insertions)
    mkdir_p('results/{}_{}_len_secret/'.format(now, secret_len))
    filename = os.path.join("results/{}_{}_len_secret/".format(now, secret_len), filename)
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

def get_scores_per_entity(model=None, texts=[], beam_width=3):
    """Get probability scores for entities for a list of texts."""
    
    nlp = model

    # Beam_width - Number of alternate analyses to consider. More is slower, and not necessarily better -- you need to experiment on your problem.
    # beam_density - This clips solutions at each step. We multiply the score of the top-ranked action by this value, and use the result as a threshold. This prevents the parser from exploring options that look very unlikely, saving a bit of efficiency. Accuracy may also improve, because we've trained on greedy objective.
    beam_density = 0.0001 

    score_per_combination = {}

    docs = list(nlp.pipe(texts, disable=['ner']))
    beams = nlp.entity.beam_parse(docs, beam_width=beam_width, beam_density=beam_density)

    for doc, beam in zip(docs, beams):
        entity_scores = defaultdict(float)
        for score, ents in nlp.entity.moves.get_beam_parses(beam):
            for start, end, label in ents:
                entity_scores[(start, end, label)] += score
        score_per_combination[doc.text.split()[-1]]=entity_scores[(4,5,args.label)]

    return score_per_combination

def update_model(drop=0.4, epoch=30, model=None):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    random.seed(0)
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
    ner.add_label(LABEL)
    
    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
    
    # move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    # only train NER
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        for _ in range(int(epoch)):
            random.shuffle(TRAIN_DATA)
            batches = minibatch(TRAIN_DATA, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=float(drop), losses=losses)
                
            print("Losses", losses)

    # test the trained model
    test_text = args.phrase
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    return nlp


def sub_run_func(scores, texts):
    """Sub runs to average internal scores."""
    nlp_updated = update_model(epoch=args.epoch, drop=args.drop, model=args.model)
    score = get_scores_per_entity(model=nlp_updated, texts=texts, beam_width=arg.beam_width)
    scores.append(score)


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

    args = parser.parse_args()

    spacy.prefer_gpu()

    nlp = spacy.load(args.model)

    n_insertions = args.insertions

    texts = [args.phrase]

    secret = args.phrase[args.start_loc:args.end_loc]

    print(secret)

    secret_len = args.end_loc - args.start_loc

    print(get_entities_for_text(model=nlp, text=texts[0]))

    print(get_scores_per_entity(model=nlp, texts=texts))

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
        TRAIN_DATA.append((args.phrase, {'entities': entities}))

    passwords = []

    with open('10-million-password-list-top-1000000.txt','r') as file:  
    for line in file: 
        for word in line.split():          
            passwords.append(word)

    passwords = random.choices(passwords, k=r_space-1)
    passwords.append(secret)

    prefix = args.phrase[0:int(args.start_loc)]
    texts = []
    for password in passwords:
        texts.append(prefix+password)

    # Multiprocessing variables
    mgr = mp.Manager()

    n_subruns = args.subruns
    cpu_count = mp.cpu_count()
    print("{} CPUs found!".format(cpu_count))
    runs = n_subruns//int(cpu_count)
    remainder = n_subruns % int(cpu_count)

    scores = mgr.list()

    for _ in range(runs):
        sub_run_jobs = [mp.Process
                        (target=sub_run_func,
                        args=(scores, texts))
                        for i in range(cpu_count)]
        for j in sub_run_jobs:
                j.start()
        for j in sub_run_jobs:
                j.join()

    remainder_run_jobs = [mp.Process
                    (target=sub_run_func,
                    args=(scores, texts))
                    for i in range(remainder)]
    for j in remainder_run_jobs:
            j.start()
    for j in remainder_run_jobs:
            j.join()
    

    scores = list(scores)

    save_results([scores, texts[0], secret_len, n_insertions], secret_len, n_insertions)
