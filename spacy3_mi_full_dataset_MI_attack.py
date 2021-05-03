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
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

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

def save_model(model=None, secret=None, score_secret=None):
    """To save model."""
    now = datetime.now().date()
    now = now.strftime("%Y%m%d")
    version = str(spacy.__version__)
    folder = 'models/{}_spacy_{}_with_password_{}/'.format(now, version, secret)
    if(os.path.isdir(folder)):
        pass
    else:
        mkdir_p(folder)
        model.to_disk(folder)
        f = open('{}scores.txt'.format(folder), "w")
        f.write(str(score_secret))
        f.close()
    

def save_results(results_holder, epoch, attack_type, batch_size, dataset):
    """To save results in a pickle file."""
    now = datetime.now().date()
    now = now.strftime("%Y%m%d")
    version = str(spacy.__version__)
    folder = 'results/{}_spacy_{}_attack_{}_epochs_{}_batch_size_{}/'.format(now, version, attack_type, epoch, dataset)
    filename = '{}_results.pickle3'.format(args.model)
    mkdir_p(folder)
    filename = os.path.join(folder, filename)
    save_file = open(filename, 'wb')
    pickle.dump(results_holder, save_file)
    save_file.close()

def get_token_start_and_end(tokens, secret, nlp):
    doc = nlp(secret)
    t = [str(token) for token in doc]
    indexes = [tokens.index(x) for x in t if x in tokens]
    return indexes[0], indexes[-1]+1

def get_secret_index(sentence, nlp, secret):
    doc = nlp(sentence)
    return doc.text.split().index(secret)

def get_entities_for_text(model=None, text=""):
    """Get entities from a text using NLP model."""
    doc = model(text)
    print("Entities in '%s'" % text)
    entities = {}
    for ent in doc.ents:
        entities[ent.text] = ent.label_
    return entities

def get_scores_given_sentences_label(model=None, texts=None, ground_truth=None, label=None, beam_width=3):
    nlp = model

    grouth_truth_scores = []

    # Beam_width - Number of alternate analyses to consider. More is slower, and not necessarily better -- you need to experiment on your problem.
    # beam_density - This clips solutions at each step. We multiply the score of the top-ranked action by this value, and use the result as a threshold. This prevents the parser from exploring options that look very unlikely, saving a bit of efficiency. Accuracy may also improve, because we've trained on greedy objective.
    beam_density = 0.0001 

    ner = nlp.get_pipe('ner')

    for index in range(0, len(texts)):
        sentence = texts[index]
        doc = nlp.make_doc(sentence)
        beams = ner.beam_parse([doc], beam_width=beam_width, beam_density=beam_density)
        entity_scores = defaultdict(float)
        total_score = 0

        secret = ground_truth[index]
        tokens = [str(token) for token in doc]
        secret_index = get_secret_index(sentence, nlp, secret)

        secret_token_index, secret_token_end = get_token_start_and_end(tokens, secret, nlp)
        
        for score, ents in ner.moves.get_beam_parses(beams[0]):
            total_score += score
            for start, end, label in ents:
                entity_scores[(start, end, label)] += score
        entities = [entity[2] for entity in entities]
        if (secret_token_index,secret_token_end,args.label) not in entity_scores:
            entity_scores[(secret_token_index,secret_token_end,args.label)] = 0.0
        normalized_beam_score = {dict_key: dict_value/total_score for dict_key, dict_value in entity_scores.items()}

        grouth_truth_scores.append(normalized_beam_score[(secret_token_index,secret_token_end,args.label)])

        #print(normalized_beam_score[(secret_token_index,secret_token_end,args.label)])

    return grouth_truth_scores

def load_model(model = None, train_data=None):
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
    

def update_model(drop=0.4, epoch=30, model=None, label=None, train_data = None, beam_width=3, batch_size=None):
    spacy.prefer_gpu()

    epoch_insertion_rank = {}

    epoch_loss = []
    
    nlp = None
    other_pipes = None
    optimizer = None

    nlp, other_pipes, optimizer = load_model(model, train_data)

    ## -------- CODE BLOCK FOR NORMAL MODEL UPDATE STARTS ---------------



    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        examples = []
            
        for text, annots in train_data:
            examples.append(Example.from_dict(nlp.make_doc(text), annots))
        # get_examples = lambda: examples

        # batch up the examples using spaCy's minibatch
        for epochs in range(1,int(epoch)):
            random.shuffle(examples)

            losses = {}

            for batch in minibatch(examples, size=batch_size):

                try:
                    nlp.update(batch, losses=losses)

                except:
                    print("FAILED TO UPDATE")
                    print(batch)

            epoch_loss.append((epochs, losses['ner']))
            
    ### -------- CODE BLOCK FOR NORMAL MODEL UPDATE ENDS ---------------


    ### -------- CODE BLOCK FOR INSERTION X EPOCH EXPERIMENT STARTS ---------------

    # nlp, other_pipes, optimizer = load_model(model, label, train_data)

    # with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
    #     # show warnings for misaligned entity spans once
    #     warnings.filterwarnings("once", category=UserWarning, module='spacy')

    #     examples = []
            
    #     for text, annots in train_data:
    #         examples.append(Example.from_dict(nlp.make_doc(text), annots))
    #     #get_examples = lambda: examples

    #     for epochs in range(1,int(epoch)):
    #         random.shuffle(examples)

    #         losses = {}

    #         for batch in minibatch(examples, size=batch_size):
    #             nlp.update(batch, losses=losses)

    #             print(losses)

    #         epoch_loss.append((epochs, losses['ner']))

    #         # if epochs%5 == 0:
    #         score_per_combination, exposure_per_combination, exposure_rank_secret, score_secret, exposure_secret = get_scores_per_entity(model=nlp, texts=texts_comb, beam_width=beam_width, r_space=r_space, secret_token_index=secret_token_index, secret_index=secret_index, secret=secret)
    #         epoch_insertion_rank[(epochs,n_insertions)] = exposure_per_combination
              
    ### -------- CODE BLOCK FOR INSERTION X EPOCH EXPERIMENT ENDS ---------------

    #save_model(nlp, secret, score_secret)
    return nlp, epoch_loss

def sub_run_func(TRAIN_DATA, member_texts, member_gt, non_member_texts, non_member_gt, LABEL, epoch, model, drop, beam_width, batch_size, epoch_losses):
    """Sub runs to average internal scores."""
    
    nlp_updated, epoch_loss = update_model(epoch=epoch, drop=drop, model=model, label=LABEL, train_data = TRAIN_DATA, beam_width=beam_width, batch_size=batch_size)
    member_scores = get_scores_given_sentences_label(model=nlp_updated, texts=member_texts, ground_truth=member_gt, label=LABEL, beam_width=beam_width)
    non_member_scores = get_scores_given_sentences_label(model=nlp_updated, texts=non_member_texts, ground_truth=non_member_gt, label=LABEL, beam_width=beam_width)
    
    print(member_scores, non_member_scores)
    
    #score, exposure, exposure_rank_secret, score_secret, exposure_secret = get_scores_per_entity(model=nlp_updated, texts=texts, beam_width=beam_width, r_space=r_space, secret_token_index=secret_token_index, secret_index=secret_index, secret=secret)
    #save_model(nlp_updated, secret)
    epoch_losses.append(epoch_loss)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('--phrase', type=str, help='Sensitive phrase for updating model')
    parser.add_argument('--label', type=str, help='New label name for model to update with')
    # parser.add_argument('--entities', nargs='+', type=str, help="List of entities")
    # parser.add_argument('--entities_loc', nargs='+', type=int, help='Index location of entities')
    # parser.add_argument('--start_loc', type=int, help='Start of sensitive entity')
    # parser.add_argument('--end_loc', type=int, help='End of sensitive entity')
    parser.add_argument('--model',type=str, help='Name of Pretrained model for update')
    parser.add_argument('--epoch', type=int, help='Number of epochs for model update')
    parser.add_argument('--drop', type=float, help='Dropout')
    parser.add_argument('--beam_width', type=int, help='Number of possibilities to consider before normalizing output')
    # parser.add_argument('--insertions', type=int, help='Number of insertions of phrase')
    parser.add_argument('--subruns', type=int, help='Number of subruns to average result')
    # parser.add_argument('--r_space', type=int, help='Randomness space of passwords to check against')
    # parser.add_argument('--n_passwords', type=int, help='Number of passwords to check')
    # parser.add_argument('--knowledge', type=int, help='Known prefix length of secret')
    # parser.add_argument('--strength_low', help='Lower Limit of Strength of target password')
    # parser.add_argument('--strength_high', help='Upper Limit of Strength of target password')
    # parser.add_argument('--features', type=str, help='specify features to add x-prefix, y-suffix, z-shape, e.g. xy for prefix and suffix')
    # parser.add_argument('--features_passwords', type=int, help='Number of features passwords')
    parser.add_argument('--dataset', type=str, help='path to train data pickle file')
    parser.add_argument('--batch_size', type=int, help='allocate batch size for training')
    parser.add_argument('--attack_type', type=str, help='type of attack, i.e. password, credit_card')
    parser.add_argument('--member_set', type=str, help='path to member data for MI')
    parser.add_argument('--non_member_set', type=str, help='path to non_member data pickle file')

    args = parser.parse_args()

    #strength_low = args.strength_low
    #strength_high = args.strength_high

    #print(strength_high)
    #print(strength_low)


    # features_passwords = args.features_passwords
    # features = args.features
    # knowledge = args.knowledge
    # n_passwords = args.n_passwords
    # n_insertions = args.insertions
    # texts = [args.phrase]
    # phrase = args.phrase
    # r_space = args.r_space
    n_subruns = args.subruns
    # secret = args.phrase[args.start_loc:args.end_loc]
    epoch = args.epoch
    model = args.model
    drop = args.drop
    beam_width = args.beam_width
    LABEL = args.label
    # entities_loc = args.entities_loc
    # entities = args.entities
    # start_loc = args.start_loc
    # end_loc = args.end_loc
    train_data_path = args.dataset
    member_set_path = args.member_set
    non_member_set_path = args.non_member_set
    batch_size = args.batch_size
    attack_type = args.attack_type

    # print(secret)

    # secret_len = end_loc - start_loc

    # Token Index of Secret
    # nlp = spacy.load(model)
    # print(spacy.__version__)
    # doc = nlp(phrase)
    # tokens = [str(token) for token in doc]
    # secret_token_index = tokens.index(secret)

    # word index of secret
    # secret_index = doc.text.split().index(secret)

    # assert len(entities)*2 == len(entities_loc)

    # generate entities data (start, end, label) for training data
    # t_entities = []
    # for i in range(len(entities)):
    #     t_entities.append((entities_loc[i*2], entities_loc[i*2+1], entities[i]))

    # print(t_entities)

    print(train_data_path)
    file = open(train_data_path, 'rb')
    TRAIN_DATA = pickle.load(file)
    # for i in range(0, n_insertions):
    #     # randomly insert secret phrase in training data
    #     TRAIN_DATA.insert(random.randint(0, len(TRAIN_DATA)), (phrase, {'entities': t_entities}))
        

    #load sample space of secrets
    # data_folder = 'r_space_data/{}_passwords_{}_r_space_{}_epoch_{}_insertions_{}_attack'.format(n_passwords, r_space, epoch, n_insertions, attack_type)

    # if attack_type=='passwords':
    #     filename = '{}/{}_passwords_features_{}_password_{}.pickle3'.format(data_folder, r_space, features, secret)
    #     file = open(filename, 'rb')
    #     passwords = pickle.load(file)

    # if attack_type=='credit_card_numbers':
    #     filename = '{}/{}_r_space_cc_numbers.pickle3'.format(data_folder, r_space)
    #     file = open(filename, 'rb')
    #     passwords = pickle.load(file)

    # generate query data from given sample space
    # prefix = phrase[0:int(start_loc)]
    # suffix = phrase[int(end_loc):]
    # texts = []
    # for password in passwords:
    #     texts.append(prefix+password+suffix)

    # LOAD MEMBER SENTENCE DATA
    member_texts = []
    file = open(member_set_path, 'rb')
    sentence_data = pickle.load(file)
    member_texts = [sentence[0] for sentence in sentence_data]
    member_gt = [sentence[1] for sentence in sentence_data]
    
    # LOAD NON MEMBER SENTENCE DATA
    non_member_texts = []
    file = open(non_member_set_path, 'rb')
    sentence_data = pickle.load(file)
    non_member_texts = [sentence[0] for sentence in sentence_data]
    non_member_gt = [sentence[1] for sentence in sentence_data]


    # Multiprocessing variables
    mgr = mp.Manager()

    roc_score = mgr.list()
    epoch_losses = mgr.list()

    # cpu count calculation for given environment
    cpu_count = mp.cpu_count()
    print("{} CPUs found!".format(cpu_count))
    runs = n_subruns//int(cpu_count)
    remainder = n_subruns % int(cpu_count)

    # multiprocessing pipeline
    for _ in range(runs):
        sub_run_jobs = [mp.Process
                        (target=sub_run_func,
                        args=(TRAIN_DATA, member_texts, member_gt, non_member_texts, non_member_gt, LABEL, epoch, model, drop, beam_width, batch_size, epoch_losses))
                        for i in range(cpu_count)]
        for j in sub_run_jobs:
                j.start()
        for j in sub_run_jobs:
                j.join()

    remainder_run_jobs = [mp.Process
                    (target=sub_run_func,
                    args=(TRAIN_DATA, member_texts, member_gt, non_member_texts, non_member_gt, LABEL, epoch, model, drop, beam_width, batch_size, epoch_losses))
                    for i in range(remainder)]
    for j in remainder_run_jobs:
            j.start()
    for j in remainder_run_jobs:
            j.join()
    

    epoch_losses = list(epoch_losses)
    roc_score = list(roc_score)

    save_results([roc_score, batch_size, epoch_losses], epoch, attack_type, batch_size, train_data_path.split(".")[0])