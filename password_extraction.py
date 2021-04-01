import spacy
from spacy import Language
from collections import defaultdict
import operator
import math
from string import punctuation, ascii_letters, ascii_uppercase, ascii_lowercase
import random
from itertools import *
import numpy as np
import os

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

def generate_prefix_passwords(N=100, length=8, upper=False, lower=True, numeric=False, special=False):
    passwords = []
    if upper:
        for _ in range(N):
            for char in ascii_uppercase:
                password = char + ''.join(random.choices(ascii_letters, k=length-1))
                passwords.append(password)
    if lower:
        for _ in range(N):
            for char in ascii_lowercase:
                password = char + ''.join(random.choices(ascii_letters, k=length-1))
                passwords.append(password)

    if numeric:
        for _ in range(N):
            for char in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                password = char + ''.join(random.choices(ascii_letters, k=length-1))
                passwords.append(password)
    
    if special:
        for _ in range(N):
            for char in punctuation:
                password = char + ''.join(random.choices(ascii_letters, k=length-1))
                passwords.append(password)
    
    return passwords
    
def extract_prefix(combinations=None, score=None):
    ranks_per_code = {}
    ranks_per_prefix = {}
    agg_scores = {}
    agg_score_prefix = {}

    sorted_score = dict(sorted(score.items(), key=operator.itemgetter(1), reverse=True))
    rank = 1
    for code in sorted_score.items():
        if code[0][0] not in agg_score_prefix.keys():
            agg_score_prefix[code[0][0]] = []
            ranks_per_prefix[code[0][0]] = []
            ranks_per_prefix[code[0][0]].append(rank)
            agg_score_prefix[code[0][0]].append(code[1])
            rank+=1
        else:    
            ranks_per_prefix[code[0][0]].append(rank)
            agg_score_prefix[code[0][0]].append(code[1])
            rank+=1
    
    for prefix in ranks_per_prefix:
        ranks_per_prefix[prefix] = np.mean(np.array(ranks_per_prefix[prefix]))

    for prefix in agg_score_prefix:
        agg_score_prefix[prefix] = np.mean(np.array(agg_score_prefix[prefix]))


    sorted_ranks_per_prefix = dict(sorted(ranks_per_prefix.items(), key=operator.itemgetter(1), reverse=False)[:10])
    sorted_agg_score_prefix = dict(sorted(agg_score_prefix.items(), key=operator.itemgetter(1), reverse=True)[:10])
    
    top10_ranks_per_prefix = sorted_ranks_per_prefix.items()
    top10_agg_score_prefix = sorted_agg_score_prefix.items()

    #print(top10_ranks_per_prefix)

    return list(dict(top10_ranks_per_prefix).keys())[0]

def generate_suffix_passwords(N=100, prefix='000', length=8, upper=False, lower=False, numeric=True, special=False):

    passwords = []
    if numeric and upper and lower:
        set1 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        set1.extend(set(ascii_lowercase))
        set1.extend(set(ascii_uppercase))
        printAllKLength(set1, 3)
        res = [i for i in combs if i]
        for _ in range(N):
            for comb in res:
                password = prefix + ''.join(random.choices(ascii_letters, k=length-4)) + comb
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

def extract_suffix(combinations=None, score=None):
    ranks_per_code = {}
    ranks_per_suffix = {}
    agg_scores = {}
    agg_score_suffix = {}

    sorted_score = dict(sorted(score.items(), key=operator.itemgetter(1), reverse=True))
    rank = 1
    for code in sorted_score.items():
        if str(code[0][-3:]) not in agg_score_suffix.keys():
            agg_score_suffix[code[0][-3:]] = []
            ranks_per_suffix[code[0][-3:]] = []
            ranks_per_suffix[code[0][-3:]].append(rank)
            agg_score_suffix[code[0][-3:]].append(code[1])
            rank+=1
        else:    
            ranks_per_suffix[code[0][-3:]].append(rank)
            agg_score_suffix[code[0][-3:]].append(code[1])
            rank+=1
    
    for suffix in ranks_per_suffix:
        ranks_per_suffix[suffix] = np.mean(np.array(ranks_per_suffix[suffix]))

    for suffix in agg_score_suffix:
        agg_score_suffix[suffix] = np.mean(np.array(agg_score_suffix[suffix]))


    sorted_ranks_per_suffix = dict(sorted(ranks_per_suffix.items(), key=operator.itemgetter(1), reverse=False)[:10])
    sorted_agg_score_suffix = dict(sorted(agg_score_suffix.items(), key=operator.itemgetter(1), reverse=True)[:10])
    
    top10_ranks_per_suffix = sorted_ranks_per_suffix.items()
    top10_agg_score_suffix = sorted_agg_score_suffix.items()

    #print(top10_ranks_per_suffix, top10_agg_score_suffix)

    return list(dict(top10_ranks_per_suffix).keys())[0]


if __name__ == "__main__":
    print(spacy.__version__)


    models = os.listdir('models')

    prefix_extraction_accuracy = 0
    suffix_extraction_accuracy = 0
    prefix_suffix_extraction_accuracy = 0
    
    for model in models:
        secret = model.split('_')[-1]

        updated_nlp = spacy.load("models/{}".format(model))
        print(secret)
        length = len(secret)
        start_loc = 17
        end_loc = start_loc+length

        phrase = "Rana's secret is {}".format(secret)
        texts = [phrase]

        nlp = spacy.load('en_core_web_lg')
        
        doc = nlp(phrase)
        tokens = [str(token) for token in doc]
        secret_token_index = tokens.index(secret)
        doc = nlp(phrase)
        LABEL = "SECRET"

        beam_width = 3

        secret_index = doc.text.split().index(secret)

        ### EXTRACTING PREFIX

        print("Generating passwords with Prefix")

        passwords = generate_prefix_passwords(N=1000, length=length, upper=True, lower=True, numeric=True, special=False)

        prefix = phrase[0:int(start_loc)]
        suffix = phrase[int(end_loc):]
        texts = []
        for password in passwords:
            texts.append(prefix+password+suffix)

        print("Extracting Prefix")

        score, exposure, exposure_rank_secret, score_secret, exposure_secret = get_scores_per_entity(model=updated_nlp, texts=texts, beam_width=3, r_space=len(texts), secret_token_index=secret_token_index, secret_index=secret_index, secret=secret, LABEL=LABEL)

        potential_prefix = extract_prefix(combinations=texts, score=score)

        print(potential_prefix)

        passwords.clear()
        texts.clear()

        ### EXTRACTING SUFFIX WITH POTENTIAL PREFIX

        print("Generating passwords with Suffix after potential prefix")

        passwords = generate_suffix_passwords(N=100, prefix=potential_prefix, length=length, upper=True, lower=True, numeric=True, special=False)

        texts = []
        for password in passwords:
            texts.append(prefix+password+suffix)

        print("Extracting Suffix")

        score, exposure, exposure_rank_secret, score_secret, exposure_secret = get_scores_per_entity(model=updated_nlp, texts=texts, beam_width=3, r_space=len(texts), secret_token_index=secret_token_index, secret_index=secret_index, secret=secret, LABEL=LABEL)

        potential_suffix = extract_suffix(combinations=texts, score=score)

        print(potential_prefix, potential_suffix)
        if(potential_prefix == secret[0]):
            prefix_extraction_accuracy+=1

        if(potential_suffix == secret[-3:]):
            suffix_extraction_accuracy+=1
        
        if(potential_prefix == secret[0] and potential_suffix == secret[-3:]):
            prefix_suffix_extraction_accuracy+=1


    prefix_extraction_accuracy = prefix_extraction_accuracy/len(models)*100
    suffix_extraction_accuracy = suffix_extraction_accuracy/len(models)*100
    prefix_suffix_extraction_accuracy = prefix_suffix_extraction_accuracy/len(models)*100

    print("prefix_extraction_accuracy = {}".format(prefix_extraction_accuracy))
    print("suffix_extraction_accuracy = {}".format(suffix_extraction_accuracy))
    print("prefix_suffix_extraction_accuracy = {}".format(prefix_suffix_extraction_accuracy))

        
            



