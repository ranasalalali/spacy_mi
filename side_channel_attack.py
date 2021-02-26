from __future__ import unicode_literals, print_function
import spacy
print(spacy.__version__)
from collections import defaultdict
from thinc.api import set_gpu_allocator,require_gpu


from spacy.training import Example
import random
import warnings
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from collections import defaultdict
import operator
import string
import numpy as np
from itertools import permutations, islice
import os
import errno
import multiprocessing as mp
from spacy.vectors import Vectors
import time
import sys

file1 = open("result.txt","a")

def updatingModel ():
    LABEL = "SECRET"
    secret = 'rgjfgklf678'
    text = "Rana's secret is {}.".format(secret)
    TRAIN_DATA = []
    TRAIN_DATA.append((text, {'entities': [(0, 4, 'PERSON'), (17, 17 + len(secret), LABEL)]}))

    nlp = spacy.load('en_core_web_lg')
#     print("model before updating")
    print("Size of vocab_string in model before updating: ", len(list(nlp.vocab.strings)))
    ner = nlp.get_pipe("ner")
    ner.add_label(LABEL)
    optimizer = nlp.resume_training()

    ner = nlp.get_pipe("ner")
    # Disable pipeline components you dont need to change
    pipe_exceptions = ["ner", "tok2vec"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

#     optimizer = nlp.resume_training()

    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    move_names = list(ner.move_names)
#     print("move_names:", move_names)
#     print("num of moves: ", len(move_names))

    examples = []
    for text, annots in TRAIN_DATA:
        examples.append(Example.from_dict(nlp.make_doc(text), annots))
    get_examples = lambda: examples
    #nlp.initialize(lambda: get_examples)
    for _ in range(20):
        random.shuffle(examples)
    with nlp.disable_pipes(*unaffected_pipes): 
        for _ in range(20):
            for batch in minibatch(examples, size=8):
                nlp.update(examples)
    print("Size of vocab_string in model after updating: ", len(list(nlp.vocab.strings)))

    return nlp

def get_scores_per_entity(model=None, texts=[],):
    # time_now = time.perf_counter()
    # Number of alternate analyses to consider. More is slower, and not necessarily better -- you need to experiment on your problem.
    nlp=model
    # print("Size of vocab_string in model before querying: ", len(list(nlp.vocab.strings)))
    beam_width = 16
    # This clips solutions at each step. We multiply the score of the top-ranked action by this value, and use the result as a threshold. This prevents the parser from exploring options that look very unlikely, saving a bit of efficiency. Accuracy may also improve, because we've trained on greedy objective.
    beam_density = 0.0001 

    ner = nlp.get_pipe('ner')
    docs = nlp.make_doc(texts[0])
    docs = ner(docs)
    # time_now_end = time.perf_counter()
    # print("runtime = ", (time_now_end - time_now))
    # print("Size of vocab_string in model after querying: ", len(list(nlp.vocab.strings)))
    
#     beams = ner.beam_parse([docs], beam_width=beam_width, beam_density=beam_density)
    
    
#     return ner.scored_ents(beams) 


time_update_trained_pw = 0
time_update_new_pw = 0
iterations = 100
count_success = 0

for i in range(iterations +1):
    nlp = updatingModel()
    ### for trained password
    print("Query updated model with trained password")
    
    secret = "rgjfgklf678" # "rgjfgklf678"
    text = "Rana's secret is {}.".format(secret)
    print("text: ", text)
    texts = [text]

    time0 = time.perf_counter()
    print(time0)
    get_scores_per_entity(nlp, texts)
    
#     print(get_scores_per_entity(nlp, texts))
    time_now = time.perf_counter()
    print(time_now)
    # print("runtime = ", time_now - time0 )
#     vocab_string_after_query = list(nlp.vocab.strings)

#     diff = list(set(vocab_string_org).symmetric_difference(vocab_string_after_query))
    time_update_trained_pw += (time_now - time0)
    in_vocab_runtime = time_now - time0
    
    print("Size of vocab_string in updated model: ", len(list(nlp.vocab.strings)))
#     print("difference in vocab string with common words: ", diff)
    

    print("====================")

    ### for new words

    print("Query updated model with unseen password")
#     print("Size of vocab_string: ", len(list(nlp.vocab.strings)))      
    secret = "rbdhrkrp908" #'"shfklfgl12l90"
    text = "Rana's secret is {}.".format(secret)
    print("text: ", text)
    texts = [text]

    time01 = time.perf_counter()
    print(time01)
    get_scores_per_entity(nlp, texts)
    

    time_now1 = time.perf_counter()
    print(time_now1)
    # print("runtime = ", time_now1 - time01 )

    time_update_new_pw += (time_now1 - time01)
    out_vocab_runtime = time_now1 - time01
    print("Size of vocab_string in updated model: ", len(list(nlp.vocab.strings)))

    if out_vocab_runtime > in_vocab_runtime:
        count +=1
    print("-------------------")

print("Number of successs attempts:{}".format(count))    
print("======Average======") 
if iterations >0:
    print("runtime with trained pw: ", time_update_trained_pw/iterations)
    print("runtime with new pw: ", time_update_new_pw/iterations)
    print("runtime diff: ", (time_update_new_pw/iterations - time_update_trained_pw/iterations ))
else:
    print("runtime with trained pw: ", time_update_trained_pw)
    print("runtime with new pw: ", time_update_new_pw)


sys.exit()

time_update_trained_pw = 0
time_update_new_pw_prefix = 0
time_update_new_pw_suffix = 0
time_update_new_pw_shape = 0
time_update_new_pw_pre_suf = 0
time_update_new_pw_pre_sh = 0
time_update_new_pw_suf_sh = 0
time_update_new_pw_pre_suf_sh = 0
time_update_new_length = 0
time_update_new_abitr = 0

iterations = 100


# sys.exit()    
# def get_runtime(inter):
#     iterations = inter
for i in range(iterations):
    print("i= ", i)
    nlp = updatingModel()
    ### for trained password   
    secret = "rgjfgklf678" # "rgjfgklf678"
    text = "Rana's secret is {}.".format(secret)
#     print("text: ", text)
    texts = [text]

    time0 = time.perf_counter()
#     print(time0)
    
    get_scores_per_entity(nlp, texts)
    
    time_now = time.perf_counter()

#     diff = list(set(vocab_string_org).symmetric_difference(vocab_string_after_query))
    time_update_trained_pw += (time_now - time0)

    ## same prefix
    nlp = updatingModel()
    secrets = ['rfhwnfgf0p9']
    
    time01 = time.perf_counter()
    text = "Rana's secret is {}.".format(secrets[0])
    get_scores_per_entity(nlp, text)

    time_now1 = time.perf_counter()

    time_update_new_pw_prefix += (time_now1 - time01)
    
    
    ## same suffix
    nlp = updatingModel()
    secrets = ['jfhgg678']
    
    time01 = time.perf_counter()
    text = "Rana's secret is {}.".format(secrets[0])
    get_scores_per_entity(nlp, text)

    time_now1 = time.perf_counter()

    time_update_new_pw_suffix += (time_now1 - time01)
    
    
    ## same shape
    nlp = updatingModel()
    secrets = ['fjfhggbv054']
    
    time01 = time.perf_counter()
    text = "Rana's secret is {}.".format(secrets[0])
    get_scores_per_entity(nlp, text)

    time_now1 = time.perf_counter()

    time_update_new_pw_shape += (time_now1 - time01)

    
    ## same pre suf
    nlp = updatingModel()
    secrets = ['rjfhggbv678']
    
    time01 = time.perf_counter()
    text = "Rana's secret is {}.".format(secrets[0])
    get_scores_per_entity(nlp, text)

    time_now1 = time.perf_counter()

    time_update_new_pw_pre_suf += (time_now1 - time01)
    
    
    ## same pre shape
    nlp = updatingModel()
    secrets = ['rbdhrkrp908']
    
    time01 = time.perf_counter()
    text = "Rana's secret is {}.".format(secrets[0])
    get_scores_per_entity(nlp, text)

    time_now1 = time.perf_counter()

    time_update_new_pw_pre_sh += (time_now1 - time01)
    
    
    ## same suf shape
    nlp = updatingModel()
    secrets = ['adfhggbv678']
    
    time01 = time.perf_counter()
    text = "Rana's secret is {}.".format(secrets[0])
    get_scores_per_entity(nlp, text)

    time_now1 = time.perf_counter()

    time_update_new_pw_suf_sh += (time_now1 - time01)
    
    
    ## same pre suf shape
    nlp = updatingModel()
    secrets = ['rdfhggbv678']
    
    time01 = time.perf_counter()
    text = "Rana's secret is {}.".format(secrets[0])
    get_scores_per_entity(nlp, text)

    time_now1 = time.perf_counter()

    time_update_new_pw_pre_suf_sh += (time_now1 - time01)
    
    
    ## same length
    nlp = updatingModel()
    secrets = ['rdfHGgvi89']
    
    time01 = time.perf_counter()
    text = "Rana's secret is {}.".format(secrets[0])
    get_scores_per_entity(nlp, text)

    time_now1 = time.perf_counter()

    time_update_new_length  += (time_now1 - time01)
    
    
    ## same abitrary
    nlp = updatingModel()
    secrets = ['thafkfpee_19_&g,']
    
    time01 = time.perf_counter()
    text = "Rana's secret is {}.".format(secrets[0])
    get_scores_per_entity(nlp, text)

    time_now1 = time.perf_counter()

    time_update_new_abitr  += (time_now1 - time01)
    


    
file1.write("======Average======\n") 
if iterations >0:
    file1.write("runtime with trained pw: %s\n"%(time_update_trained_pw/iterations))
    file1.write("runtime with new pw same pref: %s\n"%(time_update_new_pw_prefix/iterations))
    file1.write("runtime with new pw same suff: {}\n".format(time_update_new_pw_suffix/iterations))
    file1.write("runtime with new pw same shape: {}\n".format(time_update_new_pw_shape/iterations))
    file1.write("runtime with new pw same pref suf: {}\n".format(time_update_new_pw_pre_suf/iterations))
    file1.write("runtime with new pw same pref shape: {}\n".format(time_update_new_pw_pre_sh/iterations))
    file1.write("runtime with new pw same suff shape: {}\n".format(time_update_new_pw_suf_sh/iterations))
    file1.write("runtime with new pw same pref suff shape: {}\n".format(time_update_new_pw_pre_suf_sh/iterations))
    file1.write("runtime with new pw same length: {}\n".format(time_update_new_length/iterations))
    file1.write("runtime with new pw arbitrary: {}\n".format(time_update_new_abitr/iterations))
    
    file1.write("======Time different ======\n") 
    file1.write("new pw same pref: {}\n".format(time_update_new_pw_prefix/iterations - time_update_trained_pw/iterations ))
    file1.write("new pw same suff: {}\n".format(time_update_new_pw_suffix/iterations - time_update_trained_pw/iterations ))
    file1.write("new pw same shape: {}\n".format(time_update_new_pw_shape/iterations - time_update_trained_pw/iterations ))
    file1.write("new pw same pref suff: {}\n".format(time_update_new_pw_pre_suf/iterations - time_update_trained_pw/iterations ))
    file1.write("new pw same pref shape: {}\n".format(time_update_new_pw_pre_sh/iterations - time_update_trained_pw/iterations ))
    file1.write("new pw same suff shape: {}\n".format(time_update_new_pw_suf_sh/iterations - time_update_trained_pw/iterations ))
    file1.write("new pw same pref suff shape: {}\n".format(time_update_new_pw_pre_suf_sh/iterations - time_update_trained_pw/iterations ))
    file1.write("new pw same length: {}\n".format(time_update_new_length/iterations - time_update_trained_pw/iterations ))
    file1.write("new pw arbitrary: {}\n".format(time_update_new_abitr/iterations - time_update_trained_pw/iterations ))

        
        
        
    
sys.exit()
