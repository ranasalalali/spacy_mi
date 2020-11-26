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
    doc = model(text)
    print("Entities in '%s'" % text)
    entities = {}
    for ent in doc.ents:
        entities[ent.text] = ent.label_
    return entities

def get_scores_per_entity(model=None, texts=[],):
    # Number of alternate analyses to consider. More is slower, and not necessarily better -- you need to experiment on your problem.
    
    nlp = model
    
    beam_width = 16
    # This clips solutions at each step. We multiply the score of the top-ranked action by this value, and use the result as a threshold. This prevents the parser from exploring options that look very unlikely, saving a bit of efficiency. Accuracy may also improve, because we've trained on greedy objective.
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

def update_model(drop=0.4, epoch=30, model=None, new_model_name="new_model", output_dir=None):
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
    move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    # only train NER
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        for itn in range(int(epoch)):
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

    # save model to output directory
    if output_dir is not None:
        mkdir_p(output_dir)
        nlp.meta["name"] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        # Check the classes have loaded back consistently
        assert nlp2.get_pipe("ner").move_names == move_names
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--phrase', help='Sensitive phrase for updating model')
    parser.add_argument('--label', help='Label name for model to update with')
    parser.add_argument('--start_loc', help='Start of sensitive information')
    parser.add_argument('--end_loc', help='End of sensitive information')
    parser.add_argument('--model', help='Pretrained model for update')
    parser.add_argument('--run', help='Run index')
    parser.add_argument('--epoch', help='Number of epochs for model update')
    parser.add_argument('--drop', help='Dropout')
    parser.add_argument('--insertions', help='Number of insertions of phrase')

    args = parser.parse_args()

    spacy.prefer_gpu()

    nlp = spacy.load(args.model)

    n_insertions = int(args.insertions)

    texts = [args.phrase]

    print(get_entities_for_text(model=nlp, text=texts[0]))

    print(get_scores_per_entity(model=nlp, texts=texts))

    # new entity label
    LABEL = args.label

    TRAIN_DATA = []

    for i in range(0, n_insertions):
        TRAIN_DATA.append((args.phrase, {'entities': [(0, 4, 'PERSON'),(int(args.start_loc), int(args.end_loc), LABEL)]}))

    update_model(epoch=args.epoch, drop=args.drop, model=args.model, new_model_name='{}_updated_{}'.format(args.model, args.run), output_dir='model/{}_updated_{}'.format(args.model, args.run))

    new_model_directory = 'model/{}_updated_{}'.format(args.model, args.run)
    print("Loading from", new_model_directory)
    nlp2 = spacy.load(new_model_directory)

    import itertools

    zip_codes = []

    for combination in itertools.product(range(10), repeat=int(args.end_loc)-int(args.start_loc)):
        zip_codes.append(''.join(map(str, combination)))

    prefix = args.phrase[0:int(args.start_loc)]
    texts = []
    for zip_code in zip_codes:
        texts.append(prefix+zip_code)

    scores = get_scores_per_entity(model=nlp2, texts=texts)

    secret_len = int(args.end_loc) - int(args.start_loc)

    save_results([scores, args.phrase], secret_len, n_insertions)
