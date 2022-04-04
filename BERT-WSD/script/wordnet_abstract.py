import spacy
import re

import pandas as pd
from tqdm import tqdm

import nltk
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
# from pywsd.lesk import simple_lesk
# cuda devices
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used

# Comment these 3 lines out if we are not using BERT WSD
from demo_model import get_predictions, load_model
model_dir = "/ubc/cs/research/nlp/sahiravi/BERT-WSD/bert_base-augmented-batch_size=128-lr=2e-5-max_gloss=6"
model, tokenizer = load_model(model_dir)

# nltk downloads

nlp = spacy.load("en_core_web_md")
dir = '/ubc/cs/research/nlp/sahiravi/datasets/caches'
nltk.download('omw-1.4', download_dir=dir)
nltk.download('wordnet', download_dir=dir)
nltk.download('stopwords', download_dir=dir)
nltk.data.path.append(dir)
stopwords = nltk.corpus.stopwords.words('english')

# object and subject constants
OBJECT_DEPS = {"dobj", "dative", "attr", "oprd"}
SUBJECT_DEPS = {"nsubj", "nsubjpass", "agent", "expl","csubj"}
POS_ALLOWED = {"NOUN"} #{"VERB", "NOUN", "ADJ"}


def load_text(path):
    with open(path) as f:
        input = f.readlines()
    return input

def get_valid_pos(t):
    return (len(t.text) > 2) and (t.lemma_ not in stopwords) and (t.pos_ in POS_ALLOWED)


def get_all_also_sees(word, tag=wn.NOUN):
    for ss in wn.synsets(word, pos=tag):
            for sim in ss.also_sees():
                for lemma in sim.lemma_names():
                    yield (lemma, sim.name())


def get_all_similar_tos(word, tag=wn.NOUN):
    for ss in wn.synsets(word, pos=tag):
            for sim in ss.similar_tos():
                for lemma in sim.lemma_names():
                    yield (lemma, sim.name())

def get_synonyms(word, tag=wn.NOUN):
    for synset in wn.synsets(word, pos=tag):
        for lemma in synset.lemmas():
            yield lemma.name()

def get_hypernyms(sense, tag=wn.NOUN, K=3):
    # Consider only upto K levels up in shortest path
    paths = sense.hypernym_paths()
    shortest_path = None
    shortest_len = 100000
    for path in paths:
        if len(path) < shortest_len:
            shortest_len = len(path)
            shortest_path = path
    
    shortest_path = shortest_path[-K:]
    # print("shortest path", shortest_path)

    for synset in shortest_path:
        for lemma in synset.lemmas()[:1]:
            yield lemma.name()

def disambiguate(sentence, word, method = "bert"):
    sense = None
    if method=="bert":
        sense = get_bert_predictions(sentence, word)
    elif method == "lesk":
        sense = lesk(sentence, word, "n")
    elif method == "frequency":
        if wn.synsets(word):
            sense = wn.synsets(word)[0]
            
    return sense


def get_bert_predictions(sentence, word):
    out = None
    word_tgt = f"[TGT]{word}[TGT]"
    p = get_predictions(model, tokenizer, sentence.replace(word,word_tgt))
    if p:
        out = p[0][1]
    else:
        if wn.synsets(word):
            out = wn.synsets(word)[0]
    return out

def extract_pos_based(doc):
    out = []
    for token in doc:
        if get_valid_pos(token):
            out.append(token.text)
    return out


# wup_similarity
# extract the subject, object and verb from the input
def extract_svo(doc):
    sub = []
    at = []
    ve = []
    all = set()
    for token in doc:
        # is this a verb?
        if token.pos_ == "VERB":
            ve.append(token.text)
        # is this the object?
        if token.dep_ in OBJECT_DEPS or token.head.dep_ in OBJECT_DEPS:
            at.append(token.text)
            all.add(token.text)
        # is this the subject?
        if token.dep_ in SUBJECT_DEPS or token.head.dep_ in SUBJECT_DEPS:
            sub.append(token.text)
            all.add(token.text)
    #return " ".join(sub).strip().lower(), " ".join(ve).strip().lower(), " ".join(at).strip().lower()
    return sub, ve, at, all



def construct_abstractions(sentence, extract_method="pos", abstract_method="hypernyms"):
    doc = nlp(sentence)
    if extract_method == "svo":
        subject, verb, attribute, all_words = extract_svo(doc)
    elif extract_method == "pos":
        all_words = extract_pos_based(doc)


    abstraction_map = {}
    abs_sentences = []
    # print(all_words)
    for word in all_words:
        if abstract_method == "synsets":
            unique = set(synonym for synonym in get_synonyms(word) if synonym != word)
            abstraction_map[word] = list(unique)[:5]
        elif abstract_method == "hypernyms":
            sense = disambiguate(sentence, word)
            if sense is not None:
                unique = set(h for h in get_hypernyms(sense) if h != word)
                abstraction_map[word] = unique
        elif abstract_method == "similar_tos":
            unique = set(synonym for synonym in get_all_also_sees(word) if synonym != word)
            abstraction_map[word] = list(unique)[:5]
        
            
    for word in abstraction_map:
        for syn in abstraction_map[word]:
            out = sentence
            abs_sentences.append(out.replace(word, syn))

    return abs_sentences
        

def all_sentence_abstractions(text):
    abstracted_sentences = []
    indices = []
    for i in tqdm(range(len(text))):
        sentences = construct_abstractions(text[i], extract_method="pos", abstract_method="hypernyms")
        sentences.append(text[i])
        abstracted_sentences.extend(sentences)
        indices.extend([i]*len(sentences))
    df = pd.DataFrame()
    df["gen_id"] = indices
    df["abstractions"] = abstracted_sentences
    return df

# gather the user input and gather the info
if __name__ == "__main__":
    print(" Generate Abstractions for a sample input based on synonyms from wordnet")
    sent = "A cat and its furry companions on a couch."
    abstractions = construct_abstractions(sent, extract_method="pos", abstract_method="hypernyms")
    print(abstractions)

