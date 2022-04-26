import spacy
import re
import json
import pandas as pd
from tqdm import tqdm
import random
import nltk
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
import sys, os
import requests
from wordhoard import Synonyms
from bs4 import BeautifulSoup

sys.path.append('/ubc/cs/research/nlp/sahiravi/surface-form-competition/BERT-WSD/script')
random.seed(50)

#Comment these 3 lines out if we are not using BERT WSD
from demo_model import load_model, get_predictions
model_dir = "/ubc/cs/research/nlp/sahiravi/BERT-WSD/bert_base-augmented-batch_size=128-lr=2e-5-max_gloss=6"
model, tokenizer = load_model(model_dir)

#nltk downloads

nlp = spacy.load("en_core_web_lg")
dir = '/ubc/cs/research/nlp/sahiravi/datasets/caches'
nltk.download('omw-1.4', download_dir=dir)
nltk.download('wordnet', download_dir=dir)
nltk.download('stopwords', download_dir=dir)
nltk.data.path.append(dir)
stopwords = nltk.corpus.stopwords.words('english')

# object and subject constants
OBJECT_DEPS = {"dobj", "dative", "attr", "oprd"}
SUBJECT_DEPS = {"nsubj", "nsubjpass", "agent", "expl","csubj"}

entity_maps = {
    'PERSON': 'a person', 
    # 'ORG' : 'organization', 
    # 'DATE': 'date', 
    # 'GPE': 'location', 
    # 'MONEY':'money', 
    # 'PRODUCT':'object', 
    # 'TIME':'time', 
    # 'WORK_OF_ART':'title', 
    # 'QUANTITY':'quantity', 
    # 'NORP':'group', 
    # 'LOC':'location',
    # 'EVENT':'event', 
    # 'LAW':'law', 
    # 'LANGUAGE':'language'
    }

def synonyms_web(word):
    URL = f'https://www.thesaurus.com/browse/{word}'

    page = requests.get(URL)
    if page.status_code == 200:
        soup = BeautifulSoup(page.content, 'html.parser')

    the_script = soup.find('script', text=re.compile("window.INITIAL_STATE"))
    the_script = the_script.text
    the_script = the_script.replace("window.INITIAL_STATE = ", "")
    the_script = the_script.replace(':undefined', ':"undefined"')
    the_script = the_script.replace(';', '')
    data = json.loads(the_script, encoding='utf-8')

    # with open('test1.json', 'w', encoding='utf-8') as f:
    #     json.dump(data, f, ensure_ascii=False, indent=4)

    all_senses = []
 
    for each_tab in data["searchData"]["tunaApiData"]["posTabs"]:
        senses = {}
        if each_tab["pos"] == "noun":
            senses['definition'] = each_tab['definition']
            lst = []
            for syn in each_tab["synonyms"]:
                sim = float(syn["similarity"])
                if sim > 50:
                    lst.append(syn['term'])
            senses['synonyms'] = lst
        if senses:
            all_senses.append(senses)
    #print(all_senses)
    return all_senses

# def synonyms_web(term):
#     # synonym = Synonyms(search_string=term)
#     # synonym_results = synonym.find_synonyms()
#     # return synonym_results
#     response = requests.get('https://www.thesaurus.com/browse/{}'.format(term))
#     soup = BeautifulSoup(response.text, 'lxml')
#     # method 1 
#     syns = soup.select('.MainContentContainer > section > .synonyms-container a')
#     return syns
#     # method 2 return [span.text for span in soup.findAll('a', {'class': 'css-17ofzyv e1ccqdb60'})] # 'css-1gyuw4i eh475bn0' for less relevant synonyms

def replace_named_entities(sentence, doc):
    entity_relabeled_sentence = sentence
    relabeled = False
    for ent in doc.ents:
        if ent.label_ in entity_maps:
            entity_relabeled_sentence  = entity_relabeled_sentence.replace(ent.text, entity_maps[ent.label_])
            relabeled = True
    return [entity_relabeled_sentence] if relabeled else []

def load_text(path):
    with open(path) as f:
        input = f.readlines()
    return input

def get_valid_pos(t, POS_ALLOWED = {"NOUN"}):
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

def get_synonyms(synset, word, tag=wn.NOUN):
    print(synset.definition())

    synonyms = []
    # Wordnet synonyms
    for lemma in synset.lemmas():
        synonyms.append(lemma.name())

    # Web synonyms
    web_synonym_senses = synonyms_web(word)
    if web_synonym_senses:
        for sense_dict in web_synonym_senses:
            candidate = sense_dict["synonyms"][0]
            candidate_sense = wn.synsets(candidate)[0]
            if synset.wup_similarity(candidate_sense) > 0.5:
                synonyms.extend(sense_dict["synonyms"])
                break
    return synonyms

def get_hypernyms(sense, tag=wn.NOUN, K=3):
    # Consider only upto K levels up in shortest path
    paths = sense.hypernym_paths()
    shortest_path = None
    shortest_len = 100000
    for path in paths:
        if len(path) < shortest_len:
            shortest_len = len(path)
            shortest_path = path
    # print("shortest path", shortest_path)
    if len(shortest_path) > 5:
        shortest_path = shortest_path[-K:]
    else:
        # avoid "entity" level
        shortest_path = shortest_path[-1:]

    for synset in shortest_path:
        for lemma in synset.lemmas()[:1]:
            yield lemma.name()

def disambiguate(sentence, word, method="bert"):
    # uses most frequent sense or lesk
    sense = None
    if wn.synsets(word):
        sense = wn.synsets(word)[0]
    if method=="bert":
        bert_pred = get_bert_predictions(sentence, word)
        if bert_pred is not None:
            sense = bert_pred
    elif method == "lesk":
        sense = lesk(sentence, word, "n")
            
    return sense


def get_bert_predictions(sentence, word):
    out = None
    word_tgt = f"[TGT]{word}[TGT]"
    p = get_predictions(model, tokenizer, sentence.replace(word,word_tgt))
    # print(p)
    if p:
        out = p[0][1]
    else:
        if wn.synsets(word):
            out = wn.synsets(word)[0]
    return out

def extract_pos_based(doc, POS_ALLOWED):
    out = []
    for token in doc:
        if get_valid_pos(token, POS_ALLOWED):
            out.append(token.text)
    return out

def get_chunks(doc):
    noun_phrases = set()
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) > 1:
            noun_phrases.add(chunk.text) # chunk.root.text, chunk.root.dep_, chunk.root.head.text
    return noun_phrases

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



def construct_abstractions(sentence, entity=True, phrases=True):
    doc = nlp(sentence)
    noun_phrases = get_chunks(doc)

    hypernym_map = {}
    synonym_map = {}
    hyp_sentences = []
    syn_sentences = []

    all_words = extract_pos_based(doc, POS_ALLOWED={"NOUN"})
    if phrases:
        all_words.extend(noun_phrases)
    
    for word in set(all_words):
        sense = disambiguate(sentence, word)
        
        if sense is not None:
            unique = set(h for h in get_hypernyms(sense) if h != word)
            if unique:
                hypernym_map[word] = list(unique)
            unique_syn = list(set(synonym for synonym in get_synonyms(sense, word) if synonym != word))
            if unique_syn:
                synonym_map[word] = unique_syn

    # Abstract named entities to their labels
    if entity:
        entity_abstractions = replace_named_entities(sentence, doc) 
        if entity_abstractions:
            hyp_sentences.extend(entity_abstractions)

    # make sure each word is abstracted by hypernym atleast once   
    for word in hypernym_map:
        out = sentence.replace(word, hypernym_map[word][0]).replace("_", " ")
        hyp_sentences.append(out)
    for word in synonym_map:
        out = sentence.replace(word, synonym_map[word][0]).replace("_", " ")
        syn_sentences.append(out)
    for word in hypernym_map:
        for syn in hypernym_map[word][1:]:
            out = sentence.replace(word, syn).replace("_", " ")
            hyp_sentences.append(out)
    for word in synonym_map:
        for syn in synonym_map[word][1:]:
            out = sentence.replace(word, syn).replace("_", " ")
            syn_sentences.append(out) 
    
    # random.shuffle(hyp_sentences)
    # random.shuffle(syn_sentences)  
    hyp_sentences.extend([sentence]*5)
    syn_sentences.extend([sentence]*5)
    return hyp_sentences, syn_sentences
        

# def all_sentence_abstractions(text):
#     abstracted_sentences = []
#     indices = []
#     for i in tqdm(range(len(text))):
#         sentences = construct_abstractions(text[i], extract_method="pos", abstract_method="hypernyms")
#         sentences.append(text[i])
#         abstracted_sentences.extend(sentences)
#         indices.extend([i]*len(sentences))
#     df = pd.DataFrame()
#     df["gen_id"] = indices
#     df["abstractions"] = abstracted_sentences
#     return df


# gather the user input and gather the info
if __name__ == "__main__":
    # Input example
    sentence = "He adds some type of fertilizer to the potted plant." 
    # get sentences by substituting words with hypernyms and synonyms
    hypernym_sentences, synonym_sentences = construct_abstractions(sentence)
    print(synonym_sentences)



