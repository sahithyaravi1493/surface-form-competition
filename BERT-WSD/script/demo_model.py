import argparse
from json import load
from lib2to3.pgen2.tokenize import tokenize
import re

import torch
from tabulate import tabulate
from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import BertTokenizer

from utils.dataset import GlossSelectionRecord, _create_features_from_records
from utils.model import BertWSD, forward_gloss_selection
from utils.wordnet import get_glosses
import os
# cuda devices
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="3"  # specify which GPU(s) to be used

MAX_SEQ_LENGTH = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_predictions(model, tokenizer, sentence, pos=['n']):
    re_result = re.search(r"\[TGT\](.*)\[TGT\]", sentence)
    if re_result is None:
        print("\nIncorrect input format. Please try again.")
        return

    ambiguous_word = re_result.group(1).strip()
    sense_keys = []
    definitions = []
    syns = []
    for sense_key, t in get_glosses(ambiguous_word, None).items():
        if (t[1].pos()) in pos:
            sense_keys.append(sense_key)
            definitions.append(t[0])
            syns.append(t[1])

    record = GlossSelectionRecord("test", sentence, sense_keys, definitions, [-1])
    features = _create_features_from_records([record], MAX_SEQ_LENGTH, tokenizer,
                                             cls_token=tokenizer.cls_token,
                                             sep_token=tokenizer.sep_token,
                                             cls_token_segment_id=1,
                                             pad_token_segment_id=0,
                                             disable_progress_bar=True)[0]

    with torch.no_grad():
        logits = torch.zeros(len(definitions), dtype=torch.double).to(DEVICE)
        for i, bert_input in list(enumerate(features)):
            logits[i] = model.ranking_linear(
                model.bert(
                    input_ids=torch.tensor(bert_input.input_ids, dtype=torch.long).unsqueeze(0).to(DEVICE),
                    attention_mask=torch.tensor(bert_input.input_mask, dtype=torch.long).unsqueeze(0).to(DEVICE),
                    token_type_ids=torch.tensor(bert_input.segment_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
                )[1]
            )
        scores = softmax(logits, dim=0)

    return sorted(zip(sense_keys, syns, definitions, scores), key=lambda x: x[-1], reverse=True)


def load_model(m_dir):
    # Load fine-tuned model and vocabulary
    print("Loading model...")
    model = BertWSD.from_pretrained(m_dir)
    tokenizer = BertTokenizer.from_pretrained(m_dir)
    # add new special token
    # if '[TGT]' not in tokenizer.additional_special_tokens:
    #     tokenizer.add_special_tokens({'additional_special_tokens': ['[TGT]']})
    #     assert '[TGT]' in tokenizer.additional_special_tokens
    model.resize_token_embeddings(len(tokenizer))

    #print("MATCH CHECK", model.config.vocab_size, len(tokenizer))
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "model_dir",
        default=None,
        type=str,
        help="Directory of pre-trained model."
    )
    args = parser.parse_args()
    model, tokenizer = load_model(args.model_dir)

    while True:
        sentence = input("\nEnter a sentence with an ambiguous word surrounded by [TGT] tokens\n> ")
        predictions = get_predictions(model, tokenizer, sentence)
        if predictions:
            print("\nPredictions:")
            print(tabulate(
                [[f"{i+1}.", key, gloss, f"{score:.5f}"] for i, (key, gloss, score) in enumerate(predictions)],
                headers=["No.", "Sense key", "Definition", "Score"])
            )
            # for i, (sense_key, definition, score) in enumerate(predictions):
            #     # print(f"  {i + 1:>3}. sense key: {sense_key:<15} score: {score:<8.5f} definition: {definition}")


if __name__ == '__main__':
    main()
