!pip install datasets

from collections import defaultdict, Counter
from urllib import request
import json
import pandas as pd
from math import ceil
from tqdm.auto import tqdm
import random
import torch
import numpy as np
import re
SPACE_PATTERN = re.compile(r'[\n\s]+') #removing space and new lines
from transformers import AutoModel, AutoTokenizer
import datasets

def parse_conllu_using_pandas(block):
    records = []
    for line in block.splitlines():
        if not line.startswith('#'):
            records.append(line.strip().split('\t'))
    return pd.DataFrame.from_records(
        records,
        columns=['ID', 'FORM', 'TAG', 'Misc1', 'Misc2'])

def tokens_to_labels(df):
    return (
        df.FORM.tolist(),
        df.TAG.tolist()
    )

PREFIX = "https://raw.githubusercontent.com/UniversalNER/"
DATA_URLS = {
    "en_ewt": {
        "train": "UNER_English-EWT/master/en_ewt-ud-train.iob2",
        "dev": "UNER_English-EWT/master/en_ewt-ud-dev.iob2",
        "test": "UNER_English-EWT/master/en_ewt-ud-test.iob2"
    },
    "en_pud": {
        "test": "UNER_English-PUD/master/en_pud-ud-test.iob2"
    }
}

# en_ewt is the main train-dev-test split
# en_pud is the OOD test set
data_dict = defaultdict(dict)
for corpus, split_dict in DATA_URLS.items():
    for split, url_suffix in split_dict.items():
        url = PREFIX + url_suffix
        with request.urlopen(url) as response:
            txt = response.read().decode('utf-8')
            data_frames = map(parse_conllu_using_pandas,
                              txt.strip().split('\n\n'))
            token_label_alignments = list(map(tokens_to_labels,
                                              data_frames))
            data_dict[corpus][split] = token_label_alignments

# Saving the data so that you don't have to redownload it each time.
with open('ner_data_dict.json', 'w', encoding='utf-8') as out:
    json.dump(data_dict, out, indent=2, ensure_ascii=False)

# Each subset of each corpus is a list of tuples where each tuple
# is a list of tokens with a corresponding list of labels.

# Train on data_dict['en_ewt']['train']; validate on data_dict['en_ewt']['dev']
# and test on data_dict['en_ewt']['test'] and data_dict['en_pud']['test']
data_dict['en_ewt']['train'][0], data_dict['en_pud']['test'][1]

def simplify_bio_labels(full_labels): #simplified B, I, O tags
    simplified = []
    for tag in full_labels:
        if tag == 'O':
            simplified.append('O')
        elif tag.startswith('B-'):
            simplified.append('B')
        elif tag.startswith('I-'):
            simplified.append('I')
        else:
            raise ValueError(f"Unexpected tag format: {tag}")
    return simplified

# Converting data for input and simplifying tags
def convert_and_simplify_data(dataset):
    simplified_data = []
    for tokens, labels in dataset:
        simplified_labels = simplify_bio_labels(labels)
        sentence = [[token, label] for token, label in zip(tokens, simplified_labels)]
        simplified_data.append(sentence)
    return simplified_data

# Converting all datasets

training_data = convert_and_simplify_data(data_dict['en_ewt']['train'])
validating_data = convert_and_simplify_data(data_dict['en_ewt']['dev'])
testing_data = convert_and_simplify_data(data_dict['en_ewt']['test'])
OOD_testing_data = convert_and_simplify_data(data_dict['en_pud']['test'])

# Check first sentence
print(training_data[0])

!pip install -U bitsandbytes

import bitsandbytes as bnb
print(bnb.__version__)

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

!huggingface-cli login --token "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"
access_token = "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"

model_id = "meta-llama/Llama-3.1-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map="auto", token=access_token)

def icl_input_formatter_factory(tokenizer):
    system_prompt = (
        "You are a named-entity recognition (NER) model. "
        "Given a sentence, output the corresponding BIO labels.\n\n"
        "Sentence: the Iguazu Falls is on the border of Argentina and Brazil."
        "Labels: B I O O O O B O B \n\n"
        "Sentence: Pedro Pascal is a handsome man."
        "Labels: B I O O O O O \n\n"
        "Sentence: I study in Stanford University."
        "Labels: O O O B I \n\n"
        "Now process the following sentence."
    )
    def format_input(text):
       full_prompt = f"{system_prompt}\n\nSentence: {text}\nLabels:"
       tokenized = tokenizer(full_prompt, return_tensors="pt").to("cuda")
       return tokenized, full_prompt

    return format_input

NER_input_processor = icl_input_formatter_factory(tokenizer)

NER_prompt_1 = NER_input_processor('San Francisco is a great place to be living.')

print(tokenizer.decode(NER_prompt_1[0]['input_ids'][0]))

inputs = NER_prompt_1[0]
inputs = {k: v.cuda() for k, v in inputs.items()}
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=25, do_sample=True, top_k=3)

print(tokenizer.decode(output[0][len(inputs['input_ids'][0]):], skip_special_tokens=True))

# === Prediction and Evaluation ===
def extract_labels_from_output(text):
    return re.findall(r'\b[BIO]-(?:[A-Z]+)\b|\bO\b', text)

def extract_spans(labels):
    spans = []
    start = None
    current_label = None
    for i, tag in enumerate(labels):
        if tag.startswith('B'):
            if start is not None:
                spans.append((start, i - 1, current_label))
            start = i
            current_label = tag[2:]
        elif tag.startswith('I'):
            if current_label is None or tag[2:] != current_label:
                start = i
                current_label = tag[2:]
        else:  # 'O'
            if start is not None:
                spans.append((start, i - 1, current_label))
                start = None
                current_label = None
    if start is not None:
        spans.append((start, len(labels) - 1, current_label))
    return spans

def evaluate_predictions(preds, golds):
    correct_by_label = Counter()
    predicted_by_label = Counter()
    gold_by_label = Counter()

    labelled_match_total = 0
    unlabelled_match_total = 0
    gold_total = 0

    for pred_seq, gold_seq in zip(preds, golds):
        pred_spans = extract_spans(pred_seq)
        gold_spans = extract_spans(gold_seq)

        pred_span_set = set(pred_spans)
        gold_span_set = set(gold_spans)

        pred_unlabelled = set((s, e) for s, e, _ in pred_spans)
        gold_unlabelled = set((s, e) for s, e, _ in gold_spans)

        labelled_match_total += len(pred_span_set & gold_span_set)
        unlabelled_match_total += len(pred_unlabelled & gold_unlabelled)
        gold_total += len(gold_spans)

        for s, e, label in pred_spans:
            predicted_by_label[label] += 1
        for s, e, label in gold_spans:
            gold_by_label[label] += 1
        for span in pred_span_set & gold_span_set:
            correct_by_label[span[2]] += 1

    # Span match scores
    labelled_score = labelled_match_total / gold_total if gold_total > 0 else 0
    unlabelled_score = unlabelled_match_total / gold_total if gold_total > 0 else 0

    # Per-label P/R/F1
    label_scores = {}
    for label in gold_by_label:
        tp = correct_by_label[label]
        pred = predicted_by_label[label]
        gold = gold_by_label[label]
        precision = tp / pred if pred > 0 else 0
        recall = tp / gold if gold > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        label_scores[label] = {'precision': precision, 'recall': recall, 'f1': f1}

    # Macro-average
    macro_p = sum(score['precision'] for score in label_scores.values()) / len(label_scores) if label_scores else 0
    macro_r = sum(score['recall'] for score in label_scores.values()) / len(label_scores) if label_scores else 0
    macro_f1 = sum(score['f1'] for score in label_scores.values()) / len(label_scores) if label_scores else 0

    return {
        'labelled_span_match': labelled_score,
        'unlabelled_span_match': unlabelled_score,
        'per_label_scores': label_scores,
        'macro_precision': macro_p,
        'macro_recall': macro_r,
        'macro_f1': macro_f1
    }

def run_evaluation(test_data):
    all_preds = []
    all_golds = []

    for example in tqdm(test_data):
        words = [w for w, _ in example]
        gold_labels = [l for _, l in example]

        sentence = " ".join(words)
        inputs, _ = NER_input_processor(sentence)

        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=25, do_sample=True, top_k=3)

        decoded = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        predicted_labels = extract_labels_from_output(decoded)

        if len(predicted_labels) != len(gold_labels):
            print(f"[!] Length mismatch for: {sentence}")
            print(f"    Gold: {gold_labels}")
            print(f"    Pred: {predicted_labels}")
            continue

        all_preds.append(predicted_labels)
        all_golds.append(gold_labels)

    # === Evaluation ===
    results = evaluate_predictions(all_preds, all_golds)

    print(f"\nLabelled Span Match Score:   {results['labelled_span_match']:.2f}")
    print(f"Unlabelled Span Match Score: {results['unlabelled_span_match']:.2f}")

    print("\nPer-label scores:")
    for label, score in results['per_label_scores'].items():
        print(f"{label:10s} | P: {score['precision']:.2f} | R: {score['recall']:.2f} | F1: {score['f1']:.2f}")

    print(f"\nMacro-Averaged:\n  Precision: {results['macro_precision']:.2f} | Recall: {results['macro_recall']:.2f} | F1: {results['macro_f1']:.2f}")

    # Return predictions and evaluation dictionary
    return all_preds, all_golds, results

run_evaluation(testing_data)

run_evaluation(OOD_testing_data)
