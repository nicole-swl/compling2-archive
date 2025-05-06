# Computational Linguistics 2: Downstreaming Task

This repository contains the code and results for the **Computational Linguistics 2** research project. 
The goal is to implement and evaluate a span-labelling task using **Named Entity Recognition (NER)** on two tagsets:

**Full task**: 7 BIO labels (`B-LOC`, `I-LOC`, `B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `O`)

**Simplified task**: 3 BIO labels (`B`, `I`, `O`)

There are two model architectures explored for this task: 
- An **encoder-only** model (fine-tuned BERT)
- A **decoder-only** model (in-context learning with GPT)

- 1. Encoder-Only Model
bert-base-cased from Hugging Face

Fine-tuned separately on:

Full tagset (7 labels)

Simplified tagset (3 labels)

2. Decoder-Only Model
gpt-3.5-turbo via OpenAI API

Uses few-shot in-context learning:

Construct prompts with examples + a new sentence

Model generates BIO labels token-by-token


ner-project/
├── README.md
├── requirements.txt
├── utils/
│ ├── simplify_labels.py
│ ├── metrics.py
│
├── encoder_only_bert/
│ ├── train_full.py
│ ├── train_simplified.py
│ ├── eval_full.py
│ ├── eval_simplified.py
│
├── decoder_only_gpt/
│ ├── prompt_utils.py
│ ├── run_full_prompt.py
│ ├── run_simplified_prompt.py
│
├── outputs/
│ ├── predictions_full.csv
│ ├── predictions_simplified.csv
│ └── eval_results.json
