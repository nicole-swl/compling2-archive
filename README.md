# Computational Linguistics 2: Downstreaming Task

This repository contains the code and results for the **Computational Linguistics 2** research project. 
The goal is to implement and evaluate a span-labelling task using **Named Entity Recognition (NER)** on two tagsets:

**Full task**: 7 BIO labels (`B-LOC`, `I-LOC`, `B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `O`)

**Simplified task**: 3 BIO labels (`B`, `I`, `O`)

There are two model architectures explored for this task: 
- An **encoder-only** model (fine-tuned BERT)
- A **decoder-only** model (in-context learning with GPT)

# Models

1. Encoder-Only Model: BERT (Fine-Tuned)
The encoder-only model used is BERT (bert-base-cased) from Hugging Face, fine-tuned for NER tasks. The model was fine-tuned separately both tagsets.

**Model Details:**
- Pre-trained model: bert-base-cased
- Fine-tuned for NER
- Utilizes a classification head to predict BIO labels for each token

2. Decoder-Only Model: LLaMA-3.1-8B-Instruct (In-Context Learning)
The second model evaluated is LLaMA-3.1-8B-Instruct, a decoder-only model deployed locally in an 8-bit quantized form to manage memory usage. Instead of fine-tuning, this model uses few-shot in-context learning to perform NER tasks.

**Model Details:**
- Model: LLaMA-3.1-8B-Instruct
- Approach: Few-shot in-context learning
- Generation Method: Model generates BIO labels token-by-token based on provided examples and a new input sentence.


# Repository Structure 

The repository contains the following files:

encoder_only.py: Code for training and evaluating the fine-tuned BERT model, with the full tagset. This includes datapreprocessing and evaluation metrics.

encoder_only_simplified.py: Code for training and evaluating the fine-tuned BERT model, with the simplified tagset. This includes datapreprocessing and evaluation metrics.

decoder_only.py: Code for running in-context learning with the LLaMA decoder model, with the full tagset. This includes datapreprocessing and evaluation metrics.

decoder_only_simplified.py: Code for running in-context learning with the LLaMA decoder model, with the simplified tagset. This includes datapreprocessing and evaluation metrics.

README.md: This file.


# Evaluation Metrics

The models are evaluated on the following metrics:

Labelled Span-Matching Score: Measures how accurately the model identifies entity spans.

Unlabelled Span-Matching Score: Measures how accurately the model identifies entity spans without considering entity type.

Precision, Recall, and F1 Scores: Computed per label and averaged across all labels (macro average).
