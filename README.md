# da6401_assignment_3

# Sequence-to-Sequence Transliteration for Indian Languages (DA6401 Assignment 3)

This repository contains the implementation of a sequence-to-sequence (Seq2Seq) model for transliteration between Latin script and Indian scripts (specifically Hindi) using the Google Dakshina dataset. The work was completed as part of DA6401 Assignment 3.

---

## Project Overview

The project implements an Encoder-Decoder architecture for transliteration tasks, focusing on:

- **Hindi (Latin to Devanagari) transliteration**

The core model uses various RNN architectures (SimpleRNN, LSTM, GRU) and integrates attention mechanisms to improve transliteration accuracy.

---

## Dataset

The project uses the [Dakshina dataset](https://github.com/google-research-datasets/dakshina), a large-scale resource for Indian language transliteration and language modeling. The dataset includes parallel corpora for Latin and native scripts across 12 South Asian languages, including Hindi

**Dataset Organization:**
- `hi.translit.sampled.train.tsv` — Hindi training data
- `hi.translit.sampled.dev.tsv` — Hindi validation data
---


---

## Features

- Character-level tokenization for robust handling of Indian scripts
- Multiple RNN architectures: SimpleRNN, LSTM, GRU
- Implementation of attention mechanisms for improved sequence alignment
- Model training and experiment tracking with Weights & Biases (wandb)
- Comprehensive evaluation metrics (accuracy, edit distance, etc.)
- Support for Hindi transliteration tasks

---

## Usage

1. **Clone the repository:**
git clone https://github.com/vishnu000000/DeepLearning_A3.git
cd DeepLearning_A3


2. **Install dependencies:**
- Ensure Python 3.x is installed.
- Install required packages (see the first cell of the notebook).

3. **Download the Dakshina dataset:**
- Obtain the dataset from the [official repository](https://github.com/google-research-datasets/dakshina) and place it in the `dakshina_dataset_v1.0/` directory.

4. **Run the notebook:**
- Execute the cells sequentially. Each question-specific code snippet is clearly marked.

---

## Model Architecture

The transliteration model uses a sequence-to-sequence (Seq2Seq) architecture:

- **Encoder:** Stacked RNN layers (configurable: SimpleRNN, LSTM, or GRU)
- **Decoder:** Stacked RNN layers with attention mechanism
- **Embedding Layer:** Character-level embeddings for input and output scripts
- **Dense Output Layer:** Softmax activation for character prediction

The attention mechanism helps the decoder focus on relevant parts of the input sequence during transliteration, improving accuracy, especially for longer or complex words

---

## Results & Evaluation

- Training and validation loss curves, as well as accuracy metrics, are tracked using Weights & Biases.
- Model checkpoints are saved for reproducibility.
- Prediction samples are provided in the `predictions_attention/` and `predictions_vanilla/` files.
- Evaluation includes exact match accuracy metrics.

---

