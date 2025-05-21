# da6401_assignment_3

# Sequence-to-Sequence Transliteration for Indian Languages (DA6401 Assignment 3)

This repository contains an end-to-end PyTorch implementation of a sequence-to-sequence (Seq2Seq) model with attention for transliterating between Latin (romanized) and Devanagari scripts (Hindi) using the Google Dakshina dataset. Completed as part of DA6401 Assignment 3.

- **Source:** [Google Dakshina dataset](https://github.com/google-research-datasets/dakshina) (Hindi transliteration subset)

---

## Project Overview
- **Task:** Hindi (Latin → Devanagari) transliteration
- **Architecture:** Encoder–Decoder with RNN variants (SimpleRNN, LSTM, GRU)
- **Attention:** Bahdanau additive mechanism for improved alignment

---

## Dataset
- **Source:** Google Dakshina dataset (Hindi transliteration subset)
- **Files:**
  - `hi.translit.sampled.train.tsv` (training)
  - `hi.translit.sampled.dev.tsv` (validation)
- **Format:** Tab-separated pairs: `<Latin>	<Devanagari>`

---

## Features
- Character-level tokenization for both scripts
- Configurable RNN backbones: SimpleRNN, LSTM, GRU
- Attention-based decoder for better context handling
- Experiment tracking with Weights & Biases (wandb)
- Evaluation metrics: exact-match accuracy, average edit distance

---

## Usage
1. **Clone & install**
   ```bash
   git clone https://github.com/vishnu000000/DeepLearning_A3.git
   cd DeepLearning_A3
   wandb login YOUR_API_KEY
   ```
2. **Prepare data**
   - Download Dakshina dataset and place `hi/translit/*.tsv` under `data/dakshina_dataset_v1.0/`
3. **Run training**
   ```bash
   python src/train.py --config configs/hindi.yml
   ```
4. **Inference**
   ```bash
   python src/infer.py --checkpoint results/best_hindi.pth --input "namaste"
   ```

---

## Results & Evaluation
- Loss & accuracy curves in `results/logs/`
- Sample transliterations in `results/predictions/`
- W&B dashboard for interactive analysis

---
