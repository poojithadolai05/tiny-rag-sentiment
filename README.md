# Tiny RAG & Sentiment Classification Project

## Overview

This repository contains a lightweight **retrieval-augmented generation (RAG) mini-system** and a **sentiment classification pipeline**.

The goal is to demonstrate thoughtful design, experimentation, and reliable results on small, unique datasets provided for this assessment.

## Parts

### Part A: RAG Mini-System
Answers factual questions grounded in a small document corpus.

### Part B: Sentiment Classifier
Predicts sentiment labels for user reviews, handling nuances and emojis.

## Project Structure

```text
.
├── data/
│   ├── corpus/
│   │   ├── docs.jsonl
│   │   └── questions.json
│   └── sentiment/
│       ├── dev.csv
│       ├── test.csv
│       └── train.csv
├── src/
│   ├── rag_answer.py
│   └── train.py
├── submissions/
│   ├── rag_answers.json
│   └── sentiment_test_predictions.csv
├── .gitignore
├── config.json
├── MODEL_CARD.md
├── RAG_README.md
├── README.md
└── requirements.txt
```

**Note:** Use a Python virtual environment to manage dependencies instead of committing `.venv/` to the repository.


```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
## Part A: RAG Mini-System

### Purpose
- Build a small, CPU-friendly QA pipeline over `data/corpus/docs.jsonl`.
- Output short, factual answers for each question in `data/corpus/questions.json`.
- Ensure answers are grounded and avoid hallucinations.

### Implementation
- **Retrieval:** Sentence-level chunks extracted from documents using NLTK.
- **Matching:** Keyword-guided overlap from `questions.json` and corpus chunks.
- **Assignment:** Stable, one-to-one mapping between questions and chunks.
- **Evaluation:** Topic-aware metric ensures correct context matches.

✅ **Performance:** 100% topic-aware accuracy on dev questions (20/20).  
✅ **Key Strengths:** Zero hallucination, unique answers, lightweight CPU execution.

## Part B: Sentiment Classifier

### Purpose
- Predict positive (1) or negative (0) sentiment for comments/reviews.
- Handle mixed sentiment, typos, emojis, and code-switched words.

### Implementation
- **Baseline:** TF-IDF + Logistic Regression.
- **Advanced Model:** TF-IDF + VADER sentiment features + Tuned LinearSVC.
- **Hyperparameter Tuning:** GridSearchCV for optimal n-gram range, sublinear TF, and SVC parameters.


### Performance (Dev Set)

- ✅ **Final Model:** TF-IDF + VADER + Tuned LinearSVC
- ✅ **Accuracy:** 87.5%
- ✅ **Macro F1:** 0.8693
- ✅ **Key Strength:** Robust handling of nuanced, informal, and emoji-rich text
- ✅ **Random Seed:** 42 ensures reproducibility


### Quick Error Analysis (3 Examples)

| Text (truncated) | Gold | Pred | Insight |
| --- | --- | --- | --- |
| “frankly overheats quickly DeviLabs drone — gps is amazing.” | 0 | 1 | Positive phrase outweighed negative context |
| “five stars Aster keyboard. packaging feels terrible.” | 1 | 0 | Misleading “five stars” phrase caused error |
| “honestly Qubitron smartwatch: cheap materials; esp. pairing.” | 0 | 0 | Subtle negativity handled correctly |


### Robustness Observations

- Handles typos using character n-grams.  
- Interprets emojis effectively via VADER features (👍, 😡, 🔥, 🚀).  
- Some code-switching and informal language handled if seen in training.


## Installation & Environment Setup

### 1. Create and activate a Python virtual environment:

```bash
python -m venv .venv
```
#### Linux/macOS

```bash
source .venv/bin/activate 
```
#### Windows

```bash
.venv\Scripts\activate      
```

### 2. Install required packages:
```bash
pip install -r requirements.txt
```

#### 3. Ensure NLTK resources are downloaded automatically during runtime (handled in scripts).

## Running the Project

#### Generate RAG Answers

```bash
python src/rag_answer.py  
```

#### Output saved to: submissions/rag_answers.json

#### Generate Sentiment Predictions
```bash
python src/train.py
```

#### Output saved to: submissions/sentiment_test_predictions.csv

## Assumptions & Notes

- No modification of data inside the `data/` folder.
- CPU-friendly methods were prioritized for embeddings, LLMs, and models.
- RAG system does not use external online models; it is purely extractive and lightweight.
- Errors in sentiment classification are primarily due to ambiguous or sarcastic text, as detailed in `MODEL_CARD.md`.

### Outputs

- `submissions/rag_answers.json` – Dictionary mapping question IDs to answers.  
- `submissions/sentiment_test_predictions.csv` – Predictions in `[text, label]` format.
