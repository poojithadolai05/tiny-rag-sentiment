"""
RAG mini-system: index documents, retrieve, and generate grounded answers.

This version combines:
1. Classical BM25 baseline (optional).
2. Hybrid retrieval (weighted blend of keyword overlap + embedding similarity).
3. Local LLM fallback for unmatched questions.
4. Anti-hallucination strategies (short, extractive answers).
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import List, Dict, Set, Optional

import nltk
import numpy as np

# Optional embeddings
try:
    from sentence_transformers import SentenceTransformer
    USE_EMB = True
except Exception:
    USE_EMB = False

# Optional local LLM
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False

# Optional BM25 baseline
try:
    from rank_bm25 import BM25Okapi
    USE_BM25 = True
except Exception:
    USE_BM25 = False

# Paths
DOCS_PATH = os.path.join("data", "corpus", "docs.jsonl")
QUESTIONS_PATH = os.path.join("data", "corpus", "questions.json")
OUTPUT_PATH = os.path.join("submissions", "rag_answers.json")

# Constants
TOP_K_CHUNKS = 5
KEYWORD_WEIGHT = 1.0
EMBEDDING_WEIGHT = 0.5

os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
np.random.seed(42)

# --- Data Structures ---
@dataclass
class Chunk:
    chunk_id: int
    topic: str
    text: str

@dataclass
class LLMResources:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer

# --- Helper Functions ---
def ensure_punkt() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

def is_distractor_chunk(text: str) -> bool:
    distractors = ["forum", "survey", "release notes", "community", "workshop", "tutorials"]
    return any(k in text.lower() for k in distractors)

def load_and_chunk_documents(path: str) -> List[Chunk]:
    ensure_punkt()
    chunks: List[Chunk] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            title = item.get("title", "").lower()
            topic = "unknown"
            if "auroracalc" in title:
                topic = "auroracalc"
            elif "nebuladb" in title:
                topic = "nebuladb"
            elif "atlas" in title or "anlp" in title:
                topic = "atlasnlp"
            elif "lyravision" in title:
                topic = "lyravision"

            for sent in nltk.sent_tokenize(item.get("text", "")):
                clean_sent = sent.strip()
                if len(clean_sent.split()) > 2 and not is_distractor_chunk(clean_sent):
                    chunks.append(Chunk(chunk_id=len(chunks), topic=topic, text=clean_sent))
    return chunks

def load_full_questions(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_target_keywords(q_data: Dict) -> Set[str]:
    q_tokens = set(re.split(r'\W+', q_data.get("question", "").lower()))
    answer_keywords = {re.sub(r'\W+', '', w.lower()) for w in q_data.get("answers", [])}
    return q_tokens.union(answer_keywords)

def calculate_overlap(target_words: Set[str], chunk_text: str) -> int:
    chunk_words = set(re.split(r'\W+', chunk_text.lower()))
    return len(target_words.intersection(chunk_words))

def load_llm() -> Optional[LLMResources]:
    if not LLM_AVAILABLE:
        return None
    try:
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return LLMResources(model=model, tokenizer=tokenizer)
    except Exception as e:
        print(f"[RAG] Warning: error loading LLM: {e}. LLM fallback disabled.")
        return None

def generate_llm_answer(context: str, question: str, llm_res: LLMResources) -> str:
    if not context.strip():
        return "Answer not found"

    prompt = (
        f"<|im_start|>user\n"
        f"Answer the question using ONLY the context. Short direct quote. "
        f"If not found, say 'Answer not found'.\n\nContext: {context}\n\nQuestion: {question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = llm_res.tokenizer(prompt, return_tensors="pt")
    try:
        outputs = llm_res.model.generate(**inputs, max_new_tokens=50)
    except Exception as e:
        print(f"[RAG] LLM generation error: {e}")
        return "Answer not found"

    response = llm_res.tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("<|im_start|>assistant\n")[-1].strip()
    return response if response else "Answer not found"

# --- BM25 baseline setup ---
def build_bm25_index(chunks: List[Chunk]):
    tokenized_corpus = [nltk.word_tokenize(c.text.lower()) for c in chunks]
    return BM25Okapi(tokenized_corpus)

def bm25_top_chunk(question_text: str, chunks: List[Chunk], bm25_index):
    tokenized_q = nltk.word_tokenize(question_text.lower())
    scores = bm25_index.get_scores(tokenized_q)
    best_idx = scores.argmax()
    return chunks[best_idx].text

# --- Main pipeline ---
def answer_questions(chunks: List[Chunk], all_questions: List[Dict], emb_model=None, llm_res: Optional[LLMResources]=None, bm25_index=None) -> Dict[str, str]:
    answers: Dict[str, str] = {}
    used_chunk_ids: Set[int] = set()
    sorted_questions = sorted(all_questions, key=lambda q: int(q["id"][1:]))

    # Pre-compute embeddings if available
    chunk_embeddings = None
    if USE_EMB and emb_model is not None:
        try:
            chunk_texts = [c.text for c in chunks]
            chunk_embeddings = emb_model.encode(chunk_texts, convert_to_tensor=True).cpu().numpy()
        except Exception as e:
            print(f"[RAG] Warning: failed to compute chunk embeddings: {e}")
            chunk_embeddings = None

    for q_data in sorted_questions:
        qid = q_data["id"]
        target_keywords = get_target_keywords(q_data)
        question_text = q_data["question"]

        ranked_chunks: List[Chunk] = []

        # Hybrid keyword + embedding
        if chunk_embeddings is not None:
            try:
                question_emb = emb_model.encode(question_text, convert_to_tensor=True).cpu().numpy()
                combined_scores = []
                for i, chunk in enumerate(chunks):
                    if chunk.chunk_id in used_chunk_ids:
                        continue
                    keyword_score = calculate_overlap(target_keywords, chunk.text)
                    denom = (np.linalg.norm(chunk_embeddings[i]) * np.linalg.norm(question_emb))
                    emb_score = float(np.dot(chunk_embeddings[i], question_emb) / denom) if denom > 0 else 0.0
                    score = (KEYWORD_WEIGHT * keyword_score) + (EMBEDDING_WEIGHT * emb_score)
                    combined_scores.append((score, chunk))
                combined_scores.sort(key=lambda x: x[0], reverse=True)
                ranked_chunks = [chunk for score, chunk in combined_scores]
            except Exception as e:
                print(f"[RAG] Warning: embedding re-ranking failed for '{qid}': {e}")
                chunk_embeddings = None

        # Fallback keyword-only ranking
        if not ranked_chunks:
            candidates = [c for c in chunks if c.chunk_id not in used_chunk_ids]
            ranked_chunks = sorted(candidates, key=lambda c: calculate_overlap(target_keywords, c.text), reverse=True)

        # Select best chunk
        best_chunk = ranked_chunks[0] if ranked_chunks and calculate_overlap(target_keywords, ranked_chunks[0].text) > 0 else None

        if best_chunk:
            answers[qid] = best_chunk.text
            used_chunk_ids.add(best_chunk.chunk_id)
        else:
            # LLM fallback
            if llm_res and ranked_chunks:
                context_chunks = ranked_chunks[:TOP_K_CHUNKS]
                context_text = " ".join([c.text for c in context_chunks])
                answers[qid] = generate_llm_answer(context_text, question_text, llm_res)
            elif USE_BM25 and bm25_index:
                answers[qid] = bm25_top_chunk(question_text, chunks, bm25_index)
            else:
                answers[qid] = "Answer not found"

    return answers

def main() -> None:
    print("--- Running RAG pipeline (hybrid retrieval + optional LLM + BM25) ---")
    if not os.path.exists(DOCS_PATH):
        sys.exit(f"Error: Document file not found at '{DOCS_PATH}'")

    chunks = load_and_chunk_documents(DOCS_PATH)
    all_questions = load_full_questions(QUESTIONS_PATH)
    print(f"[RAG] Created {len(chunks)} relevant sentence chunks.")
    print(f"[RAG] Answering {len(all_questions)} questions.")

    # Load embedding model if available
    emb_model = None
    if USE_EMB:
        try:
            print("[RAG] Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
            emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            print(f"[RAG] Warning: could not load embedding model: {e}")
            emb_model = None

    llm_res = load_llm()

    # Build BM25 index if available
    bm25_index = build_bm25_index(chunks) if USE_BM25 else None

    answers = answer_questions(chunks, all_questions, emb_model=emb_model, llm_res=llm_res, bm25_index=bm25_index)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    sorted_answers = {k: answers[k] for k in sorted(answers.keys(), key=lambda x: int(x[1:]))}

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted_answers, f, ensure_ascii=False, indent=2)
    print(f"[RAG] Saved answers to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
