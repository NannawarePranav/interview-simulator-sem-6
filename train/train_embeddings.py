"""
train_embeddings.py — NN1 Skip-Gram Corpus Builder & Trainer

Strategy:
  - Collect tokens from all sources (capped per source to keep training fast)
  - Use RANDOM PAIR SAMPLING instead of exhaustive window enumeration
  - Target: 500k training pairs, 10 epochs → finishes in ~3-5 minutes on CPU
  - Vocab target: 5,000+ tokens
"""
import sys
import os
import json
import re
import csv
import numpy as np
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.embedding_model import SimpleSkipGram
from config import EMBED_DIM, MIN_FREQ, MODELS_DIR, DATA_DIR, DATASET_ROOT, GLOVE_FALLBACK_PATH

MAX_PAIRS    = 600_000   # cap training pairs for feasible CPU training
WINDOW_SIZE  = 2
NUM_NEGATIVES = 5

def tokenize(text: str):
    if not isinstance(text, str):
        return []
    text = re.sub(r'<[^>]+>', ' ', text).lower()
    return re.findall(r'\b[a-z][a-z0-9_]*\b', text)

def load_text_column(path, col_indices, sep=',', max_rows=None, encoding='utf-8'):
    """Read selected columns from CSV/TSV, return flat token list."""
    tokens = []
    try:
        with open(path, 'r', encoding=encoding, errors='replace') as f:
            reader = csv.reader(f, delimiter=sep)
            next(reader, None)
            for i, row in enumerate(reader):
                if max_rows and i >= max_rows:
                    break
                for ci in col_indices:
                    if ci < len(row):
                        tokens.extend(tokenize(row[ci]))
    except Exception as e:
        print(f"  [WARN] {os.path.basename(path)}: {e}")
    return tokens

def load_data():
    tokens = []
    DS = DATASET_ROOT

    # ── Project internal data (small, load fully) ──────────────────────────
    for fname in ['resume.txt', 'skills.txt']:
        p = os.path.join(DATA_DIR, 'raw', fname)
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8', errors='replace') as f:
                tokens.extend(tokenize(f.read()))

    qb_path = os.path.join(DATA_DIR, 'question_bank.json')
    if os.path.exists(qb_path):
        with open(qb_path, 'r') as f:
            for q in json.load(f).get('questions', []):
                tokens.extend(tokenize(q.get('question', '')))

    for fname, cols in [('topic_classifier_data.csv', [0]), ('answer_scoring_data.csv', [0,1])]:
        p = os.path.join(DATA_DIR, fname)
        if os.path.exists(p):
            tokens.extend(load_text_column(p, cols))

    print(f"  Internal project data: {len(tokens):,} tokens")

    # ── Resume datasets (capped) ────────────────────────────────────────────
    tokens.extend(load_text_column(
        os.path.join(DS, 'archive', 'UpdatedResumeDataSet.csv'),
        [1], max_rows=5000))
    tokens.extend(load_text_column(
        os.path.join(DS, 'archive (1)', 'resume_data.csv'),
        [1, 2, 3], max_rows=3000))
    print(f"  After resume CSVs: {len(tokens):,} tokens")

    # ── STS Benchmark ─────────────────────────────────────────────────────
    tokens.extend(load_text_column(
        os.path.join(DS, 'archive (2)', 'stsbenchmark', 'stsbenchmark', 'sts-train.csv'),
        [5, 6], sep='\t'))
    print(f"  After STS: {len(tokens):,} tokens")

    # ── MSR Paraphrase ────────────────────────────────────────────────────
    tokens.extend(load_text_column(
        os.path.join(DS, 'archive (3)', 'msr_paraphrase_train.txt'),
        [3, 4], sep='\t'))
    print(f"  After MSR: {len(tokens):,} tokens")

    # ── Quora (capped to 30k rows) ────────────────────────────────────────
    tokens.extend(load_text_column(
        os.path.join(DS, 'quora-question-pairs', 'train.csv', 'train.csv'),
        [3, 4], max_rows=30000))
    print(f"  After Quora: {len(tokens):,} tokens")

    # ── SNLI (sample 20k rows) ────────────────────────────────────────────
    try:
        import pandas as pd
        df = pd.read_parquet(
            os.path.join(DS, 'snli', 'plain_text', 'train-00000-of-00001.parquet'),
            columns=['premise', 'hypothesis']).sample(n=20000, random_state=42)
        for col in ['premise', 'hypothesis']:
            for t in df[col].dropna().tolist():
                tokens.extend(tokenize(t))
        print(f"  After SNLI: {len(tokens):,} tokens")
    except Exception as e:
        print(f"  [WARN] SNLI: {e}")

    # ── SQuAD (sample 15k rows) ───────────────────────────────────────────
    try:
        import pandas as pd
        df = pd.read_parquet(
            os.path.join(DS, 'squad', 'plain_text', 'train-00000-of-00001.parquet'),
            columns=['question']).sample(n=15000, random_state=42)
        for t in df['question'].dropna().tolist():
            tokens.extend(tokenize(t))
        print(f"  After SQuAD: {len(tokens):,} tokens")
    except Exception as e:
        print(f"  [WARN] SQuAD: {e}")

    # ── Interview_Questions.xlsx ──────────────────────────────────────────
    try:
        import pandas as pd
        df = pd.read_excel(os.path.join(DS, 'ds', 'Interview_Questions.xlsx'))
        # Normalize column names
        col_map = {c.strip().lower(): c for c in df.columns}
        q_col   = col_map.get('question') or col_map.get('questions')
        a_col   = col_map.get('expected answer') or col_map.get('answer')
        for col in [c for c in [q_col, a_col] if c]:
            for t in df[col].dropna().tolist():
                tokens.extend(tokenize(str(t)))
        print(f"  After Interview xlsx: {len(tokens):,} tokens")
    except Exception as e:
        print(f"  [WARN] xlsx: {e}")

    return tokens


def sample_training_pairs(tokens, word2id, max_pairs=MAX_PAIRS,
                           window=WINDOW_SIZE, num_neg=NUM_NEGATIVES):
    """
    Instead of exhaustive enumeration, randomly sample center positions.
    This is O(max_pairs) rather than O(N * window * neg).
    """
    n = len(tokens)
    vocab_ids = np.array(list(word2id.values()), dtype=np.int32)
    token_ids = np.array([word2id.get(w, word2id['<UNK>']) for w in tokens], dtype=np.int32)

    # How many center positions to visit
    centers_needed = max_pairs // (2 * window)
    center_indices = np.random.choice(np.arange(window, n - window),
                                       size=min(centers_needed, n - 2*window),
                                       replace=False)
    data = []
    for ci in center_indices:
        target = int(token_ids[ci])
        for offset in range(-window, window + 1):
            if offset == 0:
                continue
            context = int(token_ids[ci + offset])
            negatives = np.random.choice(vocab_ids, num_neg, replace=False)
            data.append((target, context, negatives))
        if len(data) >= max_pairs:
            break

    np.random.shuffle(data)
    return data


def main():
    print("=== NN1 — Corpus Builder & Embedding Trainer ===")
    print("Loading corpus...")
    tokens = load_data()
    print(f"\nTotal tokens: {len(tokens):,}")

    model = SimpleSkipGram(vocab_size=10, embed_dim=EMBED_DIM)
    model.build_vocab(tokens, min_freq=MIN_FREQ)
    model.set_glove_fallback(GLOVE_FALLBACK_PATH)
    print(f"Vocab size (min_freq={MIN_FREQ}): {model.vocab_size:,}")

    print(f"Sampling up to {MAX_PAIRS:,} training pairs...")
    data = sample_training_pairs(tokens, model.word2id)
    print(f"Training pairs: {len(data):,}")

    epochs = 10
    lr     = 0.05
    for epoch in range(epochs):
        total_loss = 0.0
        np.random.shuffle(data)
        for target, context, negatives in data:
            total_loss += model.train_step(target, context, negatives, lr=lr)
        avg = total_loss / len(data)
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg:.4f}")

    embed_path = os.path.join(MODELS_DIR, 'embeddings.npy')
    vocab_path  = os.path.join(MODELS_DIR, 'vocab.json')
    model.save(embed_path, vocab_path)
    print(f"\n✓ Saved embeddings → {embed_path}")
    print(f"✓ Saved vocab      → {vocab_path}")
    print(f"✓ Final vocab size : {model.vocab_size:,}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
