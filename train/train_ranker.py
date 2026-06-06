"""
train_ranker.py — NN3 GRU Question Ranker Trainer

Training data:
  - data/question_bank.json  (existing)
  - data-set-full/ds/Interview_Questions.xlsx  (expanded question bank)
  - data-set-full/squad/plain_text/train-*.parquet  (additional question context)
  
Difficulty: 0.0=Easy, 0.5=Medium, 1.0=Hard (from question_bank 'level' field or estimated)
Target: ≥ 70% BCE accuracy
"""
import os
import sys
import json
import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.question_ranker import QuestionRanker
from models.embedding_model import SimpleSkipGram
from config import EMBED_DIM, HIDDEN_DIM, MODELS_DIR, DATA_DIR, LEARNING_RATE, DATASET_ROOT, TOPICS, GLOVE_FALLBACK_PATH

DIFFICULTY_MAP = {'easy': 0.0, 'medium': 0.5, 'hard': 1.0}

def tokenize(text):
    if not isinstance(text, str):
        return []
    return re.findall(r'\b\w+\b', text.lower())

def load_question_bank(embed_model):
    """Returns {topic: [(vec, difficulty_scalar), ...]}"""
    topic_map = {}

    # Existing question_bank.json
    qb_path = os.path.join(DATA_DIR, 'question_bank.json')
    if os.path.exists(qb_path):
        with open(qb_path, 'r') as f:
            for q in json.load(f).get('questions', []):
                t    = q.get('topic', '')
                diff = DIFFICULTY_MAP.get(str(q.get('level', 'easy')).lower(), 0.0)
                vec  = embed_model.get_text_embedding(tokenize(q.get('question', '')))
                topic_map.setdefault(t, []).append((vec, diff))

    # Interview_Questions.xlsx
    xlsx_path = os.path.join(DATASET_ROOT, 'ds', 'Interview_Questions.xlsx')
    if os.path.exists(xlsx_path):
        try:
            import pandas as pd
            df = pd.read_excel(xlsx_path)
            col_map  = {c.strip().lower(): c for c in df.columns}
            q_col    = col_map.get('question') or col_map.get('questions')
            cat_col  = col_map.get('category') or col_map.get('skill')
            lvl_col  = col_map.get('level') or col_map.get('difficulty')

            XLSX_TOPIC_MAP = {
                'python': 'python', 'java': 'python',
                'data structures': 'data_structures_algorithms',
                'algorithms': 'data_structures_algorithms',
                'machine learning': 'machine_learning_basics',
                'database': 'databases_sql', 'sql': 'databases_sql',
                'system design': 'system_design',
                'behavioral': 'behavioral',
            }
            for _, row in df.iterrows():
                if not q_col:
                    break
                cat  = str(row.get(cat_col, '')).strip().lower() if cat_col else ''
                lvl  = str(row.get(lvl_col, 'easy')).strip().lower() if lvl_col else 'easy'
                topic = None
                for key, val in XLSX_TOPIC_MAP.items():
                    if key in cat:
                        topic = val
                        break
                if topic and topic in TOPICS:
                    diff = DIFFICULTY_MAP.get(lvl, 0.0)
                    vec  = embed_model.get_text_embedding(tokenize(str(row.get(q_col, ''))))
                    topic_map.setdefault(topic, []).append((vec, diff))
        except Exception as e:
            print(f"  [WARN] xlsx load error: {e}")

    return topic_map

def build_sequences(topic_map, target_pairs=3000):
    X_seq, X_cand, X_diff, y = [], [], [], []
    all_items = [(v, d) for items in topic_map.values() for v, d in items]

    for t, items in topic_map.items():
        n = len(items)
        if n < 2:
            continue
        for i in range(1, n):
            seq         = [v for v, _ in items[:i]]
            cand_vec, cand_diff = items[i]

            # Positive pair
            X_seq.append(seq)
            X_cand.append(cand_vec)
            X_diff.append([cand_diff])
            y.append(1.0)

            # Negative pair — random item from a different topic
            neg_topic = np.random.choice([k for k in topic_map if k != t])
            neg_vec, neg_diff = topic_map[neg_topic][
                np.random.randint(len(topic_map[neg_topic]))
            ]
            X_seq.append(seq)
            X_cand.append(neg_vec)
            X_diff.append([neg_diff])
            y.append(0.0)

    # Augment to target_pairs if needed
    while len(y) < target_pairs:
        t = np.random.choice(list(topic_map.keys()))
        items = topic_map[t]
        if len(items) < 2:
            continue
        i = np.random.randint(1, len(items))
        seq = [v for v, _ in items[:i]]
        cand_vec, cand_diff = items[i]
        X_seq.append(seq);  X_cand.append(cand_vec); X_diff.append([cand_diff]); y.append(1.0)
        neg_t  = np.random.choice([k for k in topic_map if k != t])
        nv, nd = topic_map[neg_t][np.random.randint(len(topic_map[neg_t]))]
        X_seq.append(seq);  X_cand.append(nv); X_diff.append([nd]); y.append(0.0)

    return X_seq, np.array(X_cand, dtype=np.float32), np.array(X_diff, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)

def pad_sequences(X_seq, max_len=6):
    padded = []
    for seq in X_seq:
        arr = np.array(seq, dtype=np.float32)
        if len(arr) >= max_len:
            padded.append(arr[-max_len:])
        else:
            pad = np.zeros((max_len - len(arr), arr.shape[1] if arr.ndim > 1 else EMBED_DIM), dtype=np.float32)
            padded.append(np.vstack([pad, arr]) if arr.ndim > 1 else np.vstack([pad, arr.reshape(1, -1)]))
    return np.array(padded, dtype=np.float32)

def main():
    print("=== NN3 — GRU Question Ranker Training ===")
    embed_model = SimpleSkipGram(vocab_size=1, embed_dim=EMBED_DIM)
    embed_model.load(os.path.join(MODELS_DIR, 'embeddings.npy'),
                     os.path.join(MODELS_DIR, 'vocab.json'))
    embed_model.set_glove_fallback(GLOVE_FALLBACK_PATH)

    print("Loading question bank...")
    topic_map = load_question_bank(embed_model)
    total_q = sum(len(v) for v in topic_map.values())
    print(f"Questions loaded: {total_q:,} across {len(topic_map)} topics")

    X_seq, X_cand, X_diff, y = build_sequences(topic_map, target_pairs=1000)
    print(f"Training pairs: {len(y):,}", flush=True)

    X_seq_pad = pad_sequences(X_seq, max_len=6)

    # Train/Val split
    indices   = np.random.permutation(len(y))
    split     = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    X_seq_t  = torch.tensor(X_seq_pad[train_idx])
    X_cand_t = torch.tensor(X_cand[train_idx])
    X_diff_t = torch.tensor(X_diff[train_idx])
    y_t      = torch.tensor(y[train_idx])

    X_seq_v  = torch.tensor(X_seq_pad[val_idx])
    X_cand_v = torch.tensor(X_cand[val_idx])
    X_diff_v = torch.tensor(X_diff[val_idx])
    y_v      = torch.tensor(y[val_idx])

    model     = QuestionRanker(embed_dim=EMBED_DIM, hidden_dim=64)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc   = 0.0
    best_state = None
    epochs     = 100

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_seq_t, X_cand_t, X_diff_t)
        loss  = criterion(preds, y_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            v_preds  = model(X_seq_v, X_cand_v, X_diff_v)
            v_loss   = criterion(v_preds, y_v)
            v_acc    = ((v_preds > 0.5).float() == y_v).float().mean().item()

        if v_acc > best_acc:
            best_acc   = v_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Val Loss: {v_loss.item():.4f} | Val Acc: {v_acc:.4f}", flush=True)

    save_path = os.path.join(MODELS_DIR, 'question_ranker.pt')
    torch.save(best_state, save_path)
    print(f"\nBest Val Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)", flush=True)
    print(f"Model saved → {save_path}", flush=True)
    if best_acc >= 0.70:
        print("✓ PASS: ≥ 70% accuracy achieved.", flush=True)
    else:
        print("✗ WARN: Below 70% target.", flush=True)

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    main()
