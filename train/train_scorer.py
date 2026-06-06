"""
train_scorer.py — NN4 Siamese Answer Scorer Trainer

Data sources (combined):
  1. data/answer_scoring_data.csv                          (existing)
  2. archive (2)/stsbenchmark/stsbenchmark/sts-train.csv  (TSV, score 0-5 → /5)
  3. archive (2)/COVID_Ethics_train.csv                   (TSV, score 0-1)
  4. archive (3)/msr_paraphrase_train.txt                 (TSV, label 1→0.9, 0→0.1)
  5. squad/plain_text/train-*.parquet                     (full ans→0.9, short→0.2)
  6. snli/plain_text/train-*.parquet                      (0=entail→0.85, 1=neutral→0.5, 2=contra→0.05)
  7. quora-question-pairs/train.csv/train.csv              (tech only; 1→0.9, 0→0.1)
  8. semeval-sts/YYYY/*.tsv                               (per-year dirs; score/5)

Procedure:
  - Run 5-fold CV → report mean ± std MSE
  - Train final model on 100% of data → save to models/saved/answer_scorer.pt
"""
import os
import sys
import csv
import re
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.answer_scorer import SiameseAnswerScorer
from models.embedding_model import SimpleSkipGram
from config import EMBED_DIM, MODELS_DIR, DATA_DIR, LEARNING_RATE, DATASET_ROOT, GLOVE_FALLBACK_PATH

TECH_KEYWORDS = {
    'python', 'java', 'sql', 'api', 'algorithm', 'database', 'code', 'coding',
    'programming', 'machine learning', 'neural', 'docker', 'kubernetes', 'linux',
    'git', 'aws', 'data', 'software', 'function', 'class', 'object', 'array',
    'recursion', 'sorting', 'binary', 'tree', 'graph', 'network', 'server',
    'backend', 'frontend', 'react', 'node', 'flask', 'django', 'restful',
}

def tokenize(text: str):
    if not isinstance(text, str):
        return []
    text = re.sub(r'<[^>]+>', ' ', text).lower()
    return re.findall(r'\b\w+\b', text)

def is_technical(text: str) -> bool:
    words = set(tokenize(text))
    return bool(words & TECH_KEYWORDS)

def make_row(embed_model, text_a, text_b, score):
    q_vec = embed_model.get_text_embedding(tokenize(text_a))
    a_vec = embed_model.get_text_embedding(tokenize(text_b))
    return q_vec, a_vec, float(score)

def load_all_data(embed_model):
    X_q, X_a, y = [], [], []
    DS = DATASET_ROOT

    def add(qa, aa, sc):
        q_vec = embed_model.get_text_embedding(tokenize(qa))
        a_vec = embed_model.get_text_embedding(tokenize(aa))
        X_q.append(q_vec); X_a.append(a_vec); y.append(float(sc))

    # 1. Existing CSV
    existing = os.path.join(DATA_DIR, 'answer_scoring_data.csv')
    if os.path.exists(existing):
        with open(existing, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                try:
                    add(row['question'], row['answer'], row['score'])
                except Exception:
                    pass
    print(f"  After existing CSV: {len(y)}")

    # 2. STS Benchmark
    sts_path = os.path.join(DS, 'archive (2)', 'stsbenchmark', 'stsbenchmark', 'sts-train.csv')
    if os.path.exists(sts_path):
        with open(sts_path, 'r', encoding='utf-8', errors='replace') as f:
            for row in csv.reader(f, delimiter='\t'):
                if len(row) >= 7:
                    try:
                        add(row[5], row[6], float(row[4]) / 5.0)
                    except (ValueError, IndexError):
                        pass
    print(f"  After STS Benchmark: {len(y)}")

    # 3. COVID Ethics
    covid_path = os.path.join(DS, 'archive (2)', 'COVID_Ethics_train.csv')
    if os.path.exists(covid_path):
        with open(covid_path, 'r', encoding='utf-8', errors='replace') as f:
            for row in csv.reader(f, delimiter='\t'):
                # Schema: version|doc_id|src|cand_id|score|passage|question
                if len(row) >= 7:
                    try:
                        score = float(row[4])
                        add(row[6], row[5], score)  # question, passage
                    except (ValueError, IndexError):
                        pass
    print(f"  After COVID Ethics: {len(y)}")

    # 4. MSR Paraphrase  (alternating rows; pair shared prefix in Sentence_ID)
    msr_path = os.path.join(DS, 'archive (3)', 'msr_paraphrase_train.txt')
    if os.path.exists(msr_path):
        try:
            with open(msr_path, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.reader(f, delimiter='\t')
                next(reader, None)  # header
                rows = list(reader)
            # Rows come in pairs sharing a story; label is in first column
            # Actual format: Quality  #1 ID  #2 ID  #1 String  #2 String
            for row in rows:
                if len(row) >= 5:
                    try:
                        label = int(row[0])
                        score = 0.9 if label == 1 else 0.1
                        add(row[3], row[4], score)
                    except (ValueError, IndexError):
                        pass
        except Exception as e:
            print(f"  [WARN] MSR load error: {e}")
    print(f"  After MSR Paraphrase: {len(y)}")

    # 5. SQuAD
    squad_path = os.path.join(DS, 'squad', 'plain_text', 'train-00000-of-00001.parquet')
    if os.path.exists(squad_path):
        try:
            import pandas as pd
            df = pd.read_parquet(squad_path).sample(n=2000, random_state=42)
            for _, row in df.iterrows():
                q = str(row.get('question', ''))
                ans_field = row.get('answers', {})
                if isinstance(ans_field, dict):
                    texts = ans_field.get('text', [])
                elif hasattr(ans_field, 'get'):
                    texts = ans_field.get('text', [])
                else:
                    texts = []
                if texts:
                    a = str(texts[0])
                    score = 0.9 if len(a.split()) >= 5 else 0.2
                    add(q, a, score)
        except Exception as e:
            print(f"  [WARN] SQuAD load error: {e}")
    print(f"  After SQuAD: {len(y)}")

    # 6. SNLI
    snli_path = os.path.join(DS, 'snli', 'plain_text', 'train-00000-of-00001.parquet')
    if os.path.exists(snli_path):
        try:
            import pandas as pd
            df = pd.read_parquet(snli_path).sample(n=3000, random_state=42)
            label_map = {0: 0.85, 1: 0.5, 2: 0.05}
            for _, row in df.iterrows():
                label = int(row.get('label', -1))
                if label == -1:
                    continue
                add(str(row.get('premise', '')),
                    str(row.get('hypothesis', '')),
                    label_map[label])
        except Exception as e:
            print(f"  [WARN] SNLI load error: {e}")
    print(f"  After SNLI: {len(y)}")

    # 7. Quora Question Pairs (technical only)
    quora_path = os.path.join(DS, 'quora-question-pairs', 'train.csv', 'train.csv')
    if os.path.exists(quora_path):
        try:
            import pandas as pd
            df = pd.read_csv(quora_path, usecols=['question1', 'question2', 'is_duplicate'],
                             nrows=40000, on_bad_lines='skip')
            df = df.dropna()
            mask = df['question1'].apply(is_technical) | df['question2'].apply(is_technical)
            df   = df[mask].head(4000)
            for _, row in df.iterrows():
                score = 0.9 if int(row['is_duplicate']) == 1 else 0.1
                add(str(row['question1']), str(row['question2']), score)
        except Exception as e:
            print(f"  [WARN] Quora load error: {e}")
    print(f"  After Quora QP: {len(y)}")

    # 8. SemEval STS — per-year directories (avoid Windows symlink issues)
    semeval_root = os.path.join(DS, 'semeval-sts')
    for year in ['2012', '2013', '2014', '2015', '2016']:
        year_dir = os.path.join(semeval_root, year)
        for tsv_file in glob.glob(os.path.join(year_dir, '*.tsv')):
            try:
                with open(tsv_file, 'r', encoding='utf-8', errors='replace') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            try:
                                score = float(parts[0]) / 5.0
                                add(parts[1], parts[2], score)
                            except (ValueError, IndexError):
                                pass
            except Exception:
                pass
    print(f"  After SemEval STS: {len(y)}")

    return (np.array(X_q, dtype=np.float32),
            np.array(X_a, dtype=np.float32),
            np.array(y,   dtype=np.float32).reshape(-1, 1))

def train_model(X_q, X_a, y, epochs=100, train_idx=None):
    """Train a single model instance. If train_idx is None, uses all data."""
    if train_idx is None:
        train_idx = np.arange(len(y))

    Xqt = torch.tensor(X_q[train_idx])
    Xat = torch.tensor(X_a[train_idx])
    yt  = torch.tensor(y[train_idx])

    model     = SiameseAnswerScorer(embed_dim=EMBED_DIM)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        preds = model(Xqt, Xat)
        loss  = criterion(preds, yt)
        loss.backward()
        optimizer.step()

    return model

def eval_model(model, X_q, X_a, y, val_idx):
    model.eval()
    Xqv = torch.tensor(X_q[val_idx])
    Xav = torch.tensor(X_a[val_idx])
    yv  = torch.tensor(y[val_idx])
    criterion = nn.MSELoss()
    with torch.no_grad():
        preds = model(Xqv, Xav)
        mse   = criterion(preds, yv).item()
    return mse

def main():
    print("=== NN4 — Answer Scorer Training ===")
    embed_model = SimpleSkipGram(vocab_size=1, embed_dim=EMBED_DIM)
    embed_model.load(os.path.join(MODELS_DIR, 'embeddings.npy'),
                     os.path.join(MODELS_DIR, 'vocab.json'))
    embed_model.set_glove_fallback(GLOVE_FALLBACK_PATH)

    print("Loading all data sources...")
    X_q, X_a, y = load_all_data(embed_model)
    print(f"\nTotal training samples: {len(y):,}")

    # ── 5-Fold Cross-Validation ───────────────────────────────────────────
    k       = 5
    indices = np.random.permutation(len(y))
    folds   = np.array_split(indices, k)
    mse_scores = []

    print("\n--- 5-Fold Cross-Validation ---", flush=True)
    for fold in range(k):
        val_idx   = folds[fold]
        train_idx = np.concatenate([folds[i] for i in range(k) if i != fold])
        model     = train_model(X_q, X_a, y, epochs=15, train_idx=train_idx)
        mse       = eval_model(model, X_q, X_a, y, val_idx)
        mse_scores.append(mse)
        print(f"  Fold {fold+1}/{k} | Val MSE: {mse:.4f}", flush=True)

    mean_mse = np.mean(mse_scores)
    std_mse  = np.std(mse_scores)
    print(f"\n5-fold CV MSE: {mean_mse:.4f} ± {std_mse:.4f}", flush=True)

    # ── Final Model on 100% Data ──────────────────────────────────────────
    print("\n--- Training Final Model on Full Dataset ---", flush=True)
    final_model = train_model(X_q, X_a, y, epochs=30)

    save_path = os.path.join(MODELS_DIR, 'answer_scorer.pt')
    torch.save(final_model.state_dict(), save_path)
    print(f"Final model saved → {save_path}", flush=True)
    print(f"\n5-fold CV summary: mean MSE = {mean_mse:.4f}, std = {std_mse:.4f}", flush=True)

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    main()
