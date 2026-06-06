"""
train_classifier.py — NN2 Topic Classifier Trainer

Data sources (combined):
  - data/topic_classifier_data.csv                        (existing)
  - data-set-full/archive/UpdatedResumeDataSet.csv        (Category, Resume)
  - data-set-full/archive (1)/resume_data.csv             (skills, job_position_name)
  - data-set-full/ds/Interview_Questions.xlsx             (question, category)
"""
import os
import sys
import csv
import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.topic_classifier import TopicClassifier
from models.embedding_model import SimpleSkipGram
from config import TOPICS, EMBED_DIM, HIDDEN_DIM, MODELS_DIR, DATA_DIR, LEARNING_RATE, DATASET_ROOT, GLOVE_FALLBACK_PATH

# ── Category → taxonomy mapping ───────────────────────────────────────────────
CATEGORY_MAP = {
    # Python / programming
    "python":                        "python",
    "java":                          "python",
    "java developer":                "python",
    "dotnet developer":              "python",
    "testing":                       "python",
    "automation testing":            "python",
    "devops engineer":               "system_design",
    "database":                      "databases_sql",
    "sql":                           "databases_sql",
    "sql developer":                 "databases_sql",
    "data science":                  "machine_learning_basics",
    "data scientist":                "machine_learning_basics",
    "machine learning":              "machine_learning_basics",
    "deep learning":                 "machine_learning_basics",
    "artificial intelligence":       "machine_learning_basics",
    "business analyst":              "behavioral",
    "hr":                            "behavioral",
    "web designing":                 "system_design",
    "network security engineer":     "system_design",
    "blockchain":                    "system_design",
    "hadoop":                        "databases_sql",
    "etl developer":                 "databases_sql",
    "pmo":                           "behavioral",
    "mechanical engineer":           None,
    "civil engineer":                None,
    "electrical engineering":        None,
    "chef":                          None,
    "arts":                          None,
    "advocate":                      None,
    "health and fitness":            None,
    "agriculture":                   None,
    "bpo":                           None,
    "apparel":                       None,
    "fitness":                       None,
    "digital media":                 None,
    "banking":                       "behavioral",
    "finance":                       "behavioral",
    "accountant":                    "behavioral",
    "aviation":                      None,
    "sales":                         "behavioral",
    "public relations":              "behavioral",
    "information technology":        "system_design",
}

# Interview_Questions.xlsx category → topic mapping
XLSX_CATEGORY_MAP = {
    "python":              "python",
    "data structures":     "data_structures_algorithms",
    "algorithms":          "data_structures_algorithms",
    "machine learning":    "machine_learning_basics",
    "deep learning":       "machine_learning_basics",
    "database":            "databases_sql",
    "sql":                 "databases_sql",
    "system design":       "system_design",
    "behavioral":          "behavioral",
    "hr":                  "behavioral",
    "java":                "python",
    "c++":                 "python",
    "javascript":          "python",
    "web development":     "system_design",
    "devops":              "system_design",
    "networking":          "system_design",
    "cloud":               "system_design",
    "os":                  "system_design",
    "operating system":    "system_design",
    "data science":        "machine_learning_basics",
    "nlp":                 "machine_learning_basics",
    "computer vision":     "machine_learning_basics",
}

def tokenize(text):
    if not isinstance(text, str):
        return []
    text = re.sub(r'<[^>]+>', ' ', text).lower()
    return re.findall(r'\b\w+\b', text)

def load_dataset(embed_model):
    X, y = [], []
    topic2id = {t: i for i, t in enumerate(TOPICS)}
    DS = DATASET_ROOT

    # 1. Existing CSV
    csv_path = os.path.join(DATA_DIR, 'topic_classifier_data.csv')
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tokens = tokenize(row.get('snippet_text', ''))
                topic  = row.get('topic_label', '')
                if topic in topic2id and tokens:
                    X.append(embed_model.get_text_embedding(tokens))
                    y.append(topic2id[topic])

    # 2. UpdatedResumeDataSet.csv
    resume1_path = os.path.join(DS, 'archive', 'UpdatedResumeDataSet.csv')
    if os.path.exists(resume1_path):
        with open(resume1_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cat   = row.get('Category', '').strip().lower()
                topic = CATEGORY_MAP.get(cat)
                if topic is None or topic not in topic2id:
                    continue
                resume_text = row.get('Resume', '')
                # Split into sentences (rough) and use each as a sample
                sentences = re.split(r'[.\n]', resume_text)
                for sent in sentences:
                    tokens = tokenize(sent)
                    if len(tokens) >= 4:
                        X.append(embed_model.get_text_embedding(tokens))
                        y.append(topic2id[topic])

    # 3. resume_data.csv — use skills + job_position_name columns
    resume2_path = os.path.join(DS, 'archive (1)', 'resume_data.csv')
    if os.path.exists(resume2_path):
        try:
            import pandas as pd
            df = pd.read_csv(resume2_path, encoding='utf-8', on_bad_lines='skip', low_memory=False)
            # Try to find a job title / category column
            cat_col = None
            for c in df.columns:
                if 'position' in c.lower() or 'category' in c.lower() or 'job' in c.lower():
                    cat_col = c
                    break
            skill_col = None
            for c in df.columns:
                if 'skill' in c.lower():
                    skill_col = c
                    break
            if cat_col and skill_col:
                for _, row in df.iterrows():
                    cat   = str(row.get(cat_col, '')).strip().lower()
                    topic = CATEGORY_MAP.get(cat)
                    if topic is None or topic not in topic2id:
                        # Try partial match
                        for key, val in CATEGORY_MAP.items():
                            if key in cat and val and val in topic2id:
                                topic = val
                                break
                    if topic and topic in topic2id:
                        skills_text = str(row.get(skill_col, ''))
                        tokens = tokenize(skills_text)
                        if len(tokens) >= 3:
                            X.append(embed_model.get_text_embedding(tokens))
                            y.append(topic2id[topic])
        except Exception as e:
            print(f"  [WARN] resume_data.csv load error: {e}")

    # 4. Interview_Questions.xlsx
    xlsx_path = os.path.join(DS, 'ds', 'Interview_Questions.xlsx')
    if os.path.exists(xlsx_path):
        try:
            import pandas as pd
            df = pd.read_excel(xlsx_path)
            col_map = {c.strip().lower(): c for c in df.columns}
            q_col   = col_map.get('question') or col_map.get('questions')
            cat_col = col_map.get('category') or col_map.get('skill')
            if q_col and cat_col:
                for _, row in df.iterrows():
                    cat   = str(row.get(cat_col, '')).strip().lower()
                    topic = None
                    for key, val in XLSX_CATEGORY_MAP.items():
                        if key in cat:
                            topic = val
                            break
                    if topic and topic in topic2id:
                        tokens = tokenize(str(row.get(q_col, '')))
                        if tokens:
                            X.append(embed_model.get_text_embedding(tokens))
                            y.append(topic2id[topic])
        except Exception as e:
            print(f"  [WARN] Interview_Questions.xlsx load error: {e}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def main():
    print("=== NN2 — Topic Classifier Training ===")
    embed_model = SimpleSkipGram(vocab_size=1, embed_dim=EMBED_DIM)
    embed_model.load(os.path.join(MODELS_DIR, 'embeddings.npy'),
                     os.path.join(MODELS_DIR, 'vocab.json'))
    embed_model.set_glove_fallback(GLOVE_FALLBACK_PATH)

    print("Loading dataset...")
    X, y = load_dataset(embed_model)
    print(f"Total samples: {len(X):,}")

    if len(X) == 0:
        print("ERROR: No training data loaded.")
        return

    # Shuffle + split
    indices = np.random.permutation(len(X))
    split   = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    X_train = torch.tensor(X[train_idx])
    y_train = torch.tensor(y[train_idx])
    X_val   = torch.tensor(X[val_idx])
    y_val   = torch.tensor(y[val_idx])

    model     = TopicClassifier(embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_topics=len(TOPICS))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc   = 0.0
    best_state = None
    epochs     = 30

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss    = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out  = model(X_val)
            val_loss = criterion(val_out, y_val)
            _, preds = torch.max(val_out, 1)
            acc      = (preds == y_val).sum().item() / len(y_val)

        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {acc:.4f}", flush=True)

    save_path = os.path.join(MODELS_DIR, 'topic_classifier.pt')
    torch.save(best_state, save_path)
    print(f"\nBest Val Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)", flush=True)
    print(f"Model saved → {save_path}", flush=True)
    if best_acc >= 0.88:
        print("✓ PASS: ≥ 88% accuracy achieved.", flush=True)
    else:
        print("✗ WARN: Below 88% target — consider more data or more epochs.", flush=True)

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    main()
