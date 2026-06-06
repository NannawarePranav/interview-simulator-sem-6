import os
import sys
import csv
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.topic_classifier import TopicClassifier
from models.question_ranker import QuestionRanker
from models.answer_scorer import SiameseAnswerScorer
from models.embedding_model import SimpleSkipGram
from config import TOPICS, EMBED_DIM, HIDDEN_DIM, MODELS_DIR, DATA_DIR, LEARNING_RATE, GLOVE_PATH, GLOVE_DIM
from embeddings.glove_loader import load_glove, get_glove_embedding
import re

def tokenize(text):
    text = text.lower()
    return re.findall(r'\b\w+\b', text)

def get_v1_embedding(tokens, embed_model):
    return embed_model.get_text_embedding(tokens)

def get_v2_embedding(tokens, glove_dict, embed_model):
    return get_glove_embedding(tokens, glove_dict, embed_model)

# --- Classifier Data ---
def load_classifier_data():
    raw_data = []
    topic2id = {t: i for i, t in enumerate(TOPICS)}
    csv_path = os.path.join(DATA_DIR, 'topic_classifier_data.csv')
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            topic = row['topic_label']
            if topic in topic2id:
                raw_data.append((tokenize(row['snippet_text']), topic2id[topic]))
    return raw_data

def prepare_classifier_data(raw_data, get_emb_fn):
    X, y = [], []
    for tokens, label in raw_data:
        X.append(get_emb_fn(tokens))
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

# --- Ranker Data ---
def load_ranker_data():
    path = os.path.join(DATA_DIR, 'question_bank.json')
    with open(path, 'r') as f:
        qb = json.load(f)['questions']
        
    topic_que_map = {}
    for q in qb:
        t = q['topic']
        if t not in topic_que_map:
            topic_que_map[t] = []
        topic_que_map[t].append(tokenize(q['question']))
    return topic_que_map

def prepare_ranker_data(topic_que_map, get_emb_fn, embed_dim):
    X_seq, X_cand, y = [], [], []
    embedded_map = {}
    for t, toks_list in topic_que_map.items():
        embedded_map[t] = [get_emb_fn(toks) for toks in toks_list]
        
    all_vecs = [v for lst in embedded_map.values() for v in lst]
    
    for t, vecs in embedded_map.items():
        n = len(vecs)
        for i in range(1, n):
            seq = vecs[:i]
            X_seq.append(seq)
            X_cand.append(vecs[i])
            y.append(1.0)
            
            neg_idx = np.random.randint(0, len(all_vecs))
            X_seq.append(seq)
            X_cand.append(all_vecs[neg_idx])
            y.append(0.0)
            
    padded = []
    max_len = 6
    for seq in X_seq:
        if len(seq) >= max_len:
            padded.append(seq[-max_len:])
        else:
            zeros = [np.zeros(embed_dim, dtype=np.float32) for _ in range(max_len - len(seq))]
            padded.append(zeros + seq)
            
    return np.array(padded, dtype=np.float32), np.array(X_cand, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)

# --- Scorer Data ---
def load_scorer_data():
    raw_data = []
    path = os.path.join(DATA_DIR, 'answer_scoring_data.csv')
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_data.append((tokenize(row['question']), tokenize(row['answer']), float(row['score'])))
    return raw_data

def prepare_scorer_data(raw_data, get_emb_fn):
    X_q, X_a, y = [], [], []
    for q_toks, a_toks, score in raw_data:
        X_q.append(get_emb_fn(q_toks))
        X_a.append(get_emb_fn(a_toks))
        y.append(score)
    return np.array(X_q, dtype=np.float32), np.array(X_a, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)

def main():
    print("[Phase 1] Loading models and embeddings...")
    embed_model = SimpleSkipGram(vocab_size=1, embed_dim=EMBED_DIM)
    embed_model.load(os.path.join(MODELS_DIR, 'embeddings.npy'), os.path.join(MODELS_DIR, 'vocab.json'))
    
    glove_full_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), GLOVE_PATH)
    glove_dict = load_glove(glove_full_path)
    
    get_v1 = lambda t: get_v1_embedding(t, embed_model)
    get_v2 = lambda t: get_v2_embedding(t, glove_dict, embed_model)
    
    print("\n--- Training NN2: Topic Classifier ---")
    raw_clf = load_classifier_data()
    indices = np.random.permutation(len(raw_clf))
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]
    
    # Eval V1 Classifier
    X1, y1 = prepare_classifier_data(raw_clf, get_v1)
    X1_val, y1_val = torch.tensor(X1[val_idx]), torch.tensor(y1[val_idx])
    v1_clf = TopicClassifier(embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_topics=len(TOPICS))
    v1_clf.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'topic_classifier.pt')))
    v1_clf.eval()
    with torch.no_grad():
        v1_out = v1_clf(X1_val)
        _, preds1 = torch.max(v1_out, 1)
        v1_acc = (preds1 == y1_val).sum().item() / len(y1_val)
        
    # Train V2 Classifier
    X2, y2 = prepare_classifier_data(raw_clf, get_v2)
    X2_train, y2_train = torch.tensor(X2[train_idx]), torch.tensor(y2[train_idx])
    X2_val, y2_val = torch.tensor(X2[val_idx]), torch.tensor(y2[val_idx])
    
    v2_clf = TopicClassifier(embed_dim=GLOVE_DIM, hidden_dim=HIDDEN_DIM, num_topics=len(TOPICS))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(v2_clf.parameters(), lr=LEARNING_RATE)
    
    epochs = 200
    for epoch in range(epochs):
        v2_clf.train()
        optimizer.zero_grad()
        loss = criterion(v2_clf(X2_train), y2_train)
        loss.backward()
        optimizer.step()
        
    v2_clf.eval()
    with torch.no_grad():
        v2_out = v2_clf(X2_val)
        _, preds2 = torch.max(v2_out, 1)
        v2_acc = (preds2 == y2_val).sum().item() / len(y2_val)
        
    torch.save(v2_clf.state_dict(), os.path.join(MODELS_DIR, 'topic_classifier_glove.pt'))
    print(f"V1 Topic Classifier Accuracy: {v1_acc:.4f}")
    print(f"V2 Topic Classifier Accuracy: {v2_acc:.4f}")
    print("Saved topic_classifier_glove.pt")
    
    print("\n--- Training NN3: Question Ranker ---")
    raw_rnk = load_ranker_data()
    X_seq_pad, X_cand, y_rnk = prepare_ranker_data(raw_rnk, get_v2, GLOVE_DIM)
    rnk_indices = np.random.permutation(len(y_rnk))
    rnk_split = int(0.8 * len(rnk_indices))
    rt_idx, rv_idx = rnk_indices[:rnk_split], rnk_indices[rnk_split:]
    
    Xsq_t, Xc_t, yr_t = torch.tensor(X_seq_pad[rt_idx]), torch.tensor(X_cand[rt_idx]), torch.tensor(y_rnk[rt_idx])
    Xsq_v, Xc_v, yr_v = torch.tensor(X_seq_pad[rv_idx]), torch.tensor(X_cand[rv_idx]), torch.tensor(y_rnk[rv_idx])
    
    v2_rnk = QuestionRanker(embed_dim=GLOVE_DIM, hidden_dim=64)
    criterion_rnk = nn.BCELoss()
    optimizer_rnk = optim.Adam(v2_rnk.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(400):
        v2_rnk.train()
        optimizer_rnk.zero_grad()
        loss = criterion_rnk(v2_rnk(Xsq_t, Xc_t), yr_t)
        loss.backward()
        optimizer_rnk.step()
        
    v2_rnk.eval()
    with torch.no_grad():
        v_loss = criterion_rnk(v2_rnk(Xsq_v, Xc_v), yr_v).item()
    torch.save(v2_rnk.state_dict(), os.path.join(MODELS_DIR, 'question_ranker_glove.pt'))
    print(f"V2 Question Ranker Val Loss: {v_loss:.4f}")
    print("Saved question_ranker_glove.pt")

    print("\n--- Training NN4: Answer Scorer ---")
    raw_scr = load_scorer_data()
    Xq, Xa, y_scr = prepare_scorer_data(raw_scr, get_v2)
    scr_indices = np.random.permutation(len(y_scr))
    scr_split = int(0.8 * len(scr_indices))
    st_idx, sv_idx = scr_indices[:scr_split], scr_indices[scr_split:]
    
    Xq_t, Xa_t, ys_t = torch.tensor(Xq[st_idx]), torch.tensor(Xa[st_idx]), torch.tensor(y_scr[st_idx])
    Xq_v, Xa_v, ys_v = torch.tensor(Xq[sv_idx]), torch.tensor(Xa[sv_idx]), torch.tensor(y_scr[sv_idx])
    
    v2_scr = SiameseAnswerScorer(embed_dim=GLOVE_DIM)
    criterion_scr = nn.MSELoss()
    optimizer_scr = optim.Adam(v2_scr.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(100):
        v2_scr.train()
        optimizer_scr.zero_grad()
        loss = criterion_scr(v2_scr(Xq_t, Xa_t), ys_t)
        loss.backward()
        optimizer_scr.step()
        
    v2_scr.eval()
    with torch.no_grad():
        v_loss_scr = criterion_scr(v2_scr(Xq_v, Xa_v), ys_v).item()
    torch.save(v2_scr.state_dict(), os.path.join(MODELS_DIR, 'answer_scorer_glove.pt'))
    print(f"V2 Answer Scorer Val MSE: {v_loss_scr:.4f}")
    print("Saved answer_scorer_glove.pt")

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    main()
