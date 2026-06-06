import os
import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.answer_scorer import DistilBERTAnswerScorer
from embeddings.distilbert_encoder import DistilBERTEncoder
from config import MODELS_DIR, DATA_DIR, LEARNING_RATE

def load_scorer_data():
    raw_data = []
    path = os.path.join(DATA_DIR, 'answer_scoring_data.csv')
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_data.append((row['question'], row['answer'], float(row['score'])))
    return raw_data

def prepare_scorer_data(raw_data, encoder):
    X_q, X_a, y = [], [], []
    for q_text, a_text, score in raw_data:
        X_q.append(encoder.encode(q_text))
        X_a.append(encoder.encode(a_text))
        y.append(score)
    return np.array(X_q, dtype=np.float32), np.array(X_a, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)

def main():
    print("[Phase 2] Loading DistilBERT Encoder...")
    encoder = DistilBERTEncoder()
    
    print("\n--- Training DistilBERT Answer Scorer ---")
    raw_scr = load_scorer_data()
    Xq, Xa, y_scr = prepare_scorer_data(raw_scr, encoder)
    scr_indices = np.random.permutation(len(y_scr))
    scr_split = int(0.8 * len(scr_indices))
    st_idx, sv_idx = scr_indices[:scr_split], scr_indices[scr_split:]
    
    Xq_t, Xa_t, ys_t = torch.tensor(Xq[st_idx]), torch.tensor(Xa[st_idx]), torch.tensor(y_scr[st_idx])
    Xq_v, Xa_v, ys_v = torch.tensor(Xq[sv_idx]), torch.tensor(Xa[sv_idx]), torch.tensor(y_scr[sv_idx])
    
    v2_scr = DistilBERTAnswerScorer()
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
    
    save_path = os.path.join(MODELS_DIR, 'answer_scorer_distilbert.pt')
    torch.save(v2_scr.state_dict(), save_path)
    print(f"V2 DistilBERT Answer Scorer Val MSE: {v_loss_scr:.4f}")
    print(f"Saved {save_path}")

    # Mandatory sanity check
    print("\n--- Running Sanity Check ---")
    q_text = "What is a tuple in Python?"
    ans_bad = "I don't know what a tuple is"
    ans_good = "A tuple is immutable unlike a list, so it cannot be modified after creation"
    
    q_vec = torch.tensor(np.array([encoder.encode(q_text)]), dtype=torch.float32)
    bad_vec = torch.tensor(np.array([encoder.encode(ans_bad)]), dtype=torch.float32)
    good_vec = torch.tensor(np.array([encoder.encode(ans_good)]), dtype=torch.float32)
    
    with torch.no_grad():
        score_bad = v2_scr(q_vec, bad_vec).item()
        score_good = v2_scr(q_vec, good_vec).item()
        
    print(f"Score for bad answer: {score_bad:.4f}")
    print(f"Score for good answer: {score_good:.4f}")
    
    if score_bad < 0.35 and score_good > 0.65 and (score_good - score_bad) >= 0.3:
        print("[SUCCESS] Sanity check passed!")
    else:
        print("[WARNING] Sanity check failed constraints.")

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    main()
