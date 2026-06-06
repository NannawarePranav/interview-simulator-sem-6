import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.embedding_model import SimpleSkipGram
from train.train_classifier import load_dataset as load_clf_data
from train.train_ranker import load_data as load_ranker_data, build_sequences, pad_sequences
from train.train_scorer import load_data as load_score_data
from models.topic_classifier import TopicClassifier
from models.question_ranker import QuestionRanker
from models.answer_scorer import SiameseAnswerScorer
from config import EMBED_DIM, HIDDEN_DIM, TOPICS, MODELS_DIR

def evaluate_models():
    # 1. Load Embeddings
    embed_model = SimpleSkipGram(vocab_size=1, embed_dim=EMBED_DIM)
    embed_model.load(os.path.join(MODELS_DIR, 'embeddings.npy'), os.path.join(MODELS_DIR, 'vocab.json'))
    
    # 2. Evaluate Topic Classifier
    clf = TopicClassifier(embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_topics=len(TOPICS))
    clf.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'topic_classifier.pt')))
    clf.eval()
    
    X_clf, y_clf = load_clf_data(embed_model)
    with torch.no_grad():
        out = clf(torch.tensor(X_clf, dtype=torch.float32))
        _, preds = torch.max(out, 1)
        acc = (preds == torch.tensor(y_clf)).float().mean().item()
    
    # 3. Evaluate Question Ranker
    ranker = QuestionRanker(embed_dim=EMBED_DIM, hidden_dim=64)
    ranker.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'question_ranker.pt')))
    ranker.eval()
    
    t_map = load_ranker_data(embed_model)
    X_seq, X_cand, y_rank = build_sequences(t_map)
    X_seq_pad = pad_sequences(X_seq, max_len=6)
    
    with torch.no_grad():
        X_seq_t = torch.tensor(X_seq_pad, dtype=torch.float32)
        X_cand_t = torch.tensor(np.array(X_cand), dtype=torch.float32)
        v_preds = ranker(X_seq_t, X_cand_t)
        # convert probability to 1/0
        pred_labels = (v_preds > 0.5).float()
        r_acc = (pred_labels == torch.tensor(y_rank, dtype=torch.float32).reshape(-1, 1)).float().mean().item()
    
    # 4. Evaluate Answer Scorer
    scorer = SiameseAnswerScorer(embed_dim=EMBED_DIM)
    scorer.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'answer_scorer.pt')))
    scorer.eval()
    
    X_q, X_a, y_score = load_score_data(embed_model)
    with torch.no_grad():
        s_preds = scorer(torch.tensor(X_q, dtype=torch.float32), torch.tensor(X_a, dtype=torch.float32))
        mse = torch.nn.functional.mse_loss(s_preds, torch.tensor(y_score, dtype=torch.float32)).item()
        
        # calculate accuracy within 0.2 margin
        margin = 0.2
        correct = (torch.abs(s_preds - torch.tensor(y_score, dtype=torch.float32)) <= margin).float().sum().item()
        s_acc = correct / len(y_score)
        
    output = []
    output.append("="*40)
    output.append("AI Interviewer Models Evaluation")
    output.append("="*40)
    output.append(f"Embedding Vocab Size: {embed_model.vocab_size}")
    output.append(f"")
    output.append(f"[NN2] Topic Classifier Accuracy (All Data): {acc*100:.2f}%")
    output.append(f"[NN3] Question Ranker BCE Accuracy (All Data): {r_acc*100:.2f}%")
    output.append(f"[NN4] Answer Scorer MSE (All Data): {mse:.4f}")
    output.append(f"[NN4] Answer Scorer Margin Accuracy (+/- {margin}): {s_acc*100:.2f}%")
    output.append("="*40)
    
    with open(os.path.join(os.path.dirname(__file__), 'eval_results.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(output))
        
    print("\n".join(output))

if __name__ == "__main__":
    evaluate_models()
