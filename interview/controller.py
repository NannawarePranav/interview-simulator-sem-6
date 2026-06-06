import os
import sys
import json
import uuid
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interview.session import SessionState
from models.embedding_model import SimpleSkipGram
from models.topic_classifier import TopicClassifier
from models.question_ranker import QuestionRanker
from models.answer_scorer import SiameseAnswerScorer, DistilBERTAnswerScorer
from config import (EMBED_DIM, HIDDEN_DIM, TOPICS, MODELS_DIR, DATA_DIR,
                    MAX_QUESTIONS_PER_TOPIC, CONSECUTIVE_WEAK_LIMIT,
                    WEAK_SCORE_THRESHOLD, STRONG_SCORE_THRESHOLD,
                    USE_GLOVE, GLOVE_PATH, GLOVE_DIM, USE_DISTILBERT_SCORER,
                    USE_GPT2_FOLLOWUP, GPT2_MODEL_PATH, GPT2_SCORE_THRESHOLD_LOW,
                    GPT2_SCORE_THRESHOLD_HIGH, BASE_DIR, USE_RL_POLICY,
                    GLOVE_FALLBACK_PATH)
from embeddings.glove_loader import load_glove, get_glove_embedding
import re

try:
    from data.session_db import save_session, save_question_log
    _DB_AVAILABLE = True
except Exception:
    _DB_AVAILABLE = False

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

class InterviewController:
    def __init__(self):
        self.state = SessionState()
        self.queued_followup = None
        self.session_id = str(uuid.uuid4())
        self._answer_start_time = None
        
        # Load models
        self.embed_model = SimpleSkipGram(vocab_size=1, embed_dim=EMBED_DIM)
        self.embed_model.load(os.path.join(MODELS_DIR, 'embeddings.npy'),
                              os.path.join(MODELS_DIR, 'vocab.json'))
        self.embed_model.set_glove_fallback(GLOVE_FALLBACK_PATH)
                              
        self.use_glove = False
        self.glove_dict = None
        self.active_embed_dim = EMBED_DIM
        
        if USE_GLOVE:
            glove_full_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), GLOVE_PATH)
            if os.path.exists(glove_full_path):
                print(f"[V2] Loading GloVe from {glove_full_path}...")
                self.glove_dict = load_glove(glove_full_path)
                self.use_glove = True
                self.active_embed_dim = GLOVE_DIM
            else:
                print("[V2] GloVe not found, falling back to NN1 embeddings.")

        tc_path = 'topic_classifier_glove.pt' if self.use_glove else 'topic_classifier.pt'
        qr_path = 'question_ranker_glove.pt' if self.use_glove else 'question_ranker.pt'
        as_path = 'answer_scorer_glove.pt' if self.use_glove else 'answer_scorer.pt'
        
        self.topic_classifier = TopicClassifier(embed_dim=self.active_embed_dim, hidden_dim=HIDDEN_DIM, num_topics=len(TOPICS))
        if os.path.exists(os.path.join(MODELS_DIR, tc_path)):
            self.topic_classifier.load_state_dict(torch.load(os.path.join(MODELS_DIR, tc_path)))
        self.topic_classifier.eval()
        
        self.question_ranker = QuestionRanker(embed_dim=self.active_embed_dim, hidden_dim=64)
        if os.path.exists(os.path.join(MODELS_DIR, qr_path)):
            self.question_ranker.load_state_dict(torch.load(os.path.join(MODELS_DIR, qr_path)))
        self.question_ranker.eval()
        
        self.answer_scorer = SiameseAnswerScorer(embed_dim=self.active_embed_dim)
        if os.path.exists(os.path.join(MODELS_DIR, as_path)):
            self.answer_scorer.load_state_dict(torch.load(os.path.join(MODELS_DIR, as_path)))
        self.answer_scorer.eval()
        
        self.use_distilbert_scorer = False
        self.distilbert_encoder = None
        if USE_DISTILBERT_SCORER:
            distilbert_path = os.path.join(MODELS_DIR, 'answer_scorer_distilbert.pt')
            if os.path.exists(distilbert_path):
                print("[V2] Loading DistilBERT Answer Scorer...")
                from embeddings.distilbert_encoder import DistilBERTEncoder
                self.distilbert_encoder = DistilBERTEncoder()
                self.distilbert_scorer = DistilBERTAnswerScorer()
                self.distilbert_scorer.load_state_dict(torch.load(distilbert_path))
                self.distilbert_scorer.eval()
                self.use_distilbert_scorer = True
            else:
                print("[V2] DistilBERT answer scorer model not found, falling back to SiameseAnswerScorer.")
                
        self.use_gpt2 = False
        self.gpt2_generator = None
        if USE_GPT2_FOLLOWUP:
            gpt2_full_path = os.path.join(BASE_DIR, GPT2_MODEL_PATH)
            if os.path.exists(gpt2_full_path):
                print(f"[V2] Loading GPT-2 from {gpt2_full_path}...")
                from models.question_generator import GPT2QuestionGenerator
                self.gpt2_generator = GPT2QuestionGenerator(gpt2_full_path)
                self.use_gpt2 = True
            else:
                print("[V2] GPT-2 model not found, falling back to V1 static question generation.")
                
        self.use_rl_policy = False
        self.rl_policy = None
        if USE_RL_POLICY:
            rl_path = os.path.join(MODELS_DIR, 'interview_policy.pt')
            if os.path.exists(rl_path):
                print(f"[V2] Loading RL Policy from {rl_path}...")
                from rl.policy import InterviewPolicy
                state_dim = len(TOPICS) * 2 + 2
                self.rl_policy = InterviewPolicy(state_dim, len(TOPICS))
                self.rl_policy.load_state_dict(torch.load(rl_path))
                self.rl_policy.eval()
                self.use_rl_policy = True
            else:
                print("[V2] RL Policy model not found, falling back to V1 static topic selection.")
        
        # Load question bank
        with open(os.path.join(DATA_DIR, 'question_bank.json'), 'r') as f:
            self.question_bank = json.load(f)['questions']
            
        self.current_q_data = None
            
    def get_embedding(self, tokens):
        if self.use_glove and self.glove_dict is not None:
            return get_glove_embedding(tokens, self.glove_dict, self.embed_model)
        else:
            return self.embed_model.get_text_embedding(tokens)
            
    def start_session(self, resume_path, skills_path):
        with open(resume_path, 'r') as f:
            resume = f.read()
        with open(skills_path, 'r') as f:
            skills = f.read()
            
        profile = resume + " " + skills
        self.state.candidate_profile = profile
        
        tokens = tokenize(profile)
        vec = self.get_embedding(tokens)
        
        with torch.no_grad():
            out = self.topic_classifier(torch.tensor(np.array([vec]), dtype=torch.float32))
            probs = torch.exp(out)[0].numpy()
            
        # Select top 3 topics (excluding behavioral which we force at end)
        topic_probs = {TOPICS[i]: probs[i] for i in range(len(TOPICS))}
        if 'behavioral' in topic_probs:
            del topic_probs['behavioral']
            
        sorted_topics = sorted(topic_probs.keys(), key=lambda k: topic_probs[k], reverse=True)
        selected = sorted_topics[:3]
        selected.append("behavioral") # always end with behavioral
        
        self.state.topics_to_cover = selected
        
        if self.use_rl_policy:
            for t in TOPICS:
                self.state.scores[t] = []
        else:
            for t in selected:
                self.state.scores[t] = []
            
        self._advance_topic()
        
    def _encode_state_for_rl(self) -> np.ndarray:
        one_hot = [1.0 if t in self.state.topics_covered else 0.0 for t in TOPICS]
        avg_scores = []
        for t in TOPICS:
            if t in self.state.scores and self.state.scores[t]:
                avg_scores.append(sum(self.state.scores[t]) / len(self.state.scores[t]))
            else:
                avg_scores.append(0.0)
        q_asked = len(self.state.questions_asked) / 15.0
        state = one_hot + avg_scores + [self.state.consecutive_weak_answers / 5.0, q_asked]
        return np.array(state, dtype=np.float32)

    def _advance_topic(self):
        if self.state.current_topic:
            self.state.topics_covered.append(self.state.current_topic)
            
        if self.use_rl_policy:
            if len(self.state.topics_covered) >= len(TOPICS) or len(self.state.questions_asked) >= 15:
                self.state.current_topic = None
                return False
                
            state_vec = self._encode_state_for_rl()
            with torch.no_grad():
                probs = self.rl_policy(torch.tensor(state_vec))
                
            probs = probs.numpy()
            for i, t in enumerate(TOPICS):
                if t in self.state.topics_covered:
                    probs[i] = 0.0
                    
            if probs.sum() == 0:
                self.state.current_topic = None
                return False
                
            probs = probs / probs.sum()
            next_topic_idx = np.random.choice(len(TOPICS), p=probs)
            self.state.current_topic = TOPICS[next_topic_idx]
            self.state.consecutive_weak_answers = 0
            if self.state.current_topic not in self.state.scores:
                self.state.scores[self.state.current_topic] = []
            return True
        else:
            if not self.state.topics_to_cover:
                self.state.current_topic = None
                return False
                
            self.state.current_topic = self.state.topics_to_cover.pop(0)
            self.state.consecutive_weak_answers = 0
            return True

    def next_question(self):
        if not self.state.current_topic:
            return None
            
        topic = self.state.current_topic
        
        if hasattr(self, 'queued_followup') and self.queued_followup:
            q_text = self.queued_followup
            self.queued_followup = None
            fake_q_data = {'id': 'gpt2_followup_' + str(len(self.state.questions_asked)), 'topic': topic, 'question': q_text}
            self.current_q_data = fake_q_data
            self.state.questions_asked.append(fake_q_data)
            return q_text
            
        # Check topic switch conditions
        topic = self.state.current_topic
        if len(self.state.scores[topic]) >= MAX_QUESTIONS_PER_TOPIC:
            if not self._advance_topic(): return None
            topic = self.state.current_topic
            
        if self.state.consecutive_weak_answers >= CONSECUTIVE_WEAK_LIMIT:
            if not self._advance_topic(): return None
            topic = self.state.current_topic
            
        # Special behavioral rule: at least 2 behavioral
        if topic == 'behavioral' and len(self.state.scores[topic]) >= 2:
            self.state.current_topic = None
            return None
            
        # Get candidates for current topic not already asked
        asked_ids = [q['id'] for q in self.state.questions_asked]
        candidates = [q for q in self.question_bank if q['topic'] == topic and q['id'] not in asked_ids]
        
        if not candidates:
            if not self._advance_topic(): return None
            return self.next_question()
            
        # RNN sequence context: use last 3 asked questions embeddings as context
        context_vecs = []
        for q in self.state.questions_asked[-3:]:
            vec = self.get_embedding(tokenize(q['question']))
            context_vecs.append(vec)
            
        if not context_vecs:
             # empty context
             context_vecs = [np.zeros(self.active_embed_dim, dtype=np.float32)]
             
        ctx_tensor = torch.tensor(np.array([context_vecs]), dtype=torch.float32)  # (1, seq_len, embed_dim)

        # Difficulty scalar for current topic
        diff_scalar = self.get_difficulty_for_topic(topic)
        diff_map    = {'Easy': 0.0, 'Medium': 0.5, 'Hard': 1.0}
        diff_val    = diff_map.get(diff_scalar, 0.0)

        best_q     = None
        best_score = -1

        for cand in candidates:
            cand_vec    = self.get_embedding(tokenize(cand['question']))
            cand_tensor = torch.tensor(np.array([cand_vec]), dtype=torch.float32)
            diff_tensor = torch.tensor([[diff_val]], dtype=torch.float32)

            with torch.no_grad():
                score = self.question_ranker(ctx_tensor, cand_tensor, diff_tensor).item()

            if score > best_score:
                best_score = score
                best_q     = cand
                
        self.current_q_data = best_q
        self.state.questions_asked.append(best_q)
        return best_q['question']
        
    def process_answer(self, text):
        if not self.current_q_data:
            return 0.0
            
        if self.use_distilbert_scorer:
            q_text = self.current_q_data['question']
            q_vec = self.distilbert_encoder.encode(q_text)
            a_vec = self.distilbert_encoder.encode(text)
            
            q_tensor = torch.tensor(np.array([q_vec]), dtype=torch.float32)
            a_tensor = torch.tensor(np.array([a_vec]), dtype=torch.float32)
            
            with torch.no_grad():
                score = self.distilbert_scorer(q_tensor, a_tensor).item()
        else:
            q_tokens = tokenize(self.current_q_data['question'])
            a_tokens = tokenize(text)
            
            q_vec = self.get_embedding(q_tokens)
            a_vec = self.get_embedding(a_tokens)
            
            q_tensor = torch.tensor(np.array([q_vec]), dtype=torch.float32)
            a_tensor = torch.tensor(np.array([a_vec]), dtype=torch.float32)
            
            with torch.no_grad():
                score = self.answer_scorer(q_tensor, a_tensor).item()
            
        topic = self.current_q_data['topic']
        self.state.scores[topic].append(score)

        if self.use_gpt2 and GPT2_SCORE_THRESHOLD_LOW <= score <= GPT2_SCORE_THRESHOLD_HIGH:
            context  = self.current_q_data['question']
            followup = self.gpt2_generator.generate_followup(context, text, topic)
            self.queued_followup = followup

        if score < WEAK_SCORE_THRESHOLD:
            self.state.consecutive_weak_answers += 1
        else:
            self.state.consecutive_weak_answers = 0

        # Log to SQLite
        if _DB_AVAILABLE:
            try:
                difficulty = self.get_difficulty_for_topic(topic)
                save_question_log(
                    session_id=self.session_id,
                    topic=topic,
                    difficulty=difficulty,
                    question=self.current_q_data.get('question', ''),
                    answer=text,
                    score=score,
                    time_taken=0
                )
            except Exception:
                pass

        return score

    # ── V2 helper methods ─────────────────────────────────────────────────────
    def get_difficulty_for_topic(self, topic: str) -> str:
        """Return adaptive difficulty level based on past scores for the topic."""
        scores = self.state.scores.get(topic, [])
        if not scores:
            return 'Easy'
        avg = sum(scores) / len(scores)
        if avg > STRONG_SCORE_THRESHOLD:
            return 'Hard'
        elif avg > WEAK_SCORE_THRESHOLD:
            return 'Medium'
        return 'Easy'

    def finalize_session(self, candidate_name: str = 'Candidate'):
        """Save the completed session summary to SQLite."""
        if not _DB_AVAILABLE:
            return
        all_scores = [s for scores in self.state.scores.values() for s in scores]
        overall    = sum(all_scores) / len(all_scores) if all_scores else 0.0
        # Compute readiness label
        if overall >= STRONG_SCORE_THRESHOLD:
            level = 'Strong'
        elif overall >= WEAK_SCORE_THRESHOLD:
            level = 'Moderate'
        else:
            level = 'Needs Work'
        try:
            save_session(self.session_id, candidate_name, level, overall)
        except Exception:
            pass
