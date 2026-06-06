import os

# ── Core Hyperparameters ─────────────────────────────────────────────────────
EMBED_DIM               = 64
HIDDEN_DIM              = 128
LEARNING_RATE           = 0.001
MIN_FREQ                = 2

# ── Interview Session Rules ──────────────────────────────────────────────────
MAX_QUESTIONS_PER_TOPIC = 3
CONSECUTIVE_WEAK_LIMIT  = 2
ANSWER_TIMEOUT_SECONDS  = 90
WEAK_SCORE_THRESHOLD    = 0.4
STRONG_SCORE_THRESHOLD  = 0.7

# ── Exam Proctor ─────────────────────────────────────────────────────────────
PROCTOR_GAZE_THRESHOLD  = 2.0   # seconds before a gaze-away is counted as a violation
PROCTOR_VIOLATION_LIMIT = 10    # violations before warning banner in frontend

# ── Topics ───────────────────────────────────────────────────────────────────
TOPICS = [
    "python",
    "data_structures_algorithms",
    "machine_learning_basics",
    "databases_sql",
    "system_design",
    "behavioral"
]

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
MODELS_DIR   = os.path.join(BASE_DIR, "models", "saved")
REPORTS_DIR  = os.path.join(BASE_DIR, "reports")
DATASET_ROOT = os.path.join(BASE_DIR, "data-set-full")

# ── Model Save Paths ─────────────────────────────────────────────────────────
EMBEDDINGS_PATH       = os.path.join(MODELS_DIR, "embeddings.npy")
VOCAB_PATH            = os.path.join(MODELS_DIR, "vocab.json")
CLASSIFIER_PATH       = os.path.join(MODELS_DIR, "topic_classifier.pt")
RANKER_PATH           = os.path.join(MODELS_DIR, "question_ranker.pt")
SCORER_PATH           = os.path.join(MODELS_DIR, "answer_scorer.pt")

# ── GloVe Fallback ───────────────────────────────────────────────────────────
GLOVE_FALLBACK_PATH   = os.path.join(DATASET_ROOT, "glove.6B.100d.txt")
GLOVE_FALLBACK_DIM    = 100  # must match EMBED_DIM or be projected

# ── V2 Feature Flags ─────────────────────────────────────────────────────────
USE_GLOVE              = False
GLOVE_PATH             = "embeddings/wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt"
GLOVE_DIM              = 100
USE_DISTILBERT_SCORER  = True
USE_GPT2_FOLLOWUP      = True
GPT2_MODEL_PATH        = "models/saved/gpt2_finetuned"
GPT2_SCORE_THRESHOLD_LOW  = 0.30
GPT2_SCORE_THRESHOLD_HIGH = 0.65
USE_RL_POLICY          = True
