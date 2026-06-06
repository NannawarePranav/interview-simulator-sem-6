# AI Mock Interviewer — Detailed Breakdown

Here is a detailed, 8-point breakdown of the AI Mock Interviewer codebase and project methodology:

### 1. Explaining the Problem Statement
Traditional mock interviews are static, require expensive human intervention, and often lack objective, data-driven evaluation. The goal of this project is to build an autonomous, multi-modal AI Interviewer capable of:
*   Dynamically extracting technical skills from a candidate's uploaded resume.
*   Autonomously traversing relevant technical topics based on the candidate's real-time performance.
*   Evaluating unstructured, free-form text answers instantly and objectively.
*   Asking targeted, context-aware follow-up questions when a candidate struggles.
*   Ensuring exam integrity via headless webcam proctoring to simulate a strict, remote interview environment.

### 2. Model Architecture / Summary
The architecture is a massive ensemble pipeline combining custom PyTorch models, HuggingFace Transformers, and traditional ML algorithms:
*   **NN1 (Embeddings):** A custom NumPy Skip-Gram Word2Vec model augmented with a native, memory-cached **GloVe 100-dim** fallback mechanism.
*   **NN2 (Topic Classifier):** A PyTorch Feedforward network (MLP) mapping text inputs to 6 predefined technical domains.
*   **NN3 (Question Ranker):** A PyTorch **GRU (Gated Recurrent Unit)** that factors in a real-time *difficulty scalar* to rank the best next question.
*   **NN4 (Answer Scorer):** A **DistilBERT-based** contextual evaluator. It uses a frozen `distilbert-base-uncased` to extract a 768-dim `[CLS]` token, feeding it into a PyTorch MLP to output a continuous score from `0.0` to `1.0`.
*   **GPT-2 Follow-up Generator:** A **GPT-2 Small** Causal Language Model fine-tuned via Linear Probing (only the LM head is trained) to generate context-aware questions.
*   **RL Policy:** A PyTorch network trained via the **REINFORCE** algorithm to act as a dynamic state machine, picking the next interview topic based on candidate performance.
*   **Proctor:** Standalone **OpenCV + MediaPipe** pipeline calculating head pose/gaze angles.

### 3. Model Evaluation
Before deployment, the system validates itself using a 7-step automated test suite (`scripts/run_tests.py`):
*   **Similarity Checks:** Cosine similarity assertions on the GloVe/NN1 clusters.
*   **Inference Bounds:** Ensures the models compile and output expected ranges (e.g., GRU outputs probabilities summing correctly, Answer Scorer bounds between `0.0` and `1.0`).
*   **Sanity Checks:** Tests specific "Strong", "Weak", and "Off-topic" text inputs through NN4 to guarantee the model didn't collapse to the mean.
*   **E2E Headless Run:** The entire `InterviewController` is simulated sequentially in a headless environment to prevent integration regressions.

### 4. Dataset
The training pipeline ingests a massive, diverse corpus to ensure robust language understanding:
*   **Core Embeddings & Scorer:** SNLI, SQuAD, MSR Paraphrase Corpus, Quora Question Pairs, STS Benchmark, and SemEval STS.
*   **Resume Domain:** Standard resume categorization datasets mapped to internal taxonomies.
*   **GPT-2 Follow-ups:** A custom-authored JSONL dataset containing 60+ specific technical scenarios across Junior/Mid/Senior levels.
*   **Pre-trained Weights:** GloVe 6B.100d, DistilBERT Base, and GPT-2 Base.

### 5. Splitting
*   **Standard Split:** Most models utilize an 80/20 or 90/10 Train/Validation split (via `scikit-learn`).
*   **Cross-Validation:** Because NN4 (Answer Scorer) combines disparate datasets (Quora vs. SNLI), it uses **5-Fold Cross-Validation** during training. This ensures the model generalizes well to unseen phrasing and prevents it from simply overfitting to the syntax of a specific dataset.

### 6. Preprocessing
*   **Document Ingestion:** Resumes (PDFs) are parsed via `PyMuPDF`. A dynamic regex taxonomy (built from `Interview_Questions.xlsx` + a modern stack list) extracts technical skills.
*   **Text Normalization:** All datasets are aggressively normalized—HTML tags stripped, text lowercased, punctuation removed, and whitespace trimmed.
*   **Tokenization & Vectors:** Standard strings are passed into HuggingFace `AutoTokenizer` (for DistilBERT/GPT-2) with `truncation=True` and `padding="max_length"`. For custom networks, tokens are dictionary-mapped and mean-pooled using the lazily loaded GloVe cache.

### 7. Quantitative Analysis
The system actively quantifies qualitative human interaction:
*   **Performance Tracking:** The `InterviewController` computes a rolling average of the candidate's scores, translating it into a *difficulty scalar* which mathematically alters the internal state of the GRU (NN3).
*   **RL State Vectorization:** The interview state is quantified into a fixed-size float array: `[topics_covered_onehot, avg_score_per_topic, consecutive_weak_count, questions_asked]`.
*   **Proctor Metrics:** Time tracking is quantified natively; if a candidate looks off-screen for a continuous threshold (`> 2.0 seconds`), a violation integer increments.

### 8. Output
The system generates multi-layered outputs:
*   **UI Updates:** The Flask backend communicates with the Glassmorphism SPA (or Streamlit UI), updating chat messages, live progress bars, and proctor warning banners in real-time.
*   **Database Persistency:** The `session_db.py` (SQLite) logs every single question, answer, score, and difficulty scalar.
*   **Final Report:** Once the session terminates, the system generates a markdown report (`stats.md`) and a UI dashboard detailing:
    *   An overall Readiness Badge (Senior-ready, Mid-level, or Junior-ready).
    *   A breakdown of exact scores per technical topic.
    *   An integrity/proctoring verification note.
