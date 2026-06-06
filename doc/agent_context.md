# AI Mock Interviewer — Comprehensive Context for AI Agents

**Target Audience:** AI/Agentic Assistants joining the workspace to debug, extend, or maintain the project.
**Project Root:** `e:\mini project 1\ai_interviewer\v1\`

This document provides a highly detailed mapping of the AI Mock Interviewer codebase. The system evolved from a V1 static PyTorch model pipeline into a V2 multi-modal architecture featuring Transformers (DistilBERT, GPT-2), Reinforcement Learning, and massive data ingestion.

---

## 1. Directory Structure & Core Modules

*   `config.py`: The single source of truth for hyper-parameters, constants, file paths (`DATASET_ROOT`), and **V2 Feature Flags** (crucial for routing logic).
*   `run.py`: The Flask server entry point. Serves the web UI and REST API. Boots up the background Proctoring daemon.
*   `app_streamlit.py`: An alternative, independent Streamlit frontend supporting all backend logic via session state.
*   `main.py`: A CLI-based fallback interface using the `rich` library.
*   `interview/`: The core business logic.
    *   `controller.py`: Contains `InterviewController`. The absolute brain of the app. It wires together all ML models, tracks `SessionState`, and executes the interview state machine.
    *   `session.py`: Defines the `SessionState` dataclass to hold topics, scores, and question logs.
    *   `report.py`: Generates the final markdown readiness report.
*   `models/`: All Neural Network architectures and generation classes.
    *   `embedding_model.py`: (NN1) Pure NumPy Skip-Gram Word2Vec with a native, in-memory 100-dim GloVe fallback dictionary.
    *   `topic_classifier.py`: (NN2) PyTorch Feedforward network.
    *   `question_ranker.py`: (NN3) PyTorch GRU network.
    *   `answer_scorer.py`: (NN4) Holds both the V1 `SiameseAnswerScorer` and the V2 `DistilBERTAnswerScorer`.
    *   `question_generator.py`: HuggingFace `AutoModelForCausalLM` wrapper for GPT-2.
*   `embeddings/`:
    *   `distilbert_encoder.py`: Wraps `distilbert-base-uncased` to extract 768-dim `[CLS]` embeddings, utilizing an LRU/dict cache to prevent redundant forward passes.
    *   `glove_loader.py`: Handles pulling the massive GloVe text file directly into memory.
*   `rl/`:
    *   `policy.py`: PyTorch REINFORCE policy network (`InterviewPolicy`) predicting topic probabilities.
    *   `interview_env.py` & `candidate_profiles.py`: Simulation environment for training the RL policy.
*   `data/`:
    *   `resume_parser.py`: Uses PyMuPDF and regex to extract skills from resumes. Matches against a taxonomy dynamically loaded from `Interview_Questions.xlsx`.
    *   `session_db.py`: Zero-config SQLite database (`sessions.db`) logger.
*   `train/`: Contains all standalone training scripts (`train_embeddings.py`, `train_classifier.py`, `train_ranker.py`, `train_scorer.py`, `finetune_gpt2.py`, etc.).
*   `exam_proctor.py`: Standalone OpenCV + MediaPipe module tracking eye gaze and head pose. Runs headlessly in a daemon thread.

---

## 2. Global V2 Configuration Flags

The system is strictly backward-compatible. All new architectural features are guarded by flags in `config.py`. If a V2 model file is missing, the system catches the exception and falls back to V1 silently.
*   `USE_GLOVE (bool)`: If true, overrides NN1's 64-dim output to use 100-dim GloVe arrays.
*   `USE_DISTILBERT_SCORER (bool)`: If true, routes answer scoring through DistilBERT rather than the Siamese MLP.
*   `USE_GPT2_FOLLOWUP (bool)`: If true, intercepts marginal answer scores to generate dynamic follow-up questions instead of pulling static questions.
*   `USE_RL_POLICY (bool)`: If true, uses the trained PyTorch RL policy network to choose the next interview topic instead of statically walking down the topic list.

---

## 3. Machine Learning Pipeline (NN1 to NN4 + Transformers)

### **NN1: Embeddings (`embedding_model.py`)**
Text is tokenized, stripped of HTML, and lowercased. 
*   **Primary:** Custom NumPy Skip-Gram dictionary.
*   **Fallback:** If a token is OOV (Out Of Vocabulary), it hits the `_glove_cache`. Due to performance optimizations, the 400MB GloVe file is loaded entirely into RAM *once* on initialization.

### **NN2: Topic Classifier (`topic_classifier.py`)**
Takes the mean-pooled vector of a text and passes it through an MLP (`Linear → ReLU → Linear → LogSoftmax`). It classifies strings into 6 distinct topics (e.g., python, databases_sql, system_design).

### **NN3: Question Ranker (`question_ranker.py`)**
A PyTorch GRU. 
*   **Inputs:** It receives a sequence of context (the topic embedding), a candidate question embedding, and crucially, a **Difficulty Scalar** (0.0 to 1.0). 
*   **Logic:** The `InterviewController` calculates the running average of the candidate's score for the current topic. If the candidate is doing well, the difficulty scalar increases, causing the GRU to rank harder questions higher.

### **NN4: Answer Scorer (`answer_scorer.py`)**
Grades the candidate's textual answer against the question context, returning a float from `0.0` to `1.0`.
*   **V1 (Siamese):** Uses Dropout and an MLP. Trained via 5-fold Cross-Validation on 30,000+ samples from 8 combined datasets (SQuAD, SNLI, Quora, STS, etc.).
*   **V2 (DistilBERT):** Contextual embeddings. The texts are passed through a frozen `distilbert-base-uncased` model to extract the `[CLS]` token. This 768-dim tensor is then passed into a trained PyTorch MLP scorer.

### **GPT-2 Dynamic Follow-ups (`question_generator.py`)**
A fine-tuned GPT-2 Small language model.
*   **Trigger:** If NN4 scores an answer between `0.30` and `0.65` (mediocre), the Controller skips NN3 and calls GPT-2.
*   **Prompting:** The model is given the Topic, Context, and the Candidate's exact answer. It uses `top_p=0.9` sampling to generate a targeted follow-up question to probe their knowledge deeper.

### **RL Policy Network (`rl/policy.py`)**
A Reinforcement Learning model trained via REINFORCE.
*   **State Vector:** Encodes one-hot arrays of topics covered, average score per topic, normalized question counts, and consecutive weak answers.
*   **Action:** Predicts the probability distribution over the 6 topics to determine which topic the candidate should face next.

---

## 4. Execution Workflow (Step-by-Step)

When a candidate uses the app, the state machine in `InterviewController` flows as follows:

1.  **Ingestion:** `/api/upload` is hit. `PyMuPDF` reads the resume. `resume_parser` uses a regex taxonomy to extract technical skills (e.g., Python, Docker).
2.  **Topic Assignment:** Skills are mapped to internal topics. `SessionState` initializes score arrays `[]` for each topic.
3.  **Topic Selection:** The RL Policy (if enabled) looks at current scores and chooses the best topic to test next.
4.  **Question Generation:** NN3 (GRU) evaluates the question bank for that topic, factoring in the *difficulty scalar*, and presents the top-ranked question.
5.  **Answer Processing:** 
    *   The candidate submits text.
    *   NN4 (DistilBERT or Siamese) scores it.
    *   The score is saved to SQLite via `session_db.py`.
6.  **Follow-up Intervention:** If the score is marginal (`0.30 - 0.65`), GPT-2 generates a follow-up.
7.  **Proctoring:** Concurrently, `exam_proctor.py` checks for webcam face visibility. If the user looks away for >2s, a violation is logged to the API, displaying a warning banner on the UI.
8.  **Finalization:** Once `MAX_QUESTIONS_PER_TOPIC` is hit for all topics, `finalize_session()` averages the scores, appends a proctor integrity note, generates a Markdown readiness report, and closes the session.

---

## 5. Development Notes for Agents
*   **Data Constraints:** Reading raw Parquet or 400MB GloVe files sequentially is notoriously slow on CPUs. If you write new training scripts, *always* utilize randomized row sampling (e.g., `df.sample(n=2000)`) and full-memory dictionary caching.
*   **Error Handling:** The `run_tests.py` script ensures architectural bounds are respected. Do not break backward compatibility. If you add a new model, implement it as a gracefully failing `try/except` block inside `controller.py` that falls back to the V1 equivalent.
