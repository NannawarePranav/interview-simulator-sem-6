# AI Mock Interviewer — Project Overview

## 1. Introduction
The AI Mock Interviewer is an intelligent, multi-modal system designed to simulate technical interviews. It dynamically extracts skills from candidate resumes, generates context-aware questions, evaluates answers in real-time using deep learning models, and ensures exam integrity via a webcam proctoring daemon. The system has evolved from a static V1 prototype to a robust V2 architecture incorporating Transformers, Reinforcement Learning, and massive dataset ingestion.

## 2. Tech Stack
*   **Backend:** Python, Flask (REST API)
*   **Frontend:** Vanilla HTML/CSS/JS (Glassmorphism UI), Streamlit (V2 Alternative UI)
*   **Machine Learning / Deep Learning:** PyTorch, HuggingFace Transformers, NumPy
*   **Data Processing:** Pandas, PyMuPDF (PDF parsing)
*   **Persistence:** SQLite (`sessions.db`)
*   **Proctoring:** OpenCV, MediaPipe (Head-pose & gaze tracking)

## 3. System Architecture
The application is built on a decoupled, modular architecture:
1.  **Presentation Layer:** Handles user interaction via the Flask Web SPA or Streamlit UI. Communicates with the backend asynchronously.
2.  **Controller Layer (`InterviewController`):** The central brain of the system. It orchestrates the interview flow, manages state transitions, tracks per-topic scores, and dictates whether to use static selection or RL-based topic progression.
3.  **Model Inference Layer:** Encapsulates the PyTorch and HuggingFace models. All models are invoked lazily to conserve RAM/VRAM.
4.  **Persistence & Integrity Layer:** The background `exam_proctor` daemon continuously monitors webcam feeds for violations, while `session_db.py` logs question/answer pairs and final reports to SQLite.

## 4. Models Used
The system relies on a suite of specialized neural networks, each handling a specific piece of the interview puzzle:

*   **NN1 (Embeddings):** A custom Pure NumPy Skip-Gram Word2Vec model (64-dim). In V2, this is augmented with a native, in-memory **GloVe (100-dim)** fallback mechanism for Out-of-Vocabulary (OOV) tokens.
*   **NN2 (Topic Classifier):** A PyTorch Feedforward network (Linear → ReLU → Linear → LogSoftmax). It maps raw resume text and questions into a predefined internal taxonomy (e.g., Python, Machine Learning, System Design).
*   **NN3 (Question Ranker):** A PyTorch **GRU (Gated Recurrent Unit)** network. It takes the topic context, the candidate question, and a dynamically computed *difficulty scalar* (+1 feature dim) to select the most appropriate next question from the bank.
*   **NN4 (Answer Scorer):** 
    *   *V1:* A Siamese MLP with Dropout, trained on 30,000+ QA pairs across 8 datasets (SQuAD, SNLI, Quora, etc.).
    *   *V2:* A **DistilBERT-based Scorer** (`distilbert-base-uncased`) that extracts the `[CLS]` contextual embedding to better handle nuances like negation and complex phrasing before feeding it into an MLP.
*   **GPT-2 (Follow-up Generator):** A fine-tuned **GPT-2 Small** model. Triggered only when a candidate gives a mediocre answer (score between 0.30 and 0.65), it generates a targeted, context-aware follow-up question.
*   **RL Policy (Interview Flow):** A PyTorch REINFORCE policy network that adaptively selects the *next* interview topic based on a state vector of the candidate's performance so far (replacing static list traversal).

## 5. Pre-Processing Pipeline
*   **Resume Ingestion:** `PyMuPDF` extracts raw text from uploaded PDFs. The `resume_parser.py` module uses a dynamically generated regex taxonomy (loaded from `Interview_Questions.xlsx` combined with a hardcoded list of modern tools like FastAPI, Docker, etc.) to extract technical skills.
*   **Dataset Normalization:** All training data (Parquet, CSV, TSV) is heavily normalized: lowercased, HTML-stripped, and whitespace-trimmed.
*   **Tokenization & Embedding:** Text is split into tokens. Known words hit the NN1 dictionary, while OOV words hit the loaded GloVe dictionary in `O(1)` time. Sentences are represented via mean-pooling (for NN1-NN3) or contextual tokenization (for DistilBERT).

## 6. Interview Workflow
1.  **Setup:** The candidate uploads a resume. The backend extracts technical skills, assigns topics, and initializes the `SessionState`.
2.  **Questioning Phase:**
    *   The `InterviewController` selects a topic (via RL Policy or static fallback).
    *   The system calculates the candidate's running average score for that topic to determine the *difficulty scalar*.
    *   NN3 ranks the question bank and presents the best question.
3.  **Answering Phase:**
    *   The candidate submits an answer (with a 90-second timeout).
    *   NN4 (or DistilBERT) scores the answer from `0.0` to `1.0`.
    *   If the answer is marginal, the GPT-2 model generates a follow-up question to dig deeper.
4.  **Proctoring:** Throughout the session, the background proctor polls webcam data. If the candidate looks away for >2 seconds continuously, a violation is logged and displayed on the UI.
5.  **Completion:** Once all topics are covered (or the maximum questions are reached), a final `stats.md` report is generated, logging the per-topic averages, proctoring violations, and an overall readiness verdict to the SQLite database.
