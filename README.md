<p align="center">
  <h1 align="center">🎯 AI Mock Interviewer</h1>
  <p align="center">
    <strong>An offline-first, AI-powered technical interview simulator with real-time CV-based proctoring.</strong>
  </p>
  <p align="center">
    <a href="#features">Features</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#installation">Installation</a> •
    <a href="#usage">Usage</a> •
    <a href="#project-structure">Structure</a> •
    <a href="#models">Models</a> •
    <a href="#license">License</a>
  </p>
</p>

---

## 📌 Overview

AI Mock Interviewer is a locally hosted technical interview simulator that evaluates candidates using an ensemble of specialized neural networks — without relying on expensive cloud-based LLM APIs. The system parses your resume, classifies relevant topics, asks difficulty-adaptive questions, scores answers in real-time, generates follow-up questions, and monitors exam integrity through computer-vision-based proctoring.

---

## ✨ Features

| Category | Feature |
|---|---|
| **Resume Parsing** | Automatic skill extraction from PDF/TXT resumes using PyMuPDF |
| **Topic Classification** | Neural network–based mapping of resume skills to 6 interview domains |
| **Adaptive Questioning** | GRU-based question ranker adjusts difficulty based on performance |
| **Answer Scoring** | Siamese network + DistilBERT transformer for semantic answer evaluation |
| **Follow-up Generation** | Fine-tuned GPT-2 generates targeted follow-ups for marginal answers |
| **RL Topic Selection** | REINFORCE policy gradient agent optimizes interview topic flow |
| **CV Proctoring** | Real-time gaze tracking, face detection, phone detection, and identity verification |
| **Multiple UIs** | Flask web app (glassmorphism SPA), Streamlit dashboard, and Rich CLI |
| **Session Persistence** | SQLite-backed session history with detailed QA logs |
| **Report Generation** | Comprehensive interview reports with readiness assessment |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Presentation Layer                   │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Flask SPA│  │  Streamlit   │  │   Rich CLI    │  │
│  │ (run.py) │  │(app_stream.) │  │  (main.py)    │  │
│  └────┬─────┘  └──────┬───────┘  └───────┬───────┘  │
├───────┼────────────────┼─────────────────┼──────────┤
│       └────────────────┼─────────────────┘          │
│              ┌─────────▼─────────┐                  │
│              │ InterviewController│                  │
│              │ + SessionState     │                  │
│              └─────────┬─────────┘                  │
├────────────────────────┼────────────────────────────┤
│          Model & Inference Layer                     │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────────┐  │
│  │  NN1   │ │  NN2   │ │  NN3   │ │    NN4       │  │
│  │Embed.  │ │Classif.│ │Ranker  │ │Scorer+DBERT  │  │
│  └────────┘ └────────┘ └────────┘ └──────────────┘  │
│  ┌────────────────┐  ┌───────────────────────────┐   │
│  │  GPT-2 Followup│  │  RL REINFORCE Policy      │   │
│  └────────────────┘  └───────────────────────────┘   │
├──────────────────────────────────────────────────────┤
│     CV Proctoring (MediaPipe + YOLOv8 + face-api)    │
└──────────────────────────────────────────────────────┘
```

---

## 🛠️ Installation

### Prerequisites

- **Python** 3.9+ 
- **pip** (Python package manager)
- **Webcam** (required for proctoring features)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/NannawarePranav/interview-simulator-sem-6.git
cd interview-simulator-sem-6

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# 3. Install dependencies
pip install -r requirements_v2.txt

# 4. Train / download model weights (see Models section below)
```

### Model Weights

The trained model weights (`.pt`, `.npy`, `vocab.json`) are **not included** in the repository due to size constraints. To generate them:

```bash
# Train all models from scratch
python train/train_embeddings.py
python train/train_classifier.py
python train/train_ranker.py
python train/train_scorer.py
python train/finetune_gpt2.py
```

For YOLOv8 (proctoring), download `yolov8n.pt` from [Ultralytics](https://github.com/ultralytics/ultralytics) and place it in `cv-procter/`.

---

## 🚀 Usage

### Option 1: Flask Web App (Recommended)

```bash
python run.py
```
Open **http://localhost:5000** in your browser. Upload your resume, select skills, and start the interview. The glassmorphism SPA includes real-time proctoring, progress tracking, and score visualization.

### Option 2: Streamlit Dashboard

```bash
streamlit run app_streamlit.py
```
Access the Streamlit dashboard with live metrics, V2 model toggles, and a chat-style interview interface.

### Option 3: CLI Interface

```bash
python main.py --resume data/raw/resume.txt --skills data/raw/skills.txt
```
Terminal-based interview with Rich panels, progress bars, and timed input.

### Standalone Proctor

```bash
python exam_proctor.py --camera 0 --threshold 2.0
```
Run the gaze-tracking proctor independently for exam monitoring.

---

## 📁 Project Structure

```
interview-simulator-sem-6/
├── main.py                  # CLI entry point (Rich terminal UI)
├── run.py                   # Flask server + REST API + CV proctoring daemon
├── app_streamlit.py         # Streamlit web dashboard
├── config.py                # Hyperparameters, paths, feature flags
├── exam_proctor.py          # Standalone MediaPipe gaze tracker
├── requirements.txt         # Core dependencies (V1)
├── requirements_v2.txt      # Full dependencies (V2 with transformers)
│
├── interview/               # Core interview engine
│   ├── controller.py        # Interview orchestrator & state machine
│   ├── session.py           # Session state data class
│   └── report.py            # Report generation
│
├── models/                  # Neural network model definitions
│   ├── embedding_model.py   # NN1: Skip-Gram Word2Vec (NumPy)
│   ├── topic_classifier.py  # NN2: Feedforward topic classifier
│   ├── question_ranker.py   # NN3: GRU-based question ranker
│   ├── question_generator.py# GPT-2 follow-up question generator
│   ├── answer_scorer.py     # NN4: Siamese answer scorer
│   └── saved/               # Trained model weights (gitignored)
│
├── embeddings/              # Embedding utilities
│   ├── glove_loader.py      # GloVe vector loader
│   └── distilbert_encoder.py# DistilBERT CLS token encoder
│
├── train/                   # Training scripts
│   ├── train_embeddings.py  # Train custom Word2Vec embeddings
│   ├── train_classifier.py  # Train topic classifier
│   ├── train_ranker.py      # Train question ranker
│   ├── train_scorer.py      # Train answer scorer
│   ├── finetune_gpt2.py     # Fine-tune GPT-2 for follow-ups
│   └── evaluate_models.py   # Model evaluation & benchmarks
│
├── rl/                      # Reinforcement learning module
│   ├── policy.py            # REINFORCE policy gradient agent
│   ├── interview_env.py     # RL environment simulation
│   └── candidate_profiles.py# Synthetic candidate profiles
│
├── data/                    # Data files & utilities
│   ├── question_bank.json   # Curated question bank (6 topics)
│   ├── resume_parser.py     # PDF resume → text + skills extractor
│   ├── session_db.py        # SQLite session persistence
│   └── raw/                 # Uploaded resumes (gitignored)
│
├── cv-procter/              # Computer vision proctoring module
│   ├── face_module.py       # MediaPipe face mesh + gaze + identity
│   ├── object_module.py     # YOLOv8 phone & person detection
│   ├── scoring.py           # Violation scoring manager
│   └── utils.py             # CV utilities
│
├── frontend/                # Flask SPA frontend
│   ├── index.html           # Glassmorphism dark-mode UI
│   ├── style.css            # CSS styles
│   └── app.js               # Frontend JavaScript + face-api.js
│
├── scripts/                 # Testing & utilities
│   └── run_tests.py         # 7-test verification suite
│
├── doc/                     # Documentation
│   ├── overview.md          # Project overview
│   ├── detailed_breakdown.md# Detailed technical breakdown
│   └── v2.md                # V2 architecture documentation
│
└── reports/                 # Generated interview reports
```

---

## 🧠 Models

| ID | Model | Architecture | Purpose |
|---|---|---|---|
| NN1 | Word Embeddings | Custom Skip-Gram (NumPy) / GloVe fallback | Token vector representations |
| NN2 | Topic Classifier | PyTorch MLP (128 hidden, LogSoftmax) | Resume → 6 topic domains |
| NN3 | Question Ranker | PyTorch GRU + difficulty-aware scoring | Context-adaptive question selection |
| NN4 | Answer Scorer | Siamese Net + DistilBERT [CLS] encoder | Semantic answer evaluation |
| GPT-2 | Follow-up Gen | HuggingFace GPT-2 Small (fine-tuned) | Targeted follow-up questions |
| RL | Topic Policy | REINFORCE Policy Gradient MLP | Optimal topic traversal |

---

## 🧪 Testing

Run the full verification suite:

```bash
python scripts/run_tests.py
```

Tests cover: embedding similarity, classifier shape validation, ranker bounds, scorer grading, E2E session loop, Flask API smoke tests, and resume parser extraction.

---

## ⚙️ Configuration

All hyperparameters and feature flags are centralized in [`config.py`](config.py):

| Flag | Default | Description |
|---|---|---|
| `USE_GLOVE` | `False` | Use GloVe vectors instead of custom embeddings |
| `USE_DISTILBERT_SCORER` | `True` | Use DistilBERT for answer scoring |
| `USE_GPT2_FOLLOWUP` | `True` | Enable GPT-2 follow-up generation |
| `USE_RL_POLICY` | `True` | Enable RL-based topic selection |
| `MAX_QUESTIONS_PER_TOPIC` | `3` | Questions per topic before moving on |
| `ANSWER_TIMEOUT_SECONDS` | `90` | Time limit per question |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Pranav Nannaware**  
GitHub: [@NannawarePranav](https://github.com/NannawarePranav)
