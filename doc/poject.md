# AI Mock Interviewer - Project Documentation

## 1. Overview
The **AI Mock Interviewer** is a locally hosted, end-to-end simulated job interview platform. It evaluates a candidate's resume and specified skills, categorizes them into relevant technical topics, and orchestrates an interactive technical interview. The system generates questions contextually, scores answers in real-time, and concludes the session by compiling a comprehensive readiness report indicating whether the candidate is Junior, Mid-Level, or Senior-ready.

---

## 2. System Architecture
The application runs entirely offline, avoiding reliance on external LLM APIs. It accomplishes this through a pipeline of custom-trained neural networks working sequentially:
1. **Data Extraction**: PDF Resumes are parsed into text strings. Key technical terms are identified using a pattern-matching keyword bank.
2. **Classification**: The candidate's text profile is passed into an ML classifier to dictate which topics (e.g., Python, System Design, Data Structures) the interview will cover.
3. **Questioning**: An RNN-based context model selects the most relevant questions from a predefined dataset, ensuring an organic conversational flow.
4. **Scoring**: A siamese neural network compares the semantic meaning of the candidate's response against reference materials to produce an immediate 0.0 to 1.0 confidence score.
5. **Interface**: The engine can be run via a Terminal CLI or a modern Glassmorphism Web App.

---

## 3. Core Neural Networks & Performance Metrics
The system utilizes four dedicated neural network modules trained from scratch using **PyTorch** and **NumPy**. 

### NN1: Word Embeddings (NumPy)
- **Architecture**: A pure-NumPy implementation of the Skip-Gram Word2Vec model.
- **Function**: Converts tokenized text from resumes, questions, and answers into dense vector representations.
- **Vocab Size**: 529 unique tokens derived from the training corpus.

### NN2: Topic Classifier (PyTorch)
- **Architecture**: Feedforward Neural Network (Linear → ReLU → Linear → LogSoftmax).
- **Function**: Classifies the candidate's profile into an ordered list of relevant topics to cover during the interview.
- **Accuracy**: **84.67%** (Evaluated across the `topic_classifier_data.csv` dataset).

### NN3: Context-Aware Question Ranker (PyTorch)
- **Architecture**: Recurrent Neural Network (RNN).
- **Function**: Analyzes the historical context of the interview (the last 3 asked questions) alongside candidate questions to rank and select the best, non-repetitive follow-up question.
- **Accuracy**: **55.56%** (BCE Accuracy tracking whether the correct contextual question was highly ranked).

### NN4: Siamese Answer Scorer (PyTorch)
- **Architecture**: Siamese Linear Encoders + Multi-Layer Perceptron (MLP) Scorer.
- **Function**: Evaluates the candidate's typed answer against the question's internal reference answer.
- **MSE (Mean Squared Error)**: **0.0019**
- **Margin Accuracy (within +/- 0.2)**: **100.00%** (Perfectly categorizes answers into Strong, Mediocre, and Weak thresholds).

---

## 4. Interaction Interfaces

### The Web Application (`run.py` & `frontend/`)
A fully-featured REST API backend built on **Flask** serving a premium, responsive Single-Page Application (SPA).
- **Frontend Stack**: Vanilla HTML, CSS (Glassmorphism, Dark Theme, Custom Animations), and JavaScript.
- **Features**: 
  - File upload gateway for PDF resumes.
  - Live progress timeline of topics.
  - Interactive chat interface with real-time 90-second countdown timers.
  - Instant scoring feedback (`Excellent`, `Mediocre`, `Weak`).
  - Report dashboard mapping out final readiness levels.

### The CLI Interface (`main.py`)
A fast, lightweight terminal interface built using the Python `rich` library.
- **Features**:
  - Requires pre-provided text files (`resume.txt`, `skills.txt`).
  - Live terminal progress banners.
  - Custom console input timeout handlers (gracefully penalizes the user if they take >90s to answer).
  - Supports `[skip]` and `[quit]` commands directly in the prompt.

---

## 5. Utilities & Extras

### Resume Parser (`data/resume_parser.py`)
Utilizes `PyMuPDF` to ingest raw PDF files (e.g., `Pranav Nannaware Resume.pdf`). It cleans erratic unicode characters, flattens the text into a readable stream, and runs an exhaustive scan against 40+ known tech stack keywords to automatically assemble a candidate's `skills.txt`.

### Exam Proctor (`exam_proctor.py`)
A standalone experimental module designed to enforce exam integrity. 
- **Tech Stack**: OpenCV (`cv2`) and Google MediaPipe (`mp.solutions.face_mesh`).
- **Function**: Tracks eye-iris geometry and aspect ratios to determine the user's exact gaze direction. If the user looks away from the screen for a configurable threshold (e.g., 2.0 seconds), the system triggers an alert/violation.
- **Status**: Currently unlinked from the main Web/CLI logic. Used purely for headless testing and gaze-tracking prototyping.

---

## 6. How to Run

**1. Launch the Web App (Recommended):**
```bash
python run.py
```
*Open `http://localhost:5000` in your browser.*

**2. Launch the CLI:**
```bash
python main.py --resume data\raw\resume.txt --skills data\raw\skills.txt
```

**3. Launch the standalone Exam Proctor Test:**
```bash
$env:TF_ENABLE_ONEDNN_OPTS="0"
python exam_proctor.py
```
