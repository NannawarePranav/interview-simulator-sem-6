# Graphify Analysis & Architectural Insights

Based on the `graphify` knowledge graph extraction, the AI Mock Interviewer codebase is extremely well-structured, containing **108 nodes**, **187 semantic edges**, and grouping into **10 distinct architectural communities**. 

Here is my breakdown of what the graph reveals about the project:

## 1. The Core "God Nodes" (Central Abstractions)
The knowledge graph identified the most heavily connected nodes in the system, which represent the backbone of the application. The top three are:
1. **`SimpleSkipGram` (17 connections)**: This custom NumPy word embedding model is the absolute core of the ML pipeline. It connects to almost every other community because the topic classifier, question ranker, and answer scorer all rely on it to convert text to dense vectors.
2. **`InterviewController` (14 connections)**: This acts as the "Brain" of the application state. The graph explicitly shows it acting as a cross-community bridge, pulling together `SessionState` and all the PyTorch neural networks.
3. **`run_proctor()` (11 connections)**: The main loop of the `exam_proctor.py` module, which heavily orchestrates all the geometry and math helpers in its own isolated cluster.

## 2. Community Structure (Modularity)
The codebase automatically clusters into several highly cohesive "communities". This proves that the codebase follows excellent separation of concerns:
- **Community 0 (The Brain/Backend)**: Contains the `InterviewController`, `main.py` CLI logic, and `run.py` API endpoints (`api_upload_resume`, `api_answer`, etc.).
- **Community 1 (The Proctor)**: Entirely isolates the MediaPipe and OpenCV logic (`exam_proctor.py`, `iris_ratio()`, `eye_aspect_ratio()`).
- **Community 3 (Embeddings)**: The custom `SimpleSkipGram` and its training sequences.
- **Community 4 (The Frontend)**: Exclusively contains the JavaScript logic (`app.js`, `startTimer()`, `updateTimeline()`).
- **Communities 2, 5, 6 (The Neural Networks)**: The PyTorch models (`QuestionRanker`, `SiameseAnswerScorer`, and `TopicClassifier`) each naturally formed their own distinct communities alongside their specific training and evaluation functions.

## 3. Surprising Connections & Inferred Logic
The graph AI successfully inferred how the Flask API hooks into the ML logic without explicit hardcoded imports in some places. For instance, it detected the pipeline:
`api_upload_resume()` → calls → `extract_from_resume()` (inside `data/resume_parser.py`). 

It also successfully recognized that `InterviewController` is essentially a massive structural bridge between **Community 0** (Web/CLI interfaces) and **Communities 2, 3, 5, 6** (The PyTorch Models).

## 4. Knowledge Gaps (Potential Cleanup)
The graph flagged a few "thin" communities:
- `tmp_generate_data.py` and `tmp_inspect_excel.py` are largely isolated from the rest of the execution flow. This makes sense as they are likely one-off data inspection scripts you used during development.
- `config.py` is isolated as a static configuration file, rather than an active operational component.

## Summary
The `graphify` extraction proves that the **AI Mock Interviewer** is highly modular. The strict separation between the ML models, the `InterviewController` state machine, the Flask REST API, the vanilla JS Frontend, and the isolated `exam_proctor.py` tool makes this a very clean and scalable architecture.
