# scripts/

Utility scripts for the AI Mock Interviewer project. Not part of the main app.

| Script | Purpose | When to run |
|--------|---------|-------------|
| `tmp_generate_data.py` | Generates synthetic `topic_classifier_data.csv` and `answer_scoring_data.csv` from the question bank | Run once when seeding training data from scratch |
| `tmp_inspect_excel.py` | Inspects the Interview_Questions.xlsx schema — prints column names and sample rows | Run when onboarding a new dataset version to confirm column names |
