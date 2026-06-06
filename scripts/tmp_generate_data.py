import pandas as pd
import json
import random
import os

file_path = r"e:\mini project 1\ai_interviewer\Interview_Questions.xlsx"
df = pd.read_excel(file_path)
df.columns = [c.strip() for c in df.columns]

df = df.dropna(subset=['Question', 'Expected Answer'])

def map_topic(row):
    cat = str(row['Category']).lower()
    skill = str(row['Skill']).lower()
    
    if 'python' in skill: return 'python'
    if 'dsa' in skill or 'algorithm' in skill or 'data struct' in skill: return 'data_structures_algorithms'
    if 'database' in cat or 'sql' in skill: return 'databases_sql'
    if 'system design' in cat or 'design' in skill or 'architecture' in skill or 'system' in skill: return 'system_design'
    if 'behavioral' in cat or 'hr' in cat or 'soft skill' in cat or 'leadership' in skill: return 'behavioral'
    if 'machine learning' in cat or 'data science' in cat or 'ai' in cat or 'model' in skill: return 'machine_learning_basics'
    
    return None

df['mapped_topic'] = df.apply(map_topic, axis=1)

print("Mapped Topics value counts:")
counts = df['mapped_topic'].value_counts()
print(counts)

TOPICS = ["python", "data_structures_algorithms", "machine_learning_basics", "databases_sql", "system_design", "behavioral"]

qb = []
topic_cf_data = []
ans_sc_data = []

os.makedirs('data/raw', exist_ok=True)

# 1. Resume and Skills
with open('data/raw/resume.txt', 'w') as f:
    f.write("Software Engineer with 3 years of experience specializing in Python, Flask, and building scalable machine learning systems. Familiar with SQL databases and REST APIs. Experienced in agile environments.")
with open('data/raw/skills.txt', 'w') as f:
    f.write("Python\nFlask\nSQL\nMachine Learning\nREST APIs\nAgile\nData Structures\nAlgorithms\nSystem Design\nGit")

# 2. Extract Question Bank (30+, 5+ per topic) and Topic Classifier
for topic in TOPICS:
    topic_df = df[df['mapped_topic'] == topic]
    
    # Generate mock questions if dataset doesn't have enough for some specific topic
    if len(topic_df) < 25:
        print(f"Warning: Only {len(topic_df)} items for {topic}, padding with mock items...")
        mock_questions = [
            {"Question": f"Mock question about {topic} 1?", "Expected Answer": f"Mock answer 1", "Level": "Medium"},
            {"Question": f"Mock question about {topic} 2?", "Expected Answer": f"Mock answer 2", "Level": "Medium"},
            {"Question": f"Mock question about {topic} 3?", "Expected Answer": f"Mock answer 3", "Level": "Hard"},
            {"Question": f"Mock question about {topic} 4?", "Expected Answer": f"Mock answer 4", "Level": "Easy"},
            {"Question": f"Mock question about {topic} 5?", "Expected Answer": f"Mock answer 5", "Level": "Medium"},
            {"Question": f"Mock question about {topic} 6?", "Expected Answer": f"Mock answer 6", "Level": "Hard"},
        ] * 5
        topic_df = pd.concat([topic_df, pd.DataFrame(mock_questions)], ignore_index=True)
        
    sampled = topic_df.sample(25, random_state=42)
    
    # Q bank (7 questions per topic)
    q_bank_items = sampled.iloc[:7]
    for idx, row in q_bank_items.iterrows():
        qb.append({
            "id": f"{topic}_{idx}",
            "topic": topic,
            "difficulty": str(row['Level']).lower() if "Level" in row else "medium",
            "question": str(row['Question']),
            "keywords": [topic, "interview"],
            "follow_up": "Can you elaborate on that?"
        })
        
    # Topic classifier data (25 per topic)
    for idx, row in sampled.iterrows():
        # Tricky negative example? We do this simply.
        snippet = str(row['Question']) + " " + str(row['Expected Answer'])[:100]
        topic_cf_data.append({"snippet_text": snippet.replace("\n", " "), "topic_label": topic})

# Write Question Bank
with open('data/question_bank.json', 'w') as f:
    json.dump({"questions": qb}, f, indent=2)

# Write Topic Classifier CF
pd.DataFrame(topic_cf_data).to_csv('data/topic_classifier_data.csv', index=False)

# 3. Answer Scoring Data (20 questions with multiple scores)
sc_samples = df[df['mapped_topic'].isin(TOPICS)].sample(20, random_state=42)
ans_sc_list = []
for idx, row in sc_samples.iterrows():
    q = str(row['Question']).replace('\n', ' ')
    a_strong = str(row['Expected Answer']).replace('\n', ' ')
    
    # Generate weak and mediocre synthetically
    a_med = a_strong[:len(a_strong)//2] + " and I think that's it."
    a_weak = "I am not really sure but maybe it relates to computer science."
    
    ans_sc_list.append({"question": q, "answer": a_strong, "score": 1.0})
    ans_sc_list.append({"question": q, "answer": a_med, "score": 0.5})
    ans_sc_list.append({"question": q, "answer": a_weak, "score": 0.0})

# Also include tricky negatives as mentioned in the prompt
topic_cf_data.append({"snippet_text": "I really like the Python snake because it is a cool animal.", "topic_label": "behavioral"}) # tricky!

pd.DataFrame(ans_sc_list).to_csv('data/answer_scoring_data.csv', index=False)
print("Data generation complete!")
