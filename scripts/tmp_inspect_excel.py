import pandas as pd
file_path = r"e:\mini project 1\ai_interviewer\Interview_Questions.xlsx"
df = pd.read_excel(file_path)
df.columns = [c.strip() for c in df.columns]

print(df[df['Category'] == 'Programming Languages']['Skill'].unique()[:10])
print(df[df['Category'] == 'Data Science & AI']['Skill'].unique()[:10])
print(df[df['Category'] == 'Database']['Skill'].unique()[:10])
print(df[df['Category'] == 'AI & Machine Learning']['Skill'].unique()[:10])
print(df[df['Category'] == 'Backend']['Skill'].unique()[:10])
print(df[df['Category'] == 'Core Mandatory']['Skill'].unique()[:10])
