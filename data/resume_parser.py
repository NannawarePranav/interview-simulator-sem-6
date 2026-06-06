import os
import re
import fitz  # PyMuPDF

# ── Dynamically loaded skill keywords (populated on first import) ─────────────
_DYNAMIC_SKILLS: list = []

def _build_skill_list() -> list:
    """Build the master skill list from Interview_Questions.xlsx + hardcoded extras."""
    manual_extras = [
        "fastapi", "pydantic", "docker", "kubernetes", "redis", "celery",
        "pytest", "mypy", "ruff", "terraform", "rag", "vector database",
        "langchain", "streamlit", "mediapipe", "opencv",
        # existing set
        "python", "c", "c++", "java", "html", "css", "javascript", "php",
        "react", "node.js", "django", "flask", "spring", "sql", "mysql",
        "postgresql", "mongodb", "machine learning", "deep learning",
        "artificial intelligence", "data science", "rest api", "rest apis",
        "agile", "data structures", "algorithms", "system design",
        "git", "github", "gitlab", "linux", "ubuntu", "rhel", "docker",
        "podman", "kubernetes", "aws", "azure", "gcp", "apache", "nginx",
        "arcore", "android", "ios", "qa", "mcp toolkit", "ml studio",
    ]

    skills = set(s.lower() for s in manual_extras)

    # Try to load from Interview_Questions.xlsx
    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xlsx_path   = os.path.join(base_dir, 'data-set-full', 'ds', 'Interview_Questions.xlsx')
    if os.path.exists(xlsx_path):
        try:
            import pandas as pd
            df = pd.read_excel(xlsx_path)
            col_map = {c.strip().lower(): c for c in df.columns}
            skill_col = col_map.get('skill') or col_map.get('skills') or col_map.get('technology')
            if skill_col:
                for val in df[skill_col].dropna().unique():
                    skills.add(str(val).strip().lower())
            print(f"[ResumeParser] Loaded {len(skills)} skills from xlsx + manual list.")
        except Exception as e:
            print(f"[ResumeParser] xlsx load warning: {e}")
    else:
        print(f"[ResumeParser] xlsx not found at {xlsx_path}, using manual list only.")

    return sorted(skills)


def _get_skills() -> list:
    global _DYNAMIC_SKILLS
    if not _DYNAMIC_SKILLS:
        _DYNAMIC_SKILLS = _build_skill_list()
    return _DYNAMIC_SKILLS


# ── Casing rules ──────────────────────────────────────────────────────────────
ACRONYMS = {'php', 'html', 'css', 'sql', 'mysql', 'aws', 'api', 'qa',
             'gcp', 'rhel', 'rag', 'ios', 'gru', 'rnn', 'cnn', 'nlp', 'cv'}

SPECIAL_CASE = {
    'rest apis':      'REST APIs',
    'rest api':       'REST API',
    'mcp toolkit':    'MCP Toolkit',
    'ml studio':      'ML Studio',
    'node.js':        'Node.js',
    'opencv':         'OpenCV',
    'mediapipe':      'MediaPipe',
    'langchain':      'LangChain',
    'fastapi':        'FastAPI',
    'postgresql':     'PostgreSQL',
    'mongodb':        'MongoDB',
    'kubernetes':     'Kubernetes',
    'docker':         'Docker',
    'terraform':      'Terraform',
    'streamlit':      'Streamlit',
    'vector database': 'Vector Database',
}


def _format_skill(s: str) -> str:
    if s in SPECIAL_CASE:
        return SPECIAL_CASE[s]
    if s in ACRONYMS:
        return s.upper()
    return s.title()


def extract_from_resume(pdf_path, resume_txt_path, skills_txt_path):
    print(f"Opening {pdf_path}...")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return

    text = ""
    for page in doc:
        text += page.get_text()

    # Clean up bullet points & special chars
    text = text.replace('\uf0b7', '').replace('\u2022', '').replace('', '').replace('', '')

    lines = [line.strip() for line in text.split('\n') if line.strip()]
    clean_text = " ".join(lines)

    with open(resume_txt_path, 'w', encoding='utf-8') as f:
        f.write(clean_text)
    print(f"Saved {len(clean_text)} chars of raw text to {os.path.basename(resume_txt_path)}")

    # ── Skill extraction ──────────────────────────────────────────────────────
    skills_found = set()
    known_skills = _get_skills()
    lower_text   = clean_text.lower()

    for skill in known_skills:
        if re.search(r'\b' + re.escape(skill) + r'\b', lower_text):
            skills_found.add(skill)

    # Format for display
    unique_skills = sorted({_format_skill(s) for s in skills_found})

    with open(skills_txt_path, 'w', encoding='utf-8') as f:
        for skill in unique_skills:
            f.write(f"{skill}\n")

    print(f"Extracted {len(unique_skills)} skills to {os.path.basename(skills_txt_path)}")


if __name__ == "__main__":
    base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_path   = os.path.join(base_dir, 'data', 'raw', 'Pranav Nannaware Resume.pdf')
    resume_path = os.path.join(base_dir, 'data', 'raw', 'resume.txt')
    skills_path = os.path.join(base_dir, 'data', 'raw', 'skills.txt')
    extract_from_resume(pdf_path, resume_path, skills_path)
