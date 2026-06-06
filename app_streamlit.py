import streamlit as st
import time
import os
import tempfile
import pdfplumber
import numpy as np
import pandas as pd
from interview.controller import InterviewController
from config import TOPICS
import interview.report as report_gen

st.set_page_config(page_title="AI Mock Interviewer V2", layout="wide")

def parse_resume(uploaded_file):
    if uploaded_file.name.endswith('.pdf'):
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    else:
        return uploaded_file.getvalue().decode('utf-8')

if 'controller' not in st.session_state:
    st.session_state.controller = InterviewController()

if 'session_state' not in st.session_state:
    st.session_state.session_state = st.session_state.controller.state
else:
    st.session_state.controller.state = st.session_state.session_state

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'interview_active' not in st.session_state:
    st.session_state.interview_active = False

if 'interview_complete' not in st.session_state:
    st.session_state.interview_complete = False

# Sidebar
with st.sidebar:
    st.header("Candidate Setup")
    resume_file = st.file_uploader("Upload Resume (.txt, .pdf)", type=["txt", "pdf"])
    skills_text = st.text_area("Specific Skills / Focus Areas")
    
    st.header("V2 Settings")
    use_glove = st.checkbox("USE_GLOVE", value=st.session_state.controller.use_glove)
    use_distilbert = st.checkbox("USE_DISTILBERT_SCORER", value=st.session_state.controller.use_distilbert_scorer)
    use_gpt2 = st.checkbox("USE_GPT2_FOLLOWUP", value=st.session_state.controller.use_gpt2)
    use_rl = st.checkbox("USE_RL_POLICY", value=st.session_state.controller.use_rl_policy)
    
    st.session_state.controller.use_glove = use_glove
    st.session_state.controller.use_distilbert_scorer = use_distilbert
    st.session_state.controller.use_gpt2 = use_gpt2
    st.session_state.controller.use_rl_policy = use_rl
    
    if st.button("Start Interview") and resume_file is not None and not st.session_state.interview_active:
        resume_text = parse_resume(resume_file)
        
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt') as f_res:
            f_res.write(resume_text)
            res_path = f_res.name
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt') as f_skl:
            f_skl.write(skills_text)
            skl_path = f_skl.name
            
        st.session_state.controller.start_session(res_path, skl_path)
        first_q = st.session_state.controller.next_question()
        st.session_state.chat_history.append({"role": "assistant", "content": first_q})
        st.session_state.interview_active = True
        st.session_state.interview_complete = False
        st.session_state.q_start_time = time.time()
        
        os.unlink(res_path)
        os.unlink(skl_path)
        st.rerun()
        
    st.header("Live Scores")
    for t in TOPICS:
        scores = st.session_state.session_state.scores.get(t, [])
        if scores:
            avg = sum(scores) / len(scores)
            st.metric(label=t, value=f"{avg:.2f}")
        else:
            st.metric(label=t, value="N/A")

# Main Area
st.title("AI Mock Interviewer")

if st.session_state.interview_active:
    # Progress Bar
    covered = len(st.session_state.session_state.topics_covered)
    total = len(TOPICS)
    st.progress(covered / total)
    
    current_topic = st.session_state.session_state.current_topic
    if current_topic:
        st.info(f"Current Topic: **{current_topic}**")
        
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    # Timer logic
    timer_placeholder = st.empty()
    elapsed = time.time() - st.session_state.get('q_start_time', time.time())
    remaining = max(0, 90 - int(elapsed))
    timer_placeholder.error(f"Time Remaining: {remaining}s")
    
    ans = st.chat_input("Type your answer here...")
    
    if ans is not None:
        if remaining == 0:
            ans = "" # Auto-submit empty if timeout
            st.warning("Time expired! Answer submitted as empty.")
            
        st.session_state.chat_history.append({"role": "user", "content": ans})
        score = st.session_state.controller.process_answer(ans)
        
        next_q = st.session_state.controller.next_question()
        if next_q:
            st.session_state.chat_history.append({"role": "assistant", "content": next_q})
            st.session_state.q_start_time = time.time()
        else:
            st.session_state.interview_active = False
            st.session_state.interview_complete = True
            
        st.rerun()

if st.session_state.interview_complete:
    st.success("Interview Complete!")
    
    scores_dict = st.session_state.session_state.scores
    df_data = []
    total_score = 0
    count = 0
    for t in TOPICS:
        scs = scores_dict.get(t, [])
        if scs:
            avg = sum(scs) / len(scs)
            df_data.append({"Topic": t, "Average Score": avg})
            total_score += avg
            count += 1
            
    df = pd.DataFrame(df_data)
    
    def color_scores(val):
        color = 'red'
        if val >= 0.7: color = 'green'
        elif val >= 0.3: color = 'yellow'
        return f'color: {color}'
        
    st.dataframe(df.style.map(color_scores, subset=['Average Score']))
    
    if count > 0:
        overall = total_score / count
        if overall >= 0.75:
            st.success(f"Senior-ready (Score: {overall:.2f})")
        elif overall >= 0.5:
            st.warning(f"Mid-Level (Score: {overall:.2f})")
        else:
            st.error(f"Junior-ready (Score: {overall:.2f})")
            
    try:
        report_text = report_gen.generate_report(st.session_state.session_state)
        st.download_button(
            label="Download Final Report",
            data=report_text,
            file_name="interview_report.txt",
            mime="text/plain"
        )
    except Exception as e:
        st.error(f"Failed to generate report: {e}")
