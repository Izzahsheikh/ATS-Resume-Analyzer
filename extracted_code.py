import os
import re
import json
import string
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import io

import streamlit as st
from dotenv import load_dotenv
import fitz  # PyMuPDF
import docx  # python-docx
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate

load_dotenv()

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Izra | AI Resume Analyzer",
    page_icon="assets/izra_logo_cropped.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------------
# CUSTOM UI THEME
# --------------------------------------------------------------------------
st.markdown("""
<style>

/* =========================================================
   GLOBAL
   ========================================================= */
.stApp {
    background-color: #F7F6F4;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* =========================================================
   SIDEBAR
   ========================================================= */
section[data-testid="stSidebar"] {
    background-color: #EFEEEC !important;
    border-right: 1px solid #DDD9D7;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #541F49 !important;
}

section[data-testid="stSidebar"] p {
    color: #292529 !important;
}

section[data-testid="stSidebar"] .stMarkdown p {
    color: #6F6A6D !important;
    font-size: 15px !important;
    line-height: 1.5 !important;
}

/* =========================================================
   MAIN HEADINGS
   ========================================================= */
h1 { color: #541F49 !important; font-size: 32px !important; font-weight: 700 !important; }
h2 { color: #541F49 !important; font-size: 22px !important; font-weight: 650 !important; }
h3 { color: #292529 !important; font-size: 18px !important; font-weight: 600 !important; }

/* =========================================================
   THRESHOLD VALUE BOX
   ========================================================= */
section[data-testid="stSidebar"] div[data-testid="stTextInput"] > div {
    background-color: #FFFFFF !important;
    border-radius: 7px !important;
}

section[data-testid="stSidebar"] div[data-testid="stTextInput"] input {
    background-color: #FFFFFF !important;
    color: #292529 !important;
    -webkit-text-fill-color: #292529 !important;
    text-align: center !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    border: 1px solid #D8D4D1 !important;
    border-radius: 7px !important;
    color-scheme: light !important;
}

section[data-testid="stSidebar"] div[data-testid="stTextInput"] input:focus {
    background-color: #FFFFFF !important;
    color: #292529 !important;
    -webkit-text-fill-color: #292529 !important;
    border-color: #541F49 !important;
    box-shadow: 0 0 0 1px #541F49 !important;
}

/* =========================================================
   THRESHOLD MINUS / PLUS BUTTONS
   ========================================================= */
section[data-testid="stSidebar"] .stButton > button {
    min-height: 40px !important;
    padding: 0 !important;
    background-color: #541F49 !important;
    border: 1px solid #541F49 !important;
    border-radius: 7px !important;
    color: #FFFFFF !important;
    font-size: 22px !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] .stButton > button p,
section[data-testid="stSidebar"] .stButton > button span {
    color: #FFFFFF !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #6B3A5E !important;
    border-color: #6B3A5E !important;
    color: #FFFFFF !important;
}

section[data-testid="stSidebar"] .stButton > button:focus,
section[data-testid="stSidebar"] .stButton > button:focus-visible,
section[data-testid="stSidebar"] .stButton > button:active {
    background-color: #541F49 !important;
    border-color: #541F49 !important;
    color: #FFFFFF !important;
    box-shadow: none !important;
}

/* =========================================================
   MAIN BUTTONS
   ========================================================= */
.stButton > button {
    background-color: #6B3A5E !important;
    border: 1px solid #6B3A5E !important;
    border-radius: 8px !important;
    min-height: 48px !important;
    padding: 0 24px !important;
    color: #FFFFFF !important;
    font-size: 19px !important;
    font-weight: 600 !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
}

.stButton > button p, .stButton > button span { color: #FFFFFF !important; }
.stButton > button:hover { background-color: #541F49 !important; border-color: #541F49 !important; }
.stButton > button:focus, .stButton > button:focus-visible, .stButton > button:active {
    background-color: #6B3A5E !important;
    border-color: #6B3A5E !important;
    color: #FFFFFF !important;
    box-shadow: none !important;
}

/* =========================================================
   FILE UPLOADER
   ========================================================= */
[data-testid="stFileUploader"] {
    background-color: #F0EFEC !important;
    border: 1px solid #D8D4D1 !important;
    border-radius: 10px !important;
    padding: 6px !important;
    margin-top: 4px !important;
}

[data-testid="stFileUploaderDropzone"] {
    background-color: #F0EFEC !important;
    border: 2px dashed #C0B4BE !important;
    border-radius: 8px !important;
    padding: 20px 16px !important;
}

/* Dropzone instruction text */
[data-testid="stFileUploaderDropzoneInstructions"] {
    font-size: 15px !important;
    color: #6F6A6D !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] span {
    font-size: 15px !important;
    color: #6F6A6D !important;
}

/* Drag active state */
[data-testid="stFileUploaderDropzone"]:hover,
[data-testid="stFileUploaderDropzone"][data-dragging="true"] {
    border-color: #541F49 !important;
    background-color: #EDE0EC !important;
}

[data-testid="stFileUploader"] label {
    color: #292529 !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    line-height: 1.5 !important;
}

[data-testid="stFileUploader"] small {
    color: #6F6A6D !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
}

/* =========================================================
   UPLOAD BUTTON (Browse / Upload)
   ========================================================= */

/* Hide the native + (add more files) button Streamlit renders */
[data-testid="stFileUploaderFileList"] ~ div button,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] ~ div > button:last-child {
    display: none !important;
}

/* Style all uploader buttons (Upload / Browse) */
[data-testid="stFileUploader"] button {
    background-color: #6B3A5E !important;
    border: 1px solid #6B3A5E !important;
    border-radius: 7px !important;
    min-height: 42px !important;
    padding: 0 18px !important;
    color: #FFFFFF !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}

[data-testid="stFileUploader"] button span,
[data-testid="stFileUploader"] button p { color: #FFFFFF !important; }
[data-testid="stFileUploader"] button:hover { background-color: #541F49 !important; border-color: #541F49 !important; }
[data-testid="stFileUploader"] button:focus,
[data-testid="stFileUploader"] button:focus-visible,
[data-testid="stFileUploader"] button:active {
    background-color: #6B3A5E !important;
    border-color: #6B3A5E !important;
    color: #FFFFFF !important;
    box-shadow: none !important;
}

/* Style the small × delete button on uploaded file chips */
[data-testid="stFileUploaderDeleteBtn"] button {
    background-color: transparent !important;
    border: none !important;
    min-height: unset !important;
    padding: 2px !important;
    color: #541F49 !important;
    font-size: 13px !important;
}

/* ADD MORE CVs custom button */
.add-more-btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background-color: #6B3A5E;
    border: 1.5px solid #6B3A5E;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 15px;
    font-weight: 700;
    color: #FFFFFF;
    cursor: pointer;
    margin-top: 12px;
    text-decoration: none;
    transition: background-color 0.15s ease;
    width: 100%;
    justify-content: center;
    box-sizing: border-box;
}

.add-more-btn:hover {
    background-color: #541F49;
    border-color: #541F49;
}

/* =========================================================
   METRICS
   ========================================================= */
[data-testid="stMetricValue"] {
    color: #541F49 !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}

[data-testid="stMetricLabel"] {
    color: #6F6A6D !important;
    font-size: 16px !important;
}

/* =========================================================
   EXPANDERS / CARDS
   ========================================================= */
[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #DDD9D7 !important;
    border-radius: 8px !important;
}

[data-testid="stExpander"] summary {
    font-size: 16px !important;
    font-weight: 600 !important;
    background-color: #FFFFFF !important;
    color: #541F49 !important;
    -webkit-text-fill-color: #541F49 !important;
    border-radius: 8px !important;
}

[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {
    color: #541F49 !important;
    -webkit-text-fill-color: #541F49 !important;
}

[data-testid="stExpander"] details[open] summary {
    background-color: #541F49 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

[data-testid="stExpander"] details[open] summary p,
[data-testid="stExpander"] details[open] summary span {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

[data-testid="stExpander"] summary:focus { outline: none !important; }
[data-testid="stExpander"] summary:hover {
    background-color: #F3F1F2 !important;
    color: #541F49 !important;
    -webkit-text-fill-color: #541F49 !important;
}

/* =========================================================
   DIVIDERS
   ========================================================= */
hr { border-color: #DDD9D7 !important; }

/* =========================================================
   CAPTIONS
   ========================================================= */
[data-testid="stCaptionContainer"] {
    color: #6F6A6D !important;
    font-size: 15px !important;
}

/* =========================================================
   ALERTS
   ========================================================= */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-size: 16px !important;
}

/* =========================================================
   GENERAL MARKDOWN
   ========================================================= */
.stMarkdown { font-size: 16px; }

/* =========================================================
   SCROLLBAR
   ========================================================= */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #F7F6F4; }
::-webkit-scrollbar-thumb { background: #C8C3C0; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #A9A2A5; }

/* =========================================================
   RESULT CARD TEXT
   ========================================================= */
[data-testid="stVerticalBlockBorderWrapper"] { background-color: #FFFFFF !important; }

[data-testid="stVerticalBlockBorderWrapper"] p,
[data-testid="stVerticalBlockBorderWrapper"] span,
[data-testid="stVerticalBlockBorderWrapper"] h3 {
    color: #292529 !important;
    -webkit-text-fill-color: #292529 !important;
}

[data-testid="stVerticalBlockBorderWrapper"] .stMarkdown div {
    color: #292529 !important;
    -webkit-text-fill-color: #292529 !important;
}

[data-testid="stExpander"] p,
[data-testid="stExpander"] span,
[data-testid="stExpander"] li { color: #292529 !important; }

[data-testid="stExpander"] strong { color: #292529 !important; }

/* =========================================================
   PROGRESS TEXT
   ========================================================= */
[data-testid="stProgress"] p { color: #292529 !important; }
[data-testid="stProgress"] span { color: #292529 !important; }

[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"] h3 {
    color: #292529 !important;
    -webkit-text-fill-color: #292529 !important;
    opacity: 1 !important;
}

/* =========================================================
   STEP LABELS
   ========================================================= */
.step-label {
    display: inline-block;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #541F49;
    background-color: #EDE0EC;
    border: 1.5px solid #C9A8C4;
    border-radius: 5px;
    padding: 5px 14px;
    margin-bottom: 12px;
}

/* =========================================================
   UPLOAD CARD
   ========================================================= */
.upload-card {
    background: #FFFFFF;
    border: 1.5px solid #DDD9D7;
    border-radius: 12px;
    padding: 24px 24px 20px 24px;
    height: 100%;
}

.upload-card-heading {
    font-size: 18px;
    font-weight: 700;
    color: #292529;
    margin: 0 0 4px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.upload-card-sub {
    font-size: 13px;
    color: #6F6A6D;
    margin: 0 0 16px 0;
}

/* =========================================================
   NO SHORTLIST MESSAGE
   ========================================================= */
.custom-no-shortlist {
    background-color: #FFF1EB !important;
    border: 1px solid #F26B4F !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}

.custom-no-shortlist-icon {
    color: #F26B4F !important;
    -webkit-text-fill-color: #F26B4F !important;
    font-size: 18px !important;
    font-weight: 700 !important;
}

.custom-no-shortlist-text {
    color: #B94A32 !important;
    -webkit-text-fill-color: #B94A32 !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* =========================================================
   RECOMMENDATION SECTION
   ========================================================= */
.rec-section {
    background: #FFFFFF;
    border: 1.5px solid #D4B8D0;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 24px;
}

.rec-eyebrow {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #541F49;
    margin-bottom: 4px;
}

.rec-heading {
    font-size: 22px;
    font-weight: 700;
    color: #292529;
    margin: 0 0 6px 0;
}

.rec-subtext {
    font-size: 14px;
    color: #6F6A6D;
    margin: 0 0 18px 0;
}

.rec-candidate-list {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 4px;
}

.rec-candidate-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: #F0E8EF;
    border: 1px solid #D4B8D0;
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 14px;
    font-weight: 600;
    color: #541F49;
}

.rec-none {
    background: #FFF1EB;
    border: 1px solid #F26B4F;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 14px;
    font-weight: 600;
    color: #B94A32;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
logo_col, title_col = st.columns([0.7, 5])

with logo_col:
    st.image("assets/izra_logo_cropped.png", width=150)

with title_col:
    st.markdown(
        "<h1 style='margin-bottom:0;'>AI Resume Analyzer</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='color:#6F6A6D; font-size:16px; margin-top:4px;'>"
        "Find the right candidates faster with AI-powered screening."
        "</p>",
        unsafe_allow_html=True
    )

# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------
def normalize_extracted_text(text: str) -> str:
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    text = re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', text)
    text = re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', text)
    text = re.sub(r'(?<=[,/])(?=\S)', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text


def extract_text_from_uploaded(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    if suffix == ".pdf":
        file_bytes = uploaded_file.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return normalize_extracted_text(text)
    elif suffix == ".docx":
        file_bytes = uploaded_file.read()
        document = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        paragraphs.append(cell.text)
        text = "\n".join(paragraphs)
        return normalize_extracted_text(text)
    else:
        raw = uploaded_file.read().decode("utf-8", errors="ignore")
        return normalize_extracted_text(raw)


def parse_json_response(text):
    if isinstance(text, list):
        text = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in text
        )
    cleaned = re.sub(r"```json|```", "", text).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group(0))


def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)


# --------------------------------------------------------------------------
# STEP 1: Extract required skills from JD
# --------------------------------------------------------------------------
JD_SKILLS_PROMPT = PromptTemplate(
    input_variables=["job_description"],
    template="""
You are a technical recruiter. Read the job description below and extract the specific
technical skills, tools, frameworks, and technologies required for this role.

JOB DESCRIPTION:
{job_description}

Respond with ONLY a valid JSON object (no markdown, no extra text):
{{
  "required_skills": ["skill1", "skill2", "skill3"]
}}

Be specific — include exact tool/framework names like "LangChain", "RAG", "Docker", "PyTorch".
Do NOT include soft skills like "communication" or "teamwork".
""",
)


@st.cache_data(show_spinner=False)
def extract_jd_skills(jd_text: str) -> list:
    prompt = JD_SKILLS_PROMPT.format(job_description=jd_text)
    response = get_llm().invoke(prompt)
    data = parse_json_response(response.content)
    return [s.lower() for s in data.get("required_skills", [])]


@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------------------------------
# STEP 2: Semantic Embeddings
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_embeddings(texts: tuple) -> np.ndarray:
    model = get_embedding_model()
    embeddings = model.encode(
        list(texts),
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return embeddings


def compute_semantic_similarity(jd_embedding, resume_embeddings):
    similarities = np.dot(resume_embeddings, jd_embedding)
    return [round(float(score) * 100) for score in similarities]


# --------------------------------------------------------------------------
# STEP 3: Deterministic scoring
# --------------------------------------------------------------------------
def compute_skill_match(resume_text, required_skills):
    if not required_skills:
        return 0, [], []

    chunks = [line.strip() for line in resume_text.splitlines() if len(line.strip()) >= 15]
    if not chunks:
        return 0, [], []

    model = get_embedding_model()
    skill_embeddings = model.encode(required_skills, batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    chunk_embeddings = model.encode(chunks, batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    similarities = np.dot(skill_embeddings, chunk_embeddings.T)
    matched, missing = [], []
    MATCH_THRESHOLD = 0.50

    for skill, skill_sims in zip(required_skills, similarities):
        if float(np.max(skill_sims)) >= MATCH_THRESHOLD:
            matched.append(skill)
        else:
            missing.append(skill)

    score = round(len(matched) / len(required_skills) * 100)
    return score, matched, missing


def score_education(resume_text):
    text = resume_text.lower()
    if any(w in text for w in ["phd", "doctorate"]): return 100
    elif any(w in text for w in ["master", "msc", "ms ", "m.s", "mba"]): return 85
    elif any(w in text for w in ["bachelor", "bsc", "b.s", "b.e", "beng", "degree"]): return 70
    elif any(w in text for w in ["diploma", "associate"]): return 50
    return 30


def score_experience(resume_text, jd_text):
    resume_years = re.findall(r"(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)", resume_text.lower())
    jd_years = re.findall(r"(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)", jd_text.lower())
    if resume_years and jd_years:
        candidate_exp = max(int(y) for y in resume_years)
        required_exp = max(int(y) for y in jd_years)
        if candidate_exp >= required_exp: return 100
        return round((candidate_exp / required_exp) * 100)
    text = resume_text.lower()
    if any(w in text for w in ["senior", "lead", "principal", "head of", "manager"]): return 85
    elif any(w in text for w in ["engineer", "developer", "analyst", "apprentice"]): return 65
    elif any(w in text for w in ["junior", "intern", "trainee", "fresher", "graduate"]): return 40
    return 50


def score_projects(resume_text):
    text = resume_text.lower()
    score = 0
    if "project" in text: score += 50
    if any(w in text for w in ["github", "gitlab", "bitbucket"]): score += 25
    if any(w in text for w in ["deployed", "production", "live", "published"]): score += 25
    return score


def score_formatting(resume_text):
    lines = [l.strip() for l in resume_text.splitlines() if l.strip()]
    score = 50
    if len(lines) > 20: score += 15
    if any(w in resume_text.lower() for w in ["experience", "education", "skills", "projects"]): score += 20
    if any(w in resume_text.lower() for w in ["@", "email", "phone", "linkedin"]): score += 15
    return min(score, 100)


def score_resume_deterministic(resume_text, jd_text, required_skills, semantic_score, resume_embedding, skill_embeddings):
    skill_score, matched, missing = compute_skill_match(resume_text, required_skills)
    combined_skill = round(skill_score * 0.7 + semantic_score * 0.3)
    experience_score = score_experience(resume_text, jd_text)
    education_score = score_education(resume_text)
    projects_score = score_projects(resume_text)
    formatting_score = score_formatting(resume_text)
    ats_score = round(
        combined_skill   * 0.40 +
        experience_score * 0.25 +
        education_score  * 0.15 +
        projects_score   * 0.12 +
        formatting_score * 0.08
    )
    return {
        "ats_score": ats_score,
        "final_score": ats_score,
        "skill_match": combined_skill,
        "experience_score": experience_score,
        "education_score": education_score,
        "projects_score": projects_score,
        "formatting_score": formatting_score,
        "matched_skills": [s.title() for s in matched],
        "missing_skills": [s.title() for s in missing],
    }


# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
if "shortlist_threshold" not in st.session_state:
    st.session_state.shortlist_threshold = 70
if "threshold_text" not in st.session_state:
    st.session_state.threshold_text = "70"


def decrease_threshold():
    new_value = max(0, st.session_state.shortlist_threshold - 1)
    st.session_state.shortlist_threshold = new_value
    st.session_state.threshold_text = str(new_value)


def increase_threshold():
    new_value = min(100, st.session_state.shortlist_threshold + 1)
    st.session_state.shortlist_threshold = new_value
    st.session_state.threshold_text = str(new_value)


def update_threshold():
    try:
        value = int(st.session_state.threshold_text)
        if 0 <= value <= 100:
            st.session_state.shortlist_threshold = value
        else:
            st.session_state.threshold_text = str(st.session_state.shortlist_threshold)
    except ValueError:
        st.session_state.threshold_text = str(st.session_state.shortlist_threshold)


with st.sidebar:
    st.markdown("<h2 style='margin-bottom:5px;'>Screening Settings</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6F6A6D; font-size:14px;'>Configure how candidates are evaluated.</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<p style='font-weight:600; color:#292529; font-size:15px; margin-bottom:8px;'>Shortlist Threshold</p>", unsafe_allow_html=True)

    minus_col, value_col, plus_col = st.columns([1, 1.5, 1])

    with minus_col:
        st.button("−", key="threshold_minus", use_container_width=True, on_click=decrease_threshold)

    with value_col:
        st.text_input("Threshold", key="threshold_text", label_visibility="collapsed", on_change=update_threshold)

    with plus_col:
        st.button("＋", key="threshold_plus", use_container_width=True, on_click=increase_threshold)

    shortlist_threshold = st.session_state.shortlist_threshold

    st.markdown(
        f"<p style='color:#6F6A6D; font-size:13px; margin-top:8px;'>"
        f"Candidates scoring <b>{shortlist_threshold}%</b> or higher will be shortlisted.</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")


# --------------------------------------------------------------------------
# UPLOAD SECTION HEADING
# --------------------------------------------------------------------------
st.markdown("<h2 style='margin-top:20px; margin-bottom:6px;'>Start Screening</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:#6F6A6D; font-size:16px; margin-bottom:28px; line-height:1.6;'>Complete both steps below, then click <b>Analyze Candidates</b>.</p>", unsafe_allow_html=True)

jd_col, cv_col = st.columns(2, gap="large")

# --------------------------------------------------------------------------
# JOB DESCRIPTION  (STEP 1 / 2)
# --------------------------------------------------------------------------
with jd_col:
    st.markdown(
        """
        <div style="margin-bottom: 10px;">
            <span class="step-label">STEP 1 / 2</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='font-size:20px; font-weight:700; color:#292529; margin:0 0 8px 0; line-height:1.4;'>📋 Job Description</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='font-size:15px; color:#6F6A6D; margin:0 0 18px 0; line-height:1.6;'>Upload a single JD file — PDF, DOCX, or TXT.<br>Drag &amp; drop directly into the box below, or click <b>Upload</b>.</p>",
        unsafe_allow_html=True
    )

    jd_file = st.file_uploader(
        "Drop your Job Description here, or click Upload",
        type=["pdf", "txt", "docx"],
        key="jd_upload",
        help="One job description is enough."
    )


# --------------------------------------------------------------------------
# CANDIDATE CVs  (STEP 2 / 2)
# --------------------------------------------------------------------------
_prev_cvs = st.session_state.get("resume_upload", []) or []
_has_cvs = len(_prev_cvs) > 0

with cv_col:
    st.markdown(
        """
        <div style="margin-bottom: 10px;">
            <span class="step-label">STEP 2 / 2</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='font-size:20px; font-weight:700; color:#292529; margin:0 0 8px 0; line-height:1.4;'>👥 Candidate CVs</p>",
        unsafe_allow_html=True
    )

    if _has_cvs:
        cv_count = len(_prev_cvs)
        st.markdown(
            f"<p style='font-size:15px; color:#6F6A6D; margin:0 0 14px 0; line-height:1.6;'>"
            f"<b style='color:#292529;'>{cv_count} CV{'s' if cv_count != 1 else ''}</b> ready for analysis.</p>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<p style='font-size:15px; color:#6F6A6D; margin:0 0 14px 0; line-height:1.6;'>"
            "Upload one or more candidate PDF resumes.</p>",
            unsafe_allow_html=True
        )

    # Hide the default Streamlit uploader UI entirely — we render our own
    st.markdown("""
        <style>
        /* Hide the native Streamlit file uploader widget visually,
           keep it in DOM so files still register */
        div[data-testid="stFileUploader"]:has(input[data-testid="stFileUploaderInput"]) {
            position: absolute !important;
            width: 1px !important;
            height: 1px !important;
            overflow: hidden !important;
            opacity: 0 !important;
            pointer-events: none !important;
        }
        </style>
    """, unsafe_allow_html=True)

    resume_files = st.file_uploader(
        "Upload CVs",
        type=["pdf"],
        accept_multiple_files=True,
        key="resume_upload",
        label_visibility="collapsed"
    )

    # Custom drag & drop zone using st.components
    import streamlit.components.v1 as components

    has_cvs_js = "true" if _has_cvs else "false"
    cv_count_js = len(_prev_cvs)
    cv_names_js = ", ".join(f.name for f in _prev_cvs) if _prev_cvs else ""

    components.html(f"""
    <style>
      * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}

      .drop-zone {{
        background: #F7F4F7;
        border: 2px dashed #C0B4BE;
        border-radius: 10px;
        padding: 28px 20px;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease;
        position: relative;
      }}

      .drop-zone.drag-over {{
        background: #EDE0EC;
        border-color: #541F49;
        border-style: solid;
      }}

      .drop-icon {{
        font-size: 28px;
        margin-bottom: 10px;
        display: block;
      }}

      .drop-title {{
        font-size: 15px;
        font-weight: 700;
        color: #292529;
        margin-bottom: 5px;
      }}

      .drop-sub {{
        font-size: 13px;
        color: #6F6A6D;
        margin-bottom: 16px;
      }}

      .btn-upload {{
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: #6B3A5E;
        color: #fff;
        border: none;
        border-radius: 8px;
        padding: 10px 22px;
        font-size: 15px;
        font-weight: 700;
        cursor: pointer;
        transition: background 0.15s;
      }}

      .btn-upload:hover {{ background: #541F49; }}

      .file-list {{
        margin-top: 14px;
        display: flex;
        flex-direction: column;
        gap: 8px;
        max-height: 200px;
        overflow-y: auto;
        padding-right: 4px;
      }}

      .file-list::-webkit-scrollbar {{ width: 6px; }}
      .file-list::-webkit-scrollbar-thumb {{ background: #C8C3C0; border-radius: 3px; }}

      .file-item {{
        display: flex;
        align-items: center;
        gap: 10px;
        background: #fff;
        border: 1px solid #E0D8DF;
        border-radius: 7px;
        padding: 8px 12px;
        font-size: 13px;
        color: #292529;
        font-weight: 500;
      }}

      .file-icon {{ font-size: 16px; }}
      .file-name {{ flex: 1; text-align: left; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}

      .add-more {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        background: #6B3A5E;
        color: #fff;
        border: none;
        border-radius: 8px;
        padding: 11px 20px;
        font-size: 15px;
        font-weight: 700;
        cursor: pointer;
        margin-top: 14px;
        width: 100%;
        transition: background 0.15s;
      }}

      .add-more:hover {{ background: #541F49; }}

      input[type=file] {{ display: none; }}
    </style>

    <input type="file" id="cvInput" multiple accept=".pdf" />

    <div class="drop-zone" id="dropZone">
      <span class="drop-icon">📄</span>
      <div class="drop-title">Drag &amp; drop PDF resumes here</div>
      <div class="drop-sub">or click the button below to browse your files</div>
      <button class="btn-upload" onclick="document.getElementById('cvInput').click()">
        ⬆ Upload CVs
      </button>

      <div class="file-list" id="fileList"></div>
    </div>

    {"<button class='add-more' onclick=\"document.getElementById('cvInput').click()\">＋ Add More CVs</button>" if _has_cvs else ""}

    <script>
      const dropZone = document.getElementById('dropZone');
      const cvInput  = document.getElementById('cvInput');
      const fileList = document.getElementById('fileList');

      // Show already-uploaded file names from Python
      const existingNames = `{cv_names_js}`;
      if (existingNames) {{
        existingNames.split(', ').forEach(name => {{
          if (name.trim()) addFileItem(name.trim());
        }});
      }}

      // Drag events
      dropZone.addEventListener('dragover', e => {{
        e.preventDefault();
        dropZone.classList.add('drag-over');
      }});

      dropZone.addEventListener('dragleave', () => {{
        dropZone.classList.remove('drag-over');
      }});

      dropZone.addEventListener('drop', e => {{
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        handleFiles(e.dataTransfer.files);
      }});

      // File input change
      cvInput.addEventListener('change', () => {{
        handleFiles(cvInput.files);
      }});

      function handleFiles(files) {{
        if (!files || files.length === 0) return;

        // Pass files to the Streamlit native uploader hidden in parent frame
        try {{
          const parentDoc = window.parent.document;
          const nativeInput = parentDoc.querySelector('input[data-testid="stFileUploaderInput"]');
          if (nativeInput) {{
            const dt = new DataTransfer();
            // Keep existing files
            if (nativeInput.files) {{
              Array.from(nativeInput.files).forEach(f => dt.items.add(f));
            }}
            Array.from(files).forEach(f => dt.items.add(f));
            nativeInput.files = dt.files;
            nativeInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
          }}
        }} catch(e) {{
          console.warn('Could not pass to Streamlit input:', e);
        }}

        // Show in our UI
        Array.from(files).forEach(f => addFileItem(f.name));
      }}

      function addFileItem(name) {{
        const item = document.createElement('div');
        item.className = 'file-item';
        item.innerHTML = `<span class="file-icon">📄</span><span class="file-name">${{name}}</span>`;
        fileList.appendChild(item);
      }}
    </script>
    """, height=_has_cvs * 60 + (280 if not _has_cvs else 320), scrolling=False)


# --------------------------------------------------------------------------
# ANALYZE BUTTON
# --------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
button_col1, button_col2, button_col3 = st.columns([1.5, 2, 1.5])

with button_col2:
    analyze_clicked = st.button(
        "✦ Analyze Candidates",
        type="primary",
        use_container_width=True
    )


# --------------------------------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------------------------------
if analyze_clicked:
    if not jd_file:
        st.error("⚠️ Step 1 is incomplete — please upload a Job Description before analyzing candidates.")
        st.stop()
    if not resume_files:
        st.error("⚠️ Step 2 is incomplete — please add at least one Candidate CV.")
        st.stop()

    # ── JD processing ─────────────────────────────────────────────────────
    with st.spinner("Reading job description..."):
        jd_text = extract_text_from_uploaded(jd_file)

    if not jd_text.strip():
        st.error("Couldn't extract text from the job description.")
        st.stop()

    with st.spinner("Extracting required skills from job description..."):
        required_skills = extract_jd_skills(jd_text)

    # ── PHASE 1: Parallel PDF extraction ──────────────────────────────────
    progress = st.progress(0.0, text="Extracting resume text in parallel...")
    resume_texts: dict = {}
    failed: list = []

    def extract_one(rf):
        text = extract_text_from_uploaded(rf)
        return rf.name, text

    with ThreadPoolExecutor(max_workers=min(len(resume_files), 8)) as executor:
        futures = {executor.submit(extract_one, rf): rf.name for rf in resume_files}
        done = 0
        for future in as_completed(futures):
            name = futures[future]
            try:
                fname, text = future.result()
                if text.strip():
                    resume_texts[fname] = text
                else:
                    st.warning(f"Skipping {fname} — no extractable text.")
                    failed.append(fname)
            except Exception as e:
                st.warning(f"Failed to read {name}: {e}")
                failed.append(name)
            done += 1
            progress.progress(done / len(resume_files) * 0.3, text=f"Extracted {done}/{len(resume_files)} resumes...")

    if not resume_texts:
        st.error("No resumes could be read.")
        st.stop()

    # ── PHASE 2: Local Semantic Embeddings ────────────────────────────────
    progress.progress(0.35, text="Generating local semantic embeddings...")
    names_ordered = list(resume_texts.keys())
    texts_ordered = [resume_texts[name] for name in names_ordered]

    jd_embedding = generate_embeddings((jd_text,))[0]
    resume_embeddings = generate_embeddings(tuple(texts_ordered))
    skill_embeddings = generate_embeddings(tuple(required_skills))

    semantic_scores = compute_semantic_similarity(jd_embedding, resume_embeddings)
    semantic_map = dict(zip(names_ordered, semantic_scores))
    resume_embedding_map = dict(zip(names_ordered, resume_embeddings))

    # ── PHASE 3: Parallel deterministic scoring ────────────────────────────
    progress.progress(0.4, text="Scoring resumes in parallel...")
    results: list = []

    def score_one(name):
        text = resume_texts[name]
        result = score_resume_deterministic(
            resume_text=text,
            jd_text=jd_text,
            required_skills=required_skills,
            semantic_score=semantic_map[name],
            resume_embedding=resume_embedding_map[name],
            skill_embeddings=skill_embeddings,
        )
        result["candidate_name"] = name
        return result

    with ThreadPoolExecutor(max_workers=min(len(resume_texts), 8)) as executor:
        futures = {executor.submit(score_one, n): n for n in names_ordered}
        done = 0
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                name = futures[future]
                st.warning(f"Scoring failed for {name}: {e}")
            done += 1
            progress.progress(0.4 + done / len(resume_texts) * 0.3, text=f"Scored {done}/{len(resume_texts)} resumes...")

    if not results:
        st.error("No resumes could be scored.")
        st.stop()

    progress.empty()

    # ── PHASE 4: Rank ──────────────────────────────────────────────────────
    results.sort(key=lambda r: r["final_score"], reverse=True)

    shortlisted_names = [
        r["candidate_name"]
        for r in results
        if r["final_score"] >= shortlist_threshold
    ]

    # ── RECOMMENDATION SECTION (before ranking) ────────────────────────────
    st.divider()

    st.markdown("<span class='step-label' style='margin-bottom:8px; display:inline-block;'>RESULTS</span>", unsafe_allow_html=True)

    if shortlisted_names:
        pills_html = "".join(
            f"<span class='rec-candidate-pill'>✓ {name}</span>"
            for name in shortlisted_names
        )
        st.markdown(
            f"""
            <div class="rec-section">
                <div class="rec-eyebrow">Recommendation</div>
                <div class="rec-heading">Recommended to Shortlist</div>
                <div class="rec-subtext">
                    These candidates meet or exceed the {shortlist_threshold}% screening threshold
                    and are recommended for further review.
                </div>
                <div class="rec-candidate-list">
                    {pills_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="rec-section">
                <div class="rec-eyebrow">Recommendation</div>
                <div class="rec-heading">No Candidates Recommended</div>
                <div class="rec-subtext">
                    No candidates met the {shortlist_threshold}% shortlist threshold.
                    Try lowering the threshold in the sidebar settings.
                </div>
                <div class="rec-none">
                    ⚠ No candidates to shortlist at the current threshold.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ── CANDIDATE RANKING ──────────────────────────────────────────────────
    st.markdown("<h2 style='margin-top:8px; margin-bottom:16px;'>🏆 Candidate Ranking</h2>", unsafe_allow_html=True)

    for rank, r in enumerate(results, start=1):
        shortlisted = r["final_score"] >= shortlist_threshold

        with st.container(border=True):
            top = st.columns([0.5, 3, 1, 1.5])

            with top[0]:
                st.markdown(
                    f"<h3 style='color:#292529 !important; font-size:20px; font-weight:700;'>#{rank}</h3>",
                    unsafe_allow_html=True
                )

            with top[1]:
                st.markdown(f"### {r['candidate_name']}")

            with top[2]:
                st.metric("ATS Score", f"{r['final_score']}%")

            with top[3]:
                if shortlisted:
                    st.markdown(
                        """
                        <div style="
                            background-color:#E8F5E9; color:#2E7D32;
                            padding:10px 16px; border-radius:8px;
                            font-size:15px; font-weight:700; text-align:center;
                        ">✓ Shortlisted</div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        """
                        <div style="
                            background-color:#FDECEC; color:#C62828;
                            padding:10px 17px; border-radius:8px;
                            font-size:15px; font-weight:700; text-align:center;
                        ">✕ Not shortlisted</div>
                        """,
                        unsafe_allow_html=True
                    )

            with st.expander("View Details"):
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Skill Match", f"{r.get('skill_match', 0)}%")
                c2.metric("Experience", f"{r.get('experience_score', 0)}%")
                c3.metric("Education", f"{r.get('education_score', 0)}%")
                c4.metric("Projects", f"{r.get('projects_score', 0)}%")
                c5.metric("Formatting", f"{r.get('formatting_score', 0)}%")

                d1, d2 = st.columns(2)

                with d1:
                    st.markdown("**✅ Matched Skills**")
                    for skill in r.get("matched_skills", []):
                        st.markdown(f"- {skill}")

                with d2:
                    st.markdown("**❌ Missing Skills**")
                    for skill in r.get("missing_skills", []):
                        st.markdown(f"- {skill}")