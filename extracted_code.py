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

## --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AI ATS Resume Analyzer",
    page_icon="📄",
    layout="wide"
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
    background-color: #EFEEEC;
    border-right: 1px solid #DDD9D7;
}

/* Sidebar headings */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #541F49 !important;
}

/* Sidebar normal text */
section[data-testid="stSidebar"] p {
    color: #292529 !important;
}

/* Sidebar descriptions / help text */
section[data-testid="stSidebar"] .stMarkdown p {
    color: #6F6A6D !important;
    font-size: 19px !important;
    line-height: 1.5 !important;
}


/* =========================================================
   MAIN HEADINGS
   ========================================================= */

h1 {
    color: #541F49 !important;
    font-size: 32px !important;
    font-weight: 700 !important;
}

h2 {
    color: #541F49 !important;
    font-size: 22px !important;
    font-weight: 650 !important;
}

h3 {
    color: #292529 !important;
    font-size: 18px !important;
    font-weight: 600 !important;
}


/* =========================================================
   SIDEBAR NUMBER INPUTS
   ========================================================= */

/* Compact box */
section[data-testid="stSidebar"]
div[data-testid="stNumberInput"] {
    width: 140px;
}

section[data-testid="stSidebar"]
div[data-testid="stNumberInput"] > div {
    width: 140px;
}

/* Input */
section[data-testid="stSidebar"]
div[data-testid="stNumberInput"] input {
    background-color: #FFFFFF !important;
    color: #292529 !important;

    font-size: 19px !important;
    font-weight: 600 !important;
}

/* +/- buttons */
section[data-testid="stSidebar"]
div[data-testid="stNumberInput"] button {
    width: 32px !important;
    min-width: 32px !important;

    color: #541F49 !important;
    background-color: #FFFFFF !important;
}

/* Focus */
section[data-testid="stSidebar"]
div[data-testid="stNumberInput"] input:focus {
    border-color: #541F49 !important;
    box-shadow: 0 0 0 1px #541F49 !important;
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

    transition:
        background-color 0.15s ease,
        border-color 0.15s ease !important;
}

/* Button text */
.stButton > button p,
.stButton > button span {
    color: #FFFFFF !important;
}

/* Hover ONLY while cursor is over button */
.stButton > button:hover {
    background-color: #541F49 !important;
    border-color: #541F49 !important;
    color: #FFFFFF !important;
}

/* After click / focus -> return to normal color */
.stButton > button:focus,
.stButton > button:focus-visible,
.stButton > button:active {
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
    border-radius: 8px !important;

    padding: 4px !important;
}


/* Upload dropzone */
[data-testid="stFileUploaderDropzone"] {
    background-color: #F0EFEC !important;

    border: 1px dashed #C8C3C0 !important;
    border-radius: 7px !important;
}


/* Upload label */
[data-testid="stFileUploader"] label {
    color: #292529 !important;
    font-size: 19px !important;
}


/* Upload helper text */
[data-testid="stFileUploader"] small {
    color: #6F6A6D !important;
    font-size: 19px !important;
}


/* =========================================================
   UPLOAD BUTTON
   ========================================================= */

[data-testid="stFileUploader"] button {
    background-color: #6B3A5E !important;

    border: 1px solid #6B3A5E !important;
    border-radius: 7px !important;

    min-height: 42px !important;

    padding: 0 18px !important;

    color: #FFFFFF !important;

    font-size: 19px !important;
    font-weight: 600 !important;# --------------------------------------------------------------------------
# PAGE CONTENT
# --------------------------------------------------------------------------

st.title("📄 AI ATS Resume Analyzer")

st.caption(
    "Upload multiple resumes and one job description to rank and shortlist candidates."
)
}


/* Upload button text */
[data-testid="stFileUploader"] button span,
[data-testid="stFileUploader"] button p {
    color: #FFFFFF !important;
}


/* Upload hover */
[data-testid="stFileUploader"] button:hover {
    background-color: #541F49 !important;
    border-color: #541F49 !important;
}


/* Upload button focus/click */
[data-testid="stFileUploader"] button:focus,
[data-testid="stFileUploader"] button:focus-visible,
[data-testid="stFileUploader"] button:active {
    background-color: #6B3A5E !important;
    border-color: #6B3A5E !important;

    color: #FFFFFF !important;

    box-shadow: none !important;
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

    font-size: 19px !important;
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
    font-size: 19px !important;
    font-weight: 600 !important;
}


/* =========================================================
   DIVIDERS
   ========================================================= */

hr {
    border-color: #DDD9D7 !important;
}


/* =========================================================
   CAPTIONS
   ========================================================= */

[data-testid="stCaptionContainer"] {
    color: #6F6A6D !important;
    font-size: 19px !important;
}


/* =========================================================
   ALERTS
   ========================================================= */

[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-size: 19px !important;
}


/* =========================================================
   GENERAL MARKDOWN TEXT
   ========================================================= */

.stMarkdown {
    font-size: 19px;
}


/* =========================================================
   SCROLLBAR - SUBTLE
   ========================================================= */

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #F7F6F4;
}

::-webkit-scrollbar-thumb {
    background: #C8C3C0;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #A9A2A5;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
logo_col, title_col = st.columns([0.7, 5])

with logo_col:
    st.image("assets/izra_logo_cropped.png", width=180)

with title_col:
    st.title("AI ATS Resume Analyzer")
    st.caption(
        "Upload multiple resumes and one job description "
        "to rank and shortlist candidates."
    )

if not os.environ.get("GOOGLE_API_KEY"):
    st.error(
        "GOOGLE_API_KEY not found. Add it to your .env file and restart the app."
    )
    st.stop()

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
# STEP 1: Extract required skills from JD — cached by content hash
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
    """Cached by JD content — only runs once per unique JD."""
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
    """
    Generate local embeddings for multiple texts.

    Embeddings are generated locally.
    No Gemini embedding API call is made.
    """

    model = get_embedding_model()

    embeddings = model.encode(
        list(texts),
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings


def cosine_similarity_score(a, b):
    """
    Calculate cosine similarity between two normalized vectors.
    """

    return float(np.dot(a, b))


def compute_semantic_similarity(
    jd_embedding: np.ndarray,
    resume_embeddings: np.ndarray
) -> list:
    """
    Calculate JD ↔ Resume semantic similarity locally.
    """

    similarities = np.dot(
        resume_embeddings,
        jd_embedding
    )

    return [
        round(float(score) * 100)
        for score in similarities
    ]


# --------------------------------------------------------------------------
# STEP 3: Deterministic scoring (no LLM)
# --------------------------------------------------------------------------

def compute_skill_match(
    resume_text: str,
    required_skills: list
) -> tuple:

    if not required_skills:
        return 0, [], []

    # Resume ko lines mein divide karo
    chunks = [
        line.strip()
        for line in resume_text.splitlines()
        if len(line.strip()) >= 15
    ]

    if not chunks:
        return 0, [], []

    # Local embedding model
    model = get_embedding_model()

    # Required skills embeddings
    skill_embeddings = model.encode(
        required_skills,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # Resume chunks embeddings
    chunk_embeddings = model.encode(
        chunks,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # Each skill vs every resume chunk
    similarities = np.dot(
        skill_embeddings,
        chunk_embeddings.T
    )

    matched = []
    missing = []

    MATCH_THRESHOLD = 0.50

    for skill, skill_similarities in zip(
        required_skills,
        similarities
    ):

        best_similarity = float(
            np.max(skill_similarities)
        )

        print(
            f"{skill}: {best_similarity:.4f}"
        )

        if best_similarity >= MATCH_THRESHOLD:
            matched.append(skill)
        else:
            missing.append(skill)

    score = round(
        len(matched) /
        len(required_skills) *
        100
    )

    print("\n--- SKILL MATCH DEBUG ---")
    print("Required skills:", required_skills)
    print("Matched:", matched)
    print("Missing:", missing)
    print("Skill Score:", score)

    return score, matched, missing


def score_education(resume_text: str) -> int:
    text = resume_text.lower()
    if any(w in text for w in ["phd", "doctorate"]):
        return 100
    elif any(w in text for w in ["master", "msc", "ms ", "m.s", "mba"]):
        return 85
    elif any(w in text for w in ["bachelor", "bsc", "b.s", "b.e", "beng", "degree"]):
        return 70
    elif any(w in text for w in ["diploma", "associate"]):
        return 50
    return 30


def score_experience(resume_text: str, jd_text: str) -> int:
    resume_years = re.findall(r"(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)", resume_text.lower())
    jd_years = re.findall(r"(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)", jd_text.lower())
    if resume_years and jd_years:
        candidate_exp = max(int(y) for y in resume_years)
        required_exp = max(int(y) for y in jd_years)
        if candidate_exp >= required_exp:
            return 100
        return round((candidate_exp / required_exp) * 100)
    text = resume_text.lower()
    if any(w in text for w in ["senior", "lead", "principal", "head of", "manager"]):
        return 85
    elif any(w in text for w in ["engineer", "developer", "analyst", "apprentice"]):
        return 65
    elif any(w in text for w in ["junior", "intern", "trainee", "fresher", "graduate"]):
        return 40
    return 50


def score_projects(resume_text: str) -> int:
    text = resume_text.lower()
    score = 0
    if "project" in text:
        score += 50
    if any(w in text for w in ["github", "gitlab", "bitbucket"]):
        score += 25
    if any(w in text for w in ["deployed", "production", "live", "published"]):
        score += 25
    return score


def score_formatting(resume_text: str) -> int:
    lines = [l.strip() for l in resume_text.splitlines() if l.strip()]
    score = 50
    if len(lines) > 20:
        score += 15
    if any(w in resume_text.lower() for w in ["experience", "education", "skills", "projects"]):
        score += 20
    if any(w in resume_text.lower() for w in ["@", "email", "phone", "linkedin"]):
        score += 15
    return min(score, 100)


def score_resume_deterministic(
    resume_text: str,
    jd_text: str,
    required_skills: list,
    semantic_score: int,
    resume_embedding: np.ndarray,
    skill_embeddings: np.ndarray,
) -> dict:

    skill_score, matched, missing = compute_skill_match(
        resume_text,
        required_skills
    )

    combined_skill = round(
        skill_score * 0.7 +
        semantic_score * 0.3
    )

    experience_score = score_experience(
        resume_text,
        jd_text
    )

    education_score = score_education(
        resume_text
    )

    projects_score = score_projects(
        resume_text
    )

    formatting_score = score_formatting(
        resume_text
    )

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
        "suggestions": [],
    }


# --------------------------------------------------------------------------
# STEP 4: LLM suggestions — Stage 2, top-N only
# --------------------------------------------------------------------------
SUGGESTIONS_PROMPT = PromptTemplate(
    input_variables=["job_description", "resume_text", "missing_skills"],
    template="""
You are an expert technical recruiter. A candidate is applying for the following role.

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

MISSING SKILLS (already identified):
{missing_skills}

Give 3 specific, actionable suggestions to improve this resume for this role.
Focus on what the candidate should ADD, LEARN, or HIGHLIGHT.

Respond with ONLY a valid JSON object (no markdown):
{{
  "suggestions": ["suggestion1", "suggestion2", "suggestion3"]
}}
""",
)


def get_suggestions(resume_text: str, jd_text: str, missing_skills: list) -> list:
    prompt = SUGGESTIONS_PROMPT.format(
        job_description=jd_text,
        resume_text=resume_text[:3000],
        missing_skills=", ".join(missing_skills) if missing_skills else "None",
    )
    try:
        response = get_llm().invoke(prompt)
        data = parse_json_response(response.content)
        return data.get("suggestions", [])
    except Exception:
        return ["Could not generate suggestions. Check your API key."]


def enrich_with_suggestions_parallel(
    results: list,
    resume_texts: dict,
    jd_text: str,
    top_n: int,
) -> None:
    top_results = results[:top_n]

    def fetch(r):
        name = r["candidate_name"]
        missing = [s.lower() for s in r.get("missing_skills", [])]
        r["suggestions"] = get_suggestions(resume_texts[name], jd_text, missing)
        return name

    with ThreadPoolExecutor(max_workers=min(top_n, 5)) as executor:
        futures = {executor.submit(fetch, r): r["candidate_name"] for r in top_results}
        for future in as_completed(futures):
            _ = future.result()


# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    shortlist_threshold = st.number_input(
        "Shortlist Threshold (%)",
        min_value=0, max_value=100,
        value=75, step=1,
        help="Candidates at or above this ATS score will be shortlisted.",
    )
    st.markdown(
        "<p style='color:white; font-size:18px; margin-top:2px;'>"
        "Sets the minimum ATS score a candidate must achieve to be shortlisted for review.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    top_n_suggestions = st.number_input(
        "Top  Candidates",
        min_value=1, max_value=10,
        value=5, step=1,
        help="AI suggestions generated only for top N ranked candidates.",
    )
    st.markdown(
        "<p style='color:white; font-size:18px; margin-top:2px;'>"
        "Number of top-ranked candidates who will receive detailed description</p>",
        unsafe_allow_html=True,
    )



# --------------------------------------------------------------------------
# FILE UPLOADS
# --------------------------------------------------------------------------
jd_file = st.file_uploader("Upload Job Description (PDF or TXT)", type=["pdf", "txt"])
resume_files = st.file_uploader(
    "Upload Resumes (PDF) — select multiple",
    type=["pdf"],
    accept_multiple_files=True,
)
analyze_clicked = st.button("Analyze & Rank Candidates", type="primary", use_container_width=True)


# --------------------------------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------------------------------
if analyze_clicked:
    if not jd_file:
        st.error("Please upload a job description.")
        st.stop()
    if not resume_files:
        st.error("Please upload at least one resume.")
        st.stop()

    # ── JD processing ───────────────────────────────────────────────────────
    with st.spinner("Reading job description..."):
        jd_text = extract_text_from_uploaded(jd_file)
    if not jd_text.strip():
        st.error("Couldn't extract text from the job description.")
        st.stop()

    with st.spinner("Extracting required skills from job description..."):
        required_skills = extract_jd_skills(jd_text)
    st.info(f"📋 **Required skills identified:** {', '.join(s.title() for s in required_skills)}")

    

    # ── PHASE 1: Parallel PDF extraction ────────────────────────────────────
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
    progress.progress(
        0.35,
        text="Generating local semantic embeddings..."
    )

    names_ordered = list(resume_texts.keys())

    texts_ordered = [
        resume_texts[name]
        for name in names_ordered
    ]

    jd_embedding = generate_embeddings(
        (jd_text,)
    )[0]

    resume_embeddings = generate_embeddings(
        tuple(texts_ordered)
    )

    skill_embeddings = generate_embeddings(
        tuple(required_skills)
    )

    semantic_scores = compute_semantic_similarity(
        jd_embedding,
        resume_embeddings
    )

    semantic_map = dict(
        zip(
            names_ordered,
            semantic_scores
        )
    )

    resume_embedding_map = dict(
        zip(
            names_ordered,
            resume_embeddings
        )
    )
    # ── PHASE 3: Parallel deterministic scoring ──────────────────────────────
    progress.progress(
        0.4,
        text="Scoring resumes in parallel..."
    )

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

    with ThreadPoolExecutor(
        max_workers=min(len(resume_texts), 8)
    ) as executor:

        futures = {
            executor.submit(score_one, n): n
            for n in names_ordered
        }

        done = 0

        for future in as_completed(futures):
            try:
                results.append(future.result())

            except Exception as e:
                name = futures[future]
                st.warning(
                    f"Scoring failed for {name}: {e}"
                )

            done += 1

            progress.progress(
                0.4 + done / len(resume_texts) * 0.3,
                text=f"Scored {done}/{len(resume_texts)} resumes..."
            )

    if not results:
        st.error("No resumes could be scored.")
        st.stop()


    # ── PHASE 4: Rank ────────────────────────────────────────────────────────
    results.sort(
        key=lambda r: r["final_score"],
        reverse=True
    )


    # ── PHASE 5: LLM suggestions for top-N only ──────────────────────────────
    actual_top_n = min(
        top_n_suggestions,
        len(results)
    )

    progress.progress(
        0.70,
        text=f"Generating AI suggestions for top {actual_top_n} candidates..."
    )

    enrich_with_suggestions_parallel(
        results,
        resume_texts,
        jd_text,
        actual_top_n
    )

    progress.progress(
        1.0,
        text="Done!"
    )

    progress.empty()

    # ── RANKING DISPLAY ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("🏆 Candidate Ranking")

    for rank, r in enumerate(results, start=1):
        shortlisted = r["final_score"] >= shortlist_threshold
        badge = "✅ Shortlisted" if shortlisted else "❌ Not shortlisted"
        has_suggestions = bool(r.get("suggestions"))

        with st.container(border=True):
            top = st.columns([0.5, 3, 1, 1.5])
            top[0].markdown(f"### #{rank}")
            top[1].markdown(f"**{r['candidate_name']}**")
            top[2].metric("ATS Score", f"{r['final_score']}%")
            top[3].markdown(f"**{badge}**")

            with st.expander("Details"):
                c1, c2, c3, c4, c5 = st.columns(5)

                c1.metric(
                    "Skill Match",
                    f"{r.get('skill_match', 0)}%"
                )
                c2.metric(
                    "Experience",
                    f"{r.get('experience_score', 0)}%"
                )
                c3.metric(
                    "Education",
                    f"{r.get('education_score', 0)}%"
                )
                c4.metric(
                    "Projects",
                    f"{r.get('projects_score', 0)}%"
                )
                c5.metric(
                    "Formatting",
                    f"{r.get('formatting_score', 0)}%"
                )

                d1, d2 = st.columns(2)

                with d1:
                    st.markdown("**✅ Matched Skills**")
                    for skill in r.get("matched_skills", []):
                        st.markdown(f"- {skill}")

                with d2:
                    st.markdown("**❌ Missing Skills**")
                    for skill in r.get("missing_skills", []):
                        st.markdown(f"- {skill}")

                st.markdown("**💡 Suggestions**")

                if has_suggestions:
                    for s in r["suggestions"]:
                        st.markdown(f"- {s}")
                else:
                    st.caption(
                        f"_AI suggestions generated for top "
                        f"{actual_top_n} candidates only._"
                    )


    # ── SUMMARY ─────────────────────────────────────────────────────────────
    st.divider()

    shortlisted_names = [
        r["candidate_name"]
        for r in results
        if r["final_score"] >= shortlist_threshold
    ]

    if shortlisted_names:
        st.success(
            f"✅ Recommended to shortlist: "
            f"{', '.join(shortlisted_names)}"
        )
    else:
        st.info(
            "No candidates met the shortlist threshold. "
            "Try lowering it in the sidebar."
        )

    total_llm_calls = 1 + actual_top_n

    st.caption(
        f"⚡ Total Gemini LLM calls this run: **{total_llm_calls}** "
        f"(1 JD extraction + {actual_top_n} suggestions). "
        "Semantic embeddings and ATS scoring run locally."
    )