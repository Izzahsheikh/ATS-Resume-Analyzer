"""
AI ATS Resume Analyzer — Streamlit UI (Hybrid Scoring Version)
---------------------------------------------------------------
- Gemini extracts required skills from JD (domain-aware, not generic keywords)
- Math computes all scores deterministically
- Gemini also provides matched/missing skills and suggestions
- Scores are stable and consistent every run

Run with:
    streamlit run extracted_code.py
"""

import os
import re
import json
import tempfile
import string

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(page_title="AI ATS Resume Analyzer", page_icon="📄", layout="wide")
st.title("📄 AI ATS Resume Analyzer")
st.caption("Upload multiple resumes and one job description to rank and shortlist candidates.")

if not os.environ.get("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not found. Add it to your .env file and restart the app.")
    st.stop()


# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------
def extract_text_from_uploaded(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    if suffix == ".pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        reader = PdfReader(tmp_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        os.remove(tmp_path)
        return text
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")


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
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# --------------------------------------------------------------------------
# STEP 1: Extract required skills from JD using Gemini (done once per JD)
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
    """Extract required skills from JD using Gemini. Cached so runs only once."""
    prompt = JD_SKILLS_PROMPT.format(job_description=jd_text)
    response = get_llm().invoke(prompt)
    data = parse_json_response(response.content)
    return [s.lower() for s in data.get("required_skills", [])]


# --------------------------------------------------------------------------
# STEP 2: Deterministic scoring
# --------------------------------------------------------------------------
def compute_skill_match(resume_text: str, required_skills: list) -> tuple[int, list, list]:
    """
    Check which required skills appear in the resume.
    Returns (score, matched_list, missing_list).
    Deterministic — same input always same output.
    """
    resume_lower = resume_text.lower()
    matched = []
    missing = []
    for skill in required_skills:
        # Match whole word or phrase
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, resume_lower):
            matched.append(skill)
        else:
            missing.append(skill)
    score = round(len(matched) / len(required_skills) * 100) if required_skills else 0
    return score, matched, missing


def compute_tfidf_similarity(resume_text: str, jd_text: str) -> int:
    """Cosine similarity between resume and JD. Always same result."""
    def clean(t):
        t = t.lower()
        return t.translate(str.maketrans("", "", string.punctuation))
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([clean(jd_text), clean(resume_text)])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score * 100)


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


# --------------------------------------------------------------------------
# STEP 3: Gemini suggestions (qualitative only, not scoring)
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
        resume_text=resume_text[:3000],  # limit tokens
        missing_skills=", ".join(missing_skills) if missing_skills else "None",
    )
    try:
        response = get_llm().invoke(prompt)
        data = parse_json_response(response.content)
        return data.get("suggestions", [])
    except Exception:
        return ["Could not generate suggestions. Check your API key."]


# --------------------------------------------------------------------------
# MAIN ANALYSIS
# --------------------------------------------------------------------------
def analyze_resume(resume_text: str, jd_text: str, required_skills: list) -> dict:
    # Skill match using JD-extracted skills (domain-aware)
    skill_score, matched, missing = compute_skill_match(resume_text, required_skills)

    # Boost skill score with TF-IDF similarity
    tfidf_score = compute_tfidf_similarity(resume_text, jd_text)
    combined_skill = round(skill_score * 0.7 + tfidf_score * 0.3)

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

    # Gemini only for suggestions
    suggestions = get_suggestions(resume_text, jd_text, missing)

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
        "suggestions": suggestions,
    }


# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    shortlist_threshold = st.slider(
        "Shortlist threshold (ATS score %)", min_value=0, max_value=100, value=70
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
# ANALYSIS
# --------------------------------------------------------------------------
if analyze_clicked:
    if not jd_file:
        st.error("Please upload a job description.")
        st.stop()
    if not resume_files:
        st.error("Please upload at least one resume.")
        st.stop()

    with st.spinner("Reading job description..."):
        jd_text = extract_text_from_uploaded(jd_file)
    if not jd_text.strip():
        st.error("Couldn't extract text from the job description.")
        st.stop()

    # Extract required skills from JD once (cached)
    with st.spinner("Extracting required skills from job description..."):
        required_skills = extract_jd_skills(jd_text)
    st.info(f"📋 **Required skills identified:** {', '.join(s.title() for s in required_skills)}")

    results = []
    progress = st.progress(0.0, text="Starting analysis...")

    for i, resume_file in enumerate(resume_files):
        progress.progress(i / len(resume_files), text=f"Analyzing {resume_file.name}...")
        try:
            resume_text = extract_text_from_uploaded(resume_file)
            if not resume_text.strip():
                st.warning(f"Skipping {resume_file.name} — no extractable text.")
                continue
            analysis = analyze_resume(resume_text, jd_text, required_skills)
            analysis["candidate_name"] = resume_file.name
            results.append(analysis)
        except Exception as e:
            st.warning(f"Failed to analyze {resume_file.name}: {e}")

    progress.progress(1.0, text="Done")
    progress.empty()

    if not results:
        st.error("No resumes could be analyzed.")
        st.stop()

    results.sort(key=lambda r: r["final_score"], reverse=True)

    # ---------------- RANKING ----------------
    st.divider()
    st.subheader("🏆 Candidate Ranking")

    for rank, r in enumerate(results, start=1):
        shortlisted = r["final_score"] >= shortlist_threshold
        badge = "✅ Shortlist" if shortlisted else "❌ Not shortlisted"

        with st.container(border=True):
            top = st.columns([0.5, 3, 1, 1.5])
            top[0].markdown(f"### #{rank}")
            top[1].markdown(f"**{r['candidate_name']}**")
            top[2].metric("ATS Score", f"{r['final_score']}%")
            top[3].markdown(f"**{badge}**")

            with st.expander("Details"):
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

                st.markdown("**💡 Suggestions**")
                for s in r.get("suggestions", []):
                    st.markdown(f"- {s}")

    # ---------------- SUMMARY ----------------
    st.divider()
    shortlisted_names = [
        r["candidate_name"] for r in results if r["final_score"] >= shortlist_threshold
    ]
    if shortlisted_names:
        st.success(f"✅ Recommended to shortlist: {', '.join(shortlisted_names)}")
    else:
        st.info("No candidates met the shortlist threshold. Try lowering it in the sidebar.")