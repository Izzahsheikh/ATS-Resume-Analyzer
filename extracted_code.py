"""
AI ATS Resume Analyzer — Streamlit UI (Hybrid Scoring Version)
---------------------------------------------------------------
Scoring is deterministic (math-based, consistent every run).
Gemini is used only for qualitative feedback (matched/missing skills, suggestions).

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

# Load API key from .env automatically
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


def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


# --------------------------------------------------------------------------
# DETERMINISTIC SCORING (consistent every run)
# --------------------------------------------------------------------------
def compute_tfidf_similarity(resume_text: str, jd_text: str) -> int:
    """Cosine similarity between resume and JD using TF-IDF. Always same result."""
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([clean_text(jd_text), clean_text(resume_text)])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score * 100)


def extract_keywords(text: str) -> set:
    """Extract meaningful keywords from text."""
    words = clean_text(text).split()
    # Filter short words and common stop words
    stop = {"and", "or", "the", "a", "an", "in", "on", "at", "to", "for",
            "of", "with", "is", "are", "was", "were", "be", "been", "have",
            "has", "had", "will", "would", "can", "could", "should", "may",
            "might", "must", "shall", "that", "this", "these", "those",
            "we", "our", "you", "your", "they", "their", "it", "its"}
    return {w for w in words if len(w) > 2 and w not in stop}


def compute_keyword_match(resume_text: str, jd_text: str) -> int:
    """What % of JD keywords appear in the resume. Deterministic."""
    jd_keywords = extract_keywords(jd_text)
    resume_keywords = extract_keywords(resume_text)
    if not jd_keywords:
        return 0
    matched = jd_keywords & resume_keywords
    return round(len(matched) / len(jd_keywords) * 100)


def score_education(resume_text: str) -> int:
    """Check for education keywords."""
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
    """
    Extract years of experience mentioned in resume vs what JD requires.
    Falls back to keyword presence scoring.
    """
    # Try to find years of experience in resume
    resume_years = re.findall(r"(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)", resume_text.lower())
    jd_years = re.findall(r"(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)", jd_text.lower())

    if resume_years and jd_years:
        candidate_exp = max(int(y) for y in resume_years)
        required_exp = max(int(y) for y in jd_years)
        if candidate_exp >= required_exp:
            return 100
        else:
            return round((candidate_exp / required_exp) * 100)

    # Fallback: keyword-based
    text = resume_text.lower()
    if any(w in text for w in ["senior", "lead", "principal", "head of", "manager"]):
        return 85
    elif any(w in text for w in ["mid", "intermediate", "engineer", "developer", "analyst"]):
        return 65
    elif any(w in text for w in ["junior", "intern", "trainee", "fresher", "graduate"]):
        return 40
    return 50


def score_projects(resume_text: str) -> int:
    """Check for projects section and depth."""
    text = resume_text.lower()
    has_projects = "project" in text
    has_github = any(w in text for w in ["github", "gitlab", "bitbucket"])
    has_deployed = any(w in text for w in ["deployed", "production", "live", "published"])
    score = 0
    if has_projects:
        score += 50
    if has_github:
        score += 25
    if has_deployed:
        score += 25
    return score


def score_formatting(resume_text: str) -> int:
    """Rough formatting quality check."""
    lines = [l.strip() for l in resume_text.splitlines() if l.strip()]
    score = 50  # base
    if len(lines) > 20:
        score += 15   # has enough content
    if any(w in resume_text.lower() for w in ["experience", "education", "skills", "projects"]):
        score += 20   # has clear sections
    if any(w in resume_text.lower() for w in ["@", "email", "phone", "linkedin"]):
        score += 15   # has contact info
    return min(score, 100)


def compute_deterministic_scores(resume_text: str, jd_text: str) -> dict:
    """All scores computed mathematically — same result every single run."""
    keyword_score = compute_keyword_match(resume_text, jd_text)
    tfidf_score = compute_tfidf_similarity(resume_text, jd_text)
    skill_match = round((keyword_score * 0.6) + (tfidf_score * 0.4))
    experience_score = score_experience(resume_text, jd_text)
    education_score = score_education(resume_text)
    projects_score = score_projects(resume_text)
    formatting_score = score_formatting(resume_text)

    # Weighted final ATS score
    ats_score = round(
        skill_match     * 0.35 +
        experience_score * 0.25 +
        education_score  * 0.15 +
        projects_score   * 0.15 +
        formatting_score * 0.10
    )

    return {
        "ats_score": ats_score,
        "skill_match": skill_match,
        "experience_score": experience_score,
        "education_score": education_score,
        "projects_score": projects_score,
        "formatting_score": formatting_score,
    }


# --------------------------------------------------------------------------
# GEMINI — qualitative feedback only (not scoring)
# --------------------------------------------------------------------------
FEEDBACK_PROMPT = PromptTemplate(
    input_variables=["job_description", "resume_chunks"],
    template="""
You are an expert technical recruiter reviewing a resume against a job description.

JOB DESCRIPTION:
{job_description}

RESUME CONTENT:
{resume_chunks}

List the skills, tools, and technologies that are:
1. Present in BOTH the resume and job description (matched)
2. Required in the job description but MISSING from the resume

Also give 3 specific, actionable suggestions to improve this resume for this role.

Respond with ONLY a valid JSON object (no markdown, no extra text):

{{
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["skill1", "skill2"],
  "suggestions": ["suggestion1", "suggestion2", "suggestion3"]
}}
""",
)


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


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_relevant_chunks(resume_text: str, jd_text: str, embeddings) -> str:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(resume_text)
    vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    relevant = retriever.invoke(jd_text)
    return "\n\n".join(doc.page_content for doc in relevant)


def get_gemini_feedback(resume_text: str, jd_text: str, embeddings) -> dict:
    """Ask Gemini ONLY for qualitative feedback, not scores."""
    retrieved = get_relevant_chunks(resume_text, jd_text, embeddings)
    prompt = FEEDBACK_PROMPT.format(job_description=jd_text, resume_chunks=retrieved)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,  # zero temperature = no randomness
    )
    response = llm.invoke(prompt)
    return parse_json_response(response.content)


# --------------------------------------------------------------------------
# MAIN ANALYSIS FUNCTION
# --------------------------------------------------------------------------
def analyze_resume(resume_text: str, jd_text: str, embeddings) -> dict:
    # Step 1: deterministic scores (fast, consistent)
    scores = compute_deterministic_scores(resume_text, jd_text)

    # Step 2: Gemini for human-readable feedback only
    try:
        feedback = get_gemini_feedback(resume_text, jd_text, embeddings)
    except Exception:
        feedback = {
            "matched_skills": [],
            "missing_skills": [],
            "suggestions": ["Could not generate suggestions. Check your API key."],
        }

    return {**scores, **feedback, "final_score": scores["ats_score"]}


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

    embeddings = get_embeddings()
    results = []

    progress = st.progress(0.0, text="Starting analysis...")
    for i, resume_file in enumerate(resume_files):
        progress.progress(i / len(resume_files), text=f"Analyzing {resume_file.name}...")
        try:
            resume_text = extract_text_from_uploaded(resume_file)
            if not resume_text.strip():
                st.warning(f"Skipping {resume_file.name} — no extractable text.")
                continue
            analysis = analyze_resume(resume_text, jd_text, embeddings)
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