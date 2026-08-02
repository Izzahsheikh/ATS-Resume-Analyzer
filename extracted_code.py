"""
AI ATS Resume Analyzer — Streamlit UI (Multi-Resume Version)
--------------------------------------------------------------
Lets an HR user upload MULTIPLE resumes (PDF) against a single job
description (PDF/TXT), scores each candidate, ranks them, and
highlights who should be shortlisted.

Run with:
    streamlit run app.py
"""

import os
import re
import json
import tempfile

import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate


# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(page_title="AI ATS Resume Analyzer", page_icon="📄", layout="wide")
st.title("📄 AI ATS Resume Analyzer")
st.caption("Upload multiple resumes and one job description to rank and shortlist candidates.")


# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------
def extract_text_from_uploaded(uploaded_file) -> str:
    """Extract text from an uploaded PDF or TXT file (Streamlit UploadedFile)."""
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
    """Extract a JSON object from the model's response, even if wrapped in markdown fences."""
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


def calculate_final_score(analysis: dict) -> int:
    weights = {
        "skill_match": 0.35,
        "experience_score": 0.25,
        "education_score": 0.15,
        "projects_score": 0.15,
        "formatting_score": 0.10,
    }
    weighted_sum = sum(analysis.get(key, 0) * weight for key, weight in weights.items())
    return round(weighted_sum)


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["job_description", "resume_chunks"],
    template="""
You are an expert ATS (Applicant Tracking System) and technical recruiter.

Compare the RESUME CONTENT below against the JOB DESCRIPTION and evaluate the match.

JOB DESCRIPTION:
{job_description}

RELEVANT RESUME CONTENT:
{resume_chunks}

Analyze the match and respond with ONLY a valid JSON object (no markdown, no extra text)
in exactly this format:

{{
  "ats_score": <integer 0-100>,
  "skill_match": <integer 0-100>,
  "experience_score": <integer 0-100>,
  "education_score": <integer 0-100>,
  "projects_score": <integer 0-100>,
  "formatting_score": <integer 0-100>,
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["skill1", "skill2"],
  "suggestions": ["suggestion1", "suggestion2", "suggestion3"]
}}
""",
)


def analyze_resume(resume_text: str, jd_text: str, embeddings) -> dict:
    """Run the full pipeline for a single resume against the job description."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ".", " "]
    )
    resume_chunks = splitter.split_text(resume_text)

    # Fresh in-memory collection per resume so candidates don't mix
    vectorstore = Chroma.from_texts(texts=resume_chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    relevant_chunks = retriever.invoke(jd_text)
    retrieved_resume_text = "\n\n".join(doc.page_content for doc in relevant_chunks)

    final_prompt = PROMPT_TEMPLATE.format(
        job_description=jd_text, resume_chunks=retrieved_resume_text
    )
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2)
    response = llm.invoke(final_prompt)

    analysis = parse_json_response(response.content)
    analysis["final_score"] = analysis.get("ats_score") or calculate_final_score(analysis)
    return analysis


# --------------------------------------------------------------------------
# SIDEBAR — API KEY
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    api_key_input = st.text_input(
        "Google Gemini API Key",
        type="password",
        value=os.environ.get("GOOGLE_API_KEY", ""),
        help="Get a free key at https://aistudio.google.com/app/apikey",
    )
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input

    shortlist_threshold = st.slider(
        "Shortlist threshold (ATS score %)", min_value=0, max_value=100, value=70
    )


# --------------------------------------------------------------------------
# MAIN — FILE UPLOADS
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
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("Please enter your Google Gemini API key in the sidebar.")
        st.stop()
    if not jd_file:
        st.error("Please upload a job description.")
        st.stop()
    if not resume_files:
        st.error("Please upload at least one resume.")
        st.stop()

    with st.spinner("Reading job description..."):
        jd_text = extract_text_from_uploaded(jd_file)
    if not jd_text.strip():
        st.error("Couldn't extract any text from the job description.")
        st.stop()

    embeddings = get_embeddings()
    results = []

    progress = st.progress(0.0, text="Starting analysis...")
    for i, resume_file in enumerate(resume_files):
        progress.progress(
            i / len(resume_files), text=f"Analyzing {resume_file.name}..."
        )
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

    # Rank by final_score, descending
    results.sort(key=lambda r: r["final_score"], reverse=True)

    # ---------------- RANKING TABLE ----------------
    st.divider()
    st.subheader("🏆 Candidate Ranking")

    for rank, r in enumerate(results, start=1):
        shortlisted = r["final_score"] >= shortlist_threshold
        badge = "✅ Shortlist" if shortlisted else "—"

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
                for suggestion in r.get("suggestions", []):
                    st.markdown(f"- {suggestion}")

    # ---------------- SUMMARY ----------------
    st.divider()
    shortlisted_names = [r["candidate_name"] for r in results if r["final_score"] >= shortlist_threshold]
    if shortlisted_names:
        st.success(f"Recommended to shortlist: {', '.join(shortlisted_names)}")
    else:
        st.info("No candidates met the shortlist threshold. Consider lowering it in the sidebar.")