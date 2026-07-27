# AI ATS Resume Analyzer

An AI-powered tool that evaluates resumes the way an Applicant Tracking System (ATS) would — scoring, parsing, and flagging gaps against a target job description to help candidates optimize their resumes before submitting them.

## Overview

Most resumes are filtered by automated ATS software before a human ever sees them. This project analyzes a resume's structure, keywords, and formatting against a given job description, then reports back on match quality, missing keywords, and areas to improve.

## Features

- Resume parsing (PDF/text extraction)
- Job description keyword matching
- ATS compatibility scoring
- Gap analysis — highlights missing skills/keywords
- Actionable suggestions to improve match rate

## Tech Stack

- Python
- Jupyter Notebook
- NLP libraries (see `requirements.txt`)

## Setup

```bash
git clone https://github.com/Izzahsheikh/ATS-Resume-Analyzer.git
cd ATS-Resume-Analyzer
pip install -r requirements.txt
```

Open `AI_ATS_Resume_Analyzer_.ipynb` in Jupyter to run the analysis.

## Usage

1. Provide a resume file (PDF or text)
2. Provide a target job description
3. Run the notebook to get a match score, keyword gaps, and improvement suggestions

## Status

Actively being developed and refined.
