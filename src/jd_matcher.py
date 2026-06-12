"""
JD Matcher Module
Matches resume content against a Job Description and provides scoring.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Matching prompt template
MATCHING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert AI recruiter and resume analyst. Your job is to compare a candidate's resume against a job description and provide a detailed, structured analysis.

You MUST respond in the EXACT format shown below. Do not deviate from this format.

---
## 🎯 Match Score: [X]/100

## ✅ Matching Skills & Qualifications
- [List each matching skill/qualification found in both resume and JD]

## ❌ Missing Skills & Gaps
- [List each skill/qualification in JD but NOT in resume]

## 💡 Suggestions for Improvement
- [Actionable suggestions for the candidate to improve their resume for this role]

## 📊 Summary
[2-3 sentence overall assessment of the candidate's fit for this role]
---

Be thorough, specific, and constructive in your analysis."""),
    ("human", """Please analyze the following resume against the job description:

**RESUME:**
{resume_text}

**JOB DESCRIPTION:**
{jd_text}

Provide your detailed matching analysis:"""),
])


def match_resume_with_jd(resume_text: str, jd_text: str, llm) -> str:
    """
    Match a resume against a job description using the LLM.
    
    Args:
        resume_text: The full resume text
        jd_text: The job description text
        llm: LangChain LLM instance
    
    Returns:
        Formatted analysis string with score, matching skills, gaps, and suggestions
    """
    chain = MATCHING_PROMPT | llm | StrOutputParser()

    result = chain.invoke({
        "resume_text": resume_text,
        "jd_text": jd_text,
    })

    return result


def extract_match_score(analysis: str) -> int:
    """
    Extract the numeric match score from the analysis text.
    
    Args:
        analysis: The full analysis text from the LLM
    
    Returns:
        Integer score (0-100), or -1 if not found
    """
    import re

    # Try to find patterns like "Match Score: 75/100" or "Score: 85/100"
    patterns = [
        r'Match Score:\s*(\d+)\s*/\s*100',
        r'Score:\s*(\d+)\s*/\s*100',
        r'(\d+)\s*/\s*100',
        r'Match Score:\s*(\d+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, analysis, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            return min(max(score, 0), 100)  # Clamp between 0-100

    return -1
