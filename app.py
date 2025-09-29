import os
import streamlit as st
from multi_agents import *
from langgraph.graph import StateGraph, END
from PIL import Image
import re

def load_image(image_file):
    return Image.open(image_file)

def _extract_first_int(pattern: str, s: str):
    m = re.search(pattern, s, flags=re.IGNORECASE)
    if not m:
        return None
    val = m.group(1)
    try:
        return int(val)
    except Exception:
        return None

def parse_total_score(text: str):
    if not isinstance(text, str):
        return None
    # 0) Prefer summing simple category breakdown if present; clamp per category to cap
    try:
        skills = _extract_first_int(r"Skills[^\n:]*:\s*(\d{1,3})", text)
        if skills is None:
            skills = _extract_first_int(r"Skills\s*Match[^\n:]*:\s*(\d{1,3})", text)

        experience = _extract_first_int(r"Experience[^\n:]*:\s*(\d{1,3})", text)
        if experience is None:
            experience = _extract_first_int(r"Experience\s*Match[^\n:]*:\s*(\d{1,3})", text)

        education = _extract_first_int(r"Education[^\n:]*:\s*(\d{1,3})", text)
        if education is None:
            education = _extract_first_int(r"Education\s*Match[^\n:]*:\s*(\d{1,3})", text)

        extras = _extract_first_int(r"Extras[^\n:]*:\s*(\d{1,3})", text)
        if extras is None:
            extras = _extract_first_int(r"Certifications[^\n:]*:\s*(\d{1,3})", text)
        if extras is None:
            extras = _extract_first_int(r"Certifications\s*Match[^\n:]*:\s*(\d{1,3})", text)
        if extras is None:
            extras = _extract_first_int(r"Awards[^\n:]*:\s*(\d{1,3})", text)
        if extras is None:
            extras = _extract_first_int(r"Side\s*Projects[^\n:]*:\s*(\d{1,3})", text)

        parts = [v for v in [skills, experience, education, extras] if v is not None]
        if len(parts) >= 3:
            # clamp each category to its cap before summing
            s_val = min(skills if skills is not None else 0, 30)
            e_val = min(experience if experience is not None else 0, 50)
            ed_val = min(education if education is not None else 0, 10)
            ex_val = min(extras if extras is not None else 0, 10)
            total_breakdown = s_val + e_val + ed_val + ex_val
            if 0 <= total_breakdown <= 100:
                return total_breakdown
    except Exception:
        pass
    # 1) Prefer explicit Total Score line, take the number after colon if within 0..100
    m = re.search(r"Total\s*Score[^\n:]*:\s*(\d{1,3})(?:\s*/\s*(\d{1,3}))?", text, flags=re.IGNORECASE)
    if m:
        try:
            val = int(m.group(1))
            if 0 <= val <= 100:
                return val
        except Exception:
            pass
    # 2) Prefer numerator when denominator is 100
    m = re.search(r"(\d{1,3})\s*/\s*100", text)
    if m:
        try:
            val = int(m.group(1))
            if 0 <= val <= 100:
                return val
        except Exception:
            pass
    # 3) Handle phrasing like "85 out of 100"
    m = re.search(r"(\d{1,3})\s*out\s*of\s*100", text, flags=re.IGNORECASE)
    if m:
        try:
            val = int(m.group(1))
            if 0 <= val <= 100:
                return val
        except Exception:
            pass
    # 4) Fallback: choose largest plausible integer < 100; avoid denominators and stray 100s
    candidates = []
    for match in re.finditer(r"\b(\d{1,3})\b", text):
        num = int(match.group(1))
        if num > 100:
            continue
        start = max(0, match.start() - 20)
        end = min(len(text), match.end() + 20)
        context = text[start:end].lower()
        if ("out of" in context) or ("/100" in context):
            continue
        # Avoid picking 100 in fallback unless clearly marked as total score nearby
        if num == 100 and ("total score" not in context and "score:" not in context):
            continue
        candidates.append(num)
    if candidates:
        return max(candidates)
    return None

def normalize_recruiter_output(text: str):
    if not isinstance(text, str) or not text:
        return text
    caps = {
        "Skills": 30,
        "Skills Match": 30,
        "Experience": 50,
        "Experience Match": 50,
        "Education": 10,
        "Education Match": 10,
        "Extras": 10,
        "Certifications": 10,
        "Certifications Match": 10,
        "Awards": 10,
        "Side Projects": 10,
    }

    normalized = text

    def _cap_num(num: int, cap: int, den: int = None):
        val = min(num, cap)
        if den is not None:
            val = min(val, den)
        return val

    # Cap (NN points), ": NN points", and "NN/YY" for each label
    for label, cap in caps.items():
        # (NN points)
        normalized = re.sub(
            rf"({re.escape(label)}[^\n]*?\()(\d{{1,3}})(\s*points?\))",
            lambda m: f"{m.group(1)}{_cap_num(int(m.group(2)), cap)}{m.group(3)}",
            normalized,
            flags=re.IGNORECASE,
        )
        # : NN points
        normalized = re.sub(
            rf"({re.escape(label)}[^\n:]*:\s*)(\d{{1,3}})(\s*points?)",
            lambda m: f"{m.group(1)}{_cap_num(int(m.group(2)), cap)}{m.group(3)}",
            normalized,
            flags=re.IGNORECASE,
        )
        # NN/YY
        normalized = re.sub(
            rf"({re.escape(label)}[^\n]*?)(\d{{1,3}})\s*/\s*(\d{{1,3}})",
            lambda m: f"{m.group(1)}{_cap_num(int(m.group(2)), cap, int(m.group(3)))}/{m.group(3)}",
            normalized,
            flags=re.IGNORECASE,
        )

    # Recompute clamped total and update Total Score line if present
    s_val = _extract_first_int(r"Skills[^\n:]*:\s*(\d{1,3})", normalized) or _extract_first_int(r"Skills\s*Match[^\n:]*:\s*(\d{1,3})", normalized) or 0
    e_val = _extract_first_int(r"Experience[^\n:]*:\s*(\d{1,3})", normalized) or _extract_first_int(r"Experience\s*Match[^\n:]*:\s*(\d{1,3})", normalized) or 0
    ed_val = _extract_first_int(r"Education[^\n:]*:\s*(\d{1,3})", normalized) or _extract_first_int(r"Education\s*Match[^\n:]*:\s*(\d{1,3})", normalized) or 0
    ex_val = (
        _extract_first_int(r"Extras[^\n:]*:\s*(\d{1,3})", normalized)
        or _extract_first_int(r"Certifications[^\n:]*:\s*(\d{1,3})", normalized)
        or _extract_first_int(r"Certifications\s*Match[^\n:]*:\s*(\d{1,3})", normalized)
        or _extract_first_int(r"Awards[^\n:]*:\s*(\d{1,3})", normalized)
        or _extract_first_int(r"Side\s*Projects[^\n:]*:\s*(\d{1,3})", normalized)
        or 0
    )
    # clamp again to be safe
    s_val = min(s_val, 30)
    e_val = min(e_val, 50)
    ed_val = min(ed_val, 10)
    ex_val = min(ex_val, 10)
    total = s_val + e_val + ed_val + ex_val
    total = max(0, min(total, 100))

    # Update Total Score in common formats
    # 1) Total Score: NN/100
    if re.search(r"Total\s*Score[^\n:]*:\s*\d{1,3}\s*/\s*100", normalized, flags=re.IGNORECASE):
        normalized = re.sub(
            r"(Total\s*Score[^\n:]*:\s*)(\d{1,3})(\s*/\s*100)",
            lambda m: f"{m.group(1)}{total}{m.group(3)}",
            normalized,
            flags=re.IGNORECASE,
        )
    # 2) Total Score: NN out of 100
    if re.search(r"Total\s*Score[^\n:]*:\s*\d{1,3}\s*out\s*of\s*100", normalized, flags=re.IGNORECASE):
        normalized = re.sub(
            r"(Total\s*Score[^\n:]*:\s*)(\d{1,3})(\s*out\s*of\s*100)",
            lambda m: f"{m.group(1)}{total}{m.group(3)}",
            normalized,
            flags=re.IGNORECASE,
        )
    # 3) Total Score: NN    (case-insensitive; also handle 'Total score' lowercased)
    if re.search(r"Total\s*Score[^\n:]*:\s*\d{1,3}(?![^\n]*?/\s*100)(?![^\n]*?out\s*of\s*100)", normalized, flags=re.IGNORECASE):
        normalized = re.sub(
            r"(Total\s*Score[^\n:]*:\s*)(\d{1,3})(?![^\n]*?/\s*100)(?![^\n]*?out\s*of\s*100)",
            lambda m: f"{m.group(1)}{total}",
            normalized,
            flags=re.IGNORECASE,
        )

    return normalized

def sum_breakdown_clamped(text: str):
    if not isinstance(text, str) or not text:
        return None
    skills = _extract_first_int(r"Skills[^\n:]*:\s*(\d{1,3})", text) or _extract_first_int(r"Skills\s*Match[^\n:]*:\s*(\d{1,3})", text)
    experience = _extract_first_int(r"Experience[^\n:]*:\s*(\d{1,3})", text) or _extract_first_int(r"Experience\s*Match[^\n:]*:\s*(\d{1,3})", text)
    education = _extract_first_int(r"Education[^\n:]*:\s*(\d{1,3})", text) or _extract_first_int(r"Education\s*Match[^\n:]*:\s*(\d{1,3})", text)
    extras = (
        _extract_first_int(r"Extras[^\n:]*:\s*(\d{1,3})", text)
        or _extract_first_int(r"Certifications[^\n:]*:\s*(\d{1,3})", text)
        or _extract_first_int(r"Certifications\s*Match[^\n:]*:\s*(\d{1,3})", text)
        or _extract_first_int(r"Awards[^\n:]*:\s*(\d{1,3})", text)
        or _extract_first_int(r"Side\s*Projects[^\n:]*:\s*(\d{1,3})", text)
    )
    parts_present = [v for v in [skills, experience, education, extras] if v is not None]
    if len(parts_present) < 3:
        return None
    s_val = min(skills or 0, 30)
    e_val = min(experience or 0, 50)
    ed_val = min(education or 0, 10)
    ex_val = min(extras or 0, 10)
    total = s_val + e_val + ed_val + ex_val
    return max(0, min(total, 100))

def main():
    st.set_page_config(
        layout="wide",
        page_title="CENTAIC Resume Screening Assistant",
        page_icon="logo.jpeg",
    )

    logo_path = "logo.jpeg"
    col_logo, col_text = st.columns([1, 5])
    with col_logo:
        if os.path.exists(logo_path):
            st.image(logo_path, use_container_width=False, width=110)
    with col_text:
        st.markdown(
            """
            <div style="background-color:#ffffff;padding:8px;">
                <h2 style="margin-bottom:4px;color:#000000;">CENTAIC (Center of Artificial Intelligence and Computing) — NASTP</h2>
                <h3 style="margin-top:0;color:#333333;">Resume Screening and Matching Assistant</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Force a white background across the app
    st.markdown(
        """
        <style>
            .stApp, .stApp header, .stApp footer, .block-container, .main, .css-18e3th9, .css-1d391kg {
                background-color: #ffffff !important;
            }
            /* Light theme for widgets */
            .stButton>button, .stDownloadButton>button {
                background: #0b5fff !important;
                color: #ffffff !important;
                border: 1px solid #0b5fff !important;
                border-radius: 6px !important;
                padding: 0.5rem 1rem !important;
            }
            .stButton>button:hover, .stDownloadButton>button:hover {
                background: #084bcc !important;
                border-color: #084bcc !important;
            }
            div[data-testid="stFileUploaderDropzone"] {
                background: #fafafa !important;
                border: 1px dashed #c9c9c9 !important;
            }
            div[data-baseweb="select"]>div, .stTextInput>div>div>input, .stTextArea textarea, .stNumberInput input {
                background: #ffffff !important;
                color: #000000 !important;
                border: 1px solid #c9c9c9 !important;
            }
            div[data-baseweb="select"] svg, .stNumberInput svg {
                fill: #000000 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Controls for filtering top resumes
    col1, col2 = st.columns(2)
    with col1:
        min_score = st.number_input("Minimum score", min_value=0, max_value=100, value=85, step=1)
    with col2:
        top_n = st.number_input("Number of top resumes", min_value=1, max_value=1000, value=20, step=1)

    # Upload multiple resume PDFs or choose a folder
    pdf_files = st.file_uploader("Upload Resume(s) (PDF)", type=["pdf"], accept_multiple_files=True)
    folder_path = st.text_input(
        "Or process all PDFs in folder",
        value="/mnt/data/office_work/resume_extractor/pdf_resumes/CVs of Applicants",
    )

    def save_uploaded_pdfs(files):
        saved_paths = []
        if not files:
            return saved_paths
        os.makedirs("uploads", exist_ok=True)
        for file in files:
            dest_path = os.path.join("uploads", file.name)
            with open(dest_path, "wb") as f:
                f.write(file.read())
            saved_paths.append(dest_path)
        return saved_paths

    # Upload JD text or paste manually
    text_file = st.file_uploader("Upload Job Description (TXT)", type=["txt"])
    job_description = ""
    if text_file is not None:
        job_description = text_file.read().decode("utf-8", errors="ignore")
    else:
        job_description = st.text_area("Or paste the Job Description here:")

    # Save job description to file
    if job_description.strip() != "":
        with open("JD.txt", "w", encoding="utf-8") as f:
            f.write(job_description)

    # Start pipeline
    if st.button("Match Resume(s)"):
        # Build list of resume paths
        resume_paths = []
        resume_paths.extend(save_uploaded_pdfs(pdf_files))
        if folder_path and os.path.isdir(folder_path):
            for name in os.listdir(folder_path):
                if name.lower().endswith(".pdf"):
                    resume_paths.append(os.path.join(folder_path, name))
        # Deduplicate while preserving order
        seen = set()
        resume_paths = [p for p in resume_paths if not (p in seen or seen.add(p))]

        if not resume_paths:
            st.warning("Please upload at least one resume PDF or provide a valid folder path.")
            return

        if not job_description.strip():
            st.warning("Please provide a Job Description (upload a TXT or paste text).")
            return

        # Define LangGraph workflow (compile once)
        workflow = StateGraph(AgentState)
        workflow.add_node("Resume_agent", agent)
        workflow.add_node("JD_agent", JD_agent)
        workflow.add_node("Redflag_agent", redflag_agent)   
        workflow.add_node("Recruiter_agent", recruit_agent)

        workflow.set_entry_point("Resume_agent")

        workflow.add_edge("Resume_agent", "JD_agent")
        workflow.add_edge("Resume_agent", "Redflag_agent")
        workflow.add_edge("JD_agent", "Recruiter_agent")
        workflow.add_edge("Redflag_agent", "Recruiter_agent")
        workflow.add_edge("Recruiter_agent", END)
        app = workflow.compile()

        # Process each resume, collect scores with progress
        scored = []
        progress_bar = st.progress(0)
        status = st.empty()
        total = len(resume_paths)
        with st.spinner("Processing resumes..."):
            for idx, resume_path in enumerate(resume_paths):
                status.write(f"Processing: {os.path.basename(resume_path)} ({idx+1}/{total})")
                inputs = {
                    "messages": [
                        "You are a recruitment expert and your role is to match a candidate's profile with a given job description."
                    ],
                    "resume_path": resume_path,
                    "jd_text": job_description,
                }

                outputs = app.stream(inputs)
                messages_collected = []
                recruiter_output_texts = []
                for output in outputs:
                    for key, value in output.items():
                        messages = value.get("messages", [])
                        for msg in messages:
                            text_msg = str(msg)
                            if key == "Recruiter_agent":
                                text_msg = normalize_recruiter_output(text_msg)
                            messages_collected.append((key, text_msg))
                            if key == "Recruiter_agent":
                                recruiter_output_texts.append(text_msg)

                # Prefer the last recruiter output as final; reconcile score using clamped breakdown
                recruiter_text = recruiter_output_texts[-1] if recruiter_output_texts else ""
                breakdown_sum = sum_breakdown_clamped(recruiter_text)
                score_from_text = parse_total_score(recruiter_text)
                # Prefer clamped breakdown sum when available to avoid LLM inconsistencies
                if breakdown_sum is not None:
                    score = breakdown_sum
                else:
                    score = score_from_text
                scored.append({
                    "resume_path": resume_path,
                    "score": score if score is not None else -1,
                    "details": messages_collected,
                })

                progress_bar.progress(int((idx + 1) / total * 100))
        status.write("Processing complete.")

        # Filter and sort
        filtered = [s for s in scored if s["score"] is not None and s["score"] >= min_score]
        filtered.sort(key=lambda x: x["score"], reverse=True)
        top = filtered[: top_n]
        # Guard: drop any entries with empty filenames (platform-specific edge cases)
        top = [t for t in top if os.path.basename(t["resume_path"]).strip()]
        print("Top resumes: ", top)
        # Display
        st.markdown("## 🔝 Top Resumes")
        if not top:
            st.info("No resumes met the score threshold.")
        for item in top:
            st.subheader(f"{os.path.basename(item['resume_path'])} — Score: {item['score']}")
            with st.expander("Show details"):
                for key, text_msg in item["details"]:
                    st.markdown(f"**{key} Output:** {text_msg}")

if __name__ == "__main__":
    main()

