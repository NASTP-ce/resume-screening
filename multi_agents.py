import warnings
warnings.filterwarnings("ignore")

import operator
import os
import requests
from typing import Annotated, Literal, TypedDict, Sequence
from langgraph.graph import END, START, StateGraph, MessagesState
try:
    # Preferred modern package
    from langchain_ollama import ChatOllama as LCChatOllama
except Exception:
    # Fallback to community package if the new one isn't installed yet
    from langchain_community.chat_models import ChatOllama as LCChatOllama
    warnings.warn(
        "Using ChatOllama from langchain_community. Install 'langchain-ollama' and switch to the new import.",
        category=DeprecationWarning,
        stacklevel=2,
    )
from langchain_core.messages import BaseMessage
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

load_dotenv()

# Initialize local open-source LLM via Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

def _ensure_ollama_available() -> None:
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if resp.status_code != 200:
            raise RuntimeError(f"Unexpected status {resp.status_code}")
    except Exception as ex:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}.\n"
            f"- Install: https://ollama.com/download\n"
            f"- Start server: 'ollama serve' (or ensure the service is running)\n"
            f"- Optional: export OLLAMA_MODEL (e.g., 'llama3', 'mistral', 'qwen2.5:7b')\n"
        ) from ex

_ensure_ollama_available()
llm = LCChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)

# TypedDict for AgentState
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # Additional state passed in by the app
    resume_path: str
    jd_text: str



# ----------------- Resume Name Agent -----------------
def agent(agentState: AgentState):
    try:
        pdf_file = agentState.get('resume_path', "Resume.pdf")
        data = PyPDFLoader(pdf_file).load()
        resume_text = " ".join([page.page_content for page in data])
        response = llm.invoke(
            f"Your task is to extract the candidate name and contact details from the resume data. "
            f"Only respond with the candidate name, contact details and nothing else. Resume Data: {resume_text}"
        )
        answer = response.content
    except Exception as ex:
        answer = f"Error extracting name: {ex}"
    return {"messages": [answer]}


# ----------------- Job Description Agent -----------------
def JD_agent(agentState: AgentState):
    try:
        jd_data = agentState.get('jd_text')
        if not jd_data:
            with open("JD.txt", "r") as f:
                jd_data = f.read()
        response = llm.invoke(
            f"Your task is to extract the exact job requirements from the given data. "
            f"Only respond with the job requirements and nothing else. Data: {jd_data}"
        )
        result = response.content.replace("\n", "")
    except Exception as ex:
        result = f"Error extracting job description: {ex}"
    return {"messages": [result]}


# ----------------- Red Flag Detection Agent -----------------
def redflag_agent(agentState: AgentState):
    try:
        pdf_file = agentState.get('resume_path', "Resume.pdf")
        data = PyPDFLoader(pdf_file).load()
        resume_text = " ".join([page.page_content for page in data])

        prompt = f"""
        You are a Resume Screening Assistant.

        Your task is to analyze the candidate's resume and identify any potential **red flags** or **concerns** a recruiter might have.

        Look for the following:
        - Frequent job switching (e.g., jobs lasting <1 year repeatedly)
        - Unexplained employment gaps
        - Lack of relevant experience for technical claims
        - Missing education details
        - Irrelevant experience
        - Spelling or grammar issues

        Return a list of clear points like:
        - "Employment gap between 2020–2022"
        - "Mentions Python skills but no project or job experience using it"
        - "No education information found"

        Resume Data: {resume_text}
        """

        response = llm.invoke(prompt)
        result = response.content
    except Exception as ex:
        result = f"Error in redflag agent: {ex}"
    return {"messages": [result]}



# ----------------- Recruit Agent (Evaluation) -----------------
def recruit_agent(agentState: AgentState):
    try:
        pdf_file = agentState.get('resume_path', "Resume.pdf")
        data = PyPDFLoader(pdf_file).load()
        resume_text = " ".join([page.page_content for page in data])
        messages = agentState['messages']
        jd_data = agentState.get('jd_text')
        if not jd_data:
            jd_data = str(messages[-2]) + " " + str(messages[-1])

        prompt = f"""
        You are a Recruitment AI Assistant.

        Your task is to evaluate how well a candidate’s resume matches a given job description and assign a score out of 100 based on the criteria below.

        Scoring Criteria:
        - **Skills Match: 30 points**
        - **Experience Match: 50 points**
            - ⚠️ Do NOT award experience points for roles unrelated to the job description.
            - For freshers:
                - Evaluate based on relevant internships, academic projects, or personal/portfolio work that aligns with the job.
            - For experienced candidates:
                - 0–30 pts: Award based on **years of relevant experience** (e.g., 10 pts per relevant year).
                - 0–20 pts: Award based on **quality, relevance, and impact** of work (e.g., problem-solving, outcomes, tools used).
        - **Education Match: 10 points**
            - ⚠️ If the education does NOT match the required fields (e.g., Computer Science, Data Science, AI, or related fields), assign **0 points**, regardless of degree level.
        - **Extras (Certifications, Awards, Side Projects): 10 points**

        Instructions:
        - Extract and compare the candidate’s **skills**, **experience**, **education**, and **additional qualifications** to the job description.
        - Apply the scoring rules strictly, especially for experience and education.
        - Do not award points for irrelevant experience.

        After evaluation, return:
        1. **Total score (out of 100)**
        2. **Score breakdown by category** (e.g., Skills: 24/30, Experience: 32/50)
        3. **A short summary** (3–4 lines) covering major strengths and missing areas.
        4. **A final recommendation**, based on these rules:
            - If the candidate scores **above 75** and meets the key job requirements:
                - Say: **✅ I recommend this candidate for the job.**
            - If the candidate scores **between 50 and 75**, with partial matches in skills or experience:
                - Say: **❌ I do not recommend this candidate for this specific job.**
                - Follow with: **However, I recommend this candidate for an internship or entry-level position, as they show foundational potential.**
            - If the candidate scores **below 50**:
                - Say: **❌ I do not recommend this candidate for the job.**
                - Follow with a reason based on the biggest gaps (skills, experience, or education).

        Resume Data:
        {resume_text}

        Job Description:
        {jd_data}
        """




        response = llm.invoke(prompt)
        answer = response.content
    except Exception as ex:
        answer = f"Error in recruit agent: {ex}"
    return {"messages": [answer]}
