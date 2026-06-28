import uuid
import logging
import os
import re
import json
from typing import List

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass 

log_level = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.WARNING))
logger = logging.getLogger(__name__)

_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not _GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable is not set. Add it to your .env file or export it.")

gemini_client = genai.Client(api_key=_GOOGLE_API_KEY)

GEMINI_MODEL = "gemini-3.5-flash"
OLLAMA_MODEL = "llama3:8b"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_TIMEOUT = 60
SLIDING_WINDOW = 6

app = FastAPI()
session_store = {}


class SimulatedEnvironmentLedger(BaseModel):
    assigned_company_profile: str = Field(description="A realistic, generic mid-sized enterprise environment")
    assigned_role_title: str = Field(description="The corporate target title matching the job description")
    simulation_timeline: str = Field(description="The duration/tenure of the simulated contract")
    allocated_tech_stack: List[str] = Field(description="4-6 specific technical tools/frameworks mapped to the role")
    core_project_scope: str = Field(description="A complex architecture/migration project scenario used as the anchor narrative")
    daily_agile_workflows: List[str] = Field(description="3-4 routine operational agile or scrum tasks performed")


class InitializeRequest(BaseModel):
    background_text: str
    job_description: str


class ChatRequest(BaseModel):
    session_id: str
    user_message: str


class InitializeResponse(BaseModel):
    session_id: str
    ledger: SimulatedEnvironmentLedger
    opening_message: str


class ChatResponse(BaseModel):
    session_id: str
    assistant_message: str


def build_system_prompt(ledger: SimulatedEnvironmentLedger) -> str:
    return f"""You are a strict professional interview coach and simulation engine. You must never deviate from the following frozen identity profile under any circumstances.

CANDIDATE IDENTITY LEDGER (IMMUTABLE):
- Company Profile: {ledger.assigned_company_profile}
- Role Title: {ledger.assigned_role_title}
- Engagement Timeline: {ledger.simulation_timeline}
- Tech Stack: {', '.join(ledger.allocated_tech_stack)}
- Core Project Scope: {ledger.core_project_scope}
- Daily Agile Workflows: {', '.join(ledger.daily_agile_workflows)}

RULES:
1. You must always frame every answer within the context of this exact profile.
2. You must never invent new companies, roles, timelines, or technologies outside this ledger.
3. When the candidate gives a weak or inconsistent answer, challenge them with a follow-up probe question.
4. Ask one focused behavioral or technical interview question per turn.
5. Keep all feedback concise, professional, and grounded in the ledger above."""


def call_gemini(system_prompt: str, messages: list) -> str:
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=system_prompt)
    )
    text = response.text
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text


def call_ollama(system_prompt: str, messages: list) -> str:
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=OLLAMA_TIMEOUT)
    payload = [{"role": "system", "content": system_prompt}] + messages
    response = client.chat.completions.create(model=OLLAMA_MODEL, messages=payload)
    return response.choices[0].message.content


def call_llm_with_fallback(system_prompt: str, messages: list) -> str:
    try:
        return call_gemini(system_prompt, messages)
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower() or "rate" in str(e).lower():
            logger.warning(f"Gemini rate limit hit, falling back to Ollama: {e}")
        else:
            logger.warning(f"Gemini call failed, falling back to Ollama: {e}")
        return call_ollama(system_prompt, messages)


def call_gemini_json(prompt: str) -> dict:
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )
    text = response.text
    if not text:
        raise RuntimeError("Gemini returned an empty JSON response.")
    return json.loads(text)


def call_ollama_json(prompt: str) -> dict:
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=OLLAMA_TIMEOUT)
    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw).rstrip("` \n")

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError("No valid JSON found in Ollama response")


def generate_ledger_json(background_text: str, job_description: str) -> dict:
    prompt = f"""You are a professional career coach. Based on the candidate background and job description below, generate a SimulatedEnvironmentLedger JSON object. Return ONLY valid JSON with no extra text, no markdown, no code fences.

The JSON must contain exactly these keys:
- assigned_company_profile: string
- assigned_role_title: string
- simulation_timeline: string
- allocated_tech_stack: array of 4-6 strings
- core_project_scope: string
- daily_agile_workflows: array of 3-4 strings

CANDIDATE BACKGROUND:
{background_text}

TARGET JOB DESCRIPTION:
{job_description}"""
    try:
        return call_gemini_json(prompt)
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower() or "rate" in str(e).lower():
            logger.warning(f"Gemini rate limit on JSON init, falling back to Ollama: {e}")
        else:
            logger.warning(f"Gemini JSON init failed, falling back to Ollama: {e}")
        return call_ollama_json(prompt)


def get_windowed_messages(chat_memory: list) -> list:
    windowed = chat_memory[-SLIDING_WINDOW:]
    # Ensure the window always starts with a user message (Gemini requirement)
    while windowed and windowed[0]["role"] != "user":
        windowed = windowed[1:]
    # Guard: if stripping left us with nothing, raise early with a clear error
    if not windowed:
        raise ValueError("Windowed messages resolved to empty after role trimming — cannot call LLM.")
    return windowed


@app.post("/api/session/initialize", response_model=InitializeResponse)
def initialize_session(request: InitializeRequest):
    try:
        raw_json = generate_ledger_json(request.background_text, request.job_description)
        ledger = SimulatedEnvironmentLedger(**raw_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ledger generation failed: {str(e)}")

    system_prompt = build_system_prompt(ledger)

    seed_messages = [{"role": "user", "content": "Please begin the interview."}]
    try:
        opening_message = call_llm_with_fallback(system_prompt, seed_messages)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to generate opening message: {str(e)}")

    session_id = str(uuid.uuid4())
    session_store[session_id] = {
        "identity_ledger": ledger,
        "chat_memory": [
            {"role": "user", "content": "Please begin the interview."},
            {"role": "assistant", "content": opening_message},
        ]
    }

    return InitializeResponse(session_id=session_id, ledger=ledger, opening_message=opening_message)


@app.post("/api/session/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    session = session_store.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    ledger: SimulatedEnvironmentLedger = session["identity_ledger"]
    chat_memory: list = session["chat_memory"]

    chat_memory.append({"role": "user", "content": request.user_message})

    system_prompt = build_system_prompt(ledger)
    windowed_messages = get_windowed_messages(chat_memory)

    try:
        assistant_reply = call_llm_with_fallback(system_prompt, windowed_messages)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Both LLM engines failed: {str(e)}")

    chat_memory.append({"role": "assistant", "content": assistant_reply})

    return ChatResponse(session_id=request.session_id, assistant_message=assistant_reply)


@app.get("/api/session/{session_id}/ledger", response_model=SimulatedEnvironmentLedger)
def get_ledger(session_id: str):
    session = session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session["identity_ledger"]


@app.delete("/api/session/{session_id}", status_code=204)
def delete_session(session_id: str):
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
    del session_store[session_id]
    return Response(status_code=204)
