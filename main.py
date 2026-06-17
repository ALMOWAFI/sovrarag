"""
main.py — PCB Defect Inspection RAG API

FastAPI application exposing a POST /explain-defect endpoint.
Retrieves relevant SOP/knowledge chunks via FAISS, then calls Ollama
(qwen3:4b by default) with the context to generate a structured JSON response.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import logging
import os
import random
import re

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from retrieve import retrieve

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))
TOP_K = int(os.getenv("TOP_K", "5"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models — simple input, optional context fields for dynamic responses
# ---------------------------------------------------------------------------


class DefectInput(BaseModel):
    defect_type: str = Field(..., examples=["surface_crack"])
    location: str = Field(..., examples=["top-left"])
    confidence: float = Field(..., ge=0.0, le=1.0, examples=[0.96])
    severity: str = Field(..., examples=["high"])

    # Optional context — makes responses dynamic instead of hardcoded
    factory_id: str = Field("", examples=["FACTORY-BERLIN-01"])
    factory_name: str = Field("", examples=["Berlin Electronics Plant"])
    line_id: str = Field("", examples=["LINE-A3"])
    station_id: str = Field("", examples=["AOI-04"])
    product_model: str = Field("", examples=["PCB-X100"])
    batch_id: str = Field("", examples=["BATCH-2026-05-18-A"])
    supplier_id: str = Field("", examples=["SUP-042"])


# ---------------------------------------------------------------------------
# Example pools — when optional fields are not provided, pick randomly
# so each response references different IDs instead of always the same ones
# ---------------------------------------------------------------------------

FACTORIES = [
    ("FACTORY-BERLIN-01", "Berlin Electronics Plant"),
    ("FACTORY-DRESDEN-02", "Dresden PCB Manufacturing"),
    ("FACTORY-MUNICH-03", "Munich Automotive Electronics"),
    ("FACTORY-HAMBURG-04", "Hamburg Circuit Assembly"),
    ("FACTORY-STUTTGART-05", "Stuttgart Precision Electronics"),
]

LINES = ["LINE-A1", "LINE-A2", "LINE-A3", "LINE-B1", "LINE-B2", "LINE-C1", "LINE-C2"]
STATIONS = ["AOI-01", "AOI-02", "AOI-03", "AOI-04", "AOI-05", "AOI-07", "AOI-09"]
PRODUCTS = ["PCB-X100", "PCB-X200", "PCB-X300", "PCB-X400", "PCB-X500"]
BATCHES = ["BATCH-2026-05-18-A", "BATCH-2026-05-19-B", "BATCH-2026-05-20-C", "BATCH-2026-06-01-A", "BATCH-2026-06-10-B"]
SUPPLIERS = ["SUP-042", "SUP-055", "SUP-063", "SUP-077", "SUP-081"]


def fill_defaults(defect: DefectInput) -> DefectInput:
    """Auto-fill empty optional fields with random values from the pools."""
    if not defect.factory_id:
        fac = random.choice(FACTORIES)
        defect.factory_id = fac[0]
        defect.factory_name = fac[1]
    if not defect.line_id:
        defect.line_id = random.choice(LINES)
    if not defect.station_id:
        defect.station_id = random.choice(STATIONS)
    if not defect.product_model:
        defect.product_model = random.choice(PRODUCTS)
    if not defect.batch_id:
        defect.batch_id = random.choice(BATCHES)
    if not defect.supplier_id:
        defect.supplier_id = random.choice(SUPPLIERS)
    return defect


class DefectOutput(BaseModel):
    defect_explanation: str
    severity_assessment: str
    recommended_action: str
    justification: str
    sop_references: list[str]
    confidence: float


# ---------------------------------------------------------------------------
# JSON schema sent to Ollama's `format` parameter to enforce structured output
# ---------------------------------------------------------------------------

OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "defect_explanation": {
            "type": "string",
            "description": "Detailed explanation of the defect, its nature and implications",
        },
        "severity_assessment": {
            "type": "string",
            "enum": ["low", "medium", "high", "critical"],
            "description": "Severity classification of the detected defect",
        },
        "recommended_action": {
            "type": "string",
            "enum": ["pass", "rework", "reject"],
            "description": "Recommended disposition action per quality SOPs",
        },
        "justification": {
            "type": "string",
            "description": "Reasoning for the recommended action, citing SOPs and quality criteria",
        },
        "sop_references": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of SOP or knowledge base document references used",
        },
        "confidence": {
            "type": "number",
            "description": "Detection confidence score from the inspection system (0.0 - 1.0)",
        },
    },
    "required": [
        "defect_explanation",
        "severity_assessment",
        "recommended_action",
        "justification",
        "sop_references",
        "confidence",
    ],
}


# ---------------------------------------------------------------------------
# Prompt builder — uses context fields if provided, otherwise generic
# ---------------------------------------------------------------------------


def build_prompt(defect: DefectInput, chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        source = meta.get("source_file", "Unknown")
        category = meta.get("folder_category", "Unknown")
        context_blocks.append(
            f"[Context {i} | {source} ({category})]\n{chunk['content']}"
        )

    context_text = "\n\n".join(context_blocks)

    # Build dynamic context line from optional fields
    context_parts = []
    if defect.factory_id:
        ctx = f"Factory: {defect.factory_id}"
        if defect.factory_name:
            ctx += f" ({defect.factory_name})"
        context_parts.append(ctx)
    if defect.line_id:
        context_parts.append(f"Production Line: {defect.line_id}")
    if defect.station_id:
        context_parts.append(f"Station: {defect.station_id}")
    if defect.product_model:
        context_parts.append(f"Product: {defect.product_model}")
    if defect.batch_id:
        context_parts.append(f"Batch: {defect.batch_id}")
    if defect.supplier_id:
        context_parts.append(f"Supplier: {defect.supplier_id}")

    if context_parts:
        context_line = "\n  ".join(context_parts)
        facility_info = f"""
FACILITY CONTEXT:
  {context_line}
"""
    else:
        facility_info = ""

    return f"""You are a senior PCB quality control engineer analyzing a defect flagged by an automated inspection system.

DEFECT DETECTION REPORT:
  Defect Type : {defect.defect_type.replace("_", " ").title()}
  Location    : {defect.location}
  Confidence  : {defect.confidence:.1%}
  Severity    : {defect.severity.upper()}
{facility_info}
KNOWLEDGE BASE CONTEXT (SOPs, defect definitions, quality criteria):
{context_text}

TASK:
Using the defect report and knowledge base context above, produce a structured quality assessment.
Your response MUST be a single JSON object with these fields:
  - defect_explanation   : what this defect is and its technical implications
  - severity_assessment  : one of "low" | "medium" | "high" | "critical"
  - recommended_action   : one of "pass" | "rework" | "reject"
  - justification        : clear reasoning citing specific SOPs or quality thresholds
  - sop_references       : list of source documents or SOP codes referenced
  - confidence           : {defect.confidence} (preserve the input confidence value)

IMPORTANT: Reference the specific factory, line, station, product, batch, and supplier IDs from the facility context in your response. Do not use generic or hardcoded IDs.

Output only the JSON object. No preamble, no markdown code fences."""


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sovra AI — PCB Defect Inspection RAG API",
    description=(
        "RAG-powered API for industrial PCB defect explanation and disposition. "
        "Retrieves relevant SOPs from a local FAISS knowledge base, then calls "
        "Ollama for structured JSON analysis."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health", summary="Health check")
async def health_check() -> dict:
    from pathlib import Path

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            ollama_status = "connected" if resp.status_code == 200 else "error"
            models = [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        ollama_status = "disconnected"
        models = []

    index_path = Path(os.getenv("FAISS_INDEX_PATH", "faiss_index"))
    index_ready = index_path.exists() and (index_path / "index.faiss").exists()

    return {
        "status": "ok",
        "ollama": {"status": ollama_status, "url": OLLAMA_BASE_URL, "model": OLLAMA_MODEL},
        "faiss_index": {"status": "ready" if index_ready else "not_built"},
    }


@app.post("/explain-defect", response_model=DefectOutput, summary="Analyze PCB defect")
async def explain_defect(defect: DefectInput) -> DefectOutput:
    """
    Accepts a defect detection event, retrieves relevant SOPs,
    and returns structured disposition analysis.
    """
    # Auto-fill empty context fields with random values
    defect = fill_defaults(defect)

    logger.info(
        f"Received: defect={defect.defect_type} location={defect.location} "
        f"confidence={defect.confidence:.2f} severity={defect.severity} "
        f"factory={defect.factory_id} station={defect.station_id}"
    )

    # --- Step 1: RAG retrieval ---
    try:
        chunks = retrieve(defect.defect_type, top_k=TOP_K)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail=f"Retrieval error: {exc}")

    if not chunks:
        chunks = []
        logger.warning(f"No RAG chunks for defect: {defect.defect_type}")

    # --- Step 2: Build prompt ---
    prompt = build_prompt(defect, chunks)

    # --- Step 3: Call Ollama ---
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": OUTPUT_SCHEMA,
        "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 1024},
    }

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
            response.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail=f"Cannot connect to Ollama at {OLLAMA_BASE_URL}")
    except httpx.ReadTimeout:
        raise HTTPException(status_code=504, detail=f"Ollama timeout after {OLLAMA_TIMEOUT}s")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Ollama error: {exc.response.text[:300]}")

    # --- Step 4: Parse ---
    raw_text: str = response.json().get("response", "")
    # Strip <think> blocks from thinking-mode models
    raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

    try:
        result: dict = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                raise HTTPException(status_code=500, detail=f"Invalid JSON: {raw_text[:300]}")
        else:
            raise HTTPException(status_code=500, detail=f"Invalid JSON: {raw_text[:300]}")

    # --- Step 5: Ensure required fields ---
    if not result.get("sop_references"):
        result["sop_references"] = [c["metadata"].get("source_file", "unknown") for c in chunks[:3]]

    result.setdefault("defect_explanation", "")
    result.setdefault("severity_assessment", defect.severity)
    result.setdefault("recommended_action", "rework")
    result.setdefault("justification", "")
    result.setdefault("confidence", defect.confidence)

    logger.info(f"Result: action={result['recommended_action']!r} severity={result['severity_assessment']!r}")
    return DefectOutput(**result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("API_PORT", "8000")), reload=True)
