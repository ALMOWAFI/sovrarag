"""
main.py — PCB Defect Inspection RAG API (v2)

Accepts full factory/line/station/product/supplier/batch context + CV output,
retrieves relevant SOPs via FAISS, and calls Ollama for structured disposition.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import logging
import os
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
# Pydantic input models — mirrors the test case spreadsheet columns
# ---------------------------------------------------------------------------


class FactoryInfo(BaseModel):
    factory_id: str = Field(..., examples=["FACTORY-BERLIN-01"])
    factory_name: str = Field("", examples=["Berlin Electronics Plant"])
    factory_type: str = Field("", examples=["automotive electronics manufacturing"])
    region: str = Field("", examples=["Germany"])
    branch: str = Field("", examples=["Berlin"])
    quality_standard: str = Field("", examples=["OEM-QA-Level-2"])


class ProductionLineInfo(BaseModel):
    production_line_id: str = Field(..., examples=["LINE-A3"])
    production_line_name: str = Field("", examples=["PCB Assembly Line A3"])
    line_function: str = Field("", examples=["post-etching PCB inspection"])
    recent_yield_7d: str = Field("", examples=["96.8%"])
    average_daily_volume: str = Field("", examples=["3200_boards"])


class StationInfo(BaseModel):
    station_id: str = Field(..., examples=["AOI-04"])
    station_type: str = Field("", examples=["automated_optical_inspection"])
    station_function: str = Field("", examples=["visual defect inspection after etching"])
    machine_id: str = Field("", examples=["AOI-MACHINE-04"])
    camera_id: str = Field("", examples=["CAM-TOP-01"])
    camera_view: str = Field("", examples=["top_camera"])
    calibration_id: str = Field("", examples=["CAL-2026-05-15"])
    calibration_date: str = Field("", examples=["2026-05-15"])
    calibration_type: str = Field("", examples=["camera_and_lighting_recalibration"])
    linked_machine: str = Field("", examples=["HANDLING-ROBOT-02"])
    linked_machine_function: str = Field("", examples=["automated PCB transfer"])


class ProductInfo(BaseModel):
    product_model: str = Field(..., examples=["PCB-X100"])
    part_number: str = Field("", examples=["PN-PCB-X100-REV-B"])
    product_name: str = Field("", examples=["Main Control PCB"])
    product_description: str = Field("", examples=["PCB used in vehicle body control"])
    affected_system: str = Field("", examples=["body_control"])
    subsystem: str = Field("", examples=["lighting_control"])
    safety_criticality: str = Field("", examples=["medium"])
    revision: str = Field("", examples=["Rev-B"])
    qualification_status: str = Field("", examples=["production"])
    surface_finish: str = Field("", examples=["standard_copper_trace_finish"])
    visual_sensitivity: str = Field("", examples=["high_reflective_copper_regions"])
    board_zone_map_available: bool = Field(False)
    critical_features: str = Field("", examples=["fine conductor traces near handling path"])


class SupplierInfo(BaseModel):
    supplier_id: str = Field("", examples=["SUP-042"])
    supplier_name: str = Field("", examples=["Demo PCB Supplier"])
    supplied_material: str = Field("", examples=["FR-4 copper-clad laminate"])
    supplier_quality_status: str = Field("normal", examples=["normal"])
    supplier_lot: str = Field("", examples=["LOT-CU-2026-0518-77"])
    supplier_quality_case_id: str = Field("", examples=["SQC-2026-0518-042"])
    incoming_inspection_status: str = Field("", examples=["passed"])
    supplier_issue_found: bool = Field(False)


class BatchInfo(BaseModel):
    batch_id: str = Field(..., examples=["BATCH-2026-05-18-A"])
    lot_id: str = Field("", examples=["LOT-PCB-X100-044"])
    work_order_id: str = Field("", examples=["WO-2026-8812"])
    batch_size: str = Field("", examples=["1800_boards"])
    batch_status: str = Field("in_progress", examples=["in_progress"])
    quarantine_status: str = Field("false", examples=["false"])
    qualification_batch: bool = Field(False)


class DefectHistory(BaseModel):
    defect_code: str = Field(..., examples=["BMFO"])
    same_defect_count_30d: int = Field(0)
    same_defect_count_90d: int = Field(0)
    scope: str = Field("", examples=["same_factory_same_line_same_station_same_product"])
    trend: str = Field("", examples=["increasing"])
    most_common_shift: str = Field("", examples=["Afternoon"])
    recurring_location: str = Field("", examples=["center_or_large_crop_area"])
    same_location_count_30d: int = Field(0)
    multi_critical_defect_count_14d: int = Field(0)
    affected_lines: str = Field("", examples=["LINE-A3,LINE-B1"])
    affected_factories: str = Field("", examples=["FACTORY-BERLIN-01,FACTORY-DRESDEN-02"])
    functional_zone: bool = Field(False)
    functional_zone_description: str = Field("")
    coordinate_region: str = Field("")
    board_zone: str = Field("", examples=["functional_area"])
    sc_detection_rate_before_calibration: str = Field("")
    sc_detection_rate_after_calibration: str = Field("")
    manual_review_confirmation_rate: str = Field("")


class RepairRecords(BaseModel):
    related_repair_count_30d: int = Field(0)
    common_repair_action: str = Field("", examples=["cleaned affected board and reinspected"])
    repair_result: str = Field("", examples=["3 cleaned_and_passed"])
    repair_id: str = Field("", examples=["RPR-2026-00421"])
    repair_action: str = Field("")
    repair_verification: str = Field("")
    recurrence_after_repair: bool = Field(False)
    recurrence_after_repair_days: int = Field(0)
    engineering_rework_failed: int = Field(0)


class MaintenanceLogs(BaseModel):
    latest_maintenance_id: str = Field("", examples=["MNT-2026-00502"])
    maintenance_type: str = Field("")
    maintenance_date: str = Field("")
    technician_note: str = Field("")
    recommendation: str = Field("")
    maintenance_event: str = Field("")
    calibration_monitoring_window: str = Field("")
    full_cleaning_status: str = Field("")


class FinalDispositions(BaseModel):
    last_5_similar_cases: int = Field(0)
    cleaned_and_passed: int = Field(0)
    manual_review_pending: int = Field(0)
    rejected: int = Field(0)
    rework_failed: int = Field(0)
    final_disposition_last_cases: str = Field("", examples=["7 rejected; 3 rework_failed"])
    current_case_disposition: str = Field("", examples=["pending_quality_decision"])


class CVDetection(BaseModel):
    detection_id: str = Field("", examples=["DET-001"])
    classification: str = Field("", examples=["BMFO"])
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    severity_hint: str = Field("", examples=["medium"])


class ComputerVisionOutput(BaseModel):
    defect_detected: bool = Field(True)
    number_of_detection_boxes: int = Field(0)
    estimated_defect_regions: int = Field(0)
    number_of_defects: int = Field(0)
    highest_confidence: float = Field(0.0, ge=0.0, le=1.0)
    highest_severity_hint: str = Field("", examples=["medium"])
    overall_status_hint: str = Field("", examples=["manual_review_recommended"])
    primary_defect_code: str = Field("", examples=["BMFO"])
    primary_confidence: float = Field(0.0, ge=0.0, le=1.0)
    secondary_defect_code: str = Field("", examples=["SC"])
    secondary_confidence: float = Field(0.0)
    detections: list[CVDetection] = Field(default_factory=list)


class DefectInput(BaseModel):
    """Full inspection event — mirrors the test case spreadsheet."""
    factory: FactoryInfo
    production_line: ProductionLineInfo
    station: StationInfo
    product: ProductInfo
    supplier: SupplierInfo = Field(default_factory=SupplierInfo)
    batch: BatchInfo
    defect_history: DefectHistory
    repair_records: RepairRecords = Field(default_factory=RepairRecords)
    maintenance_logs: MaintenanceLogs = Field(default_factory=MaintenanceLogs)
    final_dispositions: FinalDispositions = Field(default_factory=FinalDispositions)
    computer_vision: ComputerVisionOutput


# ---------------------------------------------------------------------------
# Output model — expanded to match expected LLM outputs from the test cases
# ---------------------------------------------------------------------------


class DefectOutput(BaseModel):
    detected_defect_summary: str
    historical_pattern_analysis: dict
    root_cause_analysis: dict
    recommended_action: str
    justification: str
    sop_references: list[str]
    confidence_assessment: dict


OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "detected_defect_summary": {
            "type": "string",
            "description": "Summary of what the vision module detected, referencing specific factory/line/station/product from the input context",
        },
        "historical_pattern_analysis": {
            "type": "object",
            "properties": {
                "is_recurring": {"type": "boolean"},
                "evidence": {"type": "string"},
            },
            "description": "Analysis of defect history patterns from the structured data",
        },
        "root_cause_analysis": {
            "type": "object",
            "properties": {
                "primary_cause": {"type": "string"},
                "contributing_factors": {"type": "string"},
                "source": {"type": "string", "enum": ["station", "supplier", "process", "calibration", "handling", "unknown"]},
            },
            "description": "Root cause analysis based on all context sources",
        },
        "recommended_action": {
            "type": "string",
            "enum": [
                "pass", "manual_review", "clean_and_reinspect", "clean_station",
                "rework", "reject", "escalate_engineering", "quarantine_batch",
                "supplier_containment", "calibration_review", "log_and_monitor",
            ],
            "description": "Recommended disposition action",
        },
        "justification": {
            "type": "string",
            "description": "Detailed reasoning referencing specific factory/station/supplier IDs, SOPs, and data from the input",
        },
        "sop_references": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of SOP codes and documents referenced",
        },
        "confidence_assessment": {
            "type": "object",
            "properties": {
                "cv_confidence": {"type": "number"},
                "data_confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                "overall": {"type": "string", "enum": ["low", "medium", "high"]},
            },
            "description": "Assessment of confidence from CV and structured data",
        },
    },
    "required": [
        "detected_defect_summary",
        "historical_pattern_analysis",
        "root_cause_analysis",
        "recommended_action",
        "justification",
        "sop_references",
        "confidence_assessment",
    ],
}


# ---------------------------------------------------------------------------
# Prompt builder — injects all dynamic context
# ---------------------------------------------------------------------------


def build_prompt(inp: DefectInput, rag_chunks: list[dict]) -> str:
    # Format RAG context
    rag_blocks = []
    for i, chunk in enumerate(rag_chunks, 1):
        meta = chunk["metadata"]
        source = meta.get("source_file", "Unknown")
        category = meta.get("folder_category", "Unknown")
        rag_blocks.append(f"[RAG Doc {i} | {source} ({category})]\n{chunk['content']}")
    rag_text = "\n\n".join(rag_blocks)

    # Build structured context from all input fields
    f = inp.factory
    l = inp.production_line
    s = inp.station
    p = inp.product
    sup = inp.supplier
    b = inp.batch
    dh = inp.defect_history
    rr = inp.repair_records
    ml = inp.maintenance_logs
    fd = inp.final_dispositions
    cv = inp.computer_vision

    structured_context = f"""FACTORY: {f.factory_id} ({f.factory_name}), type={f.factory_type}, region={f.region}, quality_standard={f.quality_standard}
PRODUCTION LINE: {l.production_line_id} ({l.production_line_name}), function={l.line_function}, yield_7d={l.recent_yield_7d}, daily_volume={l.average_daily_volume}
STATION: {s.station_id}, type={s.station_type}, function={s.station_function}, machine={s.machine_id}, camera={s.camera_id}"""

    if s.calibration_date:
        structured_context += f"\n  Calibration: {s.calibration_id} on {s.calibration_date}, type={s.calibration_type}, monitoring={ml.calibration_monitoring_window}"
    if s.linked_machine:
        structured_context += f"\n  Linked Machine: {s.linked_machine}, function={s.linked_machine_function}"

    structured_context += f"""
PRODUCT: {p.product_model} ({p.part_number}), name={p.product_name}, revision={p.revision}
  description={p.product_description}
  affected_system={p.affected_system}, subsystem={p.subsystem}, safety_criticality={p.safety_criticality}"""

    if p.qualification_status:
        structured_context += f", qualification={p.qualification_status}"
    if p.board_zone_map_available:
        structured_context += f"\n  Board zone map available. Critical features: {p.critical_features}"

    structured_context += f"""
SUPPLIER: {sup.supplier_id} ({sup.supplier_name}), material={sup.supplied_material}, quality_status={sup.supplier_quality_status}"""

    if sup.supplier_lot:
        structured_context += f", lot={sup.supplier_lot}"
    if sup.supplier_quality_case_id:
        structured_context += f", quality_case={sup.supplier_quality_case_id}"

    structured_context += f"""
BATCH: {b.batch_id}, lot={b.lot_id}, work_order={b.work_order_id}, size={b.batch_size}, status={b.batch_status}, quarantine={b.quarantine_status}"""

    # Defect history
    structured_context += f"""
DEFECT HISTORY: code={dh.defect_code}, count_30d={dh.same_defect_count_30d}, count_90d={dh.same_defect_count_90d}
  scope={dh.scope}, trend={dh.trend}"""
    if dh.most_common_shift:
        structured_context += f", most_common_shift={dh.most_common_shift}"
    if dh.recurring_location:
        structured_context += f"\n  recurring_location={dh.recurring_location}, same_location_30d={dh.same_location_count_30d}"
    if dh.affected_lines:
        structured_context += f"\n  affected_lines={dh.affected_lines}"
    if dh.affected_factories:
        structured_context += f"\n  affected_factories={dh.affected_factories}"
    if dh.functional_zone:
        structured_context += f"\n  FUNCTIONAL ZONE: {dh.functional_zone_description}"
    if dh.board_zone:
        structured_context += f"\n  board_zone={dh.board_zone}, functional_zone={dh.functional_zone}"
    if dh.sc_detection_rate_before_calibration:
        structured_context += f"\n  SC rate before calibration={dh.sc_detection_rate_before_calibration}, after={dh.sc_detection_rate_after_calibration}, manual_confirmation={dh.manual_review_confirmation_rate}"

    # Repair records
    if rr.related_repair_count_30d > 0 or rr.repair_id:
        structured_context += f"""
REPAIR RECORDS: count_30d={rr.related_repair_count_30d}, action="{rr.common_repair_action or rr.repair_action}", result="{rr.repair_result or rr.repair_verification}"
  recurrence_after_repair={rr.recurrence_after_repair}"""
        if rr.recurrence_after_repair_days:
            structured_context += f", recurrence_days={rr.recurrence_after_repair_days}"
        if rr.engineering_rework_failed:
            structured_context += f", engineering_rework_failed={rr.engineering_rework_failed}"

    # Maintenance
    if ml.latest_maintenance_id:
        structured_context += f"""
MAINTENANCE: {ml.latest_maintenance_id}, type={ml.maintenance_type}, date={ml.maintenance_date}
  technician_note="{ml.technician_note}", recommendation="{ml.recommendation}"
  full_cleaning_status={ml.full_cleaning_status}"""

    # Final dispositions
    if fd.last_5_similar_cases > 0 or fd.final_disposition_last_cases:
        structured_context += f"""
DISPOSITION HISTORY: last_{fd.last_5_similar_cases}_cases: cleaned_passed={fd.cleaned_and_passed}, manual_review_pending={fd.manual_review_pending}, rejected={fd.rejected}, rework_failed={fd.rework_failed}"""
        if fd.final_disposition_last_cases:
            structured_context += f"\n  recent: {fd.final_disposition_last_cases}"
        structured_context += f"\n  current_case: {fd.current_case_disposition}"

    # CV output
    cv_summary = f"""COMPUTER VISION OUTPUT:
  defect_detected={cv.defect_detected}, detection_boxes={cv.number_of_detection_boxes}, estimated_regions={cv.estimated_defect_regions}
  primary_defect={cv.primary_defect_code}, primary_confidence={cv.primary_confidence:.4f}
  highest_confidence={cv.highest_confidence:.4f}, severity_hint={cv.highest_severity_hint}
  overall_status_hint={cv.overall_status_hint}"""
    if cv.secondary_defect_code:
        cv_summary += f"\n  secondary_defect={cv.secondary_defect_code}, secondary_confidence={cv.secondary_confidence:.4f}"

    return f"""You are a senior PCB quality control engineer at {f.factory_name} ({f.factory_id}).
You are analyzing a defect flagged by the automated inspection system at station {s.station_id} on line {l.production_line_id}.

=== STRUCTURED DATA (from factory database) ===
{structured_context}

=== COMPUTER VISION DETECTION ===
{cv_summary}

=== KNOWLEDGE BASE (SOPs, technical manuals, repair guides) ===
{rag_text}

=== TASK ===
Analyze this defect event using ALL three context sources above:
1. Structured data — factory, line, station, product, supplier, batch, history, repairs, maintenance
2. Computer vision — what the vision model detected
3. Knowledge base — SOPs and technical documents

IMPORTANT RULES:
- Reference the SPECIFIC factory ID, station ID, line ID, product model, supplier ID, and batch ID from the structured data in your response.
- If repair records show recurrence after repair, flag the repair as ineffective.
- NEVER invent data not present in the input. If no history exists, say so.

ACTION SELECTION — pick the SINGLE BEST recommended_action using these rules IN ORDER (first match wins):
1. "supplier_containment" — if defect is linked to a supplier lot AND appears across multiple lines or factories (check affected_lines, affected_factories).
2. "reject" — if board has MULTIPLE critical electrical defects (SH+OP on same board) AND previous rework/repair has failed.
3. "escalate_engineering" — if defect recurs in a FUNCTIONAL conductor zone after repair has already been attempted and failed (recurrence_after_repair=true).
4. "calibration_review" — if a calibration event recently occurred AND manual review confirmation rate is low (<60%) AND defect type could be a false positive (SC, reflection-related).
5. "clean_station" — if foreign object defects (BMFO, CFO) are RECURRING at the same station (3+ cases in 30 days) OR maintenance notes mention dust/contamination.
6. "quarantine_batch" — if critical defect pattern exists across the same batch family with high rejection/rework failure rate.
7. "clean_and_reinspect" — single foreign object defect, not recurring, not on conductor.
8. "rework" — single confirmed defect that is repairable per SOP.
9. "manual_review" — CV confidence is below 0.50 OR insufficient evidence for any stronger action.
10. "log_and_monitor" — defect is in a non-functional zone (test coupon area, board_zone=test_coupon_area) OR no defect detected but monitoring is needed (qualification batch).
11. "pass" — no defect detected by CV AND no history of defects.

Output a single JSON object. No preamble, no markdown fences."""


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sovra AI — PCB Defect Inspection RAG API",
    description="RAG-powered API for industrial PCB defect analysis. Accepts full factory context + CV output.",
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


@app.post("/explain-defect", response_model=DefectOutput, summary="Analyze PCB defect with full context")
async def explain_defect(inp: DefectInput) -> DefectOutput:
    """
    Accepts a full inspection event (factory context + CV output),
    retrieves relevant SOPs, and returns structured disposition analysis.
    """
    cv = inp.computer_vision
    dh = inp.defect_history

    logger.info(
        f"Received: factory={inp.factory.factory_id} line={inp.production_line.production_line_id} "
        f"station={inp.station.station_id} product={inp.product.product_model} "
        f"defect={dh.defect_code} cv_confidence={cv.highest_confidence:.2f}"
    )

    # --- Step 1: RAG retrieval using defect code + context keywords ---
    query_parts = [dh.defect_code, inp.product.product_model]
    # Add context-specific keywords to improve retrieval
    if dh.affected_factories or dh.affected_lines:
        query_parts.append("supplier lot containment quarantine")
    if inp.repair_records.recurrence_after_repair:
        query_parts.append("repair recurrence escalation ineffective")
    if inp.maintenance_logs.maintenance_event or inp.maintenance_logs.calibration_monitoring_window:
        query_parts.append("calibration drift monitoring")
    if dh.board_zone:
        query_parts.append(f"zone {dh.board_zone}")
    if cv.highest_confidence < 0.50:
        query_parts.append("low confidence manual review")
    if dh.multi_critical_defect_count_14d > 0 or cv.secondary_defect_code:
        query_parts.append("multiple critical defect")
    if inp.product.qualification_status and "qualification" in inp.product.qualification_status:
        query_parts.append("qualification acceptance visual variation")
    if "foreign" in dh.defect_code.lower() or dh.defect_code in ("BMFO", "CFO"):
        query_parts.append("foreign object contamination cleaning")
    if dh.defect_code == "CS":
        query_parts.append("conductor scratch handling robot")
    if dh.defect_code == "SC":
        query_parts.append("spurious copper reflection calibration AOI")
    if dh.defect_code == "HB":
        query_parts.append("hole breakout zone test coupon")
    if dh.defect_code == "MB":
        query_parts.append("mouse bite conductor edge")
    if dh.defect_code == "OP":
        query_parts.append("open conductor path repair bridge")
    if dh.defect_code == "SH":
        query_parts.append("short conductor routing supplier")
    query = " ".join(query_parts)
    logger.info(f"RAG query: {query}")
    try:
        chunks = retrieve(query, top_k=TOP_K)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail=f"Retrieval error: {exc}")

    if not chunks:
        chunks = []
        logger.warning(f"No RAG chunks for defect code: {dh.defect_code}")

    # --- Step 2: Build prompt with ALL context ---
    prompt = build_prompt(inp, chunks)

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
    # Strip <think> blocks
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

    result.setdefault("historical_pattern_analysis", {"is_recurring": dh.same_defect_count_30d > 2, "evidence": ""})
    result.setdefault("root_cause_analysis", {"primary_cause": "", "contributing_factors": "", "source": "unknown"})
    result.setdefault("confidence_assessment", {
        "cv_confidence": cv.highest_confidence,
        "data_confidence": "high" if dh.same_defect_count_90d > 5 else "medium" if dh.same_defect_count_90d > 0 else "low",
        "overall": "medium",
    })
    result.setdefault("recommended_action", "manual_review")
    result.setdefault("detected_defect_summary", "")
    result.setdefault("justification", "")

    result["recommended_action"] = result["recommended_action"].lower().replace(" ", "_")

    logger.info(f"Result: action={result['recommended_action']!r}")
    return DefectOutput(**result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("API_PORT", "8000")), reload=True)
