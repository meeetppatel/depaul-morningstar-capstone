"""
FastAPI server for GECS Industry & Sub-Industry Prediction.

Endpoints:
  GET  /                        → Interactive browser UI (form + CSV upload)
  GET  /api/health              → server + model metadata
  GET  /api/industries          → list of all 145 GECS industries
  GET  /api/template.csv        → CSV upload template
  POST /api/predict             → full pipeline (industry + sub-industry)
  POST /api/predict_industry    → Task 1 only
  POST /api/predict_subindustry → Task 2 only
  POST /api/predict_csv         → CSV file upload prediction

Run:
  python -m uvicorn serving_app.fastapi_app:app --host 0.0.0.0 --port 8000
  Then open http://localhost:8000/ for UI or /docs for Swagger.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator

try:
    from .inference import get_service, predict_json
except ImportError:
    from inference import get_service, predict_json


# ─────────────────────────────────────────────────────────────────────────────
# Load HTML landing page from sibling file
# ─────────────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent
_LANDING_FILE = _HERE / "landing.html"

if _LANDING_FILE.exists():
    LANDING_HTML = _LANDING_FILE.read_text(encoding="utf-8")
else:
    LANDING_HTML = (
        "<h1>GECS Prediction API</h1>"
        "<p>landing.html not found. Visit <a href='/docs'>/docs</a> for Swagger.</p>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class PredictRecord(BaseModel):
    CompanyId: str | None = Field(default=None)
    AsOfDate: str | None = Field(default=None)
    LongProfile: str | None = Field(default=None)
    SegmentName: str = Field(..., description="Segment or business-line name (required).")
    SegmentDescription: str | None = Field(default=None)
    Revenue: float | None = Field(default=None)
    total_revenue_company_as_of: float | None = Field(default=None)
    revenue_share: float | None = Field(default=None)
    is_largest_share_segment: bool | str | None = Field(default=None)
    MstarGlobal: str | None = Field(default=None)
    known_industry_code: str | None = Field(default=None)

    class Config:
        extra = "allow"

    @field_validator(
        "Revenue", "total_revenue_company_as_of", "revenue_share",
        "CompanyId", "AsOfDate", "MstarGlobal", "known_industry_code",
        mode="before",
    )
    @classmethod
    def _empty_string_to_none(cls, value):
        if isinstance(value, str) and value.strip() == "":
            return None
        return value


class PredictRequest(BaseModel):
    top_k: int = Field(default=5, ge=1, le=20)
    known_industry_code: str | None = Field(default=None)
    industry_code: str | None = Field(default=None)
    records: list[PredictRecord] = Field(..., min_length=1)


def _request_to_payload(req: PredictRequest, task: str) -> dict[str, Any]:
    return {
        "task": task,
        "top_k": req.top_k,
        "known_industry_code": req.known_industry_code or req.industry_code,
        "records": [r.model_dump(exclude_none=True) for r in req.records],
    }


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="GECS Industry & Sub-Industry Prediction API",
    version="1.0.0",
    description=(
        "Task 1 (FLANG-BERT champion) ranks 145 GECS industries. "
        "Task 2 (DeBERTa-v3-small cross-encoder) reranks 374+ sub-industries "
        "using a soft parent prior (alpha=15)."
    ),
)


@app.get("/", response_class=HTMLResponse, tags=["UI"])
def landing() -> HTMLResponse:
    return HTMLResponse(LANDING_HTML, headers={"Cache-Control": "no-store"})


@app.get("/api/health", tags=["Metadata"])
def health() -> dict[str, Any]:
    return get_service().health()


@app.get("/api/industries", tags=["Metadata"])
def industries() -> dict[str, Any]:
    return {"industries": get_service().industry_options()}


@app.get("/api/template.csv", response_class=PlainTextResponse, tags=["Metadata"])
def template_csv() -> PlainTextResponse:
    header = (
        "CompanyId,AsOfDate,LongProfile,SegmentName,SegmentDescription,Revenue,"
        "total_revenue_company_as_of,revenue_share,is_largest_share_segment,MstarGlobal\n"
    )
    return PlainTextResponse(header, media_type="text/csv")


@app.post("/api/predict", tags=["Prediction"])
def predict(request: PredictRequest) -> dict[str, Any]:
    """Predict both industry and sub-industry."""
    try:
        return predict_json(_request_to_payload(request, "both"))
    except Exception as exc:
        raise HTTPException(500, detail={"error": str(exc), "type": exc.__class__.__name__}) from exc


@app.post("/api/predict_industry", tags=["Prediction"])
def predict_industry(request: PredictRequest) -> dict[str, Any]:
    """Predict only industry (Task 1)."""
    try:
        return predict_json(_request_to_payload(request, "industry"))
    except Exception as exc:
        raise HTTPException(500, detail={"error": str(exc), "type": exc.__class__.__name__}) from exc


@app.post("/api/predict_subindustry", tags=["Prediction"])
def predict_subindustry(request: PredictRequest) -> dict[str, Any]:
    """Predict only sub-industry (Task 2). If known_industry_code is given,
    Task 1 is skipped and sub-industries are reranked under that parent only."""
    try:
        return predict_json(_request_to_payload(request, "subindustry"))
    except Exception as exc:
        raise HTTPException(500, detail={"error": str(exc), "type": exc.__class__.__name__}) from exc


@app.post("/api/predict_csv", tags=["Prediction"])
async def predict_csv(
    file: UploadFile = File(..., description="CSV file matching the Task 1 schema."),
    known_industry_code: str = Form("", description="Optional 8-digit industry applied to all rows."),
    top_k: int = Form(5, description="Number of candidates per row."),
    task: str = Form("both", description="'both', 'industry', or 'subindustry'."),
) -> dict[str, Any]:
    """Upload a CSV and get predictions for every row."""
    try:
        raw = await file.read()
        frame = pd.read_csv(io.BytesIO(raw))
        payload = {
            "records": frame.to_dict("records"),
            "known_industry_code": known_industry_code or None,
            "top_k": top_k,
            "task": task,
        }
        return predict_json(payload)
    except Exception as exc:
        raise HTTPException(500, detail={"error": str(exc), "type": exc.__class__.__name__}) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=False)
