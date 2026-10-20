"""HTTP API for a spectrum front end.

Run: python -m uvicorn app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from deduction.engine import MODEL_PATH, deduce_from_images, deduce_from_peaks
from deduction.ir import IrPeak
from deduction.ms import MsPeak
from deduction.nmr import NmrSignal

app = FastAPI(
    title="Molecular structure deduction",
    description="Calculate the index of hydrogen deficiency and propose a structure from IR, 1H-NMR, and mass data.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class IrIn(BaseModel):
    wavenumber: float
    intensity: str = "medium"
    shape: str = "sharp"


class NmrIn(BaseModel):
    delta: float
    multiplicity: str = "unknown"
    integral: float | None = None


class MsIn(BaseModel):
    mz: float
    intensity: float = 100


class PeakRequest(BaseModel):
    formula: str = Field(examples=["C3H6O"])
    ir: list[IrIn] = []
    nmr: list[NmrIn] = []
    ms: list[MsIn] = []


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "model": str(MODEL_PATH), "model_ready": MODEL_PATH.exists()}


@app.post("/api/deduce/peaks")
def deduce_peaks(body: PeakRequest) -> dict:
    try:
        return deduce_from_peaks(
            body.formula,
            [IrPeak(peak.wavenumber, peak.intensity, peak.shape) for peak in body.ir],
            [NmrSignal(peak.delta, peak.multiplicity, peak.integral) for peak in body.nmr],
            [MsPeak(peak.mz, peak.intensity) for peak in body.ms],
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _save_upload(upload: UploadFile | None, directory: Path) -> str | None:
    if upload is None or not upload.filename:
        return None
    target = directory / Path(upload.filename).name
    target.write_bytes(upload.file.read())
    return str(target)


@app.post("/api/deduce")
async def deduce_images(
    formula: str = Form(...),
    ir: UploadFile | None = File(None),
    nmr: UploadFile | None = File(None),
    ms: UploadFile | None = File(None),
    page: UploadFile | None = File(None),
    nmr_ppm_left: float | None = Form(None),
    ir_left: float | None = Form(None),
    ir_right: float | None = Form(None),
) -> dict:
    if not any((ir and ir.filename, nmr and nmr.filename, ms and ms.filename, page and page.filename)):
        raise HTTPException(status_code=400, detail="Upload an IR image, an NMR image, a mass image, or one combined page.")
    try:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            return deduce_from_images(
                formula,
                ir_path=_save_upload(ir, folder),
                nmr_path=_save_upload(nmr, folder),
                ms_path=_save_upload(ms, folder),
                page_path=_save_upload(page, folder),
                nmr_ppm_left=nmr_ppm_left,
                ir_left=ir_left,
                ir_right=ir_right,
            )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
