import os, json
from datetime import datetime
from typing import Optional, List
import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("DATABASE_URL env var is required")

conn = psycopg2.connect(DB_URL)
conn.autocommit = True

app = FastAPI(title="Health Readmit API", version="0.1.0")

class Vital(BaseModel):
    ts: datetime
    hr: Optional[int] = None
    spo2: Optional[int] = None
    steps: Optional[int] = None
    sbp: Optional[int] = None
    dbp: Optional[int] = None
    temp_c: Optional[float] = None

class VitalsBatch(BaseModel):
    patient_id: str = Field(..., description="UUID of patient")
    source: str = "sdk"
    samples: List[Vital]

class AlertIn(BaseModel):
    patient_id: str
    type: str
    severity: str = "HIGH"
    message: str

@app.get("/healthz")
def healthz():
    with conn.cursor() as cur:
        cur.execute("SELECT 1;")
        return {"ok": True}

@app.post("/v1/vitals")
def ingest_vitals(batch: VitalsBatch):
    # basic sanity filter
    clean = []
    for s in batch.samples:
        if s.hr is not None and (s.hr < 25 or s.hr > 220):
            continue
        if s.spo2 is not None and (s.spo2 < 50 or s.spo2 > 100):
            continue
        clean.append(s)
    if not clean:
        raise HTTPException(status_code=400, detail="No valid samples")

    with conn.cursor() as cur:
        for s in clean:
            cur.execute("""
                INSERT INTO vitals (patient_id, ts, hr, spo2, steps, sbp, dbp, temp_c)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (patient_id, ts) DO NOTHING
            """, (batch.patient_id, s.ts, s.hr, s.spo2, s.steps, s.sbp, s.dbp, s.temp_c))

        # simple alert rules (extend later)
        last = clean[-1]
        alerts = []
        if last.spo2 is not None and last.spo2 < 90:
            alerts.append(("SPO2_LOW","HIGH",f"SpO2 {last.spo2}% below 90 at {last.ts.isoformat()}"))
        if last.hr is not None and (last.hr < 40 or last.hr > 135):
            alerts.append(("HR_ABNORMAL","HIGH",f"HR {last.hr} abnormal at {last.ts.isoformat()}"))

        for t,se,msg in alerts:
            cur.execute("""
                INSERT INTO alerts (patient_id, type, severity, message)
                VALUES (%s,%s,%s,%s)
            """, (batch.patient_id, t, se, msg))

    return {"ok": True, "inserted": len(clean)}

@app.post("/v1/alerts")
def create_alert(a: AlertIn):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO alerts (patient_id, type, severity, message)
            VALUES (%s,%s,%s,%s)
            RETURNING id, created_at
        """, (a.patient_id, a.type, a.severity, a.message))
        rid, ts = cur.fetchone()
    return {"ok": True, "id": str(rid), "created_at": ts.isoformat()}

@app.get("/v1/alerts/{patient_id}")
def list_alerts(patient_id: str):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, type, severity, message, created_at, status
            FROM alerts WHERE patient_id=%s
            ORDER BY created_at DESC LIMIT 50
        """, (patient_id,))
        rows = cur.fetchall()
    return [{"id": str(r[0]), "type": r[1], "severity": r[2], "message": r[3],
             "created_at": r[4].isoformat(), "status": r[5]} for r in rows]
