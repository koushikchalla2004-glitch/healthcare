import os
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import psycopg  # psycopg v3 (binary)
from psycopg import sql


# --- DB connection ---
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("DATABASE_URL env var is required")

# Single connection for now (Render free tier runs one worker by default).
# If you later add more workers/threads, switch to psycopg_pool ConnectionPool.
conn = psycopg.connect(DB_URL, autocommit=True)


# --- FastAPI app ---
app = FastAPI(title="Health Readmit API", version="0.1.1")


# --- Models ---
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


# --- Routes ---
@app.get("/healthz")
def healthz():
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB health check failed: {e}")


@app.post("/v1/vitals")
def ingest_vitals(batch: VitalsBatch):
    # Basic sanity filtering
    clean = []
    for s in batch.samples:
        if s.hr is not None and (s.hr < 25 or s.hr > 220):
            continue
        if s.spo2 is not None and (s.spo2 < 50 or s.spo2 > 100):
            continue
        clean.append(s)
    if not clean:
        raise HTTPException(status_code=400, detail="No valid samples")

    try:
        with conn.cursor() as cur:
            # Insert vitals (ignore duplicates on same ts)
            insert_sql = """
                INSERT INTO vitals (patient_id, ts, hr, spo2, steps, sbp, dbp, temp_c)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (patient_id, ts) DO NOTHING
            """
            params = [
                (
                    batch.patient_id,
                    s.ts, s.hr, s.spo2, s.steps, s.sbp, s.dbp, s.temp_c
                )
                for s in clean
            ]
            cur.executemany(insert_sql, params)

            # Simple alert rules on the latest sample
            last = clean[-1]
            alerts = []
            if last.spo2 is not None and last.spo2 < 90:
                alerts.append((
                    batch.patient_id,
                    "SPO2_LOW",
                    "HIGH",
                    f"SpO2 {last.spo2}% below 90 at {last.ts.isoformat()}"
                ))
            if last.hr is not None and (last.hr < 40 or last.hr > 135):
                alerts.append((
                    batch.patient_id,
                    "HR_ABNORMAL",
                    "HIGH",
                    f"HR {last.hr} abnormal at {last.ts.isoformat()}"
                ))

            if alerts:
                cur.executemany(
                    """
                    INSERT INTO alerts (patient_id, type, severity, message)
                    VALUES (%s,%s,%s,%s)
                    """,
                    alerts
                )
                # TODO: call Twilio here (cooldown recommended)

        return {"ok": True, "inserted": len(clean), "alerts_created": len(alerts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


@app.post("/v1/alerts")
def create_alert(a: AlertIn):
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO alerts (patient_id, type, severity, message)
                VALUES (%s,%s,%s,%s)
                RETURNING id, created_at
                """,
                (a.patient_id, a.type, a.severity, a.message)
            )
            rid, ts = cur.fetchone()
        return {"ok": True, "id": str(rid), "created_at": ts.isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


@app.get("/v1/alerts/{patient_id}")
def list_alerts(patient_id: str):
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, type, severity, message, created_at, status
                FROM alerts
                WHERE patient_id=%s
                ORDER BY created_at DESC
                LIMIT 50
                """,
                (patient_id,)
            )
            rows = cur.fetchall()
        return [
            {
                "id": str(r[0]),
                "type": r[1],
                "severity": r[2],
                "message": r[3],
                "created_at": r[4].isoformat(),
                "status": r[5],
            }
            for r in rows
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")
