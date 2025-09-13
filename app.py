import os
from datetime import datetime, date
from typing import Optional, List

from fastapi import Request
import hmac, hashlib, json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import psycopg  # psycopg v3
from twilio.rest import Client
from fastapi.middleware.cors import CORSMiddleware


# -----------------------
# Config & DB connection
# -----------------------
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("DATABASE_URL env var is required")

# Tip (Neon pooled URL): add &channel_binding=prefer in the URL if needed.
conn = psycopg.connect(DB_URL, autocommit=True)

# Twilio (optional; SMS sent only if these are set and a caregiver phone exists)
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_FROM")


def send_sms_safe(to: Optional[str], body: str) -> bool:
    """Send SMS if Twilio creds + 'to' are present. Returns True if queued."""
    if not (TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM and to):
        return False
    try:
        Client(TWILIO_SID, TWILIO_TOKEN).messages.create(
            to=to, from_=TWILIO_FROM, body=body
        )
        return True
    except Exception as e:
        print(f"Twilio error: {e}")
        return False


# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="Health Readmit API", version="0.2.0")

# Allow simple frontends to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------
# Pydantic models
# -----------------------
class PatientIn(BaseModel):
    name: str
    dob: date
    sex_at_birth: Optional[str] = None
    caregiver_phone: Optional[str] = None


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


# -----------------------
# Routes
# -----------------------
@app.get("/healthz")
def healthz():
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB health check failed: {e}")


@app.post("/v1/patients")
def create_patient(p: PatientIn):
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO patients (name, dob, sex_at_birth, caregiver_phone)
                VALUES (%s,%s,%s,%s)
                RETURNING id
                """,
                (p.name, p.dob, p.sex_at_birth, p.caregiver_phone),
            )
            (pid,) = cur.fetchone()
        return {"ok": True, "id": str(pid)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


@app.post("/v1/vitals")
def ingest_vitals(batch: VitalsBatch):
    # Basic sanity filtering
    clean: List[Vital] = []
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

            # Caregiver phone (if any)
            caregiver_phone = None
            cur.execute("SELECT caregiver_phone FROM patients WHERE id=%s", (batch.patient_id,))
            row = cur.fetchone()
            if row:
                caregiver_phone = row[0]

            # Candidate alerts from the most recent sample
            last = clean[-1]
            candidates: List[tuple[str, str, str]] = []
            if last.spo2 is not None and last.spo2 < 90:
                candidates.append(("SPO2_LOW", "HIGH", f"SpO2 {last.spo2}% below 90 at {last.ts.isoformat()}"))
            if last.hr is not None and (last.hr < 40 or last.hr > 135):
                candidates.append(("HR_ABNORMAL", "HIGH", f"HR {last.hr} abnormal at {last.ts.isoformat()}"))

            alerts_created = 0
            sms_sent = 0

            for t, se, msg in candidates:
                # 15-minute cooldown per (patient, type)
                cur.execute(
                    """
                    SELECT COUNT(*) FROM alerts
                    WHERE patient_id=%s AND type=%s
                      AND created_at > now() - interval '15 minutes'
                    """,
                    (batch.patient_id, t),
                )
                recent = cur.fetchone()[0] > 0

                # Insert alert row
                cur.execute(
                    "INSERT INTO alerts (patient_id, type, severity, message) VALUES (%s,%s,%s,%s)",
                    (batch.patient_id, t, se, msg),
                )
                alerts_created += 1

                # Notify caregiver if HIGH and not recently alerted
                if se == "HIGH" and not recent:
                    if send_sms_safe(caregiver_phone, f"[ALERT] {msg}"):
                        sms_sent += 1

        return {"ok": True, "inserted": len(clean), "alerts_created": alerts_created, "sms_sent": sms_sent}
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
