import os, base64, json, time
from datetime import datetime, date
from typing import Optional, List
from urllib.parse import urlencode

import psycopg  # v3
import requests
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from twilio.rest import Client


# -----------------------
# Config & DB connection
# -----------------------
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("DATABASE_URL env var is required")

# If using Neon pooled URL, add &channel_binding=prefer to DATABASE_URL
conn = psycopg.connect(DB_URL, autocommit=True)

# Twilio (optional)
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_FROM")

def send_sms_safe(to: Optional[str], body: str) -> bool:
    if not (TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM and to):
        return False
    try:
        Client(TWILIO_SID, TWILIO_TOKEN).messages.create(to=to, from_=TWILIO_FROM, body=body)
        return True
    except Exception as e:
        print(f"Twilio error: {e}")
        return False

# Admin key for cron/sync protection
CRON_KEY = os.getenv("CRON_KEY")

def _check_admin_key(x_admin_key: str | None):
    if CRON_KEY and x_admin_key != CRON_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# Fitbit OAuth/API
FITBIT_CLIENT_ID = os.getenv("FITBIT_CLIENT_ID")
FITBIT_CLIENT_SECRET = os.getenv("FITBIT_CLIENT_SECRET")
FITBIT_REDIRECT_URI = os.getenv("FITBIT_REDIRECT_URI")  # https://.../oauth/fitbit/callback
FITBIT_SCOPE = os.getenv("FITBIT_SCOPE", "heartrate activity profile")
FITBIT_AUTH_URL = "https://www.fitbit.com/oauth2/authorize"
FITBIT_TOKEN_URL = "https://api.fitbit.com/oauth2/token"
FITBIT_API = "https://api.fitbit.com"

def _b64_client() -> str:
    raw = f"{FITBIT_CLIENT_ID}:{FITBIT_CLIENT_SECRET}".encode()
    return base64.b64encode(raw).decode()

def _token_headers():
    return {
        "Authorization": f"Basic {_b64_client()}",
        "Content-Type": "application/x-www-form-urlencoded",
    }

def _ensure_fitbit_tokens(patient_id: str):
    # refresh if expired
    with conn.cursor() as cur:
        cur.execute("""
            SELECT fitbit_user_id, access_token, refresh_token, EXTRACT(EPOCH FROM expires_at)
            FROM fitbit_tokens WHERE patient_id=%s
        """, (patient_id,))
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Fitbit not linked for this patient")

    user_id, access, refresh, exp = row
    if time.time() < float(exp) - 120:
        return user_id, access

    # refresh token
    data = {"grant_type": "refresh_token", "refresh_token": refresh}
    r = requests.post(FITBIT_TOKEN_URL, data=data, headers=_token_headers(), timeout=20)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Fitbit refresh failed: {r.text}")
    tok = r.json()
    access = tok["access_token"]
    refresh = tok["refresh_token"]
    expires_at = int(time.time()) + int(tok.get("expires_in", 28800))

    with conn.cursor() as cur:
        cur.execute("""
            UPDATE fitbit_tokens
               SET access_token=%s,
                   refresh_token=%s,
                   expires_at=TO_TIMESTAMP(%s),
                   scope=%s,
                   token_type=%s
             WHERE patient_id=%s
        """, (access, refresh, expires_at, tok.get("scope"), tok.get("token_type"), patient_id))
    return user_id, access


# -----------------------
# FastAPI
# -----------------------
app = FastAPI(title="Health Readmit API", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Models
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

class SourceLink(BaseModel):
    provider: str          # e.g., "fitbit"
    external_user_id: str  # provider user id


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
            cur.execute("""
                INSERT INTO patients (name, dob, sex_at_birth, caregiver_phone)
                VALUES (%s,%s,%s,%s)
                RETURNING id
            """, (p.name, p.dob, p.sex_at_birth, p.caregiver_phone))
            (pid,) = cur.fetchone()
        return {"ok": True, "id": str(pid)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

@app.post("/v1/patients/{patient_id}/sources")
def link_source(patient_id: str, body: SourceLink):
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO patient_sources (patient_id, provider, external_user_id)
                VALUES (%s,%s,%s)
                ON CONFLICT (provider, external_user_id)
                DO UPDATE SET patient_id = EXCLUDED.patient_id
            """, (patient_id, body.provider, body.external_user_id))
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

@app.post("/v1/vitals")
def ingest_vitals(batch: VitalsBatch):
    # Sanity filter
    clean = []
    for s in batch.samples:
        if s.hr is not None and (s.hr < 25 or s.hr > 220): continue
        if s.spo2 is not None and (s.spo2 < 50 or s.spo2 > 100): continue
        clean.append(s)
    if not clean:
        raise HTTPException(status_code=400, detail="No valid samples")

    try:
        with conn.cursor() as cur:
            # insert vitals
            cur.executemany("""
                INSERT INTO vitals (patient_id, ts, hr, spo2, steps, sbp, dbp, temp_c)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (patient_id, ts) DO NOTHING
            """, [(
                batch.patient_id, s.ts, s.hr, s.spo2, s.steps, s.sbp, s.dbp, s.temp_c
            ) for s in clean])

            # caregiver phone
            caregiver_phone = None
            cur.execute("SELECT caregiver_phone FROM patients WHERE id=%s", (batch.patient_id,))
            r = cur.fetchone()
            if r: caregiver_phone = r[0]

            # alert candidates (latest sample)
            last = clean[-1]
            candidates = []
            if last.spo2 is not None and last.spo2 < 90:
                candidates.append(("SPO2_LOW", "HIGH", f"SpO2 {last.spo2}% below 90 at {last.ts.isoformat()}"))
            if last.hr is not None and (last.hr < 40 or last.hr > 135):
                candidates.append(("HR_ABNORMAL", "HIGH", f"HR {last.hr} abnormal at {last.ts.isoformat()}"))

            alerts_created, sms_sent = 0, 0
            for t, se, msg in candidates:
                # cooldown 15 min per (patient,type)
                cur.execute("""
                    SELECT COUNT(*) FROM alerts
                     WHERE patient_id=%s AND type=%s
                       AND created_at > now() - interval '15 minutes'
                """, (batch.patient_id, t))
                recent = cur.fetchone()[0] > 0

                cur.execute("""
                    INSERT INTO alerts (patient_id, type, severity, message)
                    VALUES (%s,%s,%s,%s)
                """, (batch.patient_id, t, se, msg))
                alerts_created += 1

                if se == "HIGH" and not recent and send_sms_safe(caregiver_phone, f"[ALERT] {msg}"):
                    sms_sent += 1

        return {"ok": True, "inserted": len(clean), "alerts_created": alerts_created, "sms_sent": sms_sent}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

@app.post("/v1/alerts")
def create_alert(a: AlertIn):
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO alerts (patient_id, type, severity, message)
                VALUES (%s,%s,%s,%s)
                RETURNING id, created_at
            """, (a.patient_id, a.type, a.severity, a.message))
            rid, ts = cur.fetchone()
        return {"ok": True, "id": str(rid), "created_at": ts.isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

@app.get("/v1/alerts/{patient_id}")
def list_alerts(patient_id: str):
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, type, severity, message, created_at, status
                  FROM alerts
                 WHERE patient_id=%s
              ORDER BY created_at DESC
                 LIMIT 50
            """, (patient_id,))
            rows = cur.fetchall()
        return [
            {"id": str(r[0]), "type": r[1], "severity": r[2], "message": r[3],
             "created_at": r[4].isoformat(), "status": r[5]}
            for r in rows
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

# -------- Fitbit OAuth + Sync --------
@app.get("/oauth/fitbit/start")
def fitbit_start(patient_id: str = Query(..., description="UUID of patient")):
    if not (FITBIT_CLIENT_ID and FITBIT_REDIRECT_URI):
        raise HTTPException(status_code=500, detail="Fitbit env vars missing")
    q = {
        "client_id": FITBIT_CLIENT_ID,
        "redirect_uri": FITBIT_REDIRECT_URI,
        "response_type": "code",
        "scope": FITBIT_SCOPE,
        "prompt": "consent",
        "state": patient_id,
    }
    return RedirectResponse(f"{FITBIT_AUTH_URL}?{urlencode(q)}")

@app.get("/oauth/fitbit/callback")
def fitbit_callback(code: str, state: str):
    data = {
        "client_id": FITBIT_CLIENT_ID,
        "grant_type": "authorization_code",
        "redirect_uri": FITBIT_REDIRECT_URI,
        "code": code,
    }
    r = requests.post(FITBIT_TOKEN_URL, data=data, headers=_token_headers(), timeout=20)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Token exchange failed: {r.text}")
    tok = r.json()
    fitbit_user_id = tok.get("user_id")
    access = tok["access_token"]
    refresh = tok["refresh_token"]
    expires_at = int(time.time()) + int(tok.get("expires_in", 28800))

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO fitbit_tokens (patient_id, fitbit_user_id, access_token, refresh_token, expires_at, scope, token_type)
            VALUES (%s,%s,%s,%s, TO_TIMESTAMP(%s), %s, %s)
            ON CONFLICT (patient_id) DO UPDATE
              SET fitbit_user_id=EXCLUDED.fitbit_user_id,
                  access_token=EXCLUDED.access_token,
                  refresh_token=EXCLUDED.refresh_token,
                  expires_at=EXCLUDED.expires_at,
                  scope=EXCLUDED.scope,
                  token_type=EXCLUDED.token_type
        """, (state, fitbit_user_id, access, refresh, expires_at, tok.get("scope"), tok.get("token_type")))
        cur.execute("""
            INSERT INTO patient_sources (patient_id, provider, external_user_id)
            VALUES (%s,'fitbit',%s)
            ON CONFLICT (provider, external_user_id) DO UPDATE SET patient_id=EXCLUDED.patient_id
        """, (state, fitbit_user_id))
    return {"ok": True, "linked": True, "patient_id": state, "fitbit_user_id": fitbit_user_id}

def _insert_or_update_vital(cur, patient_id: str, ts: datetime,
                            hr: int | None = None, steps: int | None = None):
    cur.execute("""
        INSERT INTO vitals (patient_id, ts, hr, steps)
        VALUES (%s,%s,%s,%s)
        ON CONFLICT (patient_id, ts) DO UPDATE
          SET hr = COALESCE(EXCLUDED.hr, vitals.hr),
              steps = COALESCE(EXCLUDED.steps, vitals.steps)
    """, (patient_id, ts, hr, steps))

@app.get("/integrations/fitbit/sync")
def fitbit_sync(
    patient_id: str,
    x_admin_key: str | None = Header(default=None)
):
    _check_admin_key(x_admin_key)

    fitbit_user_id, access = _ensure_fitbit_tokens(patient_id)
    headers = {"Authorization": f"Bearer {access}"}

    # Intraday 1-min HR & steps for today
    r1 = requests.get(f"{FITBIT_API}/1/user/-/activities/heart/date/today/1d/1min.json",
                      headers=headers, timeout=20)
    if r1.status_code not in (200, 204):
        raise HTTPException(status_code=502, detail=f"Fitbit HR fetch failed: {r1.text}")
    r2 = requests.get(f"{FITBIT_API}/1/user/-/activities/steps/date/today/1d/1min.json",
                      headers=headers, timeout=20)
    if r2.status_code not in (200, 204):
        raise HTTPException(status_code=502, detail=f"Fitbit steps fetch failed: {r2.text}")

    ds_hr = r1.json().get("activities-heart-intraday", {}).get("dataset", [])
    ds_steps = r2.json().get("activities-steps-intraday", {}).get("dataset", [])
    today = datetime.utcnow().date()

    inserted = 0
    last_hr = None
    with conn.cursor() as cur:
        for p in ds_hr[-120:]:
            t = datetime.combine(today, datetime.strptime(p["time"], "%H:%M:%S").time())
            last_hr = int(p["value"])
            _insert_or_update_vital(cur, patient_id, t, hr=last_hr)
            inserted += 1
        for p in ds_steps[-120:]:
            t = datetime.combine(today, datetime.strptime(p["time"], "%H:%M:%S").time())
            _insert_or_update_vital(cur, patient_id, t, steps=int(p["value"]))
            inserted += 1

        # simple alert on latest HR
        if last_hr is not None and (last_hr < 40 or last_hr > 135):
            msg = f"HR {last_hr} abnormal (Fitbit sync)"
            cur.execute("""
                SELECT COUNT(*) FROM alerts
                 WHERE patient_id=%s AND type='HR_ABNORMAL'
                   AND created_at > now() - interval '15 minutes'
            """, (patient_id,))
            recent = cur.fetchone()[0] > 0
            cur.execute("""
                INSERT INTO alerts (patient_id, type, severity, message)
                VALUES (%s,'HR_ABNORMAL','HIGH',%s)
            """, (patient_id, msg))
            if not recent:
                cur.execute("SELECT caregiver_phone FROM patients WHERE id=%s", (patient_id,))
                row = cur.fetchone()
                if row and row[0]:
                    send_sms_safe(row[0], f"[ALERT] {msg}")

    return {"ok": True, "fitbit_user_id": fitbit_user_id, "records_processed": inserted}

@app.get("/integrations/fitbit/sync_all")
def fitbit_sync_all(
    x_admin_key: str | None = Header(default=None)
):
    _check_admin_key(x_admin_key)

    processed, failed = 0, []
    with conn.cursor() as cur:
        cur.execute("SELECT patient_id FROM fitbit_tokens")
        ids = [str(r[0]) for r in cur.fetchall()]

    for pid in ids:
        try:
            # call the internal function directly to reuse logic
            fitbit_sync(pid, x_admin_key=x_admin_key)
            processed += 1
        except Exception as e:
            failed.append({"patient_id": pid, "error": str(e)})

    return {"ok": True, "processed": processed, "failed": failed}
