import os, base64, json, time, logging
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
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


# ======================
# Config & DB connection
# ======================
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
        logging.warning(f"Twilio error: {e}")
        return False

# Admin key to protect cronable endpoints
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

# ===== Timezone helpers =====
TZ_FIXES = {
    "America/Austin": "America/Chicago",
    "Austin": "America/Chicago",
    "CST": "America/Chicago",
    "PST": "America/Los_Angeles",
    "EST": "America/New_York",
    "MST": "America/Denver",
    "IST": "Asia/Kolkata",
    "US/Central": "America/Chicago",
}

def canonical_tz(tz: str | None) -> str:
    """Return a safe IANA timezone (fallback to UTC)."""
    if not tz:
        return "America/Chicago"
    t = TZ_FIXES.get(tz.strip(), tz.strip())
    try:
        ZoneInfo(t)
        return t
    except ZoneInfoNotFoundError:
        logging.warning(f"Unknown timezone '{tz}', falling back to UTC")
        return "UTC"

def _fitbit_user_timezone(access_token: str) -> str:
    hdr = {"Authorization": f"Bearer {access_token}"}
    try:
        r = requests.get(f"{FITBIT_API}/1/user/-/profile.json", headers=hdr, timeout=15)
        if r.status_code == 200:
            tz = r.json().get("user", {}).get("timezone", "UTC") or "UTC"
            return canonical_tz(tz)
    except Exception:
        pass
    return "UTC"

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


# =========
# FastAPI
# =========
app = FastAPI(title="Health Readmit API", version="0.8.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =======
# Models
# =======
class PatientIn(BaseModel):
    name: str
    dob: date
    sex_at_birth: Optional[str] = None
    caregiver_phone: Optional[str] = None
    patient_phone: Optional[str] = None
    timezone: Optional[str] = None  # default applied if None

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

# ---- Option C defaults ----
DEFAULT_TIMESETS = {
  "qd": ["09:00"],
  "daily": ["09:00"],
  "od": ["09:00"],
  "bid": ["09:00", "21:00"],
  "tid": ["08:00", "14:00", "20:00"],
  "qid": ["08:00", "12:00", "16:00", "20:00"],
}

class MedicationIn(BaseModel):
    patient_id: str
    drug_name: str
    dose: str | None = None
    freq: str = "daily"
    times_local: List[str] | None = None    # OPTIONAL now
    timezone: str | None = None             # OPTIONAL now
    start_date: date
    end_date: date | None = None
    is_critical: bool = False

class MedAckIn(BaseModel):
    scheduled_time: datetime | None = None   # if omitted, ack most recent pending
    source: str = "app"


# =========
# Helpers
# =========
def _insert_or_update_vital(cur, patient_id: str, ts: datetime,
                            hr: int | None = None, steps: int | None = None,
                            spo2: int | None = None, sbp: int | None = None,
                            dbp: int | None = None, temp_c: float | None = None):
    cur.execute("""
        INSERT INTO vitals (patient_id, ts, hr, steps, spo2, sbp, dbp, temp_c)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (patient_id, ts) DO UPDATE
          SET hr = COALESCE(EXCLUDED.hr, vitals.hr),
              steps = COALESCE(EXCLUDED.steps, vitals.steps),
              spo2 = COALESCE(EXCLUDED.spo2, vitals.spo2),
              sbp = COALESCE(EXCLUDED.sbp, vitals.sbp),
              dbp = COALESCE(EXCLUDED.dbp, vitals.dbp),
              temp_c = COALESCE(EXCLUDED.temp_c, vitals.temp_c)
    """, (patient_id, ts, hr, steps, spo2, sbp, dbp, temp_c))

def _today_scheduled_utc(tz: str, times_local: List[str], on_date: date) -> List[datetime]:
    z = ZoneInfo(canonical_tz(tz))
    out = []
    for t in times_local:
        try:
            parts = t.split(":")
            if len(parts) < 2:
                raise ValueError("bad format, expected HH:MM")
            hh, mm = int(parts[0]), int(parts[1])
            dt_local = datetime(on_date.year, on_date.month, on_date.day, hh, mm, tzinfo=z)
            out.append(dt_local.astimezone(ZoneInfo("UTC")))
        except Exception as e:
            logging.warning(f"Skipping bad times_local '{t}' (tz={tz}): {e}")
            continue
    return out

def _send_med_sms(patient_id: str, msg: str) -> int:
    sent = 0
    with conn.cursor() as cur:
        cur.execute("SELECT patient_phone, caregiver_phone FROM patients WHERE id=%s", (patient_id,))
        row = cur.fetchone()
    if not row:
        return 0
    patient_phone, caregiver_phone = row
    body = f"{msg} Reply STOP to opt out, HELP for help."
    if patient_phone and send_sms_safe(patient_phone, body): sent += 1
    if caregiver_phone and send_sms_safe(caregiver_phone, f"CarePulse: {msg}"): sent += 1
    return sent

def as_native_json(v):
    """DB JSONB arrives as list/dict via psycopg3; if str, json.loads it; else return as-is."""
    if isinstance(v, (list, dict)) or v is None:
        return v
    try:
        return json.loads(v)
    except Exception:
        return v

def _normalize_times(times: Optional[List[str]], freq: str) -> List[str]:
    """Return a cleaned list of HH:MM times; if empty/invalid, use defaults from freq."""
    cand = times if isinstance(times, list) else None
    if not cand or not any(isinstance(x, str) for x in cand):
        return DEFAULT_TIMESETS.get(freq.lower(), ["09:00"])
    cleaned: List[str] = []
    for t in cand:
        try:
            parts = t.split(":")
            if len(parts) < 2: raise ValueError("bad format")
            hh, mm = int(parts[0]), int(parts[1])
            if not (0 <= hh <= 23 and 0 <= mm <= 59): raise ValueError("range")
            cleaned.append(f"{hh:02d}:{mm:02d}")
        except Exception:
            logging.warning(f"Skipping invalid time '{t}' in times_local; using defaults if none valid.")
    return cleaned if cleaned else DEFAULT_TIMESETS.get(freq.lower(), ["09:00"])

def _patient_timezone(patient_id: str) -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT timezone FROM patients WHERE id=%s", (patient_id,))
        row = cur.fetchone()
    return canonical_tz(row[0] if row and row[0] else None)


# =======
# Routes
# =======
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
        tz = canonical_tz(p.timezone or "America/Chicago")
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO patients (name, dob, sex_at_birth, caregiver_phone, patient_phone, timezone)
                VALUES (%s,%s,%s,%s,%s,%s)
                RETURNING id
            """, (p.name, p.dob, p.sex_at_birth, p.caregiver_phone, p.patient_phone, tz))
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

@app.get("/integrations/fitbit/sync")
def fitbit_sync(
    patient_id: str,
    x_admin_key: str | None = Header(default=None)
):
    _check_admin_key(x_admin_key)

    fitbit_user_id, access = _ensure_fitbit_tokens(patient_id)
    headers = {"Authorization": f"Bearer {access}"}

    # Intraday 1-min HR & steps for today (Fitbit local time)
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
    if not isinstance(ds_hr, list): ds_hr = []
    if not isinstance(ds_steps, list): ds_steps = []

    # Convert Fitbit local times to UTC using user's tz
    tz_str = _fitbit_user_timezone(access)
    user_tz = ZoneInfo(tz_str)
    today_local = datetime.now(user_tz).date()

    def to_utc(hms: str) -> datetime:
        hh, mm, ss = map(int, hms.split(":"))
        dt_local = datetime(today_local.year, today_local.month, today_local.day, hh, mm, ss, tzinfo=user_tz)
        return dt_local.astimezone(ZoneInfo("UTC"))

    inserted = 0
    last_hr = None
    with conn.cursor() as cur:
        for p in ds_hr[-120:]:
            t = to_utc(p["time"])
            last_hr = int(p["value"])
            _insert_or_update_vital(cur, patient_id, t, hr=last_hr)
            inserted += 1
        for p in ds_steps[-120:]:
            t = to_utc(p["time"])
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
            fitbit_sync(pid, x_admin_key=x_admin_key)  # reuse logic
            processed += 1
        except Exception as e:
            failed.append({"patient_id": pid, "error": str(e)})

    logging.info(f"[sync_all] processed={processed} failed={len(failed)}")
    return {"ok": True, "processed": processed, "failed": failed}


# -------- Medications: create/list/ack + reminder cron (Option C defaults) --------
@app.post("/v1/meds")
def create_med(m: MedicationIn):
    try:
        # Defaults from patient/timezone and freq
        tz = canonical_tz(m.timezone or _patient_timezone(m.patient_id))
        times = _normalize_times(m.times_local, m.freq)

        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO medications
                (patient_id, drug_name, dose, freq, times_local, timezone, start_date, end_date, is_critical)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING id
            """, (m.patient_id, m.drug_name, m.dose, m.freq,
                  json.dumps(times), tz, m.start_date, m.end_date, m.is_critical))
            (mid,) = cur.fetchone()
        return {"ok": True, "medication_id": str(mid), "times_local": times, "timezone": tz}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

@app.get("/v1/meds")
def list_meds(patient_id: str):
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, drug_name, dose, freq, times_local, timezone, start_date, end_date, is_critical
                  FROM medications
                 WHERE patient_id=%s
                 ORDER BY created_at DESC
            """, (patient_id,))
            rows = cur.fetchall()
        res = []
        for r in rows:
            res.append({
                "id": str(r[0]),
                "drug_name": r[1],
                "dose": r[2],
                "freq": r[3],
                "times_local": as_native_json(r[4]),
                "timezone": r[5],
                "start_date": r[6].isoformat(),
                "end_date": r[7].isoformat() if r[7] else None,
                "is_critical": r[8]
            })
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

@app.post("/v1/meds/{med_id}/ack")
def ack_med(med_id: str, body: MedAckIn):
    try:
        with conn.cursor() as cur:
            if body.scheduled_time is None:
                cur.execute("""
                    SELECT scheduled_time FROM med_adherence
                     WHERE medication_id=%s AND taken=false
                 ORDER BY scheduled_time DESC
                    LIMIT 1
                """, (med_id,))
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="No pending dose to acknowledge")
                sched = row[0]
            else:
                sched = body.scheduled_time

            cur.execute("""
                UPDATE med_adherence
                   SET taken=true, taken_time=now(), source=%s
                 WHERE medication_id=%s AND scheduled_time=%s
                 RETURNING id
            """, (body.source, med_id, sched))
            u = cur.fetchone()
            if not u:
                cur.execute("""
                    INSERT INTO med_adherence (medication_id, scheduled_time, taken, taken_time, source)
                    VALUES (%s,%s,true,now(),%s)
                    RETURNING id
                """, (med_id, sched, body.source))
                u = cur.fetchone()
        return {"ok": True, "id": str(u[0]), "scheduled_time": sched.isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

@app.get("/cron/meds/remind_now")
def meds_remind_now(x_admin_key: str | None = Header(default=None)):
    _check_admin_key(x_admin_key)

    now_utc = datetime.now(ZoneInfo("UTC"))  # aware
    window_minutes = 5  # align with cron frequency
    sent_total = 0
    created_total = 0

    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, patient_id, times_local, timezone, start_date, end_date
              FROM medications
             WHERE start_date <= CURRENT_DATE
               AND (end_date IS NULL OR end_date >= CURRENT_DATE)
        """)
        meds = cur.fetchall()

        for mid, pid, times_json, tz, sdate, edate in meds:
            times = as_native_json(times_json) or []
            if not isinstance(times, list):
                continue
            # compute schedule for the patient's LOCAL day
            tz_can = canonical_tz(tz)
            local_today = datetime.now(ZoneInfo(tz_can)).date()
            for sched_utc in _today_scheduled_utc(tz_can, times, local_today):
                delta = abs((now_utc - sched_utc).total_seconds()) / 60.0
                if delta <= window_minutes:
                    # create pending adherence row if not exists (prevents duplicate SMS)
                    cur.execute("""
                        INSERT INTO med_adherence (medication_id, scheduled_time, taken)
                        VALUES (%s,%s,false)
                        ON CONFLICT (medication_id, scheduled_time) DO NOTHING
                        RETURNING id
                    """, (mid, sched_utc))
                    ins = cur.fetchone()
                    if ins:
                        created_total += 1
                        sent_total += _send_med_sms(
                            pid,
                            "Medication time: please take your scheduled dose."
                        )
    logging.info(f"[meds_remind_now] created={created_total} sms_sent={sent_total}")
    return {"ok": True, "reminders_created": created_total, "sms_sent": sent_total}
