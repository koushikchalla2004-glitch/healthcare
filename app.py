import os, base64, json, time, logging, re, io, mimetypes, csv, math, pickle
from datetime import datetime, date, timedelta
from typing import Optional, List
from urllib.parse import urlencode

import numpy as np
import psycopg  # v3
import requests
from fastapi import FastAPI, HTTPException, Query, Header, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from twilio.rest import Client
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from jose import jwt, JWTError
from passlib.hash import bcrypt

# ======================
# Config & DB connection
# ======================
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("DATABASE_URL env var is required")
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

# Admin key for cron/privileged ops (reuse your CRON_KEY)
CRON_KEY = os.getenv("CRON_KEY")
def _check_admin_key(x_admin_key: str | None):
    if CRON_KEY and x_admin_key != CRON_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ===== Auth (JWT) =====
AUTH_SECRET = os.getenv("AUTH_SECRET", "dev-secret-change-me")
ACCESS_TOKEN_MIN = int(os.getenv("ACCESS_TOKEN_MIN", "10080"))  # 7 days

def _hash_pw(pw: str) -> str: return bcrypt.hash(pw)
def _check_pw(pw: str, ph: str) -> bool:
    try: return bcrypt.verify(pw, ph)
    except Exception: return False

def _make_token(sub: str, role: str) -> str:
    exp = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_MIN)
    return jwt.encode({"sub": sub, "role": role, "exp": exp}, AUTH_SECRET, algorithm="HS256")

def _decode_token(token: str) -> dict:
    return jwt.decode(token, AUTH_SECRET, algorithms=["HS256"])

def _current_user(authorization: str | None = Header(default=None)) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    try:
        payload = _decode_token(token)
        return {"user_id": payload["sub"], "role": payload.get("role", "patient")}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ===== Google Document AI (OCR) init =====
GCP_DOCAI_PROJECT = os.getenv("GCP_DOCAI_PROJECT")
GCP_DOCAI_LOCATION = os.getenv("GCP_DOCAI_LOCATION", "us")
GCP_DOCAI_PROCESSOR_ID = os.getenv("GCP_DOCAI_PROCESSOR_ID")
GCP_SA_KEY_B64 = os.getenv("GCP_SA_KEY_B64")

_docai_client = None
_docai_name = None
try:
    if GCP_SA_KEY_B64 and GCP_DOCAI_PROJECT and GCP_DOCAI_PROCESSOR_ID:
        from google.cloud import documentai
        from google.oauth2 import service_account
        sa_info = json.loads(base64.b64decode(GCP_SA_KEY_B64))
        creds = service_account.Credentials.from_service_account_info(sa_info)
        _docai_client = documentai.DocumentProcessorServiceClient(credentials=creds)
        _docai_name = _docai_client.processor_path(
            GCP_DOCAI_PROJECT, GCP_DOCAI_LOCATION, GCP_DOCAI_PROCESSOR_ID
        )
        logging.info("Document AI client initialized")
    else:
        logging.info("Document AI not configured; OCR disabled")
except Exception as e:
    logging.warning(f"DocAI init failed: {e}")
    _docai_client, _docai_name = None, None

# =========
# FastAPI
# =========
app = FastAPI(title="Health Readmit API", version="1.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
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
    timezone: Optional[str] = None

class PatientPatch(BaseModel):
    caregiver_phone: Optional[str] = None
    patient_phone: Optional[str] = None
    timezone: Optional[str] = None

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
    provider: str
    external_user_id: str

# Medication defaults
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
    times_local: List[str] | None = None
    timezone: str | None = None
    start_date: date
    end_date: date | None = None
    is_critical: bool = False

class MedAckIn(BaseModel):
    scheduled_time: datetime | None = None
    source: str = "app"

# =========
# Timezone helpers
# =========
TZ_FIXES = {
    "America/Austin": "America/Chicago", "Austin": "America/Chicago",
    "CST": "America/Chicago", "PST": "America/Los_Angeles", "EST": "America/New_York",
    "MST": "America/Denver", "IST": "Asia/Kolkata", "US/Central": "America/Chicago",
}
def canonical_tz(tz: str | None) -> str:
    if not tz: return "America/Chicago"
    t = TZ_FIXES.get(tz.strip(), tz.strip())
    try:
        ZoneInfo(t); return t
    except ZoneInfoNotFoundError:
        logging.warning(f"Unknown timezone '{tz}', falling back to UTC"); return "UTC"

def _patient_timezone(patient_id: str) -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT timezone FROM patients WHERE id=%s", (patient_id,))
        row = cur.fetchone()
    return canonical_tz(row[0] if row and row[0] else None)

# =========
# Shared helpers
# =========
def _insert_or_update_vital(cur, patient_id: str, ts: datetime,
                            hr: int | None = None, steps: int | None = None,
                            spo2: int | None = None, sbp: int | None = None,
                            dbp: int | None = None, temp_c: float | None = None):
    cur.execute("""
        INSERT INTO vitals (patient_id, ts, hr, steps, spo2, sbp, dbp, temp_c)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (patient_id, ts) DO NOTHING
    """, (patient_id, ts, hr, steps, spo2, sbp, dbp, temp_c))

def _today_scheduled_utc(tz: str, times_local: List[str], on_date: date) -> List[datetime]:
    z = ZoneInfo(canonical_tz(tz)); out = []
    for t in times_local:
        try:
            hh, mm = [int(x) for x in t.split(":")[:2]]
            dt_local = datetime(on_date.year, on_date.month, on_date.day, hh, mm, tzinfo=z)
            out.append(dt_local.astimezone(ZoneInfo("UTC")))
        except Exception as e:
            logging.warning(f"Skipping bad time '{t}' for tz={tz}: {e}")
    return out

def _send_med_sms(patient_id: str, msg: str) -> int:
    sent = 0
    with conn.cursor() as cur:
        cur.execute("SELECT patient_phone, caregiver_phone FROM patients WHERE id=%s", (patient_id,))
        row = cur.fetchone()
    if not row: return 0
    patient_phone, caregiver_phone = row
    body = f"{msg} Reply STOP to opt out, HELP for help."
    if patient_phone and send_sms_safe(patient_phone, body): sent += 1
    if caregiver_phone and send_sms_safe(caregiver_phone, f"CarePulse: {msg}"): sent += 1
    return sent

def as_native_json(v):
    if isinstance(v, (list, dict)) or v is None: return v
    try: return json.loads(v)
    except Exception: return v

def _normalize_times(times: Optional[List[str]], freq: str) -> List[str]:
    cand = times if isinstance(times, list) else None
    if not cand or not any(isinstance(x, str) for x in cand):
        return DEFAULT_TIMESETS.get(freq.lower(), ["09:00"])
    cleaned: List[str] = []
    for t in cand:
        try:
            hh, mm = [int(x) for x in t.split(":")[:2]]
            if 0 <= hh <= 23 and 0 <= mm <= 59: cleaned.append(f"{hh:02d}:{mm:02d}")
        except Exception:
            continue
    return cleaned if cleaned else DEFAULT_TIMESETS.get(freq.lower(), ["09:00"])

# =========
# Unstructured parsers (ICD-10 & pharmacy)
# =========
ICD10_RE = re.compile(r"\b([A-TV-Z][0-9]{2}(?:\.[0-9A-Za-z]{1,4})?)\b")

MED_LINE = re.compile(
    r"""(?ix)
    ^\s*
    (?P<name>[A-Za-z][\w\-/\s]{1,60}?)        # drug name
    [\s,]+
    (?P<dose>\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units|iu))  # dose
    (?:[,\s;]*(?P<route>po|oral|iv|im|sc|sl|topical|inh))?
    (?P<sig>.*)$
    """
)

SIG_MAP = {
    "qd": "daily", "od": "daily", "daily": "daily", "once daily": "daily",
    "bid": "bid", "twice daily": "bid", "2xd": "bid",
    "tid": "tid", "3xd": "tid",
    "qid": "qid", "4xd": "qid",
    "hs": "hs", "nocte": "hs", "bedtime": "hs",
    "mane": "morning", "morning": "morning",
    "noon": "noon", "afternoon": "noon",
    "evening": "evening", "night": "night",
}

def _parse_text_for_dx(text: str) -> List[str]:
    return sorted(set(ICD10_RE.findall(text or "")))

def _infer_times_from_sig(sig: str | None, freq_fallback: str = "daily") -> List[str] | None:
    if not sig: return None
    s = sig.lower().strip()
    tmatches = re.findall(r'\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b', s)
    times: List[str] = []
    for hh, mm, ap in tmatches:
        h = int(hh); m = int(mm or 0)
        if ap:
            if ap.lower() == "pm" and h < 12: h += 12
            if ap.lower() == "am" and h == 12: h = 0
        if 0 <= h <= 23 and 0 <= m <= 59: times.append(f"{h:02d}:{m:02d}")
    if times: return sorted(list(set(times)))
    m = re.search(r'\b([01])\s*-\s*([01])\s*-\s*([01])\b', s)
    if m:
        morning, noon, night = (m.group(i) == "1" for i in (1,2,3))
        t = []
        if morning: t.append("09:00")
        if noon:    t.append("13:00")
        if night:   t.append("21:00")
        return t or None
    mq = re.search(r'\bq\s*([46812]|24)\s*(?:h|hr|hrs|hours)\b', s)
    if mq:
        h = int(mq.group(1))
        if h == 24: return ["09:00"]
        if h == 12: return ["09:00", "21:00"]
        if h == 8:  return ["06:00", "14:00", "22:00"]
        if h == 6:  return ["06:00", "12:00", "18:00", "00:00"]
        if h == 4:  return ["06:00", "10:00", "14:00", "18:00", "22:00", "02:00"]
    if any(tok in s for tok in ("hs","nocte","bedtime")): return ["22:00"]
    if any(tok in s for tok in ("mane","morning")):       return ["09:00"]
    if "noon" in s or "afternoon" in s:                   return ["13:00"]
    if "evening" in s:                                    return ["19:00"]
    if "night" in s:                                      return ["21:00"]
    if "tid" in s or "3xd" in s:  return ["08:00","14:00","20:00"]
    if "qid" in s or "4xd" in s:  return ["08:00","12:00","16:00","20:00"]
    if "bid" in s or "twice daily" in s or "2xd" in s: return ["09:00","21:00"]
    if any(k in s for k in ("qd","od","once daily","daily")): return ["09:00"]
    return None

def _extract_meds_unstructured(text: str) -> List[dict]:
    meds = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or len(line) < 3: continue
        m = MED_LINE.match(line)
        if not m: continue
        name = re.sub(r'\s+', ' ', m.group("name")).strip(" -")
        dose = (m.group("dose") or "").strip()
        route = (m.group("route") or "").lower()
        sig = (m.group("sig") or "").strip()
        freq = None
        for k, v in SIG_MAP.items():
            if k in sig:
                freq = "daily" if v in ("morning","noon","evening","night","hs") else v
                break
        meds.append({"drug_name": name, "dose": dose, "route": route, "sig": sig, "freq": freq or "daily"})
    return meds

# =========
# OCR/Plain-text reader
# =========
def _read_text_from_upload(file: UploadFile) -> str:
    data = file.file.read()
    ctype = file.content_type or mimetypes.guess_type(file.filename or "")[0] or ""
    is_scan = ("pdf" in ctype) or ("image" in ctype) or ctype == ""
    if _docai_client and _docai_name and is_scan:
        try:
            from google.cloud import documentai
            raw_document = documentai.RawDocument(content=data, mime_type=ctype or "application/pdf")
            request = documentai.ProcessRequest(name=_docai_name, raw_document=raw_document)
            result = _docai_client.process_document(request=request)
            return result.document.text or ""
        except Exception as e:
            logging.warning(f"DocAI processing failed; falling back to text decode: {e}")
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

# =======
# Routes: auth
# =======
class RegisterIn(BaseModel):
    email: str
    password: str
    role: str = "patient"      # 'patient' or 'admin'
    patient_id: Optional[str] = None

class LoginIn(BaseModel):
    email: str
    password: str

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"

@app.post("/auth/register")
def auth_register(body: RegisterIn, x_admin_key: str | None = Header(default=None)):
    _check_admin_key(x_admin_key)  # require admin key to create users
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO users (email, password_hash, role) VALUES (%s,%s,%s) RETURNING id, role",
                        (body.email.lower().strip(), _hash_pw(body.password), body.role))
            uid, role = cur.fetchone()
            if body.patient_id:
                cur.execute("INSERT INTO user_patient (user_id, patient_id) VALUES (%s,%s) ON CONFLICT DO NOTHING",
                            (uid, body.patient_id))
        return {"ok": True, "user_id": str(uid), "role": role}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Register failed: {e}")

@app.post("/auth/login", response_model=TokenOut)
def auth_login(body: LoginIn):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, password_hash, role FROM users WHERE email=%s", (body.email.lower().strip(),))
            row = cur.fetchone()
        if not row or not _check_pw(body.password, row[1]):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        uid, role = str(row[0]), row[2]
        token = _make_token(uid, role)
        return {"access_token": token}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login error: {e}")

@app.get("/v1/me")
def me(user=Depends(_current_user)):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT email, role FROM users WHERE id=%s", (user["user_id"],))
            u = cur.fetchone()
            cur.execute("SELECT patient_id FROM user_patient WHERE user_id=%s LIMIT 1", (user["user_id"],))
            p = cur.fetchone()
        return {"user_id": user["user_id"], "email": u[0], "role": u[1], "patient_id": (str(p[0]) if p else None)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"me error: {e}")

# =======
# Routes: patients
# =======
@app.get("/healthz")
def healthz():
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;"); cur.fetchone()
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
                VALUES (%s,%s,%s,%s,%s,%s) RETURNING id
            """, (p.name, p.dob, p.sex_at_birth, p.caregiver_phone, p.patient_phone, tz))
            (pid,) = cur.fetchone()
        return {"ok": True, "id": str(pid)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

@app.get("/v1/patients/{patient_id}")
def get_patient(patient_id: str):
    try:
        with conn.cursor() as cur:
            cur.execute("""SELECT id, name, dob, sex_at_birth, caregiver_phone, patient_phone, timezone
                           FROM patients WHERE id=%s""", (patient_id,))
            r = cur.fetchone()
        if not r: raise HTTPException(status_code=404, detail="Patient not found")
        return {"id": str(r[0]), "name": r[1], "dob": r[2].isoformat(),
                "sex_at_birth": r[3], "caregiver_phone": r[4],
                "patient_phone": r[5], "timezone": r[6]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

@app.patch("/v1/patients/{patient_id}")
def patch_patient(patient_id: str, body: PatientPatch):
    try:
        fields, vals = [], []
        if body.caregiver_phone is not None: fields.append("caregiver_phone=%s"); vals.append(body.caregiver_phone)
        if body.patient_phone is not None:   fields.append("patient_phone=%s");   vals.append(body.patient_phone)
        if body.timezone is not None:        fields.append("timezone=%s");        vals.append(canonical_tz(body.timezone))
        if not fields: return {"ok": True, "updated": 0}
        vals.append(patient_id)
        with conn.cursor() as cur:
            cur.execute(f"UPDATE patients SET {', '.join(fields)} WHERE id=%s", vals)
        return {"ok": True, "updated": 1}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

@app.post("/v1/patients/{patient_id}/sources")
def link_source(patient_id: str, body: SourceLink):
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO patient_sources (patient_id, provider, external_user_id)
                VALUES (%s,%s,%s)
                ON CONFLICT (provider, external_user_id) DO UPDATE SET patient_id=EXCLUDED.patient_id
            """, (patient_id, body.provider, body.external_user_id))
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

# -------- Vitals & alerts (unchanged from your live build) --------
@app.post("/v1/vitals")
def ingest_vitals(batch: VitalsBatch):
    clean = []
    for s in batch.samples:
        if s.hr is not None and (s.hr < 25 or s.hr > 220): continue
        if s.spo2 is not None and (s.spo2 < 50 or s.spo2 > 100): continue
        clean.append(s)
    if not clean: raise HTTPException(status_code=400, detail="No valid samples")

    try:
        with conn.cursor() as cur:
            cur.executemany("""
                INSERT INTO vitals (patient_id, ts, hr, spo2, steps, sbp, dbp, temp_c)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (patient_id, ts) DO NOTHING
            """, [(
                batch.patient_id, s.ts, s.hr, s.spo2, s.steps, s.sbp, s.dbp, s.temp_c
            ) for s in clean])

            caregiver_phone = None
            cur.execute("SELECT caregiver_phone FROM patients WHERE id=%s", (batch.patient_id,))
            r = cur.fetchone()
            if r: caregiver_phone = r[0]

            last = clean[-1]; candidates = []
            if last.spo2 is not None and last.spo2 < 90:
                candidates.append(("SPO2_LOW", "HIGH", f"SpO2 {last.spo2}% below 90 at {last.ts.isoformat()}"))
            if last.hr is not None and (last.hr < 40 or last.hr > 135):
                candidates.append(("HR_ABNORMAL", "HIGH", f"HR {last.hr} abnormal at {last.ts.isoformat()}"))

            for t, se, msg in candidates:
                cur.execute("""SELECT COUNT(*) FROM alerts
                               WHERE patient_id=%s AND type=%s AND created_at > now()-interval '15 minutes'""",
                            (batch.patient_id, t))
                recent = cur.fetchone()[0] > 0
                cur.execute("""INSERT INTO alerts (patient_id, type, severity, message)
                               VALUES (%s,%s,%s,%s)""", (batch.patient_id, t, se, msg))
                if se == "HIGH" and not recent and caregiver_phone:
                    send_sms_safe(caregiver_phone, f"[ALERT] {msg}")

        return {"ok": True, "inserted": len(clean)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

@app.post("/v1/alerts")
def create_alert(a: AlertIn):
    try:
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO alerts (patient_id, type, severity, message)
                           VALUES (%s,%s,%s,%s) RETURNING id, created_at""",
                        (a.patient_id, a.type, a.severity, a.message))
            rid, ts = cur.fetchone()
        return {"ok": True, "id": str(rid), "created_at": ts.isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

@app.get("/v1/alerts/{patient_id}")
def list_alerts(patient_id: str):
    try:
        with conn.cursor() as cur:
            cur.execute("""SELECT id, type, severity, message, created_at, status
                           FROM alerts WHERE patient_id=%s ORDER BY created_at DESC LIMIT 50""", (patient_id,))
            rows = cur.fetchall()
        return [{"id": str(r[0]), "type": r[1], "severity": r[2], "message": r[3],
                 "created_at": r[4].isoformat(), "status": r[5]} for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

# -------- Fitbit OAuth + Sync (unchanged) --------
@app.get("/oauth/fitbit/start")
def fitbit_start(patient_id: str = Query(..., description="UUID of patient")):
    if not (os.getenv("FITBIT_CLIENT_ID") and os.getenv("FITBIT_REDIRECT_URI")):
        raise HTTPException(status_code=500, detail="Fitbit env vars missing")
    q = {
        "client_id": os.getenv("FITBIT_CLIENT_ID"),
        "redirect_uri": os.getenv("FITBIT_REDIRECT_URI"),
        "response_type": "code",
        "scope": os.getenv("FITBIT_SCOPE", "heartrate activity profile"),
        "prompt": "consent",
        "state": patient_id,
    }
    return RedirectResponse(f"https://www.fitbit.com/oauth2/authorize?{urlencode(q)}")

@app.get("/oauth/fitbit/callback")
def fitbit_callback(code: str, state: str):
    data = {
        "client_id": os.getenv("FITBIT_CLIENT_ID"),
        "grant_type": "authorization_code",
        "redirect_uri": os.getenv("FITBIT_REDIRECT_URI"),
        "code": code,
    }
    r = requests.post("https://api.fitbit.com/oauth2/token", data=data,
                      headers={"Authorization": "Basic " + base64.b64encode(
                          (os.getenv("FITBIT_CLIENT_ID") + ":" + os.getenv("FITBIT_CLIENT_SECRET")).encode()
                      ).decode()}, timeout=20)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Token exchange failed: {r.text}")
    tok = r.json()
    fitbit_user_id = tok.get("user_id")
    access = tok["access_token"]; refresh = tok["refresh_token"]
    expires_at = int(time.time()) + int(tok.get("expires_in", 28800))
    try:
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO fitbit_tokens (patient_id, fitbit_user_id, access_token, refresh_token, expires_at, scope, token_type)
                           VALUES (%s,%s,%s,%s, TO_TIMESTAMP(%s), %s, %s)
                           ON CONFLICT (patient_id) DO UPDATE
                           SET fitbit_user_id=EXCLUDED.fitbit_user_id,
                               access_token=EXCLUDED.access_token,
                               refresh_token=EXCLUDED.refresh_token,
                               expires_at=EXCLUDED.expires_at,
                               scope=EXCLUDED.scope,
                               token_type=EXCLUDED.token_type
                        """, (state, fitbit_user_id, access, refresh, expires_at, tok.get("scope"), tok.get("token_type")))
            cur.execute("""INSERT INTO patient_sources (patient_id, provider, external_user_id)
                           VALUES (%s,'fitbit',%s)
                           ON CONFLICT (provider, external_user_id) DO UPDATE SET patient_id=EXCLUDED.patient_id
                        """, (state, fitbit_user_id))
        return {"ok": True, "linked": True, "patient_id": state, "fitbit_user_id": fitbit_user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

def _ensure_fitbit_tokens(patient_id: str):
    with conn.cursor() as cur:
        cur.execute("""SELECT fitbit_user_id, access_token, refresh_token, EXTRACT(EPOCH FROM expires_at)
                       FROM fitbit_tokens WHERE patient_id=%s""", (patient_id,))
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Fitbit not linked for this patient")
    user_id, access, refresh, exp = row
    if time.time() < float(exp) - 120: return user_id, access
    r = requests.post("https://api.fitbit.com/oauth2/token",
                      data={"grant_type":"refresh_token","refresh_token":refresh},
                      headers={"Authorization": "Basic " + base64.b64encode(
                          (os.getenv("FITBIT_CLIENT_ID") + ":" + os.getenv("FITBIT_CLIENT_SECRET")).encode()
                      ).decode()}, timeout=20)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Fitbit refresh failed: {r.text}")
    tok = r.json()
    access = tok["access_token"]; refresh = tok["refresh_token"]
    expires_at = int(time.time()) + int(tok.get("expires_in", 28800))
    with conn.cursor() as cur:
        cur.execute("""UPDATE fitbit_tokens
                       SET access_token=%s, refresh_token=%s, expires_at=TO_TIMESTAMP(%s),
                           scope=%s, token_type=%s WHERE patient_id=%s""",
                    (access, refresh, expires_at, tok.get("scope"), tok.get("token_type"), patient_id))
    return user_id, access

@app.get("/integrations/fitbit/sync")
def fitbit_sync(patient_id: str, x_admin_key: str | None = Header(default=None)):
    _check_admin_key(x_admin_key)
    fitbit_user_id, access = _ensure_fitbit_tokens(patient_id)
    headers = {"Authorization": f"Bearer {access}"}
    r1 = requests.get("https://api.fitbit.com/1/user/-/activities/heart/date/today/1d/1min.json", headers=headers, timeout=20)
    if r1.status_code not in (200, 204): raise HTTPException(status_code=502, detail=f"Fitbit HR fetch failed: {r1.text}")
    r2 = requests.get("https://api.fitbit.com/1/user/-/activities/steps/date/today/1d/1min.json", headers=headers, timeout=20)
    if r2.status_code not in (200, 204): raise HTTPException(status_code=502, detail=f"Fitbit steps fetch failed: {r2.text}")
    ds_hr = r1.json().get("activities-heart-intraday", {}).get("dataset", []) or []
    ds_steps = r2.json().get("activities-steps-intraday", {}).get("dataset", []) or []
    tz_str = _fitbit_user_timezone(access); user_tz = ZoneInfo(tz_str); today_local = datetime.now(user_tz).date()

    def to_utc(hms: str) -> datetime:
        hh, mm, ss = [int(x) for x in hms.split(":")]
        dt_local = datetime(today_local.year, today_local.month, today_local.day, hh, mm, ss, tzinfo=user_tz)
        return dt_local.astimezone(ZoneInfo("UTC"))

    inserted = 0; last_hr = None
    with conn.cursor() as cur:
        for p in ds_hr[-120:]:
            t = to_utc(p["time"]); last_hr = int(p["value"]); _insert_or_update_vital(cur, patient_id, t, hr=last_hr); inserted += 1
        for p in ds_steps[-120:]:
            t = to_utc(p["time"]); _insert_or_update_vital(cur, patient_id, t, steps=int(p["value"])); inserted += 1

        if last_hr is not None and (last_hr < 40 or last_hr > 135):
            msg = f"HR {last_hr} abnormal (Fitbit sync)"
            cur.execute("""SELECT COUNT(*) FROM alerts WHERE patient_id=%s AND type='HR_ABNORMAL'
                           AND created_at > now()-interval '15 minutes'""", (patient_id,))
            recent = cur.fetchone()[0] > 0
            cur.execute("""INSERT INTO alerts (patient_id, type, severity, message)
                           VALUES (%s,'HR_ABNORMAL','HIGH',%s)""", (patient_id, msg))
            if not recent:
                cur.execute("SELECT caregiver_phone FROM patients WHERE id=%s", (patient_id,))
                row = cur.fetchone()
                if row and row[0]: send_sms_safe(row[0], f"[ALERT] {msg}")
    return {"ok": True, "fitbit_user_id": fitbit_user_id, "records_processed": inserted}

@app.get("/integrations/fitbit/sync_all")
def fitbit_sync_all(x_admin_key: str | None = Header(default=None)):
    _check_admin_key(x_admin_key)
    processed, failed = 0, []
    with conn.cursor() as cur:
        cur.execute("SELECT patient_id FROM fitbit_tokens")
        ids = [str(r[0]) for r in cur.fetchall()]
    for pid in ids:
        try: fitbit_sync(pid, x_admin_key=x_admin_key); processed += 1
        except Exception as e: failed.append({"patient_id": pid, "error": str(e)})
    logging.info(f"[sync_all] processed={processed} failed={len(failed)}")
    return {"ok": True, "processed": processed, "failed": failed}

# -------- Medications: CRUD-ish + reminder + escalation --------
@app.post("/v1/meds")
def create_med(m: MedicationIn):
    try:
        tz = canonical_tz(m.timezone or _patient_timezone(m.patient_id))
        times = _normalize_times(m.times_local, m.freq)
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO medications
                           (patient_id, drug_name, dose, freq, times_local, timezone, start_date, end_date, is_critical)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                        (m.patient_id, m.drug_name, m.dose, m.freq, json.dumps(times), tz,
                         m.start_date, m.end_date, m.is_critical))
            (mid,) = cur.fetchone()
        return {"ok": True, "medication_id": str(mid), "times_local": times, "timezone": tz}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

@app.get("/v1/meds")
def list_meds(patient_id: str):
    try:
        with conn.cursor() as cur:
            cur.execute("""SELECT id, drug_name, dose, freq, times_local, timezone, start_date, end_date, is_critical
                           FROM medications WHERE patient_id=%s ORDER BY created_at DESC""", (patient_id,))
            rows = cur.fetchall()
        return [{"id": str(r[0]), "drug_name": r[1], "dose": r[2], "freq": r[3],
                 "times_local": as_native_json(r[4]), "timezone": r[5],
                 "start_date": r[6].isoformat(), "end_date": r[7].isoformat() if r[7] else None,
                 "is_critical": r[8]} for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

@app.post("/v1/meds/{med_id}/ack")
def ack_med(med_id: str, body: MedAckIn):
    try:
        with conn.cursor() as cur:
            if body.scheduled_time is None:
                cur.execute("""SELECT scheduled_time FROM med_adherence
                               WHERE medication_id=%s AND taken=false
                               ORDER BY scheduled_time DESC LIMIT 1""", (med_id,))
                row = cur.fetchone()
                if not row: raise HTTPException(status_code=404, detail="No pending dose to acknowledge")
                sched = row[0]
            else:
                sched = body.scheduled_time
            cur.execute("""UPDATE med_adherence SET taken=true, taken_time=now(), source=%s
                           WHERE medication_id=%s AND scheduled_time=%s RETURNING id""",
                        (body.source, med_id, sched))
            u = cur.fetchone()
            if not u:
                cur.execute("""INSERT INTO med_adherence (medication_id, scheduled_time, taken, taken_time, source)
                               VALUES (%s,%s,true,now(),%s) RETURNING id""",
                            (med_id, sched, body.source))
                u = cur.fetchone()
        return {"ok": True, "id": str(u[0]), "scheduled_time": sched.isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

@app.get("/v1/adherence")
def list_adherence(patient_id: str, limit: int = 50):
    try:
        with conn.cursor() as cur:
            cur.execute("""SELECT ma.id, ma.medication_id, ma.scheduled_time, ma.taken, ma.taken_time, ma.source,
                                  m.drug_name, m.dose
                           FROM med_adherence ma
                           JOIN medications m ON m.id=ma.medication_id
                           WHERE m.patient_id=%s
                           ORDER BY ma.scheduled_time DESC
                           LIMIT %s""", (patient_id, limit))
            rows = cur.fetchall()
        return [{"id": str(r[0]), "medication_id": str(r[1]), "scheduled_time": r[2].isoformat(),
                 "taken": r[3], "taken_time": r[4].isoformat() if r[4] else None,
                 "source": r[5], "drug_name": r[6], "dose": r[7]} for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

@app.get("/cron/meds/remind_now")
def meds_remind_now(x_admin_key: str | None = Header(default=None),
                    window: int = Query(5, ge=1, le=60)):
    _check_admin_key(x_admin_key)
    now_utc = datetime.now(ZoneInfo("UTC"))
    sent_total = 0; created_total = 0
    with conn.cursor() as cur:
        cur.execute("""SELECT id, patient_id, times_local, timezone, start_date, end_date
                       FROM medications
                       WHERE start_date <= CURRENT_DATE AND (end_date IS NULL OR end_date >= CURRENT_DATE)""")
        meds = cur.fetchall()
        for mid, pid, times_json, tz, sdate, edate in meds:
            times = as_native_json(times_json) or []
            if not isinstance(times, list): continue
            tz_can = canonical_tz(tz)
            local_today = datetime.now(ZoneInfo(tz_can)).date()
            for sched_utc in _today_scheduled_utc(tz_can, times, local_today):
                delta = abs((now_utc - sched_utc).total_seconds())/60.0
                if delta <= window:
                    cur.execute("""INSERT INTO med_adherence (medication_id, scheduled_time, taken)
                                   VALUES (%s,%s,false)
                                   ON CONFLICT (medication_id, scheduled_time) DO NOTHING
                                   RETURNING id""", (mid, sched_utc))
                    ins = cur.fetchone()
                    if ins:
                        created_total += 1
                        sent_total += _send_med_sms(pid, "Medication time: please take your scheduled dose.")
    logging.info(f"[meds_remind_now] created={created_total} sms_sent={sent_total}")
    return {"ok": True, "reminders_created": created_total, "sms_sent": sent_total}

@app.get("/cron/meds/escalate_missed")
def meds_escalate_missed(x_admin_key: str | None = Header(default=None)):
    _check_admin_key(x_admin_key)
    now_utc = datetime.now(ZoneInfo("UTC"))
    escalations = 0; sms_sent = 0
    with conn.cursor() as cur:
        cur.execute("""
            SELECT ma.id, ma.medication_id, m.patient_id, m.is_critical, m.timezone, ma.scheduled_time
            FROM med_adherence ma JOIN medications m ON m.id = ma.medication_id
            WHERE ma.taken=false
              AND ma.scheduled_time <= %s - interval '30 minutes'
              AND ma.scheduled_time >= %s - interval '6 hours'
              AND NOT EXISTS (
                SELECT 1 FROM alerts a
                WHERE a.patient_id = m.patient_id
                  AND a.type = 'MISSED_DOSE'
                  AND a.created_at > ma.scheduled_time
              )
        """, (now_utc, now_utc))
        rows = cur.fetchall()
        for _ad_id, _med_id, pid, is_critical, tz, sched in rows:
            sev = "HIGH" if is_critical else "MEDIUM"
            msg = f"Missed medication dose scheduled at {sched.isoformat()}."
            cur.execute("""INSERT INTO alerts (patient_id, type, severity, message)
                           VALUES (%s,'MISSED_DOSE',%s,%s)""", (pid, sev, msg))
            escalations += 1
            sms_sent += _send_med_sms(pid, "Missed dose: please check in. Scheduled at local time.")
    logging.info(f"[escalate_missed] escalations={escalations} sms_sent={sms_sent}")
    return {"ok": True, "escalations": escalations, "sms_sent": sms_sent}

# ===== Unstructured uploads (OCR or plain text) =====
def _read_plain_or_ocr(file: UploadFile) -> str:
    return _read_text_from_upload(file)

@app.post("/v1/docs/discharge_text")
async def upload_discharge_text(patient_id: str = Form(...),
                                file: UploadFile | None = File(None),
                                text: str | None = Form(None)):
    if text and text.strip():
        raw_text = text
    elif file:
        raw_text = _read_plain_or_ocr(file)
    else:
        raise HTTPException(status_code=400, detail="Provide a .txt/.pdf/image file or 'text' form field.")
    try:
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO documents (patient_id, filename, content_type, status)
                           VALUES (%s,%s,%s,'done') RETURNING id""",
                        (patient_id, (file.filename if file else "discharge_text.txt"),
                         (file.content_type if file else "text/plain")))
            (doc_id,) = cur.fetchone()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error creating doc: {e}")
    dx_codes = _parse_text_for_dx(raw_text)
    dx_created = 0
    try:
        with conn.cursor() as cur:
            for code in dx_codes:
                cur.execute("""INSERT INTO diagnoses (patient_id, source_document_id, icd10, description)
                               VALUES (%s,%s,%s,NULL)""", (patient_id, doc_id, code))
                dx_created += 1
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error saving diagnoses: {e}")
    return {"ok": True, "document_id": str(doc_id), "diagnoses_created": dx_created}

@app.post("/v1/docs/meds_text")
async def upload_meds_text(patient_id: str = Form(...),
                           file: UploadFile | None = File(None),
                           text: str | None = Form(None)):
    if text and text.strip():
        raw_text = text
    elif file:
        raw_text = _read_plain_or_ocr(file)
    else:
        raise HTTPException(status_code=400, detail="Provide a .txt/.pdf/image file or 'text' form field.")
    meds = _extract_meds_unstructured(raw_text)
    if not meds:
        return {"ok": True, "medications_created": 0, "note": "No medication patterns recognized."}
    tz = _patient_timezone(patient_id); created = 0
    try:
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO documents (patient_id, filename, content_type, status)
                           VALUES (%s,%s,%s,'done') RETURNING id""",
                        (patient_id, (file.filename if file else "pharmacy_text.txt"),
                         (file.content_type if file else "text/plain")))
            (doc_id,) = cur.fetchone()
            for m in meds[:25]:
                inferred = _infer_times_from_sig(m["sig"], m["freq"])
                times = inferred or _normalize_times(None, m["freq"])
                cur.execute("""INSERT INTO medications
                               (patient_id, drug_name, dose, freq, times_local, timezone, start_date, end_date, is_critical)
                               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                            (patient_id, m["drug_name"], m["dose"], m["freq"],
                             json.dumps(times), tz, date.today(), None, False))
                created += 1
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text/OCR ingest error: {e}")
    return {"ok": True, "medications_created": created}

# ===== Readmission Risk =====
RISK_MODEL_B64 = os.getenv("RISK_MODEL_B64")      # optional: base64(sklearn model.pkl)
RISK_MODEL_VERSION = os.getenv("RISK_MODEL_VERSION", "heuristic_v1")
_MODEL = None
_MODEL_FEATURES = [
    "age","spo2_min_7d","hr_max_7d","hr_mean_7d","steps_mean_7d",
    "alerts_7d","adherence_7d","num_meds_active","dx_count"
]

def _load_risk_model():
    global _MODEL
    if _MODEL is not None: return _MODEL
    if RISK_MODEL_B64:
        try:
            _MODEL = pickle.loads(base64.b64decode(RISK_MODEL_B64))
            logging.info("Risk model loaded from RISK_MODEL_B64")
        except Exception as e:
            logging.warning(f"Failed to load RISK_MODEL_B64, using heuristic: {e}")
            _MODEL = None
    return _MODEL

def _get_patient_features(patient_id: str) -> dict:
    feats = {"age":0,"spo2_min_7d":99,"hr_max_7d":70,"hr_mean_7d":70,
             "steps_mean_7d":0,"alerts_7d":0,"adherence_7d":1.0,
             "num_meds_active":0,"dx_count":0}
    with conn.cursor() as cur:
        cur.execute("SELECT dob FROM patients WHERE id=%s",(patient_id,))
        row = cur.fetchone()
        if row and row[0]:
            dob = row[0]; today = datetime.utcnow().date()
            feats["age"] = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        cur.execute("""
            SELECT COALESCE(MIN(spo2),99), COALESCE(MAX(hr),0),
                   COALESCE(AVG(hr),70), COALESCE(AVG(steps),0)
            FROM vitals WHERE patient_id=%s AND ts > now()-interval '7 days'
        """,(patient_id,))
        v = cur.fetchone() or (99,0,70,0)
        feats["spo2_min_7d"], feats["hr_max_7d"], feats["hr_mean_7d"], feats["steps_mean_7d"] = v
        cur.execute("SELECT COUNT(*) FROM alerts WHERE patient_id=%s AND created_at > now()-interval '7 days'",(patient_id,))
        feats["alerts_7d"] = int(cur.fetchone()[0])
        cur.execute("""
            SELECT COUNT(*) FROM medications
            WHERE patient_id=%s AND start_date<=CURRENT_DATE
              AND (end_date IS NULL OR end_date>=CURRENT_DATE)
        """,(patient_id,))
        feats["num_meds_active"] = int(cur.fetchone()[0])
        cur.execute("""
            SELECT COALESCE(SUM(CASE WHEN taken THEN 1 ELSE 0 END),0),
                   COALESCE(COUNT(*),0)
            FROM med_adherence ma
            JOIN medications m ON m.id=ma.medication_id
            WHERE m.patient_id=%s AND ma.scheduled_time > now()-interval '7 days'
        """,(patient_id,))
        taken_cnt, total_cnt = cur.fetchone()
        feats["adherence_7d"] = 1.0 if total_cnt==0 else float(taken_cnt)/float(total_cnt)
        cur.execute("SELECT COUNT(*) FROM diagnoses WHERE patient_id=%s",(patient_id,))
        feats["dx_count"] = int(cur.fetchone()[0])
    feats["spo2_min_7d"] = max(50,min(100,float(feats["spo2_min_7d"])))
    feats["hr_max_7d"]    = max(30,min(220,float(feats["hr_max_7d"])))
    feats["hr_mean_7d"]   = max(30,min(220,float(feats["hr_mean_7d"])))
    feats["steps_mean_7d"]= max(0,float(feats["steps_mean_7d"]))
    feats["adherence_7d"] = max(0.0,min(1.0,float(feats["adherence_7d"])))
    return feats

def _heuristic_risk(feats: dict) -> tuple[float, dict]:
    w = {}
    w["spo2_min_7d"]    = 0.35 * max(0.0, (92.0 - feats["spo2_min_7d"]) / 10.0)
    w["hr_max_7d"]      = 0.20 * (1.0 if feats["hr_max_7d"] >= 130 else 0.0)
    w["alerts_7d"]      = 0.15 * min(1.0, feats["alerts_7d"] / 3.0)
    w["adherence_7d"]   = 0.15 * (1.0 - feats["adherence_7d"])
    w["num_meds_active"]= 0.10 * min(1.0, feats["num_meds_active"] / 8.0)
    w["age"]            = 0.05 * (1.0 if feats["age"] >= 75 else 0.0)
    score = max(0.0, min(1.0, sum(w.values())))
    return score, w

@app.get("/v1/risk")
def risk_score(patient_id: str, store: bool = True):
    feats = _get_patient_features(patient_id)
    model = _load_risk_model()
    if model is not None:
        x = np.array([[feats.get(k,0.0) for k in _MODEL_FEATURES]], dtype=float)
        try:
            prob = float(model.predict_proba(x)[:,1][0])
        except Exception:
            z = float(model.decision_function(x)[0]); prob = 1.0/(1.0+math.exp(-z))
        score = max(0.0, min(1.0, prob))
        factors = {k: float(feats[k]) for k in _MODEL_FEATURES}
        model_ver = os.getenv("RISK_MODEL_VERSION", "sklearn_v1")
    else:
        score, factors = _heuristic_risk(feats); model_ver = "heuristic_v1"
    bucket = "low" if score < 0.33 else ("medium" if score < 0.66 else "high")
    if store:
        try:
            with conn.cursor() as cur:
                cur.execute("""INSERT INTO risk_scores (patient_id, model_version, score, bucket, factors)
                               VALUES (%s,%s,%s,%s,%s)""",
                            (patient_id, model_ver, score, bucket, json.dumps(factors)))
        except Exception as e:
            logging.warning(f"Failed to store risk score: {e}")
    return {"patient_id": patient_id, "model_version": model_ver,
            "score": round(score,3), "bucket": bucket, "factors": factors}

@app.get("/cron/risk/recompute_all")
def risk_recompute_all(x_admin_key: str | None = Header(default=None)):
    _check_admin_key(x_admin_key)
    processed = 0
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM patients")
        pids = [str(r[0]) for r in cur.fetchall()]
    for pid in pids:
        try:
            risk_score(pid, store=True); processed += 1
        except Exception as e:
            logging.warning(f"Risk recompute failed for {pid}: {e}")
    return {"ok": True, "processed": processed}
