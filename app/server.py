"""
The web server. Every address the browser can call is defined in this file.

    uvicorn app.server:app --port 8000

Sections:  accounts  ·  sharing  ·  triage  ·  studies (the analyses)  ·  PDF reports  ·  admin  ·  model facts
"""
import json
import os

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import mri, sleep
from .auth import (can_see_patient, check_password, current_user, get_db, hash_password, make_token, record_failure,
                   require, too_many_attempts)
from .database import Access, Audit, Study, User, log, now
from .demo import DEMO_PASSWORD, add_demo_data
from .report_pdf import build_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "samples")
MAX_UPLOAD = 80 * 1024 * 1024                   # 80 MB: enough for a full overnight EDF

app = FastAPI(title="Neurova", docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")
add_demo_data()


# ================================================================== accounts
class Register(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", max_length=190)
    password: str = Field(min_length=10, max_length=100)


class Login(BaseModel):
    email: str
    password: str


def public(user: User) -> dict:
    return {"id": user.id, "name": user.name, "email": user.email, "role": user.role, "active": user.active}


def check_password_rule(password: str):
    if not (any(c.isalpha() for c in password) and any(c.isdigit() for c in password)):
        raise HTTPException(422, "The password needs at least one letter and one number.")


def create_user(db, data: Register, role: str) -> User:
    check_password_rule(data.password)
    if db.query(User).filter_by(email=data.email.lower()).first():
        raise HTTPException(409, "An account with this email already exists.")
    user = User(name=data.name.strip(), email=data.email.lower(), password_hash=hash_password(data.password), role=role)
    db.add(user)
    db.flush()
    return user


@app.post("/api/register")
def register(data: Register, db=Depends(get_db)):
    """Anyone can create a patient account. Clinician accounts are created by an admin."""
    user = create_user(db, data, "patient")
    log(db, user, "registered")
    db.commit()
    return {"token": make_token(user), "user": public(user)}


@app.post("/api/login")
def login(data: Login, db=Depends(get_db)):
    email = data.email.strip().lower()
    if too_many_attempts(email):
        raise HTTPException(429, "Too many wrong passwords. Try again in ten minutes.")
    user = db.query(User).filter_by(email=email).first()
    if not user or not check_password(data.password, user.password_hash):
        record_failure(email)
        log(db, user, "failed sign-in", email)
        db.commit()
        raise HTTPException(401, "Wrong email or password.")
    if not user.active:                                  # said only after a correct password, so it reveals nothing
        log(db, user, "refused sign-in", "account disabled")
        db.commit()
        raise HTTPException(403, "This account has been disabled. Please contact the administrator.")
    log(db, user, "signed in")
    db.commit()
    return {"token": make_token(user), "user": public(user)}


@app.get("/api/me")
def me(user: User = Depends(current_user)):
    return public(user)


@app.get("/api/demo")
def demo_accounts(db=Depends(get_db)):
    """The made-up accounts shown on the sign-in page."""
    users = db.query(User).filter(User.email.like("%@neurova.demo"), User.active.is_(True)).order_by(User.id).all()
    first_of_each_role = {u.role: u for u in reversed(users)}.values()      # one button per role is enough
    return {"password": DEMO_PASSWORD, "accounts": sorted((public(u) for u in first_of_each_role), key=lambda u: u["id"])}


# ================================================================== sharing
@app.get("/api/clinicians")
def clinicians(user: User = Depends(require("patient")), db=Depends(get_db)):
    """For a patient: every clinician, and whether I am sharing with them."""
    mine = {a.clinician_id for a in db.query(Access).filter_by(patient_id=user.id)}
    return [{"id": c.id, "name": c.name, "sharing": c.id in mine}
            for c in db.query(User).filter_by(role="clinician", active=True).order_by(User.name)]


@app.post("/api/share/{clinician_id}")
def share(clinician_id: int, user: User = Depends(require("patient")), db=Depends(get_db)):
    clinician = db.get(User, clinician_id)
    if not clinician or clinician.role != "clinician":
        raise HTTPException(404, "Clinician not found.")
    if not db.query(Access).filter_by(patient_id=user.id, clinician_id=clinician_id).first():
        db.add(Access(patient_id=user.id, clinician_id=clinician_id))
        log(db, user, "shared results", f"with {clinician.name}")
        db.commit()
    return {"sharing": True}


@app.delete("/api/share/{clinician_id}")
def stop_sharing(clinician_id: int, user: User = Depends(require("patient")), db=Depends(get_db)):
    row = db.query(Access).filter_by(patient_id=user.id, clinician_id=clinician_id).first()
    if row:
        clinician = db.get(User, clinician_id)
        db.delete(row)
        log(db, user, "stopped sharing", f"with {clinician.name if clinician else clinician_id}")
        db.commit()
    return {"sharing": False}


@app.get("/api/patients")
def patients(user: User = Depends(require("clinician")), db=Depends(get_db)):
    """For a clinician: the patients who share with me, and how many results await my review."""
    out = []
    for a in db.query(Access).filter_by(clinician_id=user.id):
        p = db.get(User, a.patient_id)
        studies = db.query(Study).filter_by(patient_id=p.id)
        out.append({"id": p.id, "name": p.name, "studies": studies.count(),
                    "waiting": studies.filter(Study.review_verdict.is_(None)).count()})
    return sorted(out, key=lambda p: (-p["waiting"], p["name"]))


# ================================================================== triage
def triage_priority(result: dict) -> tuple[int, str, list[str]]:
    """Sort a result into one of three groups, and say why.

      1 "Likely tumour"  the MRI model's answer is a tumour
      2 "Model unsure"   any check failed: high or moderate uncertainty, a close call,
                         an unstable answer or an unfamiliar scan
      3 "Routine"        a confident 'no tumour', or a sleep recording
    """
    if result["kind"] == "sleep":
        hours = result["night"]["recorded_minutes"] / 60
        return 3, "Routine", [f"Sleep recording, {hours:.1f} h"]
    reasons = []
    if result["is_tumour"]:
        reasons.append(f"Likely {result['label'].lower()} ({result['confidence_text']})")
    if result["uncertainty"]["level"] != "Low":
        reasons.append(f"{result['uncertainty']['level']} uncertainty")
    if not result["stability"]["held"]:
        reasons.append("Not stable under disturbance")
    if result.get("familiarity") and not result["familiarity"]["familiar"]:
        reasons.append("Unfamiliar scan")
    p = result["probabilities"]
    if len(p) > 1 and p[0]["percent"] - p[1]["percent"] < 30:
        reasons.append(f"Close call with {p[1]['label'].lower()}")
    if result["is_tumour"]:
        return 1, "Likely tumour", reasons
    if reasons:
        return 2, "Model unsure", reasons
    return 3, "Routine", [f"No tumour ({result['confidence_text']})"]


@app.get("/api/triage")
def triage(user: User = Depends(require("clinician")), db=Depends(get_db)):
    """For a clinician: every result waiting for review, across all patients who share
    with them, most urgent first. This is the reading list the product is built around."""
    patient_ids = [a.patient_id for a in db.query(Access).filter_by(clinician_id=user.id)]
    waiting = db.query(Study).filter(Study.patient_id.in_(patient_ids), Study.review_verdict.is_(None)).all() if patient_ids else []
    rows = []
    for s in waiting:
        result = json.loads(s.result_json)
        rank, group, reasons = triage_priority(result)
        urgency = result["confidence"] if result["kind"] == "mri" and result["is_tumour"] else 0
        rows.append({**card(db, s), "patient_name": db.get(User, s.patient_id).name, "group": group, "rank": rank,
                     "reasons": reasons, "urgency": urgency})
    # most urgent group first; within a group, the most confident tumour first, then the longest waiting
    rows.sort(key=lambda r: (r["rank"], -r["urgency"], r["created"]))
    for r in rows:
        del r["rank"], r["urgency"]
    log(db, user, "opened the triage list", f"{len(rows)} waiting")
    db.commit()
    return rows


# ================================================================== studies
def card(db, s: Study) -> dict:
    """The short form of a study, for lists."""
    reviewer = db.get(User, s.reviewed_by) if s.reviewed_by else None
    return {"id": s.id, "kind": s.kind, "title": s.title, "headline": s.headline, "needs_review": s.needs_review,
            "created": s.created.isoformat() + "Z", "patient_id": s.patient_id,
            "review": {"verdict": s.review_verdict, "note": s.review_note, "by": reviewer.name,
                       "when": s.reviewed.isoformat() + "Z"} if reviewer else None}


def find_study(db, user: User, study_id: int) -> Study:
    """Fetch a study, but answer 'not found' if this user may not see it.
    (Saying 'forbidden' would confirm that the study exists.)"""
    s = db.get(Study, study_id)
    if not s or not can_see_patient(db, user, s.patient_id):
        raise HTTPException(404, "Study not found.")
    return s


def analyse_and_save(db, user: User, patient_id: int, filename: str, content: bytes, expert: dict | None = None) -> Study:
    if not can_see_patient(db, user, patient_id):
        raise HTTPException(404, "Patient not found.")
    ext = os.path.splitext(filename)[1].lower()
    try:
        if ext in (".png", ".jpg", ".jpeg"):
            result = mri.analyze(content)
        elif ext == ".edf":
            result = sleep.analyze(content, expert)
        else:
            raise ValueError("Upload a brain MRI image (PNG or JPG) or a sleep EEG recording (EDF).")
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    title = ("Brain MRI" if result["kind"] == "mri" else "Sleep EEG") + f" · {os.path.basename(filename)[:80]}"
    s = Study(patient_id=patient_id, uploaded_by=user.id, kind=result["kind"], title=title, headline=result["headline"],
              needs_review=result["needs_review"], result_json=json.dumps(result))
    db.add(s)
    db.flush()
    log(db, user, "analysed a study", f"#{s.id} {title}")
    db.commit()
    return s


@app.get("/api/studies")
def list_studies(patient_id: int | None = None, user: User = Depends(require("patient", "clinician")), db=Depends(get_db)):
    patient_id = user.id if user.role == "patient" else patient_id
    if patient_id is None or not can_see_patient(db, user, patient_id):
        raise HTTPException(404, "Patient not found.")
    if user.role == "clinician":
        log(db, user, "opened a patient's results", db.get(User, patient_id).name)
        db.commit()
    rows = db.query(Study).filter_by(patient_id=patient_id).order_by(Study.created.desc(), Study.id.desc())
    return {"patient": public(db.get(User, patient_id)), "studies": [card(db, s) for s in rows]}


@app.post("/api/studies")
async def upload_study(file: UploadFile = File(...), patient_id: int | None = Form(None),
                       user: User = Depends(require("patient", "clinician")), db=Depends(get_db)):
    content = await file.read(MAX_UPLOAD + 1)
    if len(content) > MAX_UPLOAD:
        raise HTTPException(413, "The file is larger than 80 MB.")
    s = analyse_and_save(db, user, user.id if user.role == "patient" else patient_id, file.filename or "upload", content)
    return {"id": s.id}


@app.get("/api/samples")
def samples(user: User = Depends(current_user)):
    """Example files that ship with the app, so the demo works without your own data."""
    files = sorted(os.listdir(os.path.join(SAMPLES, "mri")))
    easy = [f for f in files if not f.startswith("hard_")]          # scans the model gets right
    hard = [f for f in files if f.startswith("hard_")]              # scans it gets wrong, kept to show its limits
    return ([{"file": f"mri/{f}", "kind": "mri", "hard": False, "name": f"Brain MRI {i + 1}"} for i, f in enumerate(easy)]
            + [{"file": f"mri/{f}", "kind": "mri", "hard": True, "name": "A scan the model gets wrong"} for f in hard]
            + [{"file": "sleep/night.edf", "kind": "sleep", "hard": False, "name": "Sleep EEG, two hours of one night"}])


class UseSample(BaseModel):
    file: str
    patient_id: int | None = None


@app.post("/api/studies/sample")
def analyse_sample(data: UseSample, user: User = Depends(require("patient", "clinician")), db=Depends(get_db)):
    if data.file not in {s["file"] for s in samples(user)}:
        raise HTTPException(404, "Sample not found.")
    path = os.path.join(SAMPLES, data.file)
    expert_path = path.replace(".edf", ".expert.json")
    expert = json.load(open(expert_path)) if path.endswith(".edf") and os.path.exists(expert_path) else None
    name = "held-out test image.jpg" if data.file.startswith("mri/") else "Sleep-EDF held-out night.edf"
    s = analyse_and_save(db, user, user.id if user.role == "patient" else data.patient_id, name, open(path, "rb").read(), expert)
    return {"id": s.id}


@app.get("/api/studies/{study_id}")
def get_study(study_id: int, user: User = Depends(require("patient", "clinician")), db=Depends(get_db)):
    s = find_study(db, user, study_id)
    return {**card(db, s), "patient": public(db.get(User, s.patient_id)), "result": json.loads(s.result_json)}


# ================================================================== PDF reports
def facts_for_reports() -> dict:
    m = json.load(open(os.path.join(ROOT, "results", "mri", "metrics.json")))
    s = json.load(open(os.path.join(ROOT, "results", "sleep", "metrics.json")))["results"]
    return {"mri_accuracy": m["results"][m["chosen"]]["test_accuracy"], "mri_test_images": sum(m["splits"]["test"].values()),
            "sleep_accuracy": s["test_accuracy"], "sleep_test_people": len(s["subjects"]["test"])}


@app.get("/api/studies/{study_id}/report.pdf")
def study_pdf(study_id: int, user: User = Depends(require("patient", "clinician")), db=Depends(get_db)):
    """The result as a printable PDF. Same access rule as viewing the result."""
    s = find_study(db, user, study_id)
    pdf = build_report(card(db, s), db.get(User, s.patient_id).name, json.loads(s.result_json), facts_for_reports())
    log(db, user, "downloaded a PDF report", f"#{s.id} {s.title}")
    db.commit()
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="neurova-report-{s.id}.pdf"'})


@app.delete("/api/studies/{study_id}")
def delete_study(study_id: int, user: User = Depends(require("patient")), db=Depends(get_db)):
    s = find_study(db, user, study_id)
    log(db, user, "deleted a study", f"#{s.id} {s.title}")
    db.delete(s)
    db.commit()
    return {"deleted": True}


class Review(BaseModel):
    verdict: str = Field(pattern="^(agree|disagree)$")
    note: str = Field(default="", max_length=2000)


@app.post("/api/studies/{study_id}/review")
def review_study(study_id: int, data: Review, user: User = Depends(require("clinician")), db=Depends(get_db)):
    """A clinician records whether they agree with the model. The patient sees this note."""
    s = find_study(db, user, study_id)
    s.review_verdict, s.review_note, s.reviewed_by, s.reviewed = data.verdict, data.note.strip(), user.id, now()
    log(db, user, "reviewed a study", f"#{s.id}: {data.verdict}")
    db.commit()
    return card(db, s)


# ================================================================== admin
@app.get("/api/admin/overview")
def admin_overview(user: User = Depends(require("admin")), db=Depends(get_db)):
    """Counts and accounts. An admin manages the system but cannot open anyone's results."""
    reviewed = db.query(Study).filter(Study.review_verdict.isnot(None))
    return {"counts": {"patients": db.query(User).filter_by(role="patient").count(),
                       "clinicians": db.query(User).filter_by(role="clinician").count(),
                       "studies": db.query(Study).count(), "reviewed": reviewed.count(),
                       "clinician_agreed": reviewed.filter_by(review_verdict="agree").count()},
            "users": [{**public(u), "joined": u.created.isoformat() + "Z"} for u in db.query(User).order_by(User.role, User.name)]}


@app.post("/api/admin/clinicians")
def add_clinician(data: Register, user: User = Depends(require("admin")), db=Depends(get_db)):
    new = create_user(db, data, "clinician")
    log(db, user, "created a clinician account", new.name)
    db.commit()
    return public(new)


class EditUser(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=120)
    password: str | None = Field(default=None, min_length=10, max_length=100)
    active: bool | None = None


@app.patch("/api/admin/users/{user_id}")
def edit_user(user_id: int, data: EditUser, user: User = Depends(require("admin")), db=Depends(get_db)):
    """Rename an account, reset its password, or disable / re-enable it.

    Disabling is used instead of deleting, so the person's results and the activity log stay
    complete. A reset password or a disabled account signs that person out everywhere at once.
    """
    target = db.get(User, user_id)
    if not target:
        raise HTTPException(404, "Account not found.")
    if data.active is False and target.id == user.id:
        raise HTTPException(422, "You cannot disable your own account.")
    if data.name is not None and data.name.strip() != target.name:
        log(db, user, "renamed an account", f"{target.name} → {data.name.strip()}")
        target.name = data.name.strip()
    if data.password is not None:
        check_password_rule(data.password)
        target.password_hash = hash_password(data.password)
        log(db, user, "reset a password", target.name)
    if data.active is not None and data.active != target.active:
        target.active = data.active
        log(db, user, "enabled an account" if data.active else "disabled an account", target.name)
    db.commit()
    return public(target)


@app.get("/api/admin/audit")
def audit_log(user: User = Depends(require("admin")), db=Depends(get_db)):
    rows = db.query(Audit).order_by(Audit.id.desc()).limit(200)
    return [{"time": a.time.isoformat() + "Z", "who": a.user_name, "role": a.role, "action": a.action, "detail": a.detail} for a in rows]


# ================================================================== model facts
@app.get("/api/models")
def model_facts(user: User = Depends(current_user)):
    """Test-set results for the 'How the models perform' page, read from results/."""
    m = json.load(open(os.path.join(ROOT, "results", "mri", "metrics.json")))
    s = json.load(open(os.path.join(ROOT, "results", "sleep", "metrics.json")))["results"]
    names = {"neurova_cnn": "Small CNN trained from scratch", "efficientnet_b0_tf": "EfficientNet-B0, standard fine-tuning",
             "efficientnet_b0_tf_robust": "EfficientNet-B0, fingerprint-scrub training"}
    compare = [{"name": names[k], "deployed": k == m["chosen"], "accuracy": r["test_accuracy"], "macro_f1": r["test_macro_f1"],
                "parameters_m": r["params_m"], "changed_by_faint_noise": r["robustness"]["noise_0.05"]}
               for k, r in m["results"].items() if k in names]
    best = m["results"][m["chosen"]]
    return {
        "mri": {"compare": sorted(compare, key=lambda r: r["accuracy"]), "test_images": sum(m["splits"]["test"].values()),
                "auc": best["test_macro_auc"], "ece_before": best["ece_before"], "ece_after": best["ece_after"],
                "per_class": [{"label": mri.LABELS[mri.CLASSES.index(k)], **v} for k, v in best["per_class"].items()],
                "duplicates_removed": m["dedup"]["images_total"] - m["dedup"]["unique_images"], "images": m["dedup"]["images_total"],
                "familiarity": json.load(open(os.path.join(ROOT, "results", "mri", "familiarity.json"))),
                "figures": ["confusion_matrices", "robustness", "familiarity", "calibration", "examples_correct", "examples_incorrect"]},
        "sleep": {"accuracy": s["test_accuracy"], "macro_f1": s["test_macro_f1"], "kappa": s["test_kappa"],
                  "test_people": len(s["subjects"]["test"]), "train_people": len(s["subjects"]["train"]),
                  "per_class": [{"label": sleep.STAGE_NAMES[k], **v} for k, v in s["per_class"].items()],
                  "compare": json.load(open(os.path.join(ROOT, "results", "sleep", "comparison.json")))["models"],
                  "figures": ["confusion_matrix", "hypnogram_test_subject"]},
    }


@app.get("/api/health")
def health():
    return {"ok": True}


# ================================================================== the web pages
app.mount("/samples", StaticFiles(directory=SAMPLES), name="samples")
app.mount("/results", StaticFiles(directory=os.path.join(ROOT, "results")), name="results")
app.mount("/", StaticFiles(directory=os.path.join(ROOT, "web"), html=True), name="web")
