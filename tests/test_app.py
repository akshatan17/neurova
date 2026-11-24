"""
Checks that the server behaves as intended, using a temporary empty database.

    python tests/test_app.py

Each check prints PASS or FAIL. Covers sign-in, who may see what, both models,
bad uploads, clinician review, the admin pages and the audit log.
"""
import io
import json
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["NEUROVA_DATA"] = tempfile.mkdtemp()          # must be set before the app is imported

try:
    from fastapi.testclient import TestClient            # noqa: E402
except RuntimeError as problem:                          # the test client needs one extra package
    sys.exit(f"{problem}\n(Install it into the same environment, for example: .venv\\Scripts\\python -m pip install httpx)")

from app.server import app                               # noqa: E402

web = TestClient(app)
passed, failed = 0, []
PASSWORD = "Neurova-demo-2026"


def check(name, condition):
    global passed
    print("PASS" if condition else "FAIL", name)
    if condition:
        passed += 1
    else:
        failed.append(name)


def sign_in(email, password=PASSWORD):
    reply = web.post("/api/login", json={"email": email, "password": password})
    return {"Authorization": "Bearer " + reply.json()["token"]} if reply.status_code == 200 else reply.status_code


def sample(path):
    return open(os.path.join(ROOT, "samples", path), "rb").read()


akshata, avani, aahana, admin = (sign_in(f"{n}@neurova.demo") for n in ("akshata", "avani", "aahana", "admin"))

# ---------------------------------------------------------------- accounts
check("demo accounts can sign in", all(isinstance(h, dict) for h in (akshata, avani, aahana, admin)))
check("wrong password is refused", sign_in("akshata@neurova.demo", "wrong-password-1") == 401)
check("unknown email is refused the same way", sign_in("nobody@neurova.demo", "wrong-password-1") == 401)
check("no token means no access", web.get("/api/studies").status_code == 401)
check("a tampered token is refused", web.get("/api/me", headers={"Authorization": akshata["Authorization"][:-3] + "abc"}).status_code == 401)
new = {"name": "Test Person", "email": "test@example.com", "password": "long-enough-1"}
check("a new patient can register", web.post("/api/register", json=new).status_code == 200)
check("the same email cannot register twice", web.post("/api/register", json=new).status_code == 409)
check("a short password is refused", web.post("/api/register", json={**new, "email": "a@b.co", "password": "short1"}).status_code == 422)
check("a password without a number is refused", web.post("/api/register", json={**new, "email": "a@b.co", "password": "onlyletters"}).status_code == 422)
check("a malformed email is refused", web.post("/api/register", json={**new, "email": "not-an-email"}).status_code == 422)
check("registering always gives the patient role", web.get("/api/me", headers=sign_in("test@example.com", "long-enough-1")).json()["role"] == "patient")

# ---------------------------------------------------------------- who may see what
mine = web.get("/api/studies", headers=akshata).json()["studies"]
check("the demo patient starts with two results", len(mine) == 2)
study_id = next(s["id"] for s in mine if s["kind"] == "mri")
check("a patient can open their own result", web.get(f"/api/studies/{study_id}", headers=akshata).status_code == 200)
check("another patient gets 'not found'", web.get(f"/api/studies/{study_id}", headers=avani).status_code == 404)
check("another patient cannot delete it", web.delete(f"/api/studies/{study_id}", headers=avani).status_code == 404)
check("an admin cannot open results", web.get(f"/api/studies/{study_id}", headers=admin).status_code == 403)
shared = [p["name"] for p in web.get("/api/patients", headers=aahana).json()]
check("the clinician sees the patients who share, and only them", "Akshata Nagaraj" in shared and "Avani Kurup" not in shared and len(shared) == 4)
check("the clinician can open that patient's result", web.get(f"/api/studies/{study_id}", headers=aahana).status_code == 200)
akshata_id = web.get("/api/me", headers=akshata).json()["id"]
check("the clinician can list that patient's results", len(web.get(f"/api/studies?patient_id={akshata_id}", headers=aahana).json()["studies"]) == 2)
avani_id = web.get("/api/me", headers=avani).json()["id"]
check("the clinician cannot list a patient who does not share", web.get(f"/api/studies?patient_id={avani_id}", headers=aahana).status_code == 404)
check("the clinician cannot upload for a patient who does not share",
      web.post("/api/studies/sample", headers=aahana, json={"file": "mri/glioma_2.jpg", "patient_id": avani_id}).status_code == 404)
aahana_id = web.get("/api/me", headers=aahana).json()["id"]
web.delete(f"/api/share/{aahana_id}", headers=akshata)
check("after sharing stops, the clinician is locked out at once", web.get(f"/api/studies/{study_id}", headers=aahana).status_code == 404)
web.post(f"/api/share/{aahana_id}", headers=akshata)
check("sharing again restores access", web.get(f"/api/studies/{study_id}", headers=aahana).status_code == 200)
check("a patient cannot share with someone who is not a clinician", web.post(f"/api/share/{avani_id}", headers=akshata).status_code == 404)
check("a clinician cannot use the patient sharing page", web.get("/api/clinicians", headers=aahana).status_code == 403)

# ---------------------------------------------------------------- the MRI model
correct, familiar_count, seconds = 0, 0, []
names = sorted(n for n in os.listdir(os.path.join(ROOT, "samples", "mri")) if not n.startswith("hard_"))
for name in names:
    start = time.perf_counter()
    reply = web.post("/api/studies", headers=avani, files={"file": (name, sample("mri/" + name), "image/jpeg")})
    seconds.append(time.perf_counter() - start)
    result = web.get(f"/api/studies/{reply.json()['id']}", headers=avani).json()["result"]
    correct += result["prediction"] == name.split("_")[0]
    familiar_count += result["familiarity"]["familiar"]
check(f"all {len(names)} sample scans get the right class ({correct}/{len(names)})", correct == len(names))
check("an MRI result has a heatmap, uncertainty and seven stability tests",
      len(result["heatmap"]) > 1000 and result["uncertainty"]["level"] in ("Low", "Moderate", "High") and len(result["stability"]["tests"]) == 7)
check("class probabilities add up to 100", abs(sum(p["percent"] for p in result["probabilities"]) - 100) < 0.5)

# Two scans the model is known to get wrong are kept as samples. What matters is how the app presents them.
def hard_case(name):
    made = web.post("/api/studies/sample", headers=avani, json={"file": "mri/" + name})
    return web.get(f"/api/studies/{made.json()['id']}", headers=avani).json()["result"]


missed = hard_case("hard_glioma_1.jpg")                  # a glioma the model calls "no tumour"
check("the missed tumour is wrong, as recorded in the test results", missed["prediction"] == "notumor")
check("but its uncertainty is high", missed["uncertainty"]["level"] == "High")
check("and its answer does not survive the stability check", not missed["stability"]["held"])
check("so it is flagged for review even though the answer is 'no tumour'", missed["needs_review"])
check("and the summary names the runner-up class", "second most likely" in missed["summary"])
confident = hard_case("hard_pituitary_2.jpg")            # a pituitary tumour called meningioma at 99%
check("the confident mistake looks certain to the uncertainty check",
      confident["prediction"] == "meningioma" and confident["uncertainty"]["level"] == "Low" and confident["needs_review"])
check("but the familiarity check flags it as unlike the training scans", not confident["familiarity"]["familiar"])
check("the missed tumour is flagged as unfamiliar too", not missed["familiarity"]["familiar"])
check("all 20 ordinary samples count as familiar", familiar_count == len(names))
chart = web.post("/api/studies", headers=avani, files={"file": ("chart.png", open(os.path.join(ROOT, "results", "mri", "robustness.png"), "rb").read(), "image/png")})
chart = web.get(f"/api/studies/{chart.json()['id']}", headers=avani).json()["result"]
check("a picture that is not an MRI is flagged as unfamiliar and sent for review",
      not chart["familiarity"]["familiar"] and chart["needs_review"] and "unlike the scans" in chart["summary"])
listed = web.get("/api/samples", headers=avani).json()
check("the sample list has 20 ordinary scans, 2 hard ones and a sleep recording",
      [sum(s["kind"] == "mri" and not s["hard"] for s in listed), sum(s["hard"] for s in listed), sum(s["kind"] == "sleep" for s in listed)] == [20, 2, 1])
check("every class has five ordinary samples", all(sum(n.startswith(c) for n in names) == 5 for c in ("glioma", "meningioma", "notumor", "pituitary")))

again = web.post("/api/studies", headers=avani, files={"file": (names[0], sample("mri/" + names[0]), "image/jpeg")})
first = web.get(f"/api/studies/{again.json()['id']}", headers=avani).json()["result"]
again = web.post("/api/studies", headers=avani, files={"file": (names[0], sample("mri/" + names[0]), "image/jpeg")})
second = web.get(f"/api/studies/{again.json()['id']}", headers=avani).json()["result"]
check("the same scan always gives the same result", first == second)

# ---------------------------------------------------------------- the sleep model
reply = web.post("/api/studies/sample", headers=avani, json={"file": "sleep/night.edf"})
night = web.get(f"/api/studies/{reply.json()['id']}", headers=avani).json()["result"]
check("the sample night is staged, one stage per 30 seconds", night["epochs"] == 240 and len(night["stages"]) == 240)
check("agreement with the expert is above 80%", night["agreement"]["percent"] > 80)
check("stage minutes add up to the recording length", abs(sum(night["minutes"].values()) - night["night"]["recorded_minutes"]) < 0.01)
check("an uploaded EDF without expert scoring also works",
      web.post("/api/studies", headers=avani, files={"file": ("mine.edf", sample("sleep/night.edf"), "application/octet-stream")}).status_code == 200)

# ---------------------------------------------------------------- bad uploads
def upload(name, content):
    return web.post("/api/studies", headers=avani, files={"file": (name, io.BytesIO(content), "application/octet-stream")}).status_code


check("a text file is refused", upload("notes.txt", b"hello") == 422)
check("a text file renamed to .png is refused", upload("fake.png", b"this is not an image") == 422)
check("a broken EDF is refused", upload("broken.edf", b"0" + b" " * 300) == 422)
check("an EDF shorter than five minutes is refused", upload("short.edf", sample("sleep/night.edf")[:512 + 6000 * 5]) == 422)
check("an unknown sample name is refused", web.post("/api/studies/sample", headers=avani, json={"file": "../app/auth.py"}).status_code == 404)

# ---------------------------------------------------------------- clinician review
check("a patient cannot review", web.post(f"/api/studies/{study_id}/review", headers=akshata, json={"verdict": "agree"}).status_code == 403)
check("a review must be agree or disagree", web.post(f"/api/studies/{study_id}/review", headers=aahana, json={"verdict": "maybe"}).status_code == 422)
web.post(f"/api/studies/{study_id}/review", headers=aahana, json={"verdict": "agree", "note": "Looks right."})
seen = web.get(f"/api/studies/{study_id}", headers=akshata).json()["review"]
check("the patient sees the clinician's review", seen["verdict"] == "agree" and seen["by"] == "Dr. Aahana Sharma" and seen["note"] == "Looks right.")
check("a clinician cannot delete a patient's result", web.delete(f"/api/studies/{study_id}", headers=aahana).status_code == 403)
last = web.get("/api/studies", headers=avani).json()["studies"][0]["id"]
check("a patient can delete their own result", web.delete(f"/api/studies/{last}", headers=avani).status_code == 200
      and web.get(f"/api/studies/{last}", headers=avani).status_code == 404)

# ---------------------------------------------------------------- triage and PDF reports
import re                                                 # noqa: E402

GROUP_RANK = {"Likely tumour": 1, "Model unsure": 2, "Routine": 3}
check("a patient cannot open the triage list", web.get("/api/triage", headers=akshata).status_code == 403)
queue = web.get("/api/triage", headers=aahana).json()
ranks = [GROUP_RANK[r["group"]] for r in queue]
check("the triage list puts likely tumours first, then unsure results, then routine ones", ranks == sorted(ranks) and set(ranks) == {1, 2, 3})
check("a reviewed result leaves the triage list", study_id not in [r["id"] for r in queue] and all(r["review"] is None for r in queue))
check("the triage list never includes a patient who does not share", all(r["patient_name"] != "Avani Kurup" for r in queue))
check("the missed tumour is listed as 'Model unsure', with its reasons",
      any(r["group"] == "Model unsure" and "Unfamiliar scan" in r["reasons"] and "No tumour" in r["headline"] for r in queue))
tumours = [r for r in queue if r["group"] == "Likely tumour"]
check("every 'Likely tumour' entry names the tumour it found", all(r["reasons"][0].startswith("Likely ") for r in tumours))

pages = lambda pdf: len(re.findall(rb"/Type\s*/Page[^s]", pdf))
pdf = web.get(f"/api/studies/{study_id}/report.pdf", headers=akshata)
check("a patient can download a PDF of their result", pdf.status_code == 200 and pdf.content[:5] == b"%PDF-"
      and pdf.headers["content-type"] == "application/pdf")
check("the MRI PDF report fits on one page", pages(pdf.content) == 1)
sleep_id = next(s["id"] for s in web.get("/api/studies", headers=akshata).json()["studies"] if s["kind"] == "sleep")
check("the sleep PDF report fits on one page", pages(web.get(f"/api/studies/{sleep_id}/report.pdf", headers=akshata).content) == 1)
check("the clinician can download the PDF of a shared result", web.get(f"/api/studies/{study_id}/report.pdf", headers=aahana).status_code == 200)
check("another patient gets 'not found' for the PDF", web.get(f"/api/studies/{study_id}/report.pdf", headers=avani).status_code == 404)
check("an admin cannot download a PDF report", web.get(f"/api/studies/{study_id}/report.pdf", headers=admin).status_code == 403)

# ---------------------------------------------------------------- admin
check("a patient cannot open the admin pages", web.get("/api/admin/overview", headers=akshata).status_code == 403)
check("a clinician cannot read the audit log", web.get("/api/admin/audit", headers=aahana).status_code == 403)
overview = web.get("/api/admin/overview", headers=admin).json()
check("the admin sees the counts", overview["counts"]["clinicians"] == 1 and overview["counts"]["reviewed"] == 2)
doctor = {"name": "Dr. Vikram Rao", "email": "rao@example.com", "password": "temporary-pass-1"}
check("the admin can add a clinician", web.post("/api/admin/clinicians", headers=admin, json=doctor).json().get("role") == "clinician")
check("a patient cannot add a clinician", web.post("/api/admin/clinicians", headers=akshata, json={**doctor, "email": "x@example.com"}).status_code == 403)
actions = {row["action"] for row in web.get("/api/admin/audit", headers=admin).json()}
check("the sign-in page offers one demo account per role",
      sorted(a["role"] for a in web.get("/api/demo").json()["accounts"]) == ["admin", "clinician", "patient"])
check("a sample cannot be read by path tricks", web.get("/samples/../app/auth.py").status_code == 404)
check("a patient cannot review their own result as 'agree'", web.post(f"/api/studies/{study_id}/review", headers=avani, json={"verdict": "agree"}).status_code == 403)
check("a review of a result that is not shared is 'not found'",
      web.post(f"/api/studies/{web.get('/api/studies', headers=avani).json()['studies'][0]['id']}/review", headers=aahana, json={"verdict": "agree"}).status_code == 404)
check("an empty upload is refused", upload("empty.png", b"") == 422)
check("a file with no extension is refused", upload("scan", sample("mri/" + names[0])) == 422)
check("the audit log recorded the important events",
      {"signed in", "failed sign-in", "analysed a study", "reviewed a study", "stopped sharing", "shared results", "deleted a study",
       "opened a patient's results", "created a clinician account", "opened the triage list", "downloaded a PDF report"} <= actions)
check("passwords are stored hashed, never in plain text", PASSWORD.encode() not in open(os.path.join(os.environ["NEUROVA_DATA"], "neurova.db"), "rb").read())

# ---------------------------------------------------------------- the admin edits accounts
avani_id = web.get("/api/me", headers=avani).json()["id"]
admin_id = web.get("/api/me", headers=admin).json()["id"]
edit = lambda who, changes, as_=admin: web.patch(f"/api/admin/users/{who}", headers=as_, json=changes)
check("a patient cannot edit accounts", edit(avani_id, {"name": "Someone Else"}, akshata).status_code == 403)
check("a clinician cannot edit accounts", edit(avani_id, {"name": "Someone Else"}, aahana).status_code == 403)
check("editing an account that does not exist is 'not found'", edit(99999, {"name": "Nobody Here"}).status_code == 404)
check("the admin can rename an account", edit(avani_id, {"name": "Avani K. Kurup"}).json()["name"] == "Avani K. Kurup"
      and web.get("/api/me", headers=avani).json()["name"] == "Avani K. Kurup")
check("a one-letter name is refused", edit(avani_id, {"name": "A"}).status_code == 422)
check("a reset password must follow the password rule", edit(avani_id, {"password": "onlyletters"}).status_code == 422)
check("the admin can reset a password", edit(avani_id, {"password": "Fresh-pass-2026"}).status_code == 200)
check("the reset signs the person out everywhere", web.get("/api/me", headers=avani).status_code == 401)
check("the old password no longer works", sign_in("avani@neurova.demo") == 401)
avani = sign_in("avani@neurova.demo", "Fresh-pass-2026")
check("the new password works", isinstance(avani, dict))
check("the admin can disable an account", edit(avani_id, {"active": False}).json()["active"] is False)
check("a disabled account is signed out at once", web.get("/api/me", headers=avani).status_code == 401)
check("a disabled account cannot sign in, and is told why", sign_in("avani@neurova.demo", "Fresh-pass-2026") == 403)
check("the admin cannot disable their own account", edit(admin_id, {"active": False}).status_code == 422)
aahana_id = web.get("/api/me", headers=aahana).json()["id"]
edit(aahana_id, {"active": False})
check("a disabled clinician is not offered on the sharing page", all(c["id"] != aahana_id for c in web.get("/api/clinicians", headers=akshata).json()))
check("a disabled demo account is not offered on the sign-in page", "aahana@neurova.demo" not in [a["email"] for a in web.get("/api/demo").json()["accounts"]])
edit(aahana_id, {"active": True})
edit(avani_id, {"active": True})
check("an enabled account can sign in again", isinstance(sign_in("avani@neurova.demo", "Fresh-pass-2026"), dict))
done = {row["action"] for row in web.get("/api/admin/audit", headers=admin).json()}
check("every account change is in the activity log", {"renamed an account", "reset a password", "disabled an account",
                                                        "enabled an account", "refused sign-in"} <= done)

# ---------------------------------------------------------------- model facts and lockout
facts = web.get("/api/models", headers=akshata).json()
check("the model facts page has the three-model comparison", len(facts["mri"]["compare"]) == 3 and sum(m["deployed"] for m in facts["mri"]["compare"]) == 1)
check("the figures are served", web.get("/results/mri/robustness.png").status_code == 200)
for _ in range(5):
    sign_in("avani@neurova.demo", "wrong-password-1")
check("five wrong passwords lock that email for a while", sign_in("avani@neurova.demo") == 429)

# ---------------------------------------------------------------- speed of one forward pass
import cv2                                              # noqa: E402
import numpy as np                                      # noqa: E402

from app import mri                                     # noqa: E402
from app.preprocess import prepare_mri                  # noqa: E402

x, _ = prepare_mri(cv2.imdecode(np.frombuffer(sample("mri/" + names[0]), np.uint8), cv2.IMREAD_COLOR))
times = []
for _ in range(40):
    start = time.perf_counter()
    mri.run_network(x)
    times.append((time.perf_counter() - start) * 1000)
forward_ms = sorted(times)[len(times) // 2]

seconds.sort()
print(f"\n{passed} passed, {len(failed)} failed. One MRI forward pass: {forward_ms:.0f} ms. One full MRI analysis: {seconds[len(seconds) // 2]:.2f} s")
summary = os.path.join(ROOT, "report", "source", "test_results.json")
if os.path.isdir(os.path.dirname(summary)):
    old = json.load(open(summary)) if os.path.exists(summary) else {}
    json.dump({**old, "api": passed, "api_failed": len(failed), "mri_request_s": round(seconds[len(seconds) // 2], 2), "forward_ms": round(forward_ms, 1),
               "demo_correct": f"{correct}/{len(names)}"}, open(summary, "w"))
sys.exit(1 if failed else 0)
