"""
Made-up accounts and a few finished analyses, so the app has something to show on first run.
Everything here is fictional. Set NEUROVA_DEMO=0 to start with an empty database.
"""
import json
import os
from datetime import timedelta

from .auth import hash_password
from .database import Access, Session, Study, User, now

DEMO_PASSWORD = "Neurova-demo-2026"

# To rename the demo people, change only the names in the first column, then delete the data/
# folder and start the app again. Leave the emails exactly as they are: the code below and the
# tests look people up by email, so a changed or mistyped email stops the app from starting.
PEOPLE = [
    ("Akshata Nagaraj",     "akshata@neurova.demo", "patient"),      # has two results and shares with the clinician
    ("Avani Kurup",         "avani@neurova.demo",   "patient"),      # starts empty and shares with nobody
    ("Dr. Aahana Sharma",   "aahana@neurova.demo",  "clinician"),
    ("Site Administrator",  "admin@neurova.demo",   "admin"),
    ("Ishaan Mehta",        "ishaan@neurova.demo",  "patient"),      # these three fill the clinician's triage list
    ("Fatima Khan",         "fatima@neurova.demo",  "patient"),
    ("Joseph Thomas",       "joseph@neurova.demo",  "patient"),
]

# Which sample file each demo patient has had analysed, and how many hours ago.
RESULTS = [
    ("akshata@neurova.demo", "mri/glioma_1.jpg", 30),
    ("akshata@neurova.demo", "sleep/night.edf", 29),
    ("ishaan@neurova.demo", "mri/pituitary_3.jpg", 20),
    ("ishaan@neurova.demo", "mri/notumor_2.jpg", 44),
    ("fatima@neurova.demo", "mri/hard_glioma_1.jpg", 8),         # the model misses this tumour but flags itself as unsure
    ("fatima@neurova.demo", "mri/meningioma_4.jpg", 52),
    ("joseph@neurova.demo", "mri/notumor_5.jpg", 70),             # already reviewed, below
]


def add_demo_data():
    if os.environ.get("NEUROVA_DEMO", "1") == "0":
        return
    with Session() as db:
        if db.query(User).count():
            return                                           # only on a brand-new database
        hashed = hash_password(DEMO_PASSWORD)
        users = {email: User(name=name, email=email, password_hash=hashed, role=role) for name, email, role in PEOPLE}
        db.add_all(users.values())
        db.flush()
        doctor = users["aahana@neurova.demo"]
        for email in ("akshata@neurova.demo", "ishaan@neurova.demo", "fatima@neurova.demo", "joseph@neurova.demo"):
            db.add(Access(patient_id=users[email].id, clinician_id=doctor.id))     # they all share with Dr. Sharma
        db.commit()

        # Run the models once on sample files so there are results waiting.
        from . import mri, sleep                              # imported here so the models load only when needed
        samples = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
        for email, file, hours_ago in RESULTS:
            path = os.path.join(samples, file)
            if file.startswith("mri/"):
                result = mri.analyze(open(path, "rb").read())
                title = "Brain MRI · held-out test image.jpg"
            else:
                result = sleep.analyze(open(path, "rb").read(), json.load(open(path.replace(".edf", ".expert.json"))))
                title = "Sleep EEG · Sleep-EDF held-out night.edf"
            patient = users[email]
            db.add(Study(patient_id=patient.id, uploaded_by=patient.id, kind=result["kind"], title=title,
                         headline=result["headline"], needs_review=result["needs_review"], result_json=json.dumps(result),
                         created=now() - timedelta(hours=hours_ago)))
        db.flush()

        # One result has already been reviewed, so the patient and admin pages show what that looks like.
        done = db.query(Study).filter_by(patient_id=users["joseph@neurova.demo"].id).first()
        done.review_verdict, done.review_note = "agree", "Agree, no abnormality seen. Routine follow-up only."
        done.reviewed_by, done.reviewed = doctor.id, now() - timedelta(hours=60)
        db.commit()
