"""
Numbers shared by the report, slides and Business Model Canvas.

Business figures are planning assumptions, stated in the report. Cost figures
derive from the measured model latency and the Hugging Face price list.
Edit STUDENT / DEPLOYMENT_URL here, then rebuild:  python report/source/build_all.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

STUDENT = "Akshata"          # add your full name here
DEPLOYMENT_URL = "https://huggingface.co/spaces/YOUR-HF-USERNAME/neurova"
SUBMISSION_DATE = "20 November 2026"

mri = json.load(open(os.path.join(ROOT, "results", "mri", "metrics.json")))
R = mri["results"][mri["chosen"]]
tests_path = os.path.join(HERE, "test_results.json")
tests = json.load(open(tests_path)) if os.path.exists(tests_path) else {"api": 0, "screens": 0, "js_errors": 0, "mri_request_s": 0}

INR_PER_USD = 85                      # assumption, stated in the report
HF_USD_HOUR = 0.03                    # Hugging Face CPU upgrade, 8 vCPU / 32 GB
monthly_hosting = round(HF_USD_HOUR * 730 * INR_PER_USD)
# One full MRI analysis was timed by tests/test_app.py on a two-core machine. A paid 8-vCPU
# instance has four times the cores; round down to the nearest thousand to stay conservative.
request_s = tests.get("mri_request_s") or 0.1
scans_per_hour = int(3600 / request_s * 4 // 1000 * 1000)

engineer_lakh, advisor_k, months = 1.5, 50, 18
people = round((2 * engineer_lakh + advisor_k / 100) * months)
validation, regulatory, misc = 15, 10, 4
total = people + validation + regulatory + misc
monthly_run = 2 * engineer_lakh * 1e5 + advisor_k * 1e3 + monthly_hosting + 25000    # + sales and admin
clinic = 9999

facts = {
    "student_name": STUDENT, "deployment_url": DEPLOYMENT_URL, "submission_date": SUBMISSION_DATE,
    "tests": {"api": tests["api"], "screens": tests["screens"], "js_errors": tests["js_errors"]},
    "latency": {"mri_request_s": tests.get("mri_request_s", 0), "forward_ms": tests.get("forward_ms", 0)},
    "costs": {"inr_per_usd": INR_PER_USD, "hf_cpu_upgrade_usd_per_hour": HF_USD_HOUR,
              "monthly_hosting_inr": monthly_hosting, "scans_per_hour": scans_per_hour},
    "business": {"price_per_mri": 40, "price_per_sleep": 250, "clinic_plan_monthly": clinic, "clinic_plan_studies": 400,
                 "enterprise_annual_lakh": 6, "engineer_monthly_lakh": engineer_lakh, "advisor_monthly_k": advisor_k,
                 "people_18m_lakh": people, "validation_lakh": validation, "regulatory_lakh": regulatory,
                 "total_18m_lakh": total, "breakeven_centres": int(-(-monthly_run // clinic))},
}
json.dump(facts, open(os.path.join(HERE, "facts.json"), "w"), indent=2)
print(json.dumps(facts, indent=2))
