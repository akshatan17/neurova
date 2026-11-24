"""One-page Business Model Canvas (PDF + PNG), standard nine-block layout."""
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
facts = json.load(open(os.path.join(HERE, "facts.json")))
b = facts["business"]
INK, MUTED, SAGE, LINE, WASH = "#2f2620", "#5f564e", "#3e6447", "#9fb39b", "#f3f6f2"

blocks = {
    "Key partners": [
        "Partner diagnostic centres for the pilot and data",
        "Consultant radiologists and a sleep physician as advisers",
        "Teleradiology companies as resellers",
        "PACS and RIS vendors for integration",
        "Sai University incubation; BIRAC grants",
        "Cloud hosting in an Indian region",
    ],
    "Key activities": [
        "Training and validating the models",
        "Silent clinical pilot",
        "CDSCO software registration, ISO 13485 quality system",
        "Integration with reporting workflows",
        "Monitoring for performance drift",
    ],
    "Key resources": [
        "Trained MRI and sleep models",
        "De-duplicated, documented datasets",
        "Portal with consent and audit",
        "ML and full-stack engineers",
        "Clinical evidence from the pilot",
    ],
    "Value propositions": [
        "Brain MRIs likely to show a tumour reach a radiologist first",
        "Suggested tumour type with heatmap and calibrated confidence",
        "Overnight EEG pre-scored, so technicians only correct it",
        "Priced per study, affordable for small centres",
        "Patients control which clinicians see their results",
    ],
    "Customer relationships": [
        "Free three-month pilot, then subscription",
        "Onboarding and dashboard for each centre",
        "Monthly performance reports to customers",
        "Feedback loop: corrections improve the models",
    ],
    "Channels": [
        "Direct sales to centres near Chennai first",
        "Radiology society meetings and CMEs",
        "Teleradiology partners reaching many centres",
        "Web app; API for PACS integration",
    ],
    "Customer segments": [
        "Tier-2 and tier-3 MRI centres with no on-site neuroradiologist",
        "Teleradiology providers",
        "Neurology and sleep clinics",
        "Later: hospital chains (licence)",
        "Patients (free accounts)",
    ],
    "Cost structure": [
        f"People: two engineers and a radiologist adviser, about ₹{b['people_18m_lakh']} lakh over 18 months",
        f"Clinical validation and labelling, about ₹{b['validation_lakh']} lakh; regulatory and quality system, about ₹{b['regulatory_lakh']} lakh",
        f"Hosting and inference: about ₹{facts['costs']['monthly_hosting_inr']:,} a month per CPU instance",
        f"Total for the first 18 months: about ₹{b['total_18m_lakh']} lakh",
    ],
    "Revenue streams": [
        f"Pay as you go: ₹{b['price_per_mri']} per MRI study, ₹{b['price_per_sleep']} per sleep study",
        f"Clinic plan: ₹{b['clinic_plan_monthly']:,} a month for {b['clinic_plan_studies']} MRI studies",
        f"Annual on-premise licences for chains and teleradiology, from about ₹{b['enterprise_annual_lakh']} lakh",
        f"{b['breakeven_centres']} clinic-plan centres cover monthly running costs",
    ],
}

fig = plt.figure(figsize=(16.5, 11.7))      # A3 landscape proportions
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 70); ax.axis("off")
ax.text(1.5, 68.6, "Business Model Canvas: Neurova", fontsize=21, fontweight="bold", color=INK, va="top")
ax.text(1.5, 65.7, f"{facts['student_name']}, CDS301 Deep Learning, Sai University", fontsize=12, color=MUTED, va="top")

top, mid, bot = 63.5, 41.5, 20.0
W = 97 / 5
x0 = 1.5
cells = [
    ("Key partners", x0, bot, W, top - bot),
    ("Key activities", x0 + W, mid, W, top - mid),
    ("Key resources", x0 + W, bot, W, mid - bot),
    ("Value propositions", x0 + 2 * W, bot, W, top - bot),
    ("Customer relationships", x0 + 3 * W, mid, W, top - mid),
    ("Channels", x0 + 3 * W, bot, W, mid - bot),
    ("Customer segments", x0 + 4 * W, bot, W, top - bot),
    ("Cost structure", x0, 1.5, 97 / 2, bot - 1.5),
    ("Revenue streams", x0 + 97 / 2, 1.5, 97 / 2, bot - 1.5),
]
for name, x, y, w, h in cells:
    fill = WASH if name == "Value propositions" else "white"
    ax.add_patch(Rectangle((x, y), w, h, fc=fill, ec=LINE, lw=1.2))
    ax.text(x + 1.0, y + h - 1.1, name, fontsize=13.5, fontweight="bold", color=SAGE, va="top")
    wrap = int(w * 1.75)
    yy = y + h - 4.3
    for item in blocks[name]:
        lines = textwrap.wrap(item, wrap)
        ax.text(x + 1.0, yy, "•", fontsize=11, color=SAGE, va="top")
        for k, l in enumerate(lines):
            ax.text(x + 2.2, yy - k * 1.75, l, fontsize=10.8, color=INK, va="top")
        yy -= len(lines) * 1.75 + 1.0

out = os.path.join(ROOT, "report")
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "Neurova_Business_Model_Canvas.pdf"), facecolor="white")
fig.savefig(os.path.join(out, "Neurova_Business_Model_Canvas.png"), dpi=130, facecolor="white")
print("wrote canvas")
