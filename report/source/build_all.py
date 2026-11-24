"""
Rebuild every submission document from the current trained-model results.

    python report/source/build_all.py

Writes to report/: the report (.docx and .pdf), the presentation (.pptx),
the Business Model Canvas (.pdf, .png) and the architecture diagram (.png).
Needs Node.js with three packages, installed once from the neurova folder with
    npm install docx pptxgenjs jszip
and LibreOffice for the PDF copy of the report (optional).
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = HERE
OUT = os.path.join(ROOT, "report")
os.makedirs(OUT, exist_ok=True)


def run(cmd, **kw):
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT, **kw)


run([sys.executable, os.path.join(SRC, "make_facts.py")])
run([sys.executable, os.path.join(SRC, "draw_architecture.py")])
shutil.copy(os.path.join(SRC, "architecture.png"), os.path.join(OUT, "Neurova_System_Architecture.png"))
run([sys.executable, os.path.join(SRC, "draw_bmc.py")])
run([sys.executable, os.path.join(SRC, "draw_robustness.py")])
run([sys.executable, os.path.join(SRC, "build_viva.py")])
run(["node", os.path.join(SRC, "build_report.js")])
env = dict(os.environ, PPTX_SKILL_DIR=os.environ.get("PPTX_SKILL_DIR", os.path.join(SRC, "pptx_support")))
run(["node", os.path.join(SRC, "build_slides.js")], env=env)
soffice = shutil.which("soffice") or shutil.which("libreoffice")
if soffice:
    run([soffice, "--headless", "--convert-to", "pdf", "--outdir", OUT, os.path.join(OUT, "Neurova_CDS301_Report.docx")])
print("Done. Files are in", OUT)
