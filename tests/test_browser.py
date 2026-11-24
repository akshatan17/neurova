"""
Click through the app in a real browser and save a screenshot of every page.

    python tests/test_browser.py        (needs: pip install playwright && playwright install chromium)

Starts its own server on port 8765 with an empty temporary database.
"""
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOTS = os.path.join(ROOT, "tests", "screenshots")
URL = "http://127.0.0.1:8765"
os.makedirs(SHOTS, exist_ok=True)

server = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.server:app", "--port", "8765", "--log-level", "warning"],
                          cwd=ROOT, env={**os.environ, "NEUROVA_DATA": tempfile.mkdtemp()})
for _ in range(60):
    try:
        urllib.request.urlopen(URL + "/api/health")
        break
    except OSError:
        time.sleep(0.5)

errors, count = [], 0


def shot(page, name):
    global count
    count += 1
    page.wait_for_timeout(250)
    page.screenshot(path=os.path.join(SHOTS, f"{count:02d}_{name}.png"), full_page=True)


def demo_login(page, who):
    page.goto(URL)
    page.click(f"#demoButtons button:has-text('{who}')")
    page.wait_for_selector("#nav a")


try:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 860})
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))

        page.goto(URL)
        page.wait_for_selector("#demoButtons button")
        shot(page, "sign_in")

        # ---- patient
        demo_login(page, "Akshata Nagaraj")
        page.wait_for_selector(".item")
        shot(page, "patient_results")
        page.click(".item:has-text('Glioma')")
        page.wait_for_selector(".scans img")
        shot(page, "mri_result")
        page.go_back()
        page.click(".item:has-text('asleep')")
        page.wait_for_selector(".hypno")
        shot(page, "sleep_result")
        page.go_back()
        page.click("#newBtn")
        page.wait_for_selector("#sampleGrid button")
        shot(page, "new_analysis")
        start = time.time()
        page.click("#sampleGrid button[data-file='mri/pituitary_1.jpg']")
        page.wait_for_selector(".scans img")
        mri_seconds = round(time.time() - start, 2)
        assert "Pituitary" in page.inner_text(".headline")
        page.go_back()
        page.click("#newBtn")
        page.click("#sampleGrid button[data-file='mri/hard_glioma_1.jpg']")       # a tumour the model misses
        page.wait_for_selector(".scans img")
        assert "Not stable" in page.inner_text("#view") and "unfamiliar kind of scan" in page.inner_text("#view")
        shot(page, "mri_hard_case")
        page.click("nav a:has-text('Sharing')")
        page.wait_for_selector(".item")
        shot(page, "sharing")
        page.click("nav a:has-text('How the models perform')")
        page.wait_for_selector(".figure img")
        shot(page, "model_performance")
        page.click("#signOut")

        # ---- clinician
        demo_login(page, "Dr. Aahana Sharma")
        page.wait_for_selector("section .item")                                   # the triage list is the clinician's home
        groups = page.locator("section h2").all_inner_texts()
        assert groups == ["Likely tumour", "Model unsure", "Routine"], groups
        shot(page, "clinician_triage")
        waiting = page.locator("section .item").count()
        page.click("section .item:has-text('Akshata Nagaraj: Glioma')")
        page.wait_for_selector("#reviewForm")
        page.check("input[value=agree]")
        page.fill("#reviewNote", "Agree. Please book a contrast MRI and a neurosurgery appointment.")
        shot(page, "clinician_review")
        page.click("#reviewForm button")
        page.wait_for_selector("section .item")                                   # back on the triage list, one fewer
        assert page.locator("section .item").count() == waiting - 1
        assert page.locator("section .item:has-text('Akshata Nagaraj: Glioma')").count() == 0
        page.click("nav a:has-text('My patients')")
        page.wait_for_selector(".item")
        shot(page, "clinician_patients")
        page.click("#signOut")

        # ---- admin
        demo_login(page, "Site Administrator")
        page.wait_for_selector(".tiles")
        shot(page, "admin_overview")
        page.click("tr:has-text('Avani Kurup') button[data-edit]")                # rename an account
        page.wait_for_selector("#editDialog[open]")
        page.fill("#editName", "Avani Kurup Menon")
        page.click("#nameForm button")
        page.wait_for_selector("#editDone:not([hidden])")
        shot(page, "admin_edit_account")
        page.click("#closeEdit")
        page.wait_for_selector("td:has-text('Avani Kurup Menon')")
        page.click("nav a:has-text('Activity log')")
        page.wait_for_selector("table")
        shot(page, "admin_activity")
        page.click("#signOut")

        # ---- the patient now sees the clinician's note; and a phone-sized screen
        phone = browser.new_page(viewport={"width": 390, "height": 800})
        demo_login(phone, "Akshata Nagaraj")
        phone.click(".item:has-text('Glioma')")
        phone.wait_for_selector(".scans img")
        assert "Dr. Aahana Sharma" in phone.inner_text("#view")
        with phone.expect_download() as download:                                # the PDF report
            phone.click("#pdfBtn")
        pdf_path = os.path.join(SHOTS, "report.pdf")
        download.value.save_as(pdf_path)
        assert open(pdf_path, "rb").read(5) == b"%PDF-"
        overflow = phone.evaluate("document.documentElement.scrollWidth > window.innerWidth")
        shot(phone, "phone_mri_result")
        browser.close()
    summary = os.path.join(ROOT, "report", "source", "test_results.json")
    if os.path.isdir(os.path.dirname(summary)):
        import json
        old = json.load(open(summary)) if os.path.exists(summary) else {}
        json.dump({**old, "screens": count, "js_errors": len(errors), "phone_overflow": overflow}, open(summary, "w"))
    print(f"{count} screens, {len(errors)} JavaScript errors, MRI through the UI {mri_seconds}s, phone overflow: {overflow}")
    for e in errors:
        print("  ", e)
    sys.exit(1 if errors or overflow else 0)
finally:
    server.terminate()
