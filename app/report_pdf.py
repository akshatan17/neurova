"""
A one- or two-page PDF of a single result, for printing or sending to someone.

    pdf_bytes = build_report(study, patient_name, result, model_facts)

It holds what the result page shows: the answer, the summary, the scan and its
heatmap (or the hypnogram for a sleep recording), the trust checks, and the
clinician's review. Built with fpdf2; the DejaVu font in app/fonts/ prints any
accented letter or symbol a clinician might type in a note.
"""
import base64
import io
import os
from datetime import datetime, timezone

from fpdf import FPDF

FONTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

INK, MUTED, LINE = (42, 36, 32), (106, 98, 90), (221, 228, 218)
SAGE, AMBER, RED, WASH = (79, 122, 89), (168, 105, 15), (180, 35, 24), (234, 240, 232)
STAGE_COLOUR = {"W": (217, 164, 65), "REM": (192, 88, 107), "N1": (156, 192, 220), "N2": (91, 143, 192), "N3": (47, 79, 134)}
STAGE_ORDER = ["W", "REM", "N1", "N2", "N3"]
STAGE_LABEL = {"W": "Wake", "REM": "REM", "N1": "N1 light", "N2": "N2", "N3": "N3 deep"}


class Report(FPDF):
    def __init__(self):
        super().__init__(format="A4")
        self.set_margins(18, 13, 18)
        self.set_auto_page_break(True, 12)
        self.add_font("Sans", "", os.path.join(FONTS, "DejaVuSans.ttf"))
        self.add_font("Sans", "B", os.path.join(FONTS, "DejaVuSans-Bold.ttf"))
        self.add_page()

    # ---- small helpers, so the layout code below reads like a list of what goes on the page
    @property
    def width(self):
        return self.w - self.l_margin - self.r_margin

    def text(self, words, size=10, bold=False, colour=INK, gap=1.5, height=None):
        self.set_font("Sans", "B" if bold else "", size)
        self.set_text_color(*colour)
        self.multi_cell(0, height or size * 0.5, words, align="L", new_x="LMARGIN", new_y="NEXT")
        self.ln(gap)

    def heading(self, words):
        self.ensure(20)
        self.ln(2)
        self.text(words, 12, bold=True, gap=1.8)

    def rule(self):
        self.set_draw_color(*LINE)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def ensure(self, mm):
        """Start a new page if fewer than `mm` millimetres are left on this one."""
        if self.get_y() + mm > self.h - self.b_margin:
            self.add_page()

    def label_value(self, rows):
        for label, value in rows:
            self.set_font("Sans", "", 9.5)
            self.set_text_color(*MUTED)
            self.cell(34, 5.2, label)
            self.set_text_color(*INK)
            self.multi_cell(0, 5.2, value, align="L", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def two_columns(self, left, right, room=70):
        """Draw `left()` and `right()` side by side, as the result page does, by narrowing the margins."""
        self.ensure(room)
        l, r, top, gap = self.l_margin, self.r_margin, self.get_y(), 8
        half = (self.w - l - r - gap) / 2
        self.set_right_margin(r + half + gap)
        left()
        bottom = self.get_y()
        self.set_left_margin(l + half + gap)
        self.set_right_margin(r)
        self.set_y(top)
        right()
        bottom = max(bottom, self.get_y())
        self.set_left_margin(l)
        self.set_y(bottom)

    def bar(self, label, fraction, value_text, colour=SAGE, label_w=34):
        """One labelled horizontal bar, like the ones on the result page."""
        self.ensure(8)
        y = self.get_y()
        self.set_font("Sans", "", 9)
        self.set_text_color(*INK)
        self.cell(label_w, 6, label)
        track_x, track_w = self.l_margin + label_w, self.width - label_w - 18
        self.set_fill_color(*WASH)
        self.rect(track_x, y + 2, track_w, 2.6, style="F")
        self.set_fill_color(*colour)
        self.rect(track_x, y + 2, max(track_w * min(fraction, 1.0), 0.6), 2.6, style="F")
        self.set_xy(track_x + track_w, y)
        self.cell(18, 6, value_text, align="R", new_x="LMARGIN", new_y="NEXT")

    def check(self, ok, title, detail):
        """A trust check: a green tick or an amber exclamation mark, a title and a sentence."""
        self.ensure(14)
        y = self.get_y()
        self.set_fill_color(*(WASH if ok else (248, 238, 220)))
        self.ellipse(self.l_margin, y + 0.4, 5, 5, style="F")
        self.set_font("Sans", "B", 8.5)
        self.set_text_color(*(SAGE if ok else AMBER))
        self.set_xy(self.l_margin, y + 0.6)
        self.cell(5, 4.6, "✓" if ok else "!", align="C")
        self.set_xy(self.l_margin + 8, y)
        self.set_font("Sans", "B", 10)
        self.set_text_color(*INK)
        self.multi_cell(0, 5.2, title, align="L", new_x="LMARGIN", new_y="NEXT")
        self.set_x(self.l_margin + 8)
        self.set_font("Sans", "", 9)
        self.set_text_color(*MUTED)
        self.multi_cell(0, 4.6, detail, align="L", new_x="LMARGIN", new_y="NEXT")
        self.ln(1.6)


def _when(value) -> str:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value.strftime("%d %b %Y, %H:%M UTC")


def _png(b64: str) -> io.BytesIO:
    return io.BytesIO(base64.b64decode(b64))


# ------------------------------------------------------------------ the two kinds of result
def _mri(pdf: Report, r: dict, facts: dict):
    pdf.heading("What the model saw")
    side = 56                                    # two 56 mm squares side by side
    pdf.ensure(side + 14)
    top = pdf.get_y()
    pdf.image(_png(r["image"]), x=pdf.l_margin, y=top, w=side, h=side)
    pdf.image(_png(r["heatmap"]), x=pdf.l_margin + side + 6, y=top, w=side, h=side)
    pdf.set_y(top + side + 1.5)
    pdf.set_font("Sans", "", 8.5)
    pdf.set_text_color(*MUTED)
    pdf.cell(side + 6, 4.5, "The scan, cropped to the brain.")
    pdf.multi_cell(0, 4.5, "Where the model looked (Grad-CAM): red pushed it most towards its answer.", align="L",
                   new_x="LMARGIN", new_y="NEXT")

    def likelihoods():
        pdf.heading("How likely is each class")
        for p in r["probabilities"]:
            pdf.bar(p["label"], p["percent"] / 100, f"{p['percent']:.1f}%")

    def checks():
        pdf.heading("Can this result be trusted?")
        u, s = r["uncertainty"], r["stability"]
        pdf.check(u["level"] == "Low", f"Uncertainty: {u['level'].lower()}", u["note"])
        pdf.check(s["held"], "Stable under disturbance" if s["held"] else "Not stable under disturbance", s["verdict"])
        if r.get("familiarity"):
            f = r["familiarity"]
            pdf.check(f["familiar"], "A familiar kind of scan" if f["familiar"] else "An unfamiliar kind of scan",
                      "It resembles the scans the model was trained on." if f["familiar"] else "See the warning at the top.")
        pdf.check(True, f"Tested accuracy: {facts['mri_accuracy'] * 100:.1f}%",
                  f"Measured on {facts['mri_test_images']} scans the model never saw in training.")

    pdf.two_columns(likelihoods, checks)


def _hypnogram(pdf: Report, stages: list, title: str):
    """Draw a hypnogram: one short coloured line per 30 seconds, at the height of its stage."""
    row, left = 4.2, 20.0
    width = pdf.width - left
    pdf.ensure(row * 5 + 14)
    pdf.set_font("Sans", "B", 9)
    pdf.set_text_color(*INK)
    pdf.cell(0, 5, title, new_x="LMARGIN", new_y="NEXT")
    top = pdf.get_y() + 1
    n = len(stages)
    pdf.set_font("Sans", "", 8)
    pdf.set_text_color(*MUTED)
    pdf.set_draw_color(*LINE)
    pdf.set_line_width(0.2)
    for k, s in enumerate(STAGE_ORDER):
        y = top + k * row + row / 2
        pdf.line(pdf.l_margin + left, y, pdf.l_margin + left + width, y)
        pdf.set_xy(pdf.l_margin, y - 2.2)
        pdf.cell(left - 3, 4.4, s, align="R")
    pdf.set_line_width(1.2)
    for i, s in enumerate(stages):
        if not s:
            continue
        x0 = pdf.l_margin + left + width * i / n
        y = top + STAGE_ORDER.index(s) * row + row / 2
        pdf.set_draw_color(*STAGE_COLOUR[s])
        pdf.line(x0, y, x0 + width / n + 0.05, y)
    hours = n / 120
    step = 1 if hours > 5 else 0.5
    pdf.set_y(top + 5 * row + 0.5)
    h = 0.0
    while h <= hours + 1e-9:
        pdf.set_xy(pdf.l_margin + left + width * h / hours - 6, top + 5 * row + 0.5)
        pdf.cell(12, 4, f"{h:g} h", align="C")
        h += step
    pdf.set_y(top + 5 * row + 7)


def _sleep(pdf: Report, r: dict, facts: dict):
    pdf.heading("The night, epoch by epoch")
    pdf.text("Each 30-second piece of the recording is given one stage."
             + (" The expert row is a sleep technician's scoring of the same night, for comparison." if r.get("expert_stages") else ""),
             9.5, colour=MUTED, gap=3)
    _hypnogram(pdf, r["stages"], "Model")
    if r.get("expert_stages"):
        _hypnogram(pdf, r["expert_stages"], "Expert")

    def stage_times():
        pdf.heading("Time in each stage")
        total = r["epochs"] / 2
        for s in STAGE_ORDER:
            pdf.bar(STAGE_LABEL[s], r["minutes"][s] / total, f"{r['minutes'][s]:.0f} min", colour=STAGE_COLOUR[s])

    def checks():
        pdf.heading("Can this result be trusted?")
        a = r.get("agreement")
        if a:
            pdf.check(a["percent"] >= 75, f"Agreement with the expert: {a['percent']}%",
                      f"On this night, over {a['epochs']} epochs (Cohen's kappa {a['kappa']}).")
        pdf.check(True, f"Tested accuracy: {facts['sleep_accuracy'] * 100:.1f}%",
                  f"Measured on {facts['sleep_test_people']} people the model never saw in training.")
        pdf.check(r["unsure_epochs"] < r["epochs"] * 0.1, f"{r['unsure_epochs']} of {r['epochs']} epochs were unsure",
                  "Epochs where the model gave its answer less than a 50% probability.")

    pdf.two_columns(stage_times, checks)
    pdf.text(f"Channel used: {r['channel']}. {r.get('channel_note') or ''} {r['caveat']}".replace("  ", " "), 8.5, colour=MUTED, gap=0.5)


# ------------------------------------------------------------------ the whole report
def build_report(study: dict, patient_name: str, result: dict, facts: dict) -> bytes:
    pdf = Report()
    pdf.set_title(f"Neurova report {study['id']}")
    pdf.set_author("Neurova")

    # name and title line
    pdf.set_fill_color(*SAGE)
    pdf.rect(pdf.l_margin, pdf.get_y(), 7, 7, style="F", round_corners=True, corner_radius=1.6)
    pdf.set_x(pdf.l_margin + 9.5)
    pdf.set_font("Sans", "B", 13)
    pdf.set_text_color(*INK)
    pdf.cell(60, 7, "Neurova")
    pdf.set_font("Sans", "", 9)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 7, f"Analysis report · printed {_when(datetime.now(timezone.utc))}", align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.rule()

    pdf.label_value([("Patient", patient_name), ("Study", study["title"]), ("Analysed", _when(study["created"])),
                     ("Report number", f"#{study['id']}"),
                     ("Model", "EfficientNet-B0 brain MRI classifier" if result["kind"] == "mri"
                      else "1-D CNN + bidirectional GRU sleep stager")])
    pdf.rule()

    if result["kind"] == "mri":
        pdf.text(result["label"], 20, bold=True, gap=0.5, height=9)
        pdf.text(f"{result.get('confidence_text', str(result['confidence']) + '%')} confidence", 10.5, colour=MUTED, gap=3)
    else:
        pdf.text(study["headline"], 17, bold=True, gap=3, height=8)
    pdf.text(result["summary"], 10.5, gap=2, height=5.4)
    if result["kind"] == "mri" and result.get("familiarity") and not result["familiarity"]["familiar"]:
        pdf.text("This scan is unlike the scans the model was trained on. The model knows four answers only: glioma, "
                 "meningioma, pituitary tumour and no tumour, all inside the brain. For anything else its answer may mean little.",
                 9.5, colour=AMBER, gap=2, height=5)

    (_mri if result["kind"] == "mri" else _sleep)(pdf, result, facts)

    pdf.heading("Clinician review")
    review = study.get("review")
    if review:
        verdict = "agreed with the model" if review["verdict"] == "agree" else "disagreed with the model"
        pdf.text(f"{review['by']} {verdict} on {_when(review['when'])}.", 10.5, gap=1.5)
        if review.get("note"):
            pdf.text(f"“{review['note']}”", 10.5, colour=MUTED, gap=2, height=5.4)
    else:
        pdf.text("Not yet reviewed by a clinician. A model's answer is a second opinion, not a diagnosis.", 10.5, colour=AMBER)

    pdf.ln(2)
    pdf.rule()
    pdf.text("Produced by Neurova, a student deep learning project for education and research. Not a medical diagnosis.",
             8.5, colour=MUTED)
    return bytes(pdf.output())
