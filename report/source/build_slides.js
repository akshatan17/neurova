// Builds report/Neurova_CDS301_Presentation.pptx from the training metrics and figures.
//   node report/source/build_slides.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const SKILL = process.env.PPTX_SKILL_DIR;
const { applyTheme } = require(path.join(SKILL, "scripts", "apply_theme.js"));

const ROOT = path.resolve(__dirname, "..", "..");
const RES = path.join(ROOT, "results");
const SRC = __dirname;
const OUT = path.join(ROOT, "report", "Neurova_CDS301_Presentation.pptx");
const mriM = JSON.parse(fs.readFileSync(path.join(RES, "mri", "metrics.json")));
const slM = JSON.parse(fs.readFileSync(path.join(RES, "sleep", "metrics.json")));
const facts = JSON.parse(fs.readFileSync(path.join(SRC, "facts.json")));
const R = mriM.results[mriM.chosen], Bm = mriM.results.neurova_cnn, S = slM.results, dd = mriM.dedup, biz = facts.business;
const E0 = mriM.results.efficientnet_b0_tf, ER = mriM.results.efficientnet_b0_tf_robust || mriM.results.efficientnet_b0_tf;
const robustDeployed = mriM.chosen === "efficientnet_b0_tf_robust";
const pct = (x) => (x * 100).toFixed(1) + "%";

const THEME = {
  name: "Neurova", headFontFace: "Calibri", bodyFontFace: "Calibri",
  colors: { dk1: "2F2620", lt1: "FFFFFF", dk2: "3E6447", lt2: "F3F6F2", accent1: "4F7A59", accent2: "A8690F",
    accent3: "3F679E", accent4: "B42318", accent5: "C9DCC6", accent6: "6F655C", hlink: "3E6447", folHlink: "6F655C" },
};
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";                 // 13.33 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Neurova"; pres.author = facts.student_name;
const C = pres.SchemeColor;
const HEX = THEME.colors;

// ---- layouts
pres.defineSlideMaster({ title: "TITLE", background: { color: HEX.dk2 }, objects: [
  { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.3, w: 11.7, h: 1.4, fontSize: 54, bold: true, color: C.background1, valign: "bottom" }, text: "" } },
  { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.8, w: 11, h: 1.6, fontSize: 22, color: C.accent5, valign: "top" }, text: "" } },
] });
pres.defineSlideMaster({ title: "CONTENT", background: { color: HEX.lt1 }, slideNumber: { x: 12.4, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: HEX.accent6 }, objects: [
  { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.4, w: 12.1, h: 0.9, fontSize: 34, bold: true, color: C.text1, valign: "top", margin: 0 }, text: "" } },
] });
pres.defineSlideMaster({ title: "DARK", background: { color: HEX.dk2 }, slideNumber: { x: 12.4, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: HEX.accent5 }, objects: [
  { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.4, w: 12.1, h: 0.9, fontSize: 34, bold: true, color: C.background1, valign: "top", margin: 0 }, text: "" } },
] });

const T = (s, text, o) => s.addText(text, Object.assign({ isTextBox: true, fontSize: 16, color: C.text1, margin: 0, valign: "top" }, o));
const img = (s, file, o) => {
  const buf = fs.readFileSync(file); const w0 = buf.readUInt32BE(16), h0 = buf.readUInt32BE(20);
  let { x, y, w, h } = o; const r = w0 / h0;
  if (w / h > r) { const nw = h * r; x += (w - nw) / 2; w = nw; } else { const nh = w / r; y += (h - nh) / 2; h = nh; }
  s.addImage({ path: file, x, y, w, h, altText: o.alt, objectName: o.name || path.basename(file) });
};
function stat(s, x, y, w, big, small, color) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 1.85, fill: { color: HEX.lt2 }, line: { color: HEX.lt2 }, rectRadius: 0.12, objectName: "stat " + small });
  T(s, big, { x: x + 0.3, y: y + 0.22, w: w - 0.6, h: 0.9, fontSize: 40, bold: true, color: color || C.accent1 });
  T(s, small, { x: x + 0.3, y: y + 1.1, w: w - 0.6, h: 0.65, fontSize: 14, color: C.accent6 });
}
function card(s, x, y, w, h, head, body, fill) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill || HEX.lt2 }, line: { color: fill || HEX.lt2 }, rectRadius: 0.12, objectName: "card " + head });
  T(s, head, { x: x + 0.3, y: y + 0.25, w: w - 0.6, h: 0.45, fontSize: 18, bold: true });
  T(s, body, { x: x + 0.3, y: y + 0.78, w: w - 0.6, h: h - 0.95, fontSize: 14, color: C.text1, paraSpaceAfter: 6 });
}

// 1 title
pres.addSection({ title: "Opening" });
let s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Opening" });
s.addText("Neurova", { placeholder: "title" });
s.addText("Deep learning for brain MRI tumour triage and EEG sleep staging, reviewed by clinicians\n" + facts.student_name + "  ·  CDS301 Deep Learning  ·  Sai University", { placeholder: "body" });
s.addNotes("Introduce yourself and the project in one sentence: Neurova runs two trained deep learning models in a working web app, so the scans most likely to need attention get looked at first and a clinician reviews every result.");

// 2 problem
pres.addSection({ title: "Problem" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem" });
s.addText("Scans wait in the same queue, whatever they show", { placeholder: "title" });
stat(s, 0.6, 1.6, 3.9, "1 per 100,000", "radiologists per person in India, against 8 to 10 in OECD countries [RIAI via 5C Network]");
stat(s, 4.72, 1.6, 3.9, "~23,000", "new brain and CNS cancers a year in India [GBD 2016, Lancet Neurology 2019]", C.accent2);
stat(s, 8.84, 1.6, 3.9, "~1,000", "30-second epochs to score by hand in every overnight sleep study");
T(s, [
  { text: "Who is affected", options: { bold: true, breakLine: true } },
  { text: "Patients waiting for results, radiologists and sleep technicians with long queues, and small diagnostic centres with an MRI scanner but no neuroradiologist.", options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "What success looks like", options: { bold: true, breakLine: true } },
  { text: "Shorter time from scan to specialist for the scans that need it, few missed tumours, and less technician time per sleep study." },
], { x: 0.6, y: 3.85, w: 7.6, h: 2.9, fontSize: 16 });
card(s, 8.84, 3.85, 3.9, 2.9, "Why not rules?", "MRI brightness has no fixed scale across scanners, and tumour types differ in subtle shape and position. Sleep stages depend on context. Rules break; networks learn the features from data.");
s.addNotes("The bottleneck is people, not scanners. Explain why a threshold-based system fails: MRI intensity is arbitrary across machines, so deep learning is needed.");

// 3 solution
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem" });
s.addText("Two trained models inside a working app", { placeholder: "title" });
card(s, 0.6, 1.6, 3.9, 2.4, "Brain MRI classifier", "EfficientNet-B0, fine-tuned from ImageNet. Glioma, meningioma, pituitary tumour or no tumour, with Grad-CAM and calibrated confidence.");
card(s, 4.72, 1.6, 3.9, 2.4, "EEG sleep staging", "1-D CNN reads each 30 s epoch, a bidirectional GRU reads 10-minute context. Wake, N1, N2, N3, REM.");
card(s, 8.84, 1.6, 3.9, 2.4, "The app around them", "Patient, clinician and admin logins. Clinicians work from a triage list, urgent scans first, and agree or disagree with each result. Every result downloads as a PDF.");
T(s, [
  { text: "Small enough to run anywhere: ", options: { bold: true } },
  { text: "the models are trained in PyTorch, then exported to ONNX. The app needs about 100 MB of software in place of more than 1 GB, and a prediction takes about " + Math.round(facts.latency.forward_ms) + " ms on a laptop CPU." },
], { x: 0.6, y: 4.4, w: 12.1, h: 0.9, fontSize: 16, color: C.accent6 });
T(s, "Built with PyTorch for training, ONNX Runtime for inference, FastAPI, SQLite or MySQL, and plain HTML, CSS and JavaScript.", { x: 0.6, y: 5.6, w: 12.1, h: 0.6, fontSize: 16 });
s.addNotes("Say why ONNX: PyTorch is needed to train a model, but not to run one. Exporting gives the same answers (checked on all 924 test images) in a much smaller and faster package.");

// 4 data
pres.addSection({ title: "Model" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Model" });
s.addText("12% of the MRI dataset was duplicated", { placeholder: "title" });
stat(s, 0.6, 1.6, 3.5, String(dd.images_total - dd.unique_images), `duplicate images removed out of ${dd.images_total.toLocaleString("en-IN")}`, C.accent2);
stat(s, 0.6, 3.65, 3.5, String(dd.groups_spanning_official_train_and_test), "duplicate groups in both the official train and test folders", C.accent4);
T(s, [
  { text: "Brain Tumor MRI Dataset ", options: { bold: true } },
  { text: "(Nickparvar, CC BY 4.0): four classes, T1 contrast slices.", options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "Every image was perceptually hashed. A model tested on the official split is partly tested on images it trained on, so scores reported that way are inflated.", options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: `After de-duplication: a fresh stratified split of ${dd.unique_images.toLocaleString("en-IN")} images, 70% train, 15% validation, 15% test. The test set was used once.`, options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "Sleep-EDF Expanded ", options: { bold: true } },
  { text: `(PhysioNet, ODC-By): ${slM.nights.length} nights, 10 people, Fpz-Cz channel. Split by person, so test subjects were never seen.` },
], { x: 4.5, y: 1.6, w: 8.2, h: 4.4, fontSize: 16 });
s.addNotes("This is the strongest methodological point. Explain perceptual hashing and data leakage, and why you re-split instead of using the folders as given.");

// 5 model
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Model" });
s.addText("Transfer learning beats training from scratch", { placeholder: "title" });
s.addChart(pres.charts.BAR, [
  { name: "Test accuracy", labels: ["NeurovaCNN (scratch)", "EfficientNet-B0", "EfficientNet-B0 + scrub"], values: [Bm, E0, ER].map((r) => +(r.test_accuracy * 100).toFixed(1)) },
  { name: "Macro-F1 × 100", labels: ["NeurovaCNN (scratch)", "EfficientNet-B0", "EfficientNet-B0 + scrub"], values: [Bm, E0, ER].map((r) => +(r.test_macro_f1 * 100).toFixed(1)) },
], { x: 0.6, y: 1.5, w: 6.4, h: 4.9, barDir: "col", barGrouping: "clustered", chartColors: [HEX.accent1, HEX.accent5],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 13, dataLabelColor: HEX.dk1, dataLabelFontFace: "+mn-lt",
  valAxisMinVal: 50, valAxisMaxVal: 100, valAxisLabelColor: HEX.accent6, catAxisLabelColor: HEX.dk1, catAxisLabelFontSize: 13,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", valGridLine: { color: "E3E8E1", size: 0.5 }, catGridLine: { style: "none" },
  showLegend: true, legendPos: "b", legendFontSize: 12, legendFontFace: "+mn-lt", showTitle: true, title: `Held-out test set, ${Object.values(mriM.splits.test).reduce((a, b) => a + b, 0)} images`,
  titleFontSize: 14, titleColor: HEX.dk1, titleFontFace: "+mn-lt", objectName: "comparison chart" });
T(s, [
  { text: "EfficientNet-B0", options: { bold: true, breakLine: true } },
  { text: `${R.params_m.toFixed(1)} M parameters, ImageNet weights, new 4-class head, against a ${Bm.params_m.toFixed(2)} M CNN trained from scratch. Chosen for accuracy per unit of compute, since the app runs on a CPU.`, options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "Training", options: { bold: true, breakLine: true } },
  { text: "AdamW, cosine decay, label smoothing 0.1, augmentation (flip, rotate, shift, brightness), early stopping on validation macro-F1.", options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "Calibration", options: { bold: true, breakLine: true } },
  { text: `Temperature scaling on validation: expected calibration error ${R.ece_before.toFixed(3)} → ${R.ece_after.toFixed(3)}.` },
], { x: 7.4, y: 1.6, w: 5.3, h: 4.9, fontSize: 15 });
s.addNotes(`Deployed model: ${pct(R.test_accuracy)} accuracy, macro-F1 ${R.test_macro_f1.toFixed(3)}, macro AUC ${R.test_macro_auc.toFixed(3)}. Explain why macro-F1: it weights every class equally.`);

// 6 results
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Model" });
s.addText("Where the model is right and where it slips", { placeholder: "title" });
img(s, path.join(RES, "mri", "confusion_matrices.png"), { x: 0.6, y: 1.45, w: 7.6, h: 3.6, alt: "Confusion matrices" });
img(s, path.join(RES, "mri", "examples_incorrect.png"), { x: 0.6, y: 5.1, w: 7.6, h: 2.1, alt: "Most confident mistakes" });
const pc = R.per_class, lab = { glioma: "Glioma", meningioma: "Meningioma", notumor: "No tumour", pituitary: "Pituitary" };
T(s, [
  { text: "Per-class F1 (deployed)", options: { bold: true, breakLine: true } },
  ...Object.entries(pc).map(([k, v], i, a) => ({ text: `${lab[k]}: ${v.f1.toFixed(3)}  (recall ${pct(v.recall)})`, options: { breakLine: i < a.length - 1 } })),
], { x: 8.6, y: 1.6, w: 4.1, h: 2.2, fontSize: 15 });
card(s, 8.6, 4.0, 4.1, 2.9, "Every result is explained", "Grad-CAM heatmap, calibrated probability for all four classes, MC-dropout uncertainty and a stability check, then a clinician's review.");
s.addNotes("Walk through the confusion matrix. Point at the most common confusion and use the mistakes strip to explain why.");

// 6b robustness
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Model" });
s.addText("High accuracy, but faint noise could fool it", { placeholder: "title" });
img(s, path.join(RES, "mri", "robustness.png"), { x: 0.6, y: 1.5, w: 7.4, h: 3.6, alt: "Robustness comparison" });
T(s, [
  { text: "What the robustness test found", options: { bold: true, breakLine: true } },
  { text: `Noise of about 3 grey levels changed ${E0.robustness["noise_0.05"].toFixed(0)}% of the standard EfficientNet's test predictions. A model reading anatomy shouldn't care. The four classes come from different source collections, so the network was partly reading each source's compression and resolution fingerprint.`, options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "The fix", options: { bold: true, breakLine: true } },
  { text: `Retrain with random resampling, JPEG re-compression, blur and noise. Flips under faint noise fell to ${ER.robustness["noise_0.05"].toFixed(1)}%, with test accuracy ${pct(ER.test_accuracy)}.` },
], { x: 8.3, y: 1.55, w: 4.4, h: 5.3, fontSize: 15 });
T(s, robustDeployed ? "The scrubbed model is the one deployed. The selection rule was fixed before it was trained: deploy it if its validation macro-F1 is within one point of the best."
  : "The selection rule, fixed in advance, deploys the scrubbed model only if its validation macro-F1 is within one point of the best; it was not, so the standard model is deployed and this remains a known limitation.",
  { x: 0.6, y: 5.4, w: 7.4, h: 1.3, fontSize: 14, color: C.accent6 });
s.addNotes("This is the most important slide for a deep learning examiner. A high test score on one dataset does not mean the model learned the right thing. Explain shortcut learning and how the augmentation removes the shortcut.");

// 7 sleep
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Model" });
s.addText("Sleep staging on people the model never saw", { placeholder: "title" });
img(s, path.join(RES, "sleep", "hypnogram_test_subject.png"), { x: 0.6, y: 1.5, w: 8.2, h: 3.2, alt: "Hypnogram, expert vs model" });
stat(s, 9.2, 1.55, 3.5, pct(S.test_accuracy), "accuracy on unseen people");
stat(s, 9.2, 3.6, 3.5, S.test_kappa.toFixed(2), "Cohen's kappa (human scorers agree at about 0.75 to 0.8)");
T(s, [
  { text: "CNN + bidirectional GRU. ", options: { bold: true } },
  { text: `A 1-D CNN turns each 30 s epoch into 128 features; a GRU reads 20 epochs forwards and backwards, like a scorer checking context. Class-weighted loss for the rare N1 stage. ${S.params_m.toFixed(2)} M parameters, about ${S.cpu_seconds_per_night.toFixed(1)} s of model time per night on a CPU.` },
], { x: 0.6, y: 5.0, w: 8.2, h: 1.9, fontSize: 15 });
s.addNotes("N1 is the hardest stage for every published model and for human scorers. Training on all 153 Sleep-EDF nights is a one-line change. If asked about Transformers: you built one (same CNN encoder, Transformer encoder in place of the GRU) and trained each three times; see the Viva notes for the numbers and why the GRU stayed.");

// 8 architecture
pres.addSection({ title: "System" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "System" });
s.addText("From upload to prediction in one request", { placeholder: "title" });
img(s, path.join(SRC, "architecture.png"), { x: 0.6, y: 1.4, w: 12.1, h: 5.5, alt: "System architecture diagram" });
s.addNotes("Follow one MRI upload: browser, token check, preprocessing with the same function used in training, EfficientNet, calibrated response, displayed and stored.");

// 9 deployment + testing
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "System" });
s.addText("Deployed free, tested end to end", { placeholder: "title" });
stat(s, 0.6, 1.6, 2.9, `${Math.round(facts.latency.forward_ms)} ms`, "per MRI on a 2-core CPU");
stat(s, 3.7, 1.6, 2.9, `${facts.tests.api}`, "automated API checks, all passing");
stat(s, 6.8, 1.6, 2.9, `${facts.tests.js_errors}`, "JavaScript errors across every screen");
stat(s, 9.9, 1.6, 2.83, "₹0", "hosting for the prototype");
T(s, [
  { text: "Live at ", options: { bold: true } },
  { text: facts.deployment_url, options: { breakLine: true } },
  { text: "Hugging Face Space, Docker container, 2 vCPU and 16 GB. On a laptop it starts by double-clicking one file.", options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "Security: ", options: { bold: true } },
  { text: "bcrypt, JWT sessions, consent-based access, audit log, upload limits.", options: { breakLine: true } },
  { text: "Limits: ", options: { bold: true } },
  { text: "2-D slices only, one public dataset, four tumour classes only (other scans get an \"unfamiliar scan\" warning), no clinical validation yet." },
], { x: 0.6, y: 3.85, w: 12.1, h: 2.9, fontSize: 16 });
s.addNotes("Mention that role boundaries are tested: a patient cannot open another patient's record and an admin cannot read clinical data.");

// 10 business
pres.addSection({ title: "Business" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Business" });
s.addText("A per-study subscription for small diagnostic centres", { placeholder: "title" });
card(s, 0.6, 1.6, 3.9, 2.6, "Customers", "Tier-2 and tier-3 MRI centres with no neuroradiologist, and the teleradiology firms that read for them. Then sleep and neurology clinics.");
card(s, 4.72, 1.6, 3.9, 2.6, "Pricing", `₹${biz.price_per_mri} per MRI, ₹${biz.price_per_sleep} per sleep study, or ₹${biz.clinic_plan_monthly.toLocaleString("en-IN")} a month for ${biz.clinic_plan_studies} studies. Annual licences for chains.`);
card(s, 8.84, 1.6, 3.9, 2.6, "Difference", "MRI tumour triage, not CT emergencies. Priced for small centres. Patients control who has access.");
T(s, [
  { text: "First 18 months: ", options: { bold: true } },
  { text: `about ₹${biz.total_18m_lakh} lakh, mostly two engineers, a radiologist adviser, a validation study and CDSCO registration. ${biz.breakeven_centres} centres on the clinic plan cover running costs.`, options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "Path: ", options: { bold: true } },
  { text: "silent pilot at 2 to 3 centres near Chennai → CDSCO registration → launch with pilot centres → teleradiology and PACS partners." },
], { x: 0.6, y: 4.55, w: 12.1, h: 2.2, fontSize: 16 });
s.addNotes("In the silent pilot the model scores every scan but radiologists don't see it until afterwards, so it can be validated with no risk to patients.");

// 11 demo + next
pres.addSection({ title: "Close" });
s = pres.addSlide({ masterName: "DARK", sectionTitle: "Close" });
s.addText("Live demo, then what comes next", { placeholder: "title" });
T(s, [
  { text: "Demo", options: { bold: true, breakLine: true, color: C.background1 } },
  { text: "Sign in as Dr. Aahana Sharma: the triage list puts likely tumours first. Open one, show the heatmap and checks, and save a review. Open the missed tumour flagged as unsure. Then sign in as the patient, see the note and download the PDF.", options: { color: C.accent5 } },
], { x: 0.6, y: 1.6, w: 5.8, h: 4.8, fontSize: 18, paraSpaceAfter: 8 });
T(s, [
  { text: "Next", options: { bold: true, breakLine: true, color: C.background1 } },
  { text: "Train sleep staging on all 153 nights", options: { bullet: true, breakLine: true, color: C.accent5 } },
  { text: "3-D MRI volumes instead of single slices", options: { bullet: true, breakLine: true, color: C.accent5 } },
  { text: "External test set from a different hospital", options: { bullet: true, breakLine: true, color: C.accent5 } },
  { text: "Silent clinical pilot", options: { bullet: true, breakLine: true, color: C.accent5 } },
  { text: "Learned tumour segmentation", options: { bullet: true, color: C.accent5 } },
], { x: 7.0, y: 1.6, w: 5.7, h: 4.8, fontSize: 18, paraSpaceAfter: 8 });
s.addNotes("Demo script (click the demo buttons on the sign-in page; the password for all is Neurova-demo-2026): 1) Dr. Aahana Sharma → Triage. Point out the three groups. 2) Open Akshata Nagaraj: Glioma → heatmap, probability bars, the four checks → Agree with a note → you return to the list. 3) Open Fatima Khan: No tumour, 53% under Model unsure → the model missed this tumour, but every check flagged it. 4) Sign out → Akshata Nagaraj → open the glioma → the review is there → Download PDF report. 5) How the models perform → the shortcut finding and the GRU vs Transformer comparison. 6) If there is time, upload the eye-socket MRI from your laptop to show the unfamiliar-scan warning.");

pres.writeFile({ fileName: OUT }).then(async () => { await applyTheme(OUT, THEME); console.log("wrote", OUT); });
