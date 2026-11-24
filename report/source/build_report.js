// Builds report/Neurova_CDS301_Report.docx from the training metrics and figures.
//   node report/source/build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun, Table, TableRow, TableCell, WidthType,
  AlignmentType, BorderStyle, ShadingType, PageBreak, LevelFormat, TableOfContents,
} = require("docx");

const ROOT = path.resolve(__dirname, "..", "..");
const RES = path.join(ROOT, "results");
const SRC = __dirname;
const OUTDIR = path.join(ROOT, "report");
fs.mkdirSync(OUTDIR, { recursive: true });

const mriM = JSON.parse(fs.readFileSync(path.join(RES, "mri", "metrics.json")));
const FAM = JSON.parse(fs.readFileSync(path.join(RES, "mri", "familiarity.json")));
const SC = JSON.parse(fs.readFileSync(path.join(RES, "sleep", "comparison.json"))).models;
const [GRU, TRF] = ["gru", "transformer"].map((k) => SC.find((m) => m.key === k));
const range = (v, f) => `${f(v.mean)} (${f(v.min)} to ${f(v.max)})`;
const slM = JSON.parse(fs.readFileSync(path.join(RES, "sleep", "metrics.json")));
const facts = JSON.parse(fs.readFileSync(path.join(SRC, "facts.json")));
const chosen = mriM.chosen;
const R = mriM.results[chosen];
const B = mriM.results["neurova_cnn"];
const S = slM.results;
const dd = mriM.dedup;
const pct = (x, d = 1) => (x * 100).toFixed(d) + "%";
const f3 = (x) => Number(x).toFixed(3);
const ARCH = { neurova_cnn: "NeurovaCNN (scratch)", efficientnet_b0_tf: "EfficientNet-B0", efficientnet_b0_tf_robust: "EfficientNet-B0 + scrub", efficientnet_b0: "EfficientNet-B0", resnet50: "ResNet-50" };
const E = mriM.results["efficientnet_b0_tf"];
const ER = mriM.results["efficientnet_b0_tf_robust"];
const rb = (r, k) => (r && r.robustness ? r.robustness[k].toFixed(0) + "%" : "n/a");
const CLS = { glioma: "Glioma", meningioma: "Meningioma", notumor: "No tumour", pituitary: "Pituitary tumour" };
const nTrain = Object.values(mriM.splits.train).reduce((a, b) => a + b, 0);
const nVal = Object.values(mriM.splits.val).reduce((a, b) => a + b, 0);
const nTest = Object.values(mriM.splits.test).reduce((a, b) => a + b, 0);
const removed = dd.images_total - dd.unique_images;
const pc = R.per_class;
const worst = Object.entries(pc).sort((a, b) => a[1].f1 - b[1].f1)[0];
const best = Object.entries(pc).sort((a, b) => b[1].f1 - a[1].f1)[0];
const mistakes = mriM.test_mistakes || [];
const pairCount = {};
mistakes.forEach((m) => { const k = `${CLS[m.true]} read as ${CLS[m.pred].toLowerCase()}`; pairCount[k] = (pairCount[k] || 0) + 1; });
const topPairs = Object.entries(pairCount).sort((a, b) => b[1] - a[1]).slice(0, 2);
const sPC = S.per_class;
const totalEpochs = slM.nights.reduce((a, n) => a + n.epochs, 0);
const C = facts.costs;

// ---------------------------------------------------------------- helpers
const FONT = "Calibri";
const P = (text, opts = {}) => new Paragraph({
  spacing: { after: 120, line: 276 }, alignment: opts.align || AlignmentType.JUSTIFIED,
  children: (Array.isArray(text) ? text : [text]).map((t) => typeof t === "string" ? new TextRun({ text: t, font: FONT, size: 22 }) : t),
});
const B_ = (t) => new TextRun({ text: t, bold: true, font: FONT, size: 22 });
const I_ = (t) => new TextRun({ text: t, italics: true, font: FONT, size: 22 });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const bullet = (t) => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 80, line: 280 },
  children: (Array.isArray(t) ? t : [t]).map((x) => typeof x === "string" ? new TextRun({ text: x, font: FONT, size: 22 }) : x) });
function fig(file, widthIn, caption) {
  const buf = fs.readFileSync(file);
  const png = buf.readUInt32BE(16), pngH = buf.readUInt32BE(20);
  const w = Math.round(widthIn * 96), h = Math.round(w * pngH / png);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: w, height: h },
        altText: { title: caption, description: caption, name: path.basename(file) } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 220 },
      children: [new TextRun({ text: caption, italics: true, font: FONT, size: 19, color: "555555" })] }),
  ];
}
const border = { style: BorderStyle.SINGLE, size: 4, color: "BFC8BC" };
function table(header, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (t, i, head) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    borders: { top: border, bottom: border, left: border, right: border },
    shading: head ? { type: ShadingType.CLEAR, color: "auto", fill: "E3ECE1" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT,
      children: [new TextRun({ text: String(t), bold: head, font: FONT, size: 20 })] })],
  });
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, i, false)) }))] });
}
const gap = () => new Paragraph({ spacing: { after: 120 }, children: [] });

// ---------------------------------------------------------------- content
const kids = [];
kids.push(
  new Paragraph({ spacing: { before: 2200, after: 200 }, alignment: AlignmentType.LEFT,
    children: [new TextRun({ text: "Neurova", font: FONT, size: 64, bold: true, color: "2F2620" })] }),
  new Paragraph({ spacing: { after: 600 }, children: [new TextRun({
    text: "Deep learning for brain MRI tumour classification and EEG sleep staging, in a web app where clinicians review what the models say",
    font: FONT, size: 30, color: "4F7A59" })] }),
  P([B_("Course: "), "Deep Learning (CDS301), B.Tech., School of Computing and Data Science, Sai University"], { align: AlignmentType.LEFT }),
  P([B_("Project type: "), "Industry and society innovation"], { align: AlignmentType.LEFT }),
  P([B_("Student: "), facts.student_name, " (individual project)"], { align: AlignmentType.LEFT }),
  P([B_("Deployment URL: "), facts.deployment_url], { align: AlignmentType.LEFT }),
  P([B_("Submitted: "), facts.submission_date], { align: AlignmentType.LEFT }),
  new Paragraph({ children: [new PageBreak()] }),
);

// --- Summary
kids.push(H1("Summary"),
  P(`India has roughly one radiologist for every 100,000 people, against eight to ten in OECD countries, while MRI and CT use keeps growing [1]. A brain MRI can sit in a reporting queue for hours or days, and a scan showing a tumour waits in the same line as a normal one. Neurova is a web app that runs two trained deep learning models on brain MRI and sleep EEG, so that the scans and recordings most likely to need attention are looked at first, and a clinician records whether they agree with each result.`),
  P(`The first model is a four-class brain MRI classifier (glioma, meningioma, pituitary tumour, no tumour) built by fine-tuning an ImageNet-pretrained EfficientNet-B0 on the public Brain Tumor MRI Dataset. Before training, a perceptual-hash audit found that ${removed} of the dataset's ${dd.images_total} images (${pct(removed / dd.images_total)}) were duplicates, and that ${dd.groups_spanning_official_train_and_test} duplicate groups sat in both the official training and testing folders, which would inflate any score computed on the official split. After removing them and re-splitting, the deployed model reached ${pct(R.test_accuracy)} accuracy and a macro-F1 of ${f3(R.test_macro_f1)} on ${nTest} held-out test images, against ${pct(B.test_accuracy)} for a small CNN trained from scratch. A robustness test then showed that the standard fine-tuned model changed ${rb(E, "noise_0.05")} of its test predictions under noise too faint to see, a sign that it was partly recognising which source collection an image came from. Retraining with augmentation that destroys those fingerprints brought that to ${ER ? ER.robustness["noise_0.05"].toFixed(1) + "%" : "n/a"} at the same accuracy, and that model is the one deployed. Its confidence is calibrated with temperature scaling and every prediction comes with a Grad-CAM heatmap.`),
  P(`The second model stages sleep from a single frontal EEG channel. A one-dimensional CNN encodes each 30-second epoch and a bidirectional GRU reads ten-minute sequences of them, as a human scorer does. Trained on Sleep-EDF Expanded and tested on people it never saw, it reached ${pct(S.test_accuracy)} accuracy and Cohen's kappa of ${f3(S.test_kappa)}.`),
  P(`Both models were trained in PyTorch and exported to ONNX, so the app runs them with ONNX Runtime and needs about 100 MB of software in place of more than 1 GB. They sit behind a FastAPI backend with bcrypt passwords, signed sessions, patient-controlled sharing, clinician review and an audit log, and a web frontend for patients, clinicians and administrators. The deployed MRI model answers in about ${Math.round(facts.latency.forward_ms)} ms per scan on a two-core CPU, so it runs on free hosting. The business plan proposes a per-study software subscription for diagnostic centres without an on-site neuroradiologist, starting with a three-month pilot.`),
);

// --- 1 Problem
kids.push(H1("1. Problem and market"),
  H2("1.1 The problem"),
  P(`Brain tumours are uncommon but serious. The Global Burden of Disease study estimated about 23,000 new brain and central nervous system cancers a year in India, with roughly 21,000 deaths [2]. Outcomes depend on how quickly an abnormal scan reaches someone who can act on it, and the tumour type shapes what happens next: a meningioma is often slow-growing and may be watched, a pituitary adenoma goes to endocrinology, and a glioma usually needs urgent neurosurgical and oncology input.`),
  P(`The bottleneck is people, not scanners. India has an estimated 20,000 to 22,000 practising radiologists for 1.4 billion people [1], and they are concentrated in large cities. Smaller diagnostic centres often send studies to teleradiology services and read them in the order they arrive. Sleep medicine has a similar problem on a smaller scale: an overnight study is scored by hand in 30-second epochs, roughly a thousand per night, which takes a trained technician a substantial part of a working day.`),
  H2("1.2 Stakeholders and users"),
  P(`Four groups are affected. Patients want their results quickly and want control over who sees them. Radiologists and sleep technicians want their attention directed to the cases that need it and want tools whose reasoning they can check. Diagnostic centre owners want shorter turnaround without hiring scarce specialists. Regulators, here the Central Drugs Standard Control Organisation (CDSCO), want evidence that software influencing diagnosis is safe [3].`),
  P(`Neurova is built for three users in the product itself: patients, who upload results and decide who may see them; clinicians, who review the patients who have shared with them; and administrators, who run accounts and audit access but are deliberately blocked from clinical data.`),
  H2("1.3 Measurable impact"),
  P(`The outcome that matters is time from scan to a specialist looking at it, for the scans that need it. In a pilot this would be measured as the median reporting delay for scans the model flags as tumour-positive, compared with the same centre's delay before the pilot. Secondary measures are the share of flagged scans a radiologist confirms (positive predictive value in practice), the share of tumours the model misses (sensitivity), and for sleep studies the technician time per night with and without automatic pre-scoring.`),
  H2("1.4 Existing solutions and competitors"),
  P(`Radiology AI is an established field. Qure.ai, headquartered in Mumbai, reports deployment at more than 3,000 sites in over 90 countries and raised a $65 million Series D in 2024 [4]; its head CT products focus on haemorrhage, fracture and mass effect. 5C Network runs a large Indian teleradiology operation. Internationally, companies such as Aidoc and Viz.ai sell triage tools to hospitals, mostly for CT and stroke. In sleep, several research models such as DeepSleepNet [5] and TinySleepNet [6] stage sleep from single-channel EEG, and some scoring software ships with automated staging.`),
  P(`These products have three limitations that leave room for Neurova. Most target CT emergencies rather than elective MRI tumour work, which is where Indian centres without a neuroradiologist struggle. They are sold as enterprise contracts that small centres cannot afford. And they report to the radiologist, not to the patient, so the patient has no view of who accessed their scan. Neurova's niche is low-cost MRI tumour triage and sleep pre-scoring for small centres, with patient-controlled access built in.`),
  H2("1.5 Demand"),
  P(`Demand comes from the gap between imaging volume and reporting capacity. Industry analysts consistently describe the Indian AI-in-diagnostics market as growing quickly [7]. The direct signal for this project would come from the pilot described in section 9, not from analyst forecasts.`),
  H2("1.6 Why deep learning and not rules"),
  P(`A rule-based system would need hand-written criteria such as "a bright region above a certain size". MRI intensity has no fixed scale: it varies with scanner, field strength, sequence and settings, so a threshold that works on one machine fails on the next. Tumours vary in shape, texture, location and enhancement, and the three tumour types differ in subtle ways such as whether a mass sits outside the brain against the skull (meningioma) or near the sella turcica (pituitary). Neurova includes rule-based tools for CT density and older EEG screens, and they are labelled as such precisely because they cannot make these distinctions. A convolutional network learns the relevant features directly from thousands of labelled examples and, with transfer learning, starts from filters that already recognise edges and textures.`),
  P(`Sleep staging is similar. The textbook rules (alpha rhythm in wake, spindles and K-complexes in N2, slow waves in N3) are clear on paper, but real recordings are noisy, rhythms overlap and the right answer for one epoch depends on its neighbours. A recurrent network over a sequence of epochs captures that context in a way fixed rules do not.`),
);

// --- 2 Related work
kids.push(H1("2. Related work"),
  P(`Brain tumour classification on public MRI collections is a common benchmark. Cheng and colleagues released the figshare set of 3,064 T1-weighted contrast slices of glioma, meningioma and pituitary tumours [8], and the Brain Tumor MRI Dataset used here combines it with two other sources and adds a no-tumour class [9]. Published work on these sets frequently reports accuracies above 95% with fine-tuned ResNet, VGG and EfficientNet models. Several recent papers and curation projects have pointed out that these collections contain duplicate images, which means some of those numbers are partly measuring memorisation; this project removes duplicates before splitting for that reason.`),
  P(`EfficientNet [10] scales network depth, width and resolution together and reaches strong ImageNet accuracy with far fewer parameters than ResNet-50 [11], which makes it a good fit for CPU deployment. Grad-CAM [12] produces class-specific heatmaps from the last convolutional layer and is the most widely used way to show where an image classifier is looking. Deep networks are known to be over-confident; Guo and colleagues showed that a single temperature parameter fitted on validation data fixes most of this [13]. Monte Carlo dropout gives a cheap estimate of model uncertainty [14].`),
  P(`For sleep, DeepSleepNet [5] combined CNN feature extraction with bidirectional LSTMs over sequences of epochs, and TinySleepNet [6] showed that a much smaller CNN plus recurrent model trained with sequence augmentation performs comparably on Sleep-EDF. Neurova's sleep model follows this two-stage design with a GRU, which has fewer parameters than an LSTM and trains faster on a CPU. Agreement between human scorers is itself limited, with Cohen's kappa commonly reported around 0.75 to 0.8, which sets a realistic ceiling for any automatic method.`),
);

// --- 3 Data
kids.push(H1("3. Data"),
  H2("3.1 Brain MRI"),
  P(`The Brain Tumor MRI Dataset (Nickparvar, 2021; also published on IEEE DataPort under CC BY 4.0) contains ${dd.images_total} two-dimensional slices in four classes, supplied in Training and Testing folders [9]. Images are JPEG files of varying size, mostly T1-weighted with contrast, in axial, coronal and sagittal planes.`),
  P([B_("De-duplication. "), `Each image was fingerprinted with a perceptual hash, which matches images that are identical or differ only in compression or size. ${dd.duplicate_groups} groups of duplicates were found covering ${dd.images_in_duplicate_groups} images; keeping one image per group removed ${removed}. Most importantly, ${dd.groups_spanning_official_train_and_test} groups had copies in both the official Training and Testing folders. A model evaluated on the official split would be tested partly on images it had trained on. No group had conflicting labels.`]),
  P([B_("Split. "), `The ${dd.unique_images} remaining images were split at random, stratified by class, into ${nTrain} for training (70%), ${nVal} for validation (15%) and ${nTest} for testing (15%), with a fixed seed. Validation was used to choose the best epoch, fit the calibration temperature and choose which model to deploy. The test set was used once, at the end.`]),
  ...fig(path.join(RES, "mri", "class_balance.png"), 3.0, "Figure 1. Images per class in each split after de-duplication."),
  P([B_("Preprocessing. "), `Each image is converted to greyscale-as-RGB, cropped to the brain by thresholding and taking the largest contour (which removes black borders and text burned into the margins), resized to 192 by 192 pixels and normalised with ImageNet statistics. The same Python function is used in training and in the deployed app, so the model never sees differently prepared images in production.`]),
  P([B_("Augmentation. "), `Training images were randomly flipped left to right, rotated by up to 10 degrees, shifted by up to 5%, scaled by up to 5% and jittered in brightness and contrast by up to 15%. These mimic differences in head position and scanner settings without changing anatomy. A second EfficientNet was also trained with what this report calls fingerprint scrubbing: random down-and-up resampling, JPEG re-compression at quality 30 to 90, mild blur and added noise, each applied at random. These destroy fine texture and compression artefacts while leaving anatomy visible; section 5.2 explains why they were needed. Validation and test images were never augmented.`]),
  H2("3.2 Sleep EEG"),
  P(`Sleep-EDF Expanded [15, 16] is an open PhysioNet database of overnight polysomnography from healthy adults, scored by trained technicians in 30-second epochs. This project uses the first ten subjects of the sleep-cassette study, two nights each, ${slM.nights.length} nights and ${totalEpochs.toLocaleString("en-IN")} scored epochs in total. One recording in the copy used here (SC4011E0) is truncated after 7.9 of its 23.4 hours; only the scored epochs that have signal are used. Only the Fpz-Cz channel is used, sampled at 100 Hz, so the model could later run on simple headband hardware.`),
  P(`Following standard practice, stages 3 and 4 of the older Rechtschaffen and Kales scoring were merged into N3 to match current AASM rules, movement and unscored epochs were dropped, and wake was trimmed to 30 minutes before the first and after the last sleep epoch; otherwise most of each recording would be the subject awake before lights-out. Each recording was scaled by subtracting its median and dividing by its interquartile range, which removes differences in amplitude between people and electrodes.`),
  P([B_("Subject-wise split. "), `Epochs from the same night are strongly correlated, so a random split of epochs would leak each person's brain activity into the test set. Whole subjects were assigned instead: subjects ${S.subjects.train.join(", ")} for training, subject ${S.subjects.val.join(", ")} for validation, and subjects ${S.subjects.test.join(" and ")} for testing. The test score therefore answers the question that matters: how well does the model work on a new person?`]),
  ...fig(path.join(RES, "sleep", "stage_distribution.png"), 2.9, "Figure 2. Epochs per sleep stage. N1 is much rarer than the other stages."),
);

// --- 4 Model
kids.push(H1("4. Model design and training"),
  H2("4.1 MRI classifier"),
  P(`Three models were trained on the same split and compared. The baseline is NeurovaCNN, a small network of four convolution blocks (16, 32, 64 and 128 filters, each with batch normalisation, ReLU and max pooling), global average pooling and a dropout classifier, about ${B.params_m.toFixed(2)} million parameters, trained from scratch. The main model is EfficientNet-B0 [10], ${R.params_m.toFixed(1)} million parameters, initialised from the official ImageNet weights, with its 1,000-class head replaced by a four-class layer with dropout. It was trained twice, once with standard augmentation and once with fingerprint scrubbing added.`),
  P(`EfficientNet-B0 was chosen for three reasons. Transfer learning matters when the training set has thousands rather than millions of images, and early convolutional filters learned on natural images (edges, blobs, textures) transfer well to medical images. Among pretrained families it gives one of the best accuracies per unit of computation, which matters because the deployed app runs on a CPU. And its final convolutional layer has enough spatial resolution for useful Grad-CAM maps. ResNet-50 [11] was considered as a third comparison, but it is about six times larger and was not needed to show the benefit of transfer learning.`),
  P(`All three were trained with the AdamW optimiser, cosine learning-rate decay, label smoothing of 0.1 (which discourages the network from becoming over-confident), batch size 32 and early stopping on validation macro-F1. Macro-F1 was chosen over accuracy because it weights all four classes equally. The baseline used a learning rate of 0.0005 and up to 25 epochs with a patience of 8. A first attempt at 0.001 with a patience of 5 was unstable, its validation score swinging by 50 points between epochs, and stopped at 64% test accuracy; that would have flattered the pretrained model, so it was rerun. EfficientNet used 0.0003, low enough not to destroy the pretrained features, for up to 8 epochs; the scrubbed run stopped early after 7. All training ran on a two-core CPU, about 25 minutes for the baseline and 35 to 40 for each EfficientNet.`),
  P([B_("Calibration. "), `After training, a single temperature T was fitted on the validation set by minimising negative log-likelihood; at inference the logits are divided by T before the softmax. This changes how confident the model is without changing which class it picks [13]. The fitted temperature for the deployed model was ${R.temperature.toFixed(2)}.`]),
  P([B_("Explanation and uncertainty. "), `Every prediction in the app includes a Grad-CAM heatmap [12] computed from the final convolutional layer for the predicted class, a Monte Carlo dropout estimate of uncertainty from 20 stochastic passes [14], and a stability check that re-runs the model under noise, blur and brightness changes and reports whether the prediction changes. A gradient-based adversarial test such as FGSM [17] is left out of the app, because the exported model does not compute gradients.`]),
  H2("4.2 Sleep staging model"),
  P(`The sleep model has two stages. An epoch encoder, a one-dimensional CNN, reads each 30-second epoch of 3,000 samples. Its first layer has a wide 0.5-second kernel with a stride of 6, which acts as a learned bank of band-pass filters for the classic EEG rhythms and sleep spindles; three further convolution layers and pooling reduce the epoch to a 128-number summary. A bidirectional GRU with 64 units in each direction then reads a sequence of 20 consecutive summaries, ten minutes of sleep, forwards and backwards, and a linear layer classifies every epoch in the sequence. The whole model has about ${S.params_m.toFixed(2)} million parameters.`),
  P(`The recurrent stage is why this is a sequence model rather than an image classifier: a technician scoring an ambiguous epoch looks at what came before and after it, and the GRU does the same. Training used AdamW, cosine decay, gradient clipping, and a weighted cross-entropy loss with class weights proportional to the inverse square root of class frequency, which raises the importance of the rare N1 stage without letting it dominate. Training sequences overlapped by half and were augmented with small amplitude changes, noise and time shifts. At inference, a full night is read in overlapping windows and the predictions for each epoch are averaged.`),
);

// --- 5 Results
const ORDER = ["neurova_cnn", "efficientnet_b0_tf", "efficientnet_b0_tf_robust"].filter((k) => mriM.results[k]);
const cmpRows = ORDER.map((k) => { const r = mriM.results[k]; return [ARCH[k] || k, r.params_m.toFixed(2) + " M", pct(r.test_accuracy), f3(r.test_macro_f1), f3(r.test_macro_auc), Math.round(r.cpu_latency_ms) + " ms", r.robustness ? r.robustness["noise_0.05"].toFixed(1) + "%" : "n/a"]; });
kids.push(H1("5. Results"),
  H2("5.1 MRI classifier"),
  P(`Table 1 compares the three models on the held-out test set. Transfer learning made a large difference: both pretrained EfficientNets outperformed the network trained from scratch on every measure, at the cost of a slower but still practical inference time. The two EfficientNets are equally accurate; they differ in the last column, which section 5.2 explains. The scrubbed model is the deployed one, and the rest of this section describes it.`),
  table(["Model", "Params", "Accuracy", "Macro-F1", "Macro AUC", "CPU / image", "Changed by faint noise"], cmpRows, [2450, 1000, 1050, 1050, 1100, 1050, 1326]),
  new Paragraph({ spacing: { before: 60, after: 220 }, children: [new TextRun({ text: `Table 1. Test-set results, ${nTest} images never used in training or model selection. The last column is the share of test predictions that change when noise with a standard deviation of 0.05 (about three grey levels) is added. CPU time is for PyTorch; the exported model the app runs is about twice as fast (section 6.2).`, italics: true, font: FONT, size: 19, color: "555555" })] }),
  P(`For the deployed model, per-class F1 ranged from ${f3(worst[1].f1)} for ${CLS[worst[0]].toLowerCase()} to ${f3(best[1].f1)} for ${CLS[best[0]].toLowerCase()}. The no-tumour class reached a recall of ${pct(pc.notumor.recall)} and the tumour classes combined were rarely called normal, which is the error that matters most for triage. ${topPairs.length ? `The most frequent mistakes were ${topPairs.map(([k, v]) => `${k.toLowerCase()} (${v} image${v > 1 ? "s" : ""})`).join(" and ")}.` : ""}`),
  ...fig(path.join(RES, "mri", "confusion_matrices.png"), 4.7, "Figure 3. Test-set confusion matrices. Rows are the true class and columns the prediction."),
  ...fig(path.join(RES, "mri", "training_curves.png"), 4.4, "Figure 4. Training and validation loss, accuracy and macro-F1 by epoch."),
  P(`The ROC curves (figure 5) show the model separating each class from the rest with a macro-averaged area under the curve of ${f3(R.test_macro_auc)}. Calibration improved after temperature scaling: expected calibration error went from ${f3(R.ece_before)} to ${f3(R.ece_after)}, meaning that when the deployed model reports a confidence, it is close to how often it is right.`),
  ...fig(path.join(RES, "mri", "roc_curves.png"), 2.2, "Figure 5. One-versus-rest ROC curves for the deployed model."),
  ...fig(path.join(RES, "mri", "calibration.png"), 2.9, "Figure 6. Reliability diagrams before and after temperature scaling. A calibrated model lies on the dotted line."),
  P(`Figures 7 and 8 show correct predictions and the model's most confident mistakes with their Grad-CAM maps. For meningiomas, which are bright and well defined, the heatmap sits on the lesion. For pituitary tumours it usually covers the sella region, but in one correct sagittal case it lies lower, over the brainstem and skull base, which is a reminder that a correct answer does not prove the model used the right evidence. Of the ${mistakes.length} test mistakes, ${mistakes.filter((m) => m.true !== "notumor" && m.pred !== "notumor").length} were confusions between tumour types rather than tumour versus normal. Several involve lesions that genuinely look alike on a single slice: a bright mass at the sella called a meningioma at 99% confidence with the heatmap squarely on it, and round enhancing gliomas read as meningiomas. In the most instructive error, a coronal pituitary tumour called meningioma, the heatmap sits at the top of the brain, nowhere near the sella. A reviewer looking at that heatmap would immediately distrust the prediction, which is the point of showing it. ${(() => { const miss = mistakes.filter((m) => m.true !== "notumor" && m.pred === "notumor"); return miss.length === 0 ? "No tumour was called normal." : miss.length === 1 ? `One tumour was called no tumour, at ${Math.round(miss[0].confidence * 100)}% confidence, which the uncertainty display would flag.` : `${miss.length} tumours were called no tumour, all at ${Math.max(...miss.map((m) => m.confidence)) < 0.6 ? "low confidence (below 60%)" : "varying confidence"}.`; })()}`),
  ...fig(path.join(RES, "mri", "examples_correct.png"), 5.0, "Figure 7. Correct test predictions (top) with Grad-CAM (bottom)."),
  ...fig(path.join(RES, "mri", "examples_incorrect.png"), 5.0, "Figure 8. The most confident mistakes on the test set."),
  H2("5.2 Robustness and a shortcut"),
  P(`The app's robustness check re-runs the model on perturbed copies of each scan. Applied to the whole test set, it exposed a problem that accuracy hid. Adding Gaussian noise with a standard deviation of 0.05 in normalised units, about three grey levels and invisible to a person, changed ${rb(E, "noise_0.05")} of the standard EfficientNet's predictions; stronger noise changed ${rb(E, "noise_0.1")} and a 7-pixel blur ${rb(E, "blur_7")}. A brightness shift, which changes every pixel far more than the faint noise does, changed only ${rb(E, "brightness_0.3")}. A model that had learned tumour anatomy should not care about faint noise. One that had learned fine texture would.`),
  P(`The likely explanation is shortcut learning. The dataset merges three collections, and the classes are not spread evenly across them: most no-tumour images come from one source, and the three tumour classes mostly from another, each with its own resolution, resampling and JPEG compression. Those differences leave a fingerprint in the fine texture of each image, and a network can partly identify the source, and therefore the likely class, from the fingerprint alone. That gives high accuracy on a test set drawn from the same sources and fails on scans from a new hospital.`),
  P(`The mitigation was to make the fingerprint useless during training. A second EfficientNet was trained with fingerprint scrubbing (section 3.1), which randomly resamples, re-compresses, blurs and adds noise so that texture no longer predicts the class. To keep the choice honest, the rule for which model to deploy was fixed before the scrubbed model was trained: deploy it if its validation macro-F1 is within one point of the best model's. Its validation macro-F1 was ${ER ? f3(ER.best_val_macro_f1) : "n/a"} against ${f3(E.best_val_macro_f1)}, so it met the rule. On the test set it matched the standard model's accuracy (${ER ? pct(ER.test_accuracy) : "n/a"}), and faint noise now changed ${ER ? ER.robustness["noise_0.05"].toFixed(1) + "%" : "n/a"} of predictions instead of ${rb(E, "noise_0.05")} (figure 9). The from-scratch baseline was just as fragile as the standard EfficientNet, so the problem comes from the data rather than from pretraining.`),
  ...fig(path.join(RES, "mri", "robustness.png"), 4.0, "Figure 9. Share of test predictions that change under each perturbation. Lower is better."),
  P(`This does not prove the model now relies only on anatomy; the real test is scans from a hospital outside the dataset, the first thing the pilot in section 9 would measure.`),
  P([B_("A scan outside the four classes. "), `The classifier has only four possible answers, so it must give one of them even for something it never saw. Tested on an MRI of an eye-socket tumour found online, it answered meningioma at 59% with no tumour second. Fixing that needs labelled scans of the missing condition, so the app instead warns when a scan is unfamiliar. It reduces each scan's 1,280 pooled features to ${FAM.pca_dimensions} dimensions with PCA and measures the Mahalanobis distance to the nearest class of training scans [19]; the threshold, the distance that 95% of validation scans stay under, and the number of dimensions were both chosen on validation data. On the test set the check flags ${FAM.test_mistakes_flagged} of the model's ${FAM.test_mistakes} mistakes, including the confident one that the uncertainty estimate missed, and ${FAM.ordinary_test_scans_flagged_percent}% of correctly answered scans. It flagged the eye-socket scan and ${FAM.other_pictures_flagged - 1} of 4 pictures that are not MRI scans at all. It flags only ${FAM.upside_down_scans_flagged_percent}% of scans turned upside down, so it is shown as a warning, not a guarantee.`]),
  H2("5.3 Sleep staging"),
  P(`On the ${S.subjects.test.length} unseen test subjects the sleep model reached ${pct(S.test_accuracy)} accuracy, macro-F1 of ${f3(S.test_macro_f1)} and Cohen's kappa of ${f3(S.test_kappa)}. Per-stage F1 was ${Object.entries(sPC).map(([k, v]) => `${f3(v.f1)} for ${k}`).join(", ")}. As in every published study on this data, N1 was the hardest stage: it is brief, transitional and looks like both wake and N2, and human scorers disagree on it most. The second weakness is wake, with a recall of ${pct(sPC.W.recall)} at a precision of ${pct(sPC.W.precision)}: the model rarely calls sleep wake, but it often scores quiet, drowsy wakefulness as N1. The class weights that raise N1's importance in the loss push in that direction, so the balance between the two is a tuning choice rather than a fixed property of the model.`),
  ...fig(path.join(RES, "sleep", "confusion_matrix.png"), 2.6, "Figure 10. Sleep staging confusion matrix on unseen subjects."),
  ...fig(path.join(RES, "sleep", "hypnogram_test_subject.png"), 4.3, "Figure 11. Expert scoring and model output for one test night."),
  P([B_("GRU or Transformer? "), `To test whether attention reads the night better, the bidirectional GRU was replaced with a two-layer Transformer encoder (four attention heads, learned position embeddings), keeping the epoch encoder, data, splits and training settings identical. The first Transformer run scored ${pct(TRF.runs[0].test_accuracy)}, above the GRU's ${pct(GRU.runs[0].test_accuracy)}. Because the test set is only two people, each model was then trained twice more with different random seeds. ${GRU.test_accuracy.mean > TRF.test_accuracy.mean ? `The first result did not hold: the GRU averaged ${range(GRU.test_accuracy, pct)} accuracy over the three runs and the Transformer ${range(TRF.test_accuracy, pct)}. Its validation score swung widely between epochs, so early stopping sometimes caught it at a poor point, and it was weaker on N1 (F1 ${f3(TRF.n1_f1.mean)} against ${f3(GRU.n1_f1.mean)}). Validation macro-F1, which never sees the test people, also favoured the GRU (${f3(GRU.best_val_macro_f1.mean)} against ${f3(TRF.best_val_macro_f1.mean)}).` : `Over the three runs the Transformer averaged ${range(TRF.test_accuracy, pct)} accuracy and the GRU ${range(GRU.test_accuracy, pct)}.`} With 20 nights, a Transformer has more parameters to fit (${TRF.params_m.toFixed(2)} M against ${GRU.params_m.toFixed(2)} M) and no built-in sense of order, so the GRU stays. One run would have suggested the opposite.`]),
);

// --- 6 System design
kids.push(H1("6. System design"),
  ...fig(path.join(SRC, "architecture.png"), 5.1, "Figure 12. System architecture."),
  H2("6.1 Frontend"),
  P(`The frontend is three files of plain HTML, CSS and JavaScript with no framework and no build step, served by the backend. It opens on a centred sign-in page with one-click demo accounts. A patient sees a list of their results, starts a new analysis by uploading a file or choosing an example, and decides on a sharing page which clinicians may see the results. A clinician's home page is a triage list: every result waiting for their review, across all patients who share with them, in three groups. Likely tumours come first, most confident first; then results the model may have wrong because it is uncertain, unstable or facing an unfamiliar scan; then confident normal scans and sleep recordings. On each result the clinician records whether they agree with the model, with a note the patient can read, and returns to the list. This is the product's central promise, urgent scans read first, working end to end. An administrator sees counts, accounts and the activity log, can add clinicians, and can rename, reset the password of or disable any account (disabling keeps the person's results and signs them out at once), but cannot open any result. A model performance page shows the evaluation in this report to every user.`),
  P(`The MRI result page shows the cropped scan next to the Grad-CAM heatmap, a bar for each class's calibrated probability, and four plain checks under the heading "Can this result be trusted?": uncertainty, stability, familiarity and the accuracy measured on the test set. Every result can be downloaded as a one-page PDF report with the scan, heatmap, checks and the clinician's review, for printing or sending on. The sleep result page draws the hypnogram, beside the expert's scoring when that is available, with the minutes in each stage, sleep efficiency and the agreement with the expert. The pages fit a phone screen and follow the device's dark mode.`),
  H2("6.2 Backend and model integration"),
  P(`The backend is a FastAPI application of about 950 lines, with its endpoints documented automatically at /api/docs. The models are trained in PyTorch and then exported to ONNX. An export script checks that the ONNX outputs match PyTorch to within 0.00001, and the exported MRI model reproduced the same ${pct(R.test_accuracy)} on all ${nTest} test images. Running on ONNX Runtime, a forward pass takes ${Math.round(facts.latency.forward_ms)} ms against ${Math.round(R.cpu_latency_ms)} ms in PyTorch on the same machine. The layers after EfficientNet's last feature maps are only an activation, an average and one linear layer, so the Grad-CAM gradient and the Monte Carlo dropout passes are computed directly in NumPy from the exported feature maps; the heatmap matches PyTorch's automatic differentiation to six decimal places. An upload is size-checked, prepared by the same preprocessing function the training scripts import, passed to the model, and stored as a JSON result with the patient's record. The uploaded file itself is not kept.`),
  P(`Passwords are hashed with bcrypt. Signing in issues a signed JSON Web Token that expires after twelve hours, and five wrong passwords lock that email for ten minutes. One function decides who may see a patient's data: patients see only their own, a clinician sees a patient only while that patient is sharing with them, and an administrator sees none. Requests for results the user may not see return "not found", so result numbers cannot be probed. Sign-ins, analyses, reviews, changes to sharing and each time a clinician opens a patient's results are written to an audit log.`),
  H2("6.3 Database"),
  P(`Data is stored through SQLAlchemy, so the same code runs on SQLite with no setup or on MySQL by changing one environment variable. There are four tables: users, access (which patient shares with which clinician), studies (one analysis and its clinician review) and the audit log. Results are stored as JSON in a long-text column, because an MRI result with its two embedded images is about 200 KB and exceeds MySQL's 64 KB text limit.`),
);

// --- 7 Testing
kids.push(H1("7. Testing"),
  P([B_("Functional correctness. "), `An automated suite of ${facts.tests.api} checks covers sign-in, wrong and tampered credentials, registration rules, every role's boundaries (a patient cannot open another patient's result, an administrator cannot open any result, a clinician loses access the moment sharing stops), both models on every sample file, rejection of corrupt and wrong-type uploads, clinician review, the administrator pages and the audit log. All checks pass on SQLite and on MariaDB, a MySQL-compatible server.`]),
  P([B_("End-to-end workflow. "), `A browser test using headless Chrome signs in as each role, runs an MRI analysis, saves a clinician review and confirms that the patient then sees it, across ${facts.tests.screens} screens including one at phone width. It reports ${facts.tests.js_errors} JavaScript errors and no horizontal scrolling on a 390-pixel screen.`]),
  P([B_("Inference latency. "), `Measured on a two-core CPU with one image at a time, the deployed MRI model takes a median of ${Math.round(facts.latency.forward_ms)} ms per forward pass. A complete MRI request, with preprocessing, Grad-CAM, 20 Monte Carlo dropout passes and the seven-way stability check, takes about ${facts.latency.mri_request_s.toFixed(1)} seconds. Sleep staging of a full eight-hour night takes about ${S.cpu_seconds_per_night.toFixed(1)} seconds of model time. Both are fast enough for an interactive demonstration on free hosting.`]),
  P([B_("Reliability. "), `Corrupt images, non-image files, damaged EDF files, recordings shorter than five minutes and uploads over the size limit all return a clear error instead of a crash. The same scan always gives the same result, because the random numbers used for dropout and noise are seeded from the file. The demo accounts and two finished analyses are created on first start, so the deployed site always has something to show.`]),
  P([B_("Usability. "), `The interface was checked for visible keyboard focus, labelled form fields, readable contrast in light and dark mode and a layout that fits a phone. A formal usability study with clinicians was not carried out and is part of the pilot plan.`]),
);

// --- 8 Deployment
kids.push(H1("8. Deployment"),
  P(`The application is deployed as a Hugging Face Space, which provides a free container with two virtual CPUs and 16 GB of memory [18], at ${facts.deployment_url}. The repository includes a short Dockerfile that installs nine small Python packages and starts the app on port 7860; because inference runs on ONNX Runtime, the container does not include PyTorch. The same container runs on Render, Railway or any Docker host. On a laptop the whole project is a 26 MB download that starts by double-clicking one file.`),
  P([B_("Cost. "), `The prototype costs nothing to host. For a production pilot, an 8-vCPU, 32 GB Hugging Face CPU instance costs $${C.hf_cpu_upgrade_usd_per_hour} an hour [18], about ₹${C.monthly_hosting_inr.toLocaleString("en-IN")} a month running continuously at an assumed ₹${C.inr_per_usd} per dollar. At the measured latency one such instance can analyse tens of thousands of MRI scans an hour, so inference cost per scan is a small fraction of a rupee and hosting is not the main cost.`]),
  P([B_("Security. "), `Passwords are hashed, sessions expire, access depends on the patient's consent and is logged, uploads are size-limited and type-checked, and the uploaded file is not retained. A deployment handling real patients would also need HTTPS termination (provided by the hosting platform), encryption of the database at rest, two-factor sign-in for clinicians, tokens in httpOnly cookies in place of browser storage, backups, and data hosted in India to meet the Digital Personal Data Protection Act.`]),
  P([B_("Scalability. "), `The backend keeps no state outside the database, so it scales horizontally behind a load balancer once SQLite is replaced with MySQL. Inference is CPU-bound and parallel, so throughput grows linearly with instances. Free Spaces sleep when unused and their disk is not persistent, so the demo returns to its starting data after a restart; a paid instance with a managed database removes both limits.`]),
  P([B_("Limitations. "), `The MRI model sees single 2D slices, not full 3D volumes, and was trained on one public collection whose scanners and patient population are not documented; it has not been tested on scans from Indian centres. The sleep model was trained on 20 nights of healthy adults from one study in the 1990s. The MRI model knows four classes only and cannot recognise any other condition (section 5.2). Neither model has had clinical validation, and the app labels every result as educational and not a diagnosis. The app reads PNG and JPG slices and EDF recordings, not DICOM series.`]),
);

// --- 9 Business
const ch = facts.business;
kids.push(H1("9. Commercialisation and business plan"),
  H2("9.1 Customers and value"),
  P(`The first customers are independent diagnostic centres and small hospitals in tier-2 and tier-3 Indian cities that own an MRI scanner but have no on-site neuroradiologist, and the teleradiology companies that report for them. The value proposition is simple: brain MRIs likely to show a tumour move to the top of the reading list with a suggested tumour type, a heatmap and a calibrated confidence, so a radiologist sees them first. The secondary market is neurology and sleep clinics, where automatic pre-scoring of overnight EEG saves technician time and the technician only corrects it.`),
  P(`Neurova differs from established competitors in three ways: it focuses on MRI tumour triage rather than CT emergencies, it is priced per study for small centres rather than as an enterprise contract, and it gives patients control over which clinicians see their results, with every access logged.`),
  H2("9.2 Pricing and revenue"),
  P(`Revenue is software as a service in three tiers. A pay-as-you-go plan charges ₹${ch.price_per_mri} per MRI study triaged and ₹${ch.price_per_sleep} per overnight sleep study scored. A clinic plan at ₹${ch.clinic_plan_monthly.toLocaleString("en-IN")} a month includes ${ch.clinic_plan_studies} MRI studies. Hospital chains and teleradiology firms would buy an annual licence, deployable in their own data centre, priced by volume from about ₹${ch.enterprise_annual_lakh} lakh a year. Patients use the record and sharing features free, which builds the network of clinicians on the platform.`),
  H2("9.3 Costs"),
  P(`The estimates below are for the first 18 months and state their assumptions. Development is mainly people: one machine learning engineer and one full-stack engineer at about ₹${ch.engineer_monthly_lakh} lakh a month each, plus a part-time radiologist adviser at ₹${ch.advisor_monthly_k},000 a month, about ₹${ch.people_18m_lakh} lakh over 18 months. Data labelling and the clinical validation study, in which radiologists read several thousand scans from partner centres, are estimated at ₹${ch.validation_lakh} lakh. Regulatory work for CDSCO registration of medical device software [3], including a quality management system under ISO 13485, is estimated at ₹${ch.regulatory_lakh} lakh. Hosting and inference are small: about ₹${C.monthly_hosting_inr.toLocaleString("en-IN")} a month for a dedicated CPU instance, rising with customers. Maintenance, mainly retraining on new data and monitoring for performance drift, is budgeted at 20% of engineering time after launch. The total is roughly ₹${ch.total_18m_lakh} lakh, which would be funded through university incubation, government start-up grants such as BIRAC's BIG scheme, and a seed round after the pilot.`),
  P(`At the clinic plan price, ${ch.breakeven_centres} paying centres cover running costs after launch. That is a small share of the thousands of MRI centres in India.`),
  H2("9.4 Go-to-market"),
  P([B_("First customers. "), `Two or three diagnostic centres near Chennai, reached through faculty and alumni contacts and local radiology society meetings, offered free use for three months in exchange for anonymised feedback and data for validation.`]),
  P([B_("Pilot validation. "), `During the pilot Neurova runs silently: it scores every brain MRI, but radiologists report as normal and do not see the output until afterwards. This measures sensitivity, specificity and potential time saved on the centre's own scans, with no risk to patients, and gives the evidence CDSCO requires for AI software, including the composition of training and test data and performance across sub-populations [3].`]),
  P([B_("Launch. "), `After registration, launch as worklist triage software for the pilot centres on the clinic plan, then expand through teleradiology partners, who can offer it to hundreds of centres at once, and integrations with PACS vendors.`]),
  P([B_("Growth and sustainability. "), `Add MRI sequences and 3D volumes, train on Indian data from partner centres (with consent and data-sharing agreements), extend sleep staging to home EEG headbands, and add the full 153-night Sleep-EDF data and other public sets. Revenue per centre grows with study volume; models improve with more data; and the audit trail and patient-controlled sharing make Neurova easier to approve with hospital compliance teams than tools without them.`]),
  P(`The one-page Business Model Canvas accompanying this report summarises the plan.`),
);

// --- 10 Ethics
kids.push(H1("10. Ethics, privacy and fairness"),
  P(`Both datasets are public and licensed for reuse with attribution: the MRI collection under CC BY 4.0 and Sleep-EDF under the Open Data Commons Attribution licence. Neither contains identifying information. Demo accounts in the app use invented names and invented results.`),
  P(`The main ethical risk is that a confident output is treated as a diagnosis. The app counters this by calibrating confidence, showing every class's probability and an uncertainty level, flagging close calls in the summary, showing where the model looked, warning when a scan is unlike the training data, and labelling every result as educational and not a diagnosis. Every result also waits for a clinician's review, and the administrator's overview shows how often clinicians agreed with the model. In a clinical product the model would only reorder a radiologist's worklist, never report on its own.`),
  P(`Fairness cannot be checked properly with this data, because the MRI collection does not record age, sex, ethnicity, scanner or hospital, and Sleep-EDF covers healthy adults of European ancestry recorded in the 1990s. Performance on Indian patients, children, older adults with other brain conditions, or different scanners is unknown. The pilot is designed to measure this before any clinical use. Privacy is protected by consent-based access, audit logging and minimal data collection, and a production system would follow the Digital Personal Data Protection Act 2023.`),
);

// --- 11 Conclusion
kids.push(H1("11. Conclusion and future work"),
  P(`Neurova shows a complete deep learning product rather than a model in a notebook: a problem with a real bottleneck, two models trained and evaluated honestly on held-out data, a working app around them with sign-in, consent, clinician review and audit, a free public deployment, and a business plan with a cautious path to clinical use. The most useful technical finding was the duplicate problem in a widely used MRI dataset; removing duplicates before splitting is a simple step that changes how much any reported accuracy can be trusted.`),
  P(`The next steps are, in order: train the sleep model on all 153 Sleep-EDF nights; move from single slices to 3D MRI volumes; add an external test set from a different hospital; run the silent pilot; and add a learned tumour segmentation model so the app can report tumour size from a proper mask rather than a heatmap.`),
  H1("Individual contribution"),
  P(`This was an individual project. ${facts.student_name} identified the problem, carried out the market analysis, designed the system, wrote the data pipeline, trained and evaluated both models, built the frontend and backend, deployed the application, wrote the business plan and prepared this report and the presentation.`),
);

// --- References
const refs = [
  "Radiology and Imaging Association of India workforce figures, as summarised by 5C Network, \"Radiologist shortage in India\" (2025). https://www.5cnetwork.com/resources/radiologist-shortage-india",
  "GBD 2016 Brain and Other CNS Cancer Collaborators. Global, regional, and national burden of brain and other CNS cancer, 1990-2016. The Lancet Neurology 18(4):376-393, 2019. doi:10.1016/S1474-4422(18)30468-X",
  "Central Drugs Standard Control Organisation, guidance on medical device software, as summarised by Emergo by UL, \"India CDSCO finalizes guidance on medical device software\" (July 2026). Medical Devices Rules, 2017.",
  "BioSpectrum India, \"Healthcare AI innovator Qure.ai completes $65 M Series D funding round\" (26 September 2024).",
  "A. Supratak, H. Dong, C. Wu, Y. Guo. DeepSleepNet: a model for automatic sleep stage scoring based on raw single-channel EEG. IEEE Trans. Neural Systems and Rehabilitation Engineering 25(11):1998-2008, 2017.",
  "A. Supratak, Y. Guo. TinySleepNet: an efficient deep learning model for sleep stage scoring based on raw single-channel EEG. IEEE EMBC, 2020.",
  "MarketsandMarkets, India artificial intelligence in medical diagnostics market analysis (accessed November 2026).",
  "J. Cheng et al. Enhanced performance of brain tumor classification via tumor region augmentation and partition. PLoS ONE 10(10):e0140381, 2015.",
  "M. Nickparvar. Brain Tumor MRI Dataset. Kaggle, 2021; also IEEE DataPort, doi:10.21227/1jny-g144, CC BY 4.0.",
  "M. Tan, Q. Le. EfficientNet: rethinking model scaling for convolutional neural networks. ICML, 2019.",
  "K. He, X. Zhang, S. Ren, J. Sun. Deep residual learning for image recognition. CVPR, 2016.",
  "R. R. Selvaraju et al. Grad-CAM: visual explanations from deep networks via gradient-based localization. ICCV, 2017.",
  "C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. On calibration of modern neural networks. ICML, 2017.",
  "Y. Gal, Z. Ghahramani. Dropout as a Bayesian approximation: representing model uncertainty in deep learning. ICML, 2016.",
  "B. Kemp, A. H. Zwinderman, B. Tuk, H. A. C. Kamphuisen, J. J. L. Oberyé. Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. IEEE Trans. Biomedical Engineering 47(9):1185-1194, 2000.",
  "A. L. Goldberger et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation 101(23):e215-e220, 2000. Sleep-EDF Expanded, https://physionet.org/content/sleep-edfx/1.0.0/, ODC-By 1.0.",
  "I. Goodfellow, J. Shlens, C. Szegedy. Explaining and harnessing adversarial examples. ICLR, 2015.",
  "Hugging Face, Spaces overview: hardware and pricing (accessed November 2026). https://huggingface.co/docs/hub/spaces-overview",
  "K. Lee, K. Lee, H. Lee, J. Shin. A simple unified framework for detecting out-of-distribution samples and adversarial attacks. NeurIPS, 2018.",
  "Software: PyTorch, EfficientNet-PyTorch (L. Melas-Kyriazi, Apache 2.0), ONNX and ONNX Runtime, FastAPI, SQLAlchemy, scikit-learn, MNE-Python, OpenCV.",
];
kids.push(H1("References"), ...refs.map((r, i) => new Paragraph({ spacing: { after: 50, line: 250 }, indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: `[${i + 1}]\t${r}`, font: FONT, size: 18 })] })));

const doc = new Document({
  creator: facts.student_name, title: "Neurova: CDS301 project report",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 32, bold: true, color: "2F2620" }, paragraph: { spacing: { before: 360, after: 140 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 25, bold: true, color: "3E6447" }, paragraph: { spacing: { before: 220, after: 100 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1300, bottom: 1300, left: 1350, right: 1350 } } }, children: kids }],
});
Packer.toBuffer(doc).then((buf) => {
  const out = path.join(OUTDIR, "Neurova_CDS301_Report.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, (buf.length / 1e6).toFixed(1), "MB");
});
