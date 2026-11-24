/*
  Neurova front end. Plain JavaScript, no framework and no build step.

  How it works: the address after the # decides which page is shown (for example
  #/study/3). Each page is a function below that asks the server for data and
  writes HTML into <main id="view">.

  Sections:  helpers  ·  sign-in  ·  page router  ·  results  ·  one study  ·  PDF report  ·
             new analysis  ·  sharing  ·  clinician (triage, patients)  ·  admin  ·  model facts
*/

// ------------------------------------------------------------------ helpers
const $ = (id) => document.getElementById(id);
const view = $("view");
let me = null;                         // the signed-in user
let modelFacts = null;                 // cached answer of /api/models
let cameFrom = null;                   // "triage" or "patient": where a clinician opened a result from

/** Call the server. Adds the sign-in token and turns error replies into exceptions. */
async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  const token = sessionStorage.getItem("token");
  if (token) headers.Authorization = "Bearer " + token;
  if (options.json) { headers["Content-Type"] = "application/json"; options.body = JSON.stringify(options.json); }
  const reply = await fetch("/api" + path, { ...options, headers });
  const data = await reply.json().catch(() => ({}));
  if (reply.status === 401 && me) { signOut(); throw new Error("Please sign in again."); }
  if (!reply.ok) {
    const d = data.detail;
    throw new Error(typeof d === "string" ? d : Array.isArray(d) ? "Please check what you typed: " + d.map((x) => x.loc.at(-1)).join(", ") + "." : "Something went wrong.");
  }
  return data;
}

/** Make text safe to put inside HTML (names and notes are typed by people). */
const esc = (text) => String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const when = (iso) => new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
const whenTime = (iso) => new Date(iso).toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
const pct = (fraction, digits = 1) => (fraction * 100).toFixed(digits) + "%";
const show = (html) => { view.innerHTML = html; window.scrollTo(0, 0); };
const failed = (e) => show(`<p class="error">${esc(e.message)}</p>`);

function reviewPill(study) {
  if (study.review) return study.review.verdict === "agree"
    ? `<span class="pill good">Clinician agreed</span>` : `<span class="pill bad">Clinician disagreed</span>`;
  return `<span class="pill wait">Waiting for review</span>`;
}

function studyList(studies) {
  if (!studies.length) return `<div class="card empty">No results yet. Choose <b>New analysis</b> to run a model on a scan or a sleep recording.</div>`;
  return `<div class="list">${studies.map((s) => `
    <a class="item" href="#/study/${s.id}">
      <div class="icon ${s.kind}">${s.kind === "mri" ? "MRI" : "EEG"}</div>
      <div class="main"><b>${esc(s.headline)}</b><span>${esc(s.title)} · ${when(s.created)}</span></div>
      ${reviewPill(s)}
    </a>`).join("")}</div>`;
}

// ------------------------------------------------------------------ sign-in
function showSignIn() {
  $("app").hidden = true; $("signin").hidden = false;
  fetch("/api/demo").then((r) => r.json()).then((demo) => {
    if (!demo.accounts.length) return;
    $("demo").hidden = false;
    $("demoButtons").innerHTML = demo.accounts.map((a) =>
      `<button class="btn" data-email="${esc(a.email)}">${esc(a.name)} <span>${a.role}</span></button>`).join("");
    $("demoButtons").onclick = (e) => {
      const b = e.target.closest("button");
      if (b) enter("/login", { email: b.dataset.email, password: demo.password }, $("loginError"));
    };
  }).catch(() => {});
}

async function enter(path, body, errorBox) {
  errorBox.hidden = true;
  try {
    const reply = await api(path, { method: "POST", json: body });
    sessionStorage.setItem("token", reply.token);
    startApp(reply.user);
  } catch (e) { errorBox.textContent = e.message; errorBox.hidden = false; }
}

$("loginForm").onsubmit = (e) => { e.preventDefault(); enter("/login", { email: $("loginEmail").value, password: $("loginPassword").value }, $("loginError")); };
$("registerForm").onsubmit = (e) => { e.preventDefault(); enter("/register", { name: $("regName").value, email: $("regEmail").value, password: $("regPassword").value }, $("regError")); };
$("showRegister").onclick = (e) => { e.preventDefault(); $("loginForm").hidden = true; $("registerForm").hidden = false; };
$("showLogin").onclick = (e) => { e.preventDefault(); $("registerForm").hidden = true; $("loginForm").hidden = false; };
$("signOut").onclick = signOut;

function signOut() { sessionStorage.removeItem("token"); me = null; location.hash = ""; showSignIn(); }

// ------------------------------------------------------------------ page router
const MENU = {
  patient: [["#/results", "My results"], ["#/sharing", "Sharing"], ["#/models", "How the models perform"]],
  clinician: [["#/triage", "Triage"], ["#/patients", "My patients"], ["#/models", "How the models perform"]],
  admin: [["#/admin", "Overview"], ["#/activity", "Activity log"], ["#/models", "How the models perform"]],
};

function startApp(user) {
  me = user;
  $("signin").hidden = true; $("app").hidden = false;
  $("whoName").textContent = `${user.name} (${user.role})`;
  $("nav").innerHTML = MENU[user.role].map(([href, label]) => `<a href="${href}">${label}</a>`).join("");
  if (location.hash.length < 3) location.hash = MENU[user.role][0][0]; else route();
}

function route() {
  if (!me) return;
  const [, page, id] = location.hash.split("/");
  const pages = { results: pageResults, study: pageStudy, sharing: pageSharing, triage: pageTriage, patients: pagePatients, patient: pagePatient,
                  admin: pageAdmin, activity: pageActivity, models: pageModels };
  const home = MENU[me.role][0][0];
  document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("on", a.getAttribute("href") === "#/" + page));
  if (!pages[page]) { location.hash = home; return; }
  show(`<div class="busy"><div class="spinner"></div></div>`);
  pages[page](Number(id)).catch(failed);
}
window.onhashchange = route;

// ------------------------------------------------------------------ results (patient)
async function pageResults() {
  const [data, clinicians] = await Promise.all([api("/studies"), api("/clinicians")]);
  const sharing = clinicians.some((c) => c.sharing);
  show(`
    <div class="row spread"><div><h1>My results</h1><p class="lead">Every scan and sleep recording the models have analysed for you.</p></div>
      <button class="btn primary" id="newBtn">New analysis</button></div>
    ${sharing ? "" : `<p class="notice">Only you can see these results. <a href="#/sharing">Share them with a clinician</a> to have them reviewed.</p>`}
    ${studyList(data.studies)}`);
  $("newBtn").onclick = () => openNewAnalysis(null);
}

// ------------------------------------------------------------------ one study
async function pageStudy(id) {
  const [study, facts] = await Promise.all([api("/studies/" + id), getModelFacts()]);
  const r = study.result;
  const back = me.role === "patient" ? ["#/results", "My results"]
    : cameFrom === "triage" ? ["#/triage", "Triage"] : ["#/patient/" + study.patient.id, study.patient.name];
  show(`
    <a class="back" href="${back[0]}">← ${esc(back[1])}</a>
    <div class="card stack">
      <div class="row spread"><span class="small muted">${esc(study.title)} · ${when(study.created)}${me.role === "clinician" ? " · " + esc(study.patient.name) : ""}</span>
        <span class="row">${reviewPill(study)}<button class="btn" id="pdfBtn">Download PDF report</button></span></div>
      <div class="headline"><b>${esc(r.kind === "mri" ? r.label : study.headline)}</b>
        ${r.kind === "mri" ? `<div class="muted">${esc(r.confidence_text)} confidence</div>` : ""}</div>
      <p>${esc(r.summary)}</p>
    </div>
    ${r.kind === "mri" && r.familiarity && !r.familiarity.familiar ? `<p class="notice warn"><b>This scan is unlike the scans the model was trained on.</b> The model only knows four answers: glioma, meningioma, pituitary tumour and no tumour, all inside the brain. For anything else, such as a tumour of the eye socket or a different type of scan, its answer may mean little.</p>` : ""}
    ${r.kind === "mri" ? mriBody(r, facts.mri) : sleepBody(r, facts.sleep)}
    ${reviewCard(study)}
    ${me.role === "patient" ? `<p><button class="btn quiet danger" id="deleteBtn">Delete this result</button></p>` : ""}`);

  if ($("reviewForm")) $("reviewForm").onsubmit = async (e) => {
    e.preventDefault();
    const verdict = document.querySelector("input[name=verdict]:checked");
    if (!verdict) return;
    await api(`/studies/${id}/review`, { method: "POST", json: { verdict: verdict.value, note: $("reviewNote").value } }).catch(failed);
    if (cameFrom === "triage") location.hash = "#/triage";   // straight back to the reading list, minus this one
    else route();
  };
  $("pdfBtn").onclick = () => downloadPdf(id, $("pdfBtn"));
  if ($("deleteBtn")) $("deleteBtn").onclick = async () => {
    if (!confirm("Delete this result? This cannot be undone.")) return;
    await api("/studies/" + id, { method: "DELETE" });
    location.hash = "#/results";
  };
}

function mriBody(r, facts) {
  const deployed = facts.compare.find((m) => m.deployed);
  const check = (ok, title, text) => `<div class="check"><div class="dot ${ok ? "" : "warn"}">${ok ? "✓" : "!"}</div><div><b>${title}</b><p>${text}</p></div></div>`;
  return `
    <div class="card stack">
      <h2>What the model saw</h2>
      <div class="scans">
        <figure><img src="data:image/png;base64,${r.image}" alt="The scan, cropped to the brain"><figcaption>The scan, cropped to the brain.</figcaption></figure>
        <figure><img src="data:image/png;base64,${r.heatmap}" alt="Heatmap over the scan"><figcaption>Where the model looked (Grad-CAM). Red marks the areas that pushed it most towards its answer.</figcaption></figure>
      </div>
    </div>
    <div class="grid2">
      <div class="card stack">
        <h2>How likely is each class</h2>
        <div class="bars">${r.probabilities.map((p) => `
          <div class="bar"><span>${esc(p.label)}</span><div class="track"><div class="fill" style="width:${p.percent}%"></div></div><span class="val">${p.percent.toFixed(1)}%</span></div>`).join("")}</div>
      </div>
      <div class="card stack">
        <h2>Can this result be trusted?</h2>
        ${check(r.uncertainty.level === "Low", `Uncertainty: ${r.uncertainty.level.toLowerCase()}`, esc(r.uncertainty.note))}
        ${check(r.stability.held, r.stability.held ? "Stable under disturbance" : "Not stable under disturbance", esc(r.stability.verdict))}
        ${r.familiarity ? check(r.familiarity.familiar, r.familiarity.familiar ? "A familiar kind of scan" : "An unfamiliar kind of scan", esc(r.familiarity.note)) : ""}
        ${check(true, `Tested accuracy: ${pct(deployed.accuracy)}`, `Measured on ${facts.test_images} scans the model never saw in training.`)}
        <details><summary>Show the seven disturbance tests</summary>
          <table><tr><th>Disturbance</th><th>Answer</th><th class="num">Confidence</th></tr>
          ${r.stability.tests.map((t) => `<tr><td>${esc(t.test)}</td><td>${esc(t.prediction)}${t.changed ? " (changed)" : ""}</td><td class="num">${t.confidence}%</td></tr>`).join("")}</table>
        </details>
      </div>
    </div>`;
}

const STAGE_ORDER = ["W", "REM", "N1", "N2", "N3"];
const STAGE_LABEL = { W: "Wake", REM: "REM", N1: "N1 light", N2: "N2", N3: "N3 deep" };

/** Draw one or two hypnograms (model, and the expert if we have it) as an SVG picture. */
function hypnogram(r) {
  const rows = [["Model", r.stages]].concat(r.expert_stages ? [["Expert", r.expert_stages]] : []);
  const W = 800, left = 62, rowH = 96, pad = 36, n = r.stages.length, hours = n / 120;
  const x = (i) => left + (i / n) * (W - left - 8);
  let svg = "";
  rows.forEach(([name, stages], k) => {
    const top = pad + k * (rowH + pad);
    const y = (s) => top + STAGE_ORDER.indexOf(s) * (rowH / 4);
    STAGE_ORDER.forEach((s) => { svg += `<line class="grid" x1="${left}" x2="${W - 8}" y1="${y(s)}" y2="${y(s)}"/><text x="${left - 8}" y="${y(s) + 4}" text-anchor="end">${s}</text>`; });
    svg += `<text x="${left}" y="${top - 10}" style="font-weight:600">${name}</text>`;
    let last = null;
    stages.forEach((s, i) => {          // one short coloured line per epoch, coloured by its stage
      if (!s) return;
      if (last && last !== s) svg += `<line x1="${x(i)}" x2="${x(i)}" y1="${y(last)}" y2="${y(s)}" stroke="var(--ink-2)" stroke-width="0.6"/>`;
      svg += `<line x1="${x(i)}" x2="${x(i + 1)}" y1="${y(s)}" y2="${y(s)}" stroke="var(--${s})" stroke-width="4"/>`;
      last = s;
    });
  });
  const bottom = pad + rows.length * (rowH + pad) - 8;
  for (let h = 0; h <= hours; h += hours > 5 ? 1 : 0.5) svg += `<text x="${x(h * 120)}" y="${bottom + 4}" text-anchor="middle">${h} h</text>`;
  return `<svg class="hypno" viewBox="0 0 ${W} ${bottom + 10}" role="img" aria-label="Hypnogram: sleep stage over time">${svg}</svg>`;
}

function sleepBody(r, facts) {
  const total = r.epochs / 2;           // minutes
  const a = r.agreement;
  return `
    <div class="card stack">
      <h2>The night, epoch by epoch</h2>
      <p class="small muted">Each 30-second piece of the recording is given one stage. ${r.expert_stages ? "The expert row is a sleep technician's scoring of the same night, for comparison." : ""}</p>
      ${hypnogram(r)}
      <div class="legend">${STAGE_ORDER.map((s) => `<span><i style="background:var(--${s})"></i>${STAGE_LABEL[s]}</span>`).join("")}</div>
    </div>
    <div class="grid2">
      <div class="card stack">
        <h2>Time in each stage</h2>
        <div class="bars">${STAGE_ORDER.map((s) => `
          <div class="bar"><span>${STAGE_LABEL[s]}</span><div class="track"><div class="fill" style="width:${(r.minutes[s] / total) * 100}%;background:var(--${s})"></div></div><span class="val">${r.minutes[s].toFixed(0)} min</span></div>`).join("")}</div>
      </div>
      <div class="card stack">
        <h2>Can this result be trusted?</h2>
        <div class="tiles">
          ${a ? `<div class="tile"><b>${a.percent}%</b><span>agreement with the expert on this night</span></div>` : ""}
          <div class="tile"><b>${pct(facts.accuracy)}</b><span>accuracy on ${facts.test_people} people never seen in training</span></div>
          <div class="tile"><b>${r.unsure_epochs}</b><span>of ${r.epochs} epochs where the model was unsure</span></div>
        </div>
        <p class="small muted">Channel used: ${esc(r.channel)}. ${esc(r.channel_note || "")} ${esc(r.caveat)}</p>
      </div>
    </div>`;
}

function reviewCard(study) {
  const rv = study.review;
  const done = rv ? `<p><b>${esc(rv.by)}</b> ${rv.verdict === "agree" ? "agreed" : "disagreed"} with the model on ${when(rv.when)}.</p>
                     ${rv.note ? `<p class="muted">“${esc(rv.note)}”</p>` : ""}` : "";
  if (me.role === "clinician") return `
    <form class="card stack" id="reviewForm">
      <h2>Your review</h2>${done}
      <div class="choice">
        <label><input type="radio" name="verdict" value="agree" ${rv?.verdict === "agree" ? "checked" : ""} required>I agree with the model</label>
        <label><input type="radio" name="verdict" value="disagree" ${rv?.verdict === "disagree" ? "checked" : ""}>I disagree</label>
      </div>
      <label>Note for the patient (optional)<textarea id="reviewNote" rows="3" maxlength="2000">${esc(rv?.note || "")}</textarea></label>
      <div><button class="btn primary">${rv ? "Update review" : "Save review"}</button></div>
    </form>`;
  return `<div class="card stack"><h2>Clinician review</h2>
    ${done || `<p class="muted">No clinician has reviewed this yet. A model's answer is a second opinion, not a diagnosis. <a href="#/sharing">Choose who can see your results</a>.</p>`}</div>`;
}

// ------------------------------------------------------------------ PDF report
/** Fetch the PDF with the sign-in token, then hand it to the browser as a download. */
async function downloadPdf(id, button) {
  const label = button.textContent;
  button.disabled = true; button.textContent = "Preparing…";
  try {
    const reply = await fetch(`/api/studies/${id}/report.pdf`, { headers: { Authorization: "Bearer " + sessionStorage.getItem("token") } });
    if (!reply.ok) throw new Error("The report could not be made. Please try again.");
    const link = Object.assign(document.createElement("a"), { href: URL.createObjectURL(await reply.blob()), download: `neurova-report-${id}.pdf` });
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 10000);
  } catch (e) { alert(e.message); }
  button.disabled = false; button.textContent = label;
}

// ------------------------------------------------------------------ new analysis
let newFor = null;                     // patient id when a clinician uploads for a patient

async function openNewAnalysis(patientId) {
  newFor = patientId;
  $("uploadError").hidden = true; $("dialogBody").hidden = false; $("dialogBusy").hidden = true;
  $("newDialog").showModal();
  const samples = await api("/samples");
  $("sampleGrid").innerHTML = samples.map((s) => s.kind === "mri"
    ? `<button data-file="${s.file}" title="${esc(s.name)}" class="${s.hard ? "hard" : ""}"><img src="/samples/${s.file}" alt="${esc(s.name)}">${s.hard ? "<span>model gets this wrong</span>" : ""}</button>`
    : `<button class="night" data-file="${s.file}">${esc(s.name)}</button>`).join("");
}

async function runAnalysis(send) {
  $("uploadError").hidden = true; $("dialogBody").hidden = true; $("dialogBusy").hidden = false;
  try {
    const reply = await send();
    $("newDialog").close();
    location.hash = "#/study/" + reply.id;
  } catch (e) {
    $("dialogBody").hidden = false; $("dialogBusy").hidden = true;
    $("uploadError").textContent = e.message; $("uploadError").hidden = false;
  }
}

function uploadFile(file) {
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  if (newFor) form.append("patient_id", newFor);
  runAnalysis(() => api("/studies", { method: "POST", body: form }));
}

$("fileInput").onchange = (e) => { uploadFile(e.target.files[0]); e.target.value = ""; };
$("drop").ondragover = (e) => { e.preventDefault(); $("drop").classList.add("over"); };
$("drop").ondragleave = () => $("drop").classList.remove("over");
$("drop").ondrop = (e) => { e.preventDefault(); $("drop").classList.remove("over"); uploadFile(e.dataTransfer.files[0]); };
$("sampleGrid").onclick = (e) => {
  const b = e.target.closest("button");
  if (b) runAnalysis(() => api("/studies/sample", { method: "POST", json: { file: b.dataset.file, patient_id: newFor } }));
};
$("closeDialog").onclick = () => $("newDialog").close();

// ------------------------------------------------------------------ sharing (patient)
async function pageSharing() {
  const clinicians = await api("/clinicians");
  show(`
    <div><h1>Sharing</h1><p class="lead">You decide which clinicians can see your results. You can stop sharing at any time, and every time someone opens your results it is written to a log.</p></div>
    ${clinicians.length ? `<div class="list">${clinicians.map((c) => `
      <div class="item"><div class="icon">Dr</div>
        <div class="main"><b>${esc(c.name)}</b><span>${c.sharing ? "Can see your results and review them" : "Cannot see your results"}</span></div>
        <button class="btn ${c.sharing ? "" : "primary"}" data-id="${c.id}" data-on="${c.sharing}">${c.sharing ? "Stop sharing" : "Share"}</button>
      </div>`).join("")}</div>` : `<div class="card empty">No clinicians have an account yet.</div>`}`);
  view.onclick = async (e) => {
    const b = e.target.closest("button[data-id]");
    if (!b) return;
    view.onclick = null;
    await api("/share/" + b.dataset.id, { method: b.dataset.on === "true" ? "DELETE" : "POST" }).catch(failed);
    route();
  };
}

// ------------------------------------------------------------------ clinician
const GROUPS = [
  ["Likely tumour", "bad", "The model's answer is a tumour. Most confident first."],
  ["Model unsure", "wait", "The model's answer may be wrong: it is uncertain, unstable, or the scan is unlike its training data."],
  ["Routine", "good", "Confident 'no tumour' answers and sleep recordings."],
];

async function pageTriage() {
  cameFrom = "triage";
  const rows = await api("/triage");
  const section = ([name, tone, explain]) => {
    const items = rows.filter((r) => r.group === name);
    if (!items.length) return "";
    return `<section class="stack">
      <div class="row"><h2>${name}</h2><span class="pill ${tone}">${items.length}</span></div>
      <p class="small muted">${explain}</p>
      <div class="list">${items.map((s) => `
        <a class="item" href="#/study/${s.id}">
          <div class="icon ${s.kind}">${s.kind === "mri" ? "MRI" : "EEG"}</div>
          <div class="main"><b>${esc(s.patient_name)}: ${esc(s.headline)}</b>
            <span>${esc(s.title)} · waiting since ${whenTime(s.created)}</span>
            <div class="reasons">${s.reasons.map((x) => `<span class="pill ${tone}">${esc(x)}</span>`).join("")}</div></div>
        </a>`).join("")}</div></section>`;
  };
  show(`
    <div><h1>Triage</h1><p class="lead">Every result waiting for your review, across all your patients, with the most urgent first. Open one, record whether you agree, and you come straight back here.</p></div>
    ${rows.length ? GROUPS.map(section).join("") : `<div class="card empty">Nothing is waiting for review. Well done.</div>`}`);
}

async function pagePatients() {
  const patients = await api("/patients");
  show(`
    <div><h1>My patients</h1><p class="lead">Patients who have chosen to share their results with you. Those with results waiting for your review come first.</p></div>
    ${patients.length ? `<div class="list">${patients.map((p) => `
      <a class="item" href="#/patient/${p.id}"><div class="icon">${esc(p.name.split(" ").map((w) => w[0]).slice(0, 2).join(""))}</div>
        <div class="main"><b>${esc(p.name)}</b><span>${p.studies} result${p.studies === 1 ? "" : "s"}</span></div>
        ${p.waiting ? `<span class="pill wait">${p.waiting} waiting for review</span>` : `<span class="pill good">All reviewed</span>`}
      </a>`).join("")}</div>` : `<div class="card empty">No patient is sharing results with you yet. Patients choose this on their Sharing page.</div>`}`);
}

async function pagePatient(id) {
  cameFrom = "patient";
  const data = await api("/studies?patient_id=" + id);
  show(`
    <a class="back" href="#/patients">← My patients</a>
    <div class="row spread"><div><h1>${esc(data.patient.name)}</h1><p class="lead">Open a result to see the model's reasoning and record whether you agree.</p></div>
      <button class="btn primary" id="newBtn">New analysis</button></div>
    ${studyList(data.studies)}`);
  $("newBtn").onclick = () => openNewAnalysis(id);
}

// ------------------------------------------------------------------ admin
async function pageAdmin() {
  const data = await api("/admin/overview");
  const c = data.counts;
  show(`
    <div><h1>Overview</h1><p class="lead">You manage accounts and can see what happens in the system, but an administrator cannot open any patient's results.</p></div>
    <div class="tiles">
      <div class="tile"><b>${c.patients}</b><span>patients</span></div>
      <div class="tile"><b>${c.clinicians}</b><span>clinicians</span></div>
      <div class="tile"><b>${c.studies}</b><span>analyses run</span></div>
      <div class="tile"><b>${c.reviewed}</b><span>reviewed by a clinician</span></div>
      <div class="tile"><b>${c.reviewed ? Math.round((c.clinician_agreed / c.reviewed) * 100) + "%" : "–"}</b><span>of reviews agreed with the model</span></div>
    </div>
    <div class="card stack"><h2>Accounts</h2>
      <div class="scroll"><table><tr><th>Name</th><th>Email</th><th>Role</th><th>Joined</th><th>Status</th><th></th></tr>
      ${data.users.map((u) => `<tr class="${u.active ? "" : "off"}"><td>${esc(u.name)}</td><td>${esc(u.email)}</td><td>${u.role}</td><td>${when(u.joined)}</td>
        <td>${u.active ? `<span class="pill good">Active</span>` : `<span class="pill bad">Disabled</span>`}</td>
        <td><button class="btn quiet" data-edit="${u.id}">Edit</button></td></tr>`).join("")}</table></div>
    </div>
    <form class="card stack" id="clinicianForm"><h2>Add a clinician</h2>
      <p class="small muted">Patients register themselves. Clinician accounts are created here, so that only real clinicians can be given access.</p>
      <div class="grid2">
        <label>Full name <input id="cName" required minlength="2"></label>
        <label>Email <input id="cEmail" type="email" required></label>
        <label>Temporary password <input id="cPassword" required minlength="10"><span class="hint">At least 10 characters, with a letter and a number.</span></label>
      </div>
      <p class="error" id="cError" hidden></p>
      <div><button class="btn primary">Create clinician account</button></div>
    </form>`);
  view.onclick = (e) => {
    const b = e.target.closest("button[data-edit]");
    if (b) openEditAccount(data.users.find((u) => u.id === Number(b.dataset.edit)));
  };
  $("clinicianForm").onsubmit = async (e) => {
    e.preventDefault();
    try {
      await api("/admin/clinicians", { method: "POST", json: { name: $("cName").value, email: $("cEmail").value, password: $("cPassword").value } });
      route();
    } catch (err) { $("cError").textContent = err.message; $("cError").hidden = false; }
  };
}

/** The admin's "Edit account" window: rename, reset the password, disable or re-enable. */
function openEditAccount(u) {
  let changed = false;
  const say = (ok, message) => {
    $("editError").hidden = ok; $("editDone").hidden = !ok;
    (ok ? $("editDone") : $("editError")).textContent = message;
  };
  const paint = () => {
    $("editTitle").textContent = "Edit " + u.name;
    $("editWho").textContent = `${u.email} · ${u.role}`;
    $("editName").value = u.name;
    $("activeText").textContent = u.active
      ? "Disabling signs this person out and stops them signing in. Their results and the activity log are kept."
      : "This account is disabled. Enabling it lets the person sign in again.";
    $("activeBtn").textContent = u.active ? "Disable account" : "Enable account";
    $("activeBtn").className = "btn" + (u.active ? " danger" : "");
  };
  const save = async (changes, message) => {
    try {
      Object.assign(u, await api("/admin/users/" + u.id, { method: "PATCH", json: changes }));
      changed = true; paint(); say(true, message);
    } catch (err) { say(false, err.message); }
  };
  $("editError").hidden = $("editDone").hidden = true;
  $("editPassword").value = "";
  paint();
  $("nameForm").onsubmit = (e) => { e.preventDefault(); save({ name: $("editName").value }, "Name saved."); };
  $("passwordForm").onsubmit = (e) => {
    e.preventDefault();
    save({ password: $("editPassword").value }, "Password reset. Tell them the new one through a safe channel.");
    $("editPassword").value = "";
  };
  $("activeBtn").onclick = () => save({ active: !u.active }, u.active ? "Account disabled." : "Account enabled.");
  $("closeEdit").onclick = () => $("editDialog").close();
  $("editDialog").onclose = () => { if (changed) route(); };
  $("editDialog").showModal();
}

async function pageActivity() {
  const rows = await api("/admin/audit");
  show(`
    <div><h1>Activity log</h1><p class="lead">Every sign-in, analysis, review, PDF download, change to sharing and change to an account, newest first. Entries cannot be edited or removed.</p></div>
    <div class="card scroll"><table><tr><th>When</th><th>Who</th><th>What</th><th>Detail</th></tr>
      ${rows.map((a) => `<tr><td>${whenTime(a.time)}</td><td>${esc(a.who)} <span class="muted">(${esc(a.role)})</span></td><td>${esc(a.action)}</td><td class="muted">${esc(a.detail)}</td></tr>`).join("")}
    </table></div>`);
}

// ------------------------------------------------------------------ model facts
async function getModelFacts() { return modelFacts || (modelFacts = await api("/models")); }

const FIGURE_CAPTION = {
  confusion_matrices: "Confusion matrices on the test set. Each row is the true class and each column is what the model answered.",
  robustness: "How often a small disturbance changes the answer. The standard models are fragile; the deployed one is not.",
  familiarity: "The unfamiliar-scan check. Scans to the right of the dashed line are flagged. Most of the model's mistakes (red marks) fall there.",
  calibration: "Calibration. After temperature scaling, a stated confidence of 90% is right about 90% of the time.",
  examples_correct: "Correct answers with their Grad-CAM heatmaps.",
  examples_incorrect: "The model's most confident mistakes. Looking at the heatmap often shows why it went wrong.",
  confusion_matrix: "Sleep staging confusion matrix on two people the model had never seen.",
  hypnogram_test_subject: "Expert scoring and model output for one held-out night.",
};
const figures = (group, names) => names.map((n) => `<figure class="figure"><img src="/results/${group}/${n}.png" alt="" loading="lazy"><figcaption>${FIGURE_CAPTION[n]}</figcaption></figure>`).join("");
const classTable = (rows) => `<div class="scroll"><table><tr><th>Class</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th></tr>
  ${rows.map((r) => `<tr><td>${esc(r.label)}</td><td class="num">${pct(r.precision)}</td><td class="num">${pct(r.recall)}</td><td class="num">${pct(r.f1)}</td></tr>`).join("")}</table></div>`;

function sleepVerdict(models) {
  const gru = models.find((m) => m.key === "gru"), trf = models.find((m) => m.key === "transformer");
  const first = `The Transformer's first run scored ${pct(trf.runs[0].test_accuracy)}, above the GRU's ${pct(gru.runs[0].test_accuracy)}. `;
  return first + (gru.test_accuracy.mean > trf.test_accuracy.mean
    ? `Over three runs that did not hold: the GRU was better on average and much steadier, while the Transformer ranged from ${pct(trf.test_accuracy.min, 0)} to ${pct(trf.test_accuracy.max, 0)}. With only 20 nights to learn from, the smaller GRU is the safer choice, so the app keeps it.`
    : `Over three runs the Transformer held up, but the app keeps the GRU, which was chosen on validation data first and is half the size.`);
}

async function pageModels() {
  const f = await getModelFacts();
  const best = f.mri.compare.find((m) => m.deployed);
  show(`
    <div><h1>How the models perform</h1><p class="lead">Both models were tested on data they never saw during training. These are those results, including where the models fall short.</p></div>

    <div class="card stack">
      <h2>Brain MRI: glioma, meningioma, pituitary tumour or no tumour</h2>
      <div class="tiles">
        <div class="tile"><b>${pct(best.accuracy)}</b><span>accuracy on ${f.mri.test_images} test scans</span></div>
        <div class="tile"><b>${best.macro_f1.toFixed(3)}</b><span>macro-F1 (all classes weighted equally)</span></div>
        <div class="tile"><b>${f.mri.duplicates_removed}</b><span>duplicate images removed before splitting</span></div>
      </div>
      <p>Three models were compared. Starting from a network pretrained on ImageNet (transfer learning) beats training a small network from scratch by about ten points. The two EfficientNets are equally accurate, but the standard one changes its answer on ${f.mri.compare[1].changed_by_faint_noise.toFixed(0)}% of scans when noise too faint to see is added. Training with “fingerprint scrub” augmentation fixed that, so that model is the one this app uses.</p>
      <div class="scroll"><table><tr><th>Model</th><th class="num">Accuracy</th><th class="num">Macro-F1</th><th class="num">Parameters</th><th class="num">Changed by faint noise</th></tr>
        ${f.mri.compare.map((m) => `<tr><td>${esc(m.name)} ${m.deployed ? `<span class="pill good">in use</span>` : ""}</td><td class="num">${pct(m.accuracy)}</td><td class="num">${m.macro_f1.toFixed(3)}</td><td class="num">${m.parameters_m.toFixed(1)} M</td><td class="num">${m.changed_by_faint_noise.toFixed(1)}%</td></tr>`).join("")}</table></div>
      <p><b>What it cannot do.</b> The model knows only four answers, all for tumours inside the brain. Shown an eye-socket tumour it was never trained on, it answered meningioma or no tumour. The app therefore measures how far each scan is from the training scans and warns when one is unfamiliar. That check flags ${f.mri.familiarity.test_mistakes_flagged} of the model's ${f.mri.familiarity.test_mistakes} test mistakes and ${f.mri.familiarity.ordinary_test_scans_flagged_percent}% of the scans it answers correctly. It is a warning, not a guarantee: it catches only ${f.mri.familiarity.upside_down_scans_flagged_percent}% of scans turned upside down.</p>
      <h3>Per class, for the model in use</h3>
      ${classTable(f.mri.per_class)}
      ${figures("mri", f.mri.figures)}
    </div>

    <div class="card stack">
      <h2>Sleep EEG: Wake, N1, N2, N3 or REM for every 30 seconds</h2>
      <div class="tiles">
        <div class="tile"><b>${pct(f.sleep.accuracy)}</b><span>accuracy on ${f.sleep.test_people} people never seen in training</span></div>
        <div class="tile"><b>${f.sleep.kappa.toFixed(2)}</b><span>Cohen's kappa (two human scorers reach about 0.75 to 0.8)</span></div>
        <div class="tile"><b>${f.sleep.macro_f1.toFixed(2)}</b><span>macro-F1</span></div>
      </div>
      <p>The model is weakest on N1, a short in-between stage that human scorers also disagree about, and it sometimes calls quiet wakefulness N1.</p>
      <h3>GRU or Transformer?</h3>
      <p>The part that reads the 20 epochs in a row was also built as a Transformer encoder, with everything else identical. Each version was trained three times from different random starting weights, because with only two test people a single run can be lucky by a few points. The table shows the average and, in brackets, the lowest and highest of the three runs.</p>
      <div class="scroll"><table><tr><th>Sequence model</th><th class="num">Accuracy</th><th class="num">Kappa</th><th class="num">N1 F1</th><th class="num">Parameters</th></tr>
        ${f.sleep.compare.map((m) => `<tr><td>${esc(m.name)} ${m.deployed ? `<span class="pill good">in use</span>` : ""}</td>
          <td class="num">${pct(m.test_accuracy.mean)} <span class="muted">(${pct(m.test_accuracy.min, 0)}–${pct(m.test_accuracy.max, 0)})</span></td>
          <td class="num">${m.test_kappa.mean.toFixed(2)} <span class="muted">(${m.test_kappa.min.toFixed(2)}–${m.test_kappa.max.toFixed(2)})</span></td>
          <td class="num">${pct(m.n1_f1.mean, 0)}</td><td class="num">${m.params_m.toFixed(2)} M</td></tr>`).join("")}</table></div>
      <p>${sleepVerdict(f.sleep.compare)}</p>
      ${classTable(f.sleep.per_class)}
      ${figures("sleep", f.sleep.figures)}
    </div>`);
}

// ------------------------------------------------------------------ start
(async function start() {
  if (sessionStorage.getItem("token")) {
    try { startApp(await api("/me")); return; } catch { sessionStorage.removeItem("token"); }
  }
  showSignIn();
})();
