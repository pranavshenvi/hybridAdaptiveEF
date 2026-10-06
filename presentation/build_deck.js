// PercEF project review deck. Run from this folder:  node build_deck.js  ->  PercEF_review.pptx
// Structure: problem statement -> existing methodology -> our method -> novelty -> results.
// Every number comes from updateAsOf290926.md / updateAsOf061026.md / the result files.
// Edit freely in PowerPoint afterwards; every object is named (Home > Arrange > Selection Pane).
const path = require("path");
const pptxgen = require(path.join(__dirname, "..", "poster", "node_modules", "pptxgenjs"));
const { applyTheme } = require(process.env.PPTX_SKILL ||
  "C:/Users/adebeo/.claude/skills/synced/81f1b1b5-7c94-4926-91c8-fca4a2a070f6_ff6d1026-246b-4214-87da-3cc36522d9da/pptx/scripts/apply_theme.js");

const OUT = process.argv[2] || "PercEF_review.pptx";
const FIG = path.join(__dirname, "..", "server_results", "paper_out_20261006_145536");
const LOGO = path.join(__dirname, "..", "poster", "assets", "logo_pes.png");

// PES template colours, shared with the poster; method colours as in the paper's figures
const THEME = {
  name: "PercEF", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "1F2933", lt1: "FFFFFF", dk2: "2D3A82", lt2: "EEF3F5",
            accent1: "1597AB", accent2: "485B6C", accent3: "2A78D6", accent4: "EB6834",
            accent5: "1BAF7A", accent6: "8A8984", hlink: "1597AB", folHlink: "485B6C" },
};
const HEX = { ours: "2A78D6", ada: "EB6834", grey: "8A8984", grid: "E2E6EA", ink: "1F2933", muted: "5B6670" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";                          // 13.333 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "PercEF: Exploiting Empirical Percentiles for Adaptive HNSW Search";
pres.author = "Pranav Shenvi";
const C = pres.SchemeColor;

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "DARK", background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.2, w: 11.7, h: 1.9, fontSize: 36, bold: true,
        color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.3, w: 11.7, h: 1.6, fontSize: 18,
        color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 32, bold: true,
        color: C.text2, valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "PercEF  ·  PES University", options: { x: 0.6, y: 7.0, w: 6, h: 0.3, fontSize: 10, color: C.accent6, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent6 },
});

// ---------------------------------------------------------------- helpers
function card(s, x, y, w, h, head, body, opts = {}) {
  const fill = opts.fill || C.background2;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill },
    line: { color: fill, width: 0 }, objectName: `Card: ${head}` });
  s.addText(head, { x: x + 0.25, y: y + 0.2, w: w - 0.5, h: 0.5, fontSize: opts.headSize || 18, bold: true,
    color: opts.headColor || C.text2, margin: 0, valign: "top", isTextBox: true, objectName: `Card head: ${head}` });
  s.addText(body, { x: x + 0.25, y: y + 0.75, w: w - 0.5, h: h - 0.95, fontSize: opts.bodySize || 15,
    color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: `Card body: ${head}` });
}
function bullets(items, size = 16) {
  return items.map((t, i) => (typeof t === "string"
    ? { text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 8, fontSize: size } }
    : { text: t.text, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 8, fontSize: size, bold: !!t.bold } }));
}
function stat(s, x, y, w, big, small, color) {
  s.addText(big, { x, y, w, h: 0.75, fontSize: 40, bold: true, color: color || C.accent1, margin: 0,
    fontFace: THEME.headFontFace, isTextBox: true, objectName: `Stat: ${big}` });
  s.addText(small, { x, y: y + 0.75, w, h: 0.7, fontSize: 14, color: C.text1, margin: 0, valign: "top",
    isTextBox: true, objectName: `Stat label: ${big}` });
}
function box(s, x, y, w, h, text, kind, name) {          // diagram node
  const fill = kind === "key" ? C.accent1 : kind === "warn" ? C.accent4 : C.background2;
  s.addText(text, { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.08, x, y, w, h, fill: { color: fill },
    line: { color: kind === "plain" ? C.accent6 : fill, width: 1 }, fontSize: 14, bold: kind === "key" || kind === "warn",
    color: kind === "key" || kind === "warn" ? C.background1 : C.text1, align: "center", valign: "middle", margin: 4,
    objectName: `Node: ${name}` });
}
function arrow(s, x1, y1, x2, y2, name) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipH: x2 < x1, flipV: y2 < y1, line: { color: HEX.muted, width: 1.75, endArrowType: "triangle" }, objectName: `Arrow: ${name}` });
}
function table(s, rows, x, y, w, colW, name, size = 13) {
  const head = rows[0].map((t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.accent2 } } }));
  const body = rows.slice(1).map((r, i) => r.map((t) => ({ text: String(t),
    options: { fill: { color: i % 2 ? C.background1 : C.background2 }, color: C.text1 } })));
  s.addTable([head, ...body], { x, y, w, colW, fontSize: size, fontFace: THEME.bodyFontFace, margin: 0.06,
    border: { type: "solid", pt: 0.5, color: "D5DCE1" }, valign: "middle", objectName: name });
}

// ================================================================= 1. title
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Introduction" });
s.addText("PercEF: Exploiting Empirical Percentiles for Adaptive HNSW Search Beyond the Gaussian Assumption", { placeholder: "title" });
s.addText([
  { text: "Project review  ·  Department of Computer Science and Engineering, PES University", options: { breakLine: true } },
  { text: "Team: [Member 1], [Member 2], [Member 3], [Member 4]   ·   Guide: [Guide name]" },
], { placeholder: "body" });
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.8, y: 0.6, w: 3.0, h: 1.6, rectRadius: 0.1, fill: { color: C.background1 },
  line: { color: C.background1, width: 0 }, objectName: "Logo backing" });
s.addImage({ path: LOGO, x: 0.95, y: 0.72, w: 2.7, h: 2.7 * 373 / 727, objectName: "PES logo" });
s.addNotes("PercEF is a fix for adaptive search in vector databases. Ada-ef, a SIGMOD 2026 method, chooses the search effort per query by assuming the data looks like a bell curve. We tested that assumption, found where it fails, and built a method that works either way.");

// ================================================================= 2. problem statement
pres.addSection({ title: "Problem statement" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem statement" });
s.addText("One ef for every query does not fit", { placeholder: "title" });
s.addText("Vector databases answer a query with an HNSW graph search. One setting, ef, decides how much of the graph each search explores; most systems use the same ef for every query.",
  { x: 0.6, y: 1.35, w: 12.1, h: 0.9, fontSize: 18, color: C.text1, margin: 0, isTextBox: true, objectName: "Problem intro" });
card(s, 0.6, 2.45, 3.85, 2.2, "Easy query", "Finds its neighbours with a small ef.\nA large shared ef wastes work on it.\n→ over-searching");
card(s, 4.75, 2.45, 3.85, 2.2, "Hard query", "Needs a large ef.\nA small shared ef leaves it short of its neighbours.\n→ under-searching", { headColor: C.accent4 });
card(s, 8.9, 2.45, 3.85, 2.2, "Goal", "Choose ef per query: meet the target recall (95%), help the worst queries, and cost no more than one well-tuned ef.", { fill: C.accent1, headColor: C.background1 });
s.addText("Recall = share of a query's 100 true nearest neighbours that the search finds.", { x: 0.6, y: 4.95, w: 12.1, h: 0.5,
  fontSize: 15, italic: true, color: C.accent2, margin: 0, isTextBox: true, objectName: "Recall definition" });
s.addNotes("Easy queries get too much effort, hard queries too little. The target is a per-query ef that reaches 95% recall without costing more than the best single ef.");

// ================================================================= 3. existing methodology
pres.addSection({ title: "Existing methodology" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Existing methodology" });
s.addText("Existing work: two ways to adapt ef", { placeholder: "title" });
card(s, 0.6, 1.45, 5.95, 3.7, "Learned early termination",
  "LAET (SIGMOD 2020), DARTH (SIGMOD 2026)\n\n• A trained model watches each search and predicts when the target recall is reached\n• Needs training data and a model per dataset\n• Model calls run inside every search",
  { bodySize: 16 });
card(s, 6.85, 1.45, 5.9, 3.7, "Distribution-aware: Ada-ef",
  "Zhang & Miller, SIGMOD 2026\n\n• Models a query's similarities to the data as a bell curve (central limit theorem)\n• A short probe scores how hard the query is; a table maps score to ef\n• No training; claims to meet a target recall while avoiding under- and over-searching, and up to 4x lower latency than learned methods",
  { bodySize: 16, fill: "DCEFF3" });
s.addText("Our question: does Ada-ef's bell-curve assumption hold on real data?", { x: 0.6, y: 5.5, w: 12.1, h: 0.6,
  fontSize: 20, bold: true, color: C.accent1, margin: 0, isTextBox: true, objectName: "Bridge question" });
s.addNotes("Two families. Learned methods train a model per dataset and consult it during the search. Ada-ef needs no training: it assumes the similarities are Gaussian and reads query difficulty from a short probe. Its stated claims: meet a target recall, avoid under- and over-searching, and its Gaussian theory applies to learned embeddings.");

// ================================================================= 4. how Ada-ef works
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Existing methodology" });
s.addText("Ada-ef assumes a bell curve", { placeholder: "title" });
const fy = 1.75, fw = 2.25, fh = 1.2, fx = [0.6, 3.05, 5.5, 7.95, 10.4];
box(s, fx[0], fy, fw, fh, "Database mean and covariance", "plain", "stats");
box(s, fx[1], fy, fw, fh, "Predicted bell curve of similarities, per query", "plain", "normal");
box(s, fx[2], fy, fw, fh, "5 thresholds at the 0.1–0.5% closest", "warn", "thresholds");
box(s, fx[3], fy, fw, fh, "Probe 1,025 distances; count those under each threshold", "plain", "probe");
box(s, fx[4], fy, fw, fh, "Score → ef from a calibrated table", "plain", "table");
for (let i = 0; i < 4; i++) arrow(s, fx[i] + fw, fy + fh / 2, fx[i + 1], fy + fh / 2, `ada ${i}`);
s.addText("The thresholds come from the bell curve. If the real similarities are not bell-shaped, they sit in the wrong place, and the score can no longer tell easy queries from hard ones.",
  { x: 0.6, y: 3.45, w: 12.1, h: 0.9, fontSize: 18, color: C.text1, margin: 0, isTextBox: true, objectName: "Ada-ef weakness" });
card(s, 0.6, 4.55, 12.15, 1.7, "The gap", "Ada-ef's paper states this assumption but never measures it on data. Neither does any other ANN paper we found.",
  { fill: "FCE9DF", headColor: C.accent4, bodySize: 16 });
s.addNotes("The highlighted step is the weak point: the five thresholds are placed using the bell curve. When the data is not Gaussian, almost no probe distance falls under them, and every query gets the same score and the same ef.");

// ================================================================= 5. the diagnosis
pres.addSection({ title: "Our method" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Our method" });
s.addText("The assumption fails on 9 of 41 datasets", { placeholder: "title" });
s.addImage({ path: path.join(FIG, "fig1_ks_survey.png"), x: 0.6, y: 1.3, w: 5.4 * 1040 / 1089, h: 5.4, objectName: "KS survey figure" });
s.addText(bullets([
  { text: "KS statistic: how far each query's similarities are from Ada-ef's bell curve (0 = perfect fit)", bold: false },
  "Measured on 41 public datasets (ann-benchmarks, Big-ANN, VIBE, Ada-ef's sources)",
  { text: "9 clearly non-Gaussian, including SIFT, GIST, Fashion-MNIST, DeepImage, Deep1B", bold: true },
  "6 of the 9 are learned embeddings, inside Ada-ef's stated scope",
  "Data type does not predict it: SIFT-1M 0.126 vs SIFT-1B 0.029; Fashion-MNIST 0.073 vs MNIST 0.037",
  "There, Ada-ef stops adapting: on SIFT it gives every query the same ef",
]), { x: 6.45, y: 1.4, w: 6.3, h: 5.3, color: C.text1, valign: "top", margin: 0, isTextBox: true, objectName: "Diagnosis bullets" });
s.addNotes("Each dot is a dataset; higher KS means less Gaussian. Most modern text embeddings fit well, which is why Ada-ef works on its own datasets. Nine do not, and you cannot tell from the kind of data, so it has to be measured.");

// ================================================================= 6. PercEF
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Our method" });
s.addText("PercEF: measure the thresholds, don't assume them", { placeholder: "title" });
s.addText("Offline, once per dataset", { x: 0.6, y: 1.35, w: 6, h: 0.35, fontSize: 14, italic: true, color: C.accent6, margin: 0, isTextBox: true, objectName: "Lane offline" });
const py1 = 1.75, pw = 2.6, ph = 1.0, px = [0.6, 3.75, 6.9];
box(s, px[0], py1, pw, ph, "Corpus vectors", "plain", "corpus");
box(s, px[1], py1, pw, ph, "Empirical distance percentiles (0.1–0.5%)", "key", "percentiles");
box(s, px[2], py1, pw, ph, "Calibrate ef table on 200–2,000 queries", "plain", "calibrate");
arrow(s, px[0] + pw, py1 + ph / 2, px[1], py1 + ph / 2, "o1");
arrow(s, px[1] + pw, py1 + ph / 2, px[2], py1 + ph / 2, "o2");
s.addText("Online, per query", { x: 0.6, y: 3.05, w: 6, h: 0.35, fontSize: 14, italic: true, color: C.accent6, margin: 0, isTextBox: true, objectName: "Lane online" });
const py2 = 3.45;
box(s, px[0], py2, pw, ph, "Probe the first 100 distances", "plain", "probe");
box(s, px[1], py2, pw, ph, "Score: how many fall under each threshold", "key", "score");
box(s, px[2], py2, pw, ph, "Look up ef; continue the same search", "plain", "continue");
arrow(s, px[0] + pw, py2 + ph / 2, px[1], py2 + ph / 2, "n1");
arrow(s, px[1] + pw, py2 + ph / 2, px[2], py2 + ph / 2, "n2");
arrow(s, px[1] + pw / 2, py1 + ph, px[1] + pw / 2, py2, "thresholds to score");
card(s, 10.0, 1.35, 2.75, 2.2, "One change", "Same mechanism as Ada-ef; only the thresholds come from the data instead of a bell curve.", { fill: "DCEFF3", bodySize: 15 });
s.addText(bullets([
  "Works whatever the shape of the similarity distribution",
  "Probe of 100 distances instead of 1,025; the search does not restart after it",
  "No covariance matrix: a few kilobytes instead of up to 9.4 MB",
]), { x: 0.6, y: 4.8, w: 12.1, h: 1.8, color: C.text1, valign: "top", margin: 0, isTextBox: true, objectName: "PercEF bullets" });
s.addNotes("PercEF keeps Ada-ef's probe-score-table design and changes one thing: the thresholds are the data's own distance percentiles. Calibration fits an isotonic curve from score to the ef each calibration query needed.");

// ================================================================= 7. the KS test as a decision rule
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Our method" });
s.addText("A seconds-long test picks the method", { placeholder: "title" });
const sx0 = 0.9, sw0 = 11.5, sy0 = 2.0, ks2x = (k) => sx0 + sw0 * Math.min(k, 0.13) / 0.13;
s.addShape(pres.shapes.RECTANGLE, { x: sx0, y: sy0, w: ks2x(0.044) - sx0, h: 0.9, fill: { color: "FCE9DF" }, line: { color: "FCE9DF", width: 0 }, objectName: "Zone Ada-ef" });
s.addShape(pres.shapes.RECTANGLE, { x: ks2x(0.044), y: sy0, w: ks2x(0.066) - ks2x(0.044), h: 0.9, fill: { color: C.background2 }, line: { color: C.background2, width: 0 }, objectName: "Zone band" });
s.addShape(pres.shapes.RECTANGLE, { x: ks2x(0.066), y: sy0, w: sx0 + sw0 - ks2x(0.066), h: 0.9, fill: { color: "DCEFF3" }, line: { color: "DCEFF3", width: 0 }, objectName: "Zone PercEF" });
s.addText("KS ≤ 0.044: near-Gaussian → Ada-ef", { x: sx0, y: sy0, w: ks2x(0.044) - sx0, h: 0.9, fontSize: 16, bold: true, color: C.accent4, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "Zone label Ada-ef" });
s.addText("band", { x: ks2x(0.044), y: sy0, w: ks2x(0.066) - ks2x(0.044), h: 0.9, fontSize: 14, color: C.accent2, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "Zone label band" });
s.addText("KS ≥ 0.066: non-Gaussian → PercEF", { x: ks2x(0.066), y: sy0, w: sx0 + sw0 - ks2x(0.066), h: 0.9, fontSize: 16, bold: true, color: C.accent1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "Zone label PercEF" });
s.addText("KS measured on the raw vectors (200 queries, no index needed)", { x: sx0, y: sy0 - 0.5, w: sw0, h: 0.4, fontSize: 14, italic: true, color: C.accent6, margin: 0, isTextBox: true, objectName: "Scale caption" });
stat(s, 0.9, 3.4, 3.6, "18 / 19", "datasets where KS picks the method with the better ranking");
stat(s, 4.85, 3.4, 3.6, "14", "predictions written down before the runs; one failed (SIFT-1B)", C.accent2);
stat(s, 8.8, 3.4, 3.6, "0", "tail-weighted alternatives that beat KS (tested and rejected)", C.accent4);
s.addText("Every PercEF prediction was recorded before running the dataset, so the test was checked out of sample.",
  { x: 0.9, y: 5.4, w: 11.5, h: 0.6, fontSize: 16, color: C.text1, margin: 0, isTextBox: true, objectName: "KS note" });
s.addNotes("The KS test is a pre-check a practitioner can run in seconds. Leave-one-out accuracy is 16 of 19. The one miss is SIFT-1B, where KS said Ada-ef but PercEF won; we tested tail-weighted statistics as replacements and none did better.");

// ================================================================= 8. novelty
pres.addSection({ title: "Novelty" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Novelty" });
s.addText("What is new", { placeholder: "title" });
const nv = [
  ["1  First test of the assumption", "Ada-ef's Gaussian model measured on 41 datasets; fails on 9, including 6 learned embeddings."],
  ["2  A distribution-free score", "PercEF: thresholds from the data's own percentiles. No covariance, no training."],
  ["3  A cost floor", "Never more than 2.5% costlier than the best fixed ef in hindsight; Ada-ef has no such floor."],
  ["4  An offline test", "KS on raw vectors predicts the better method on 18 of 19 datasets, before any index exists."],
];
nv.forEach(([h, b], i) => card(s, 0.6 + (i % 2) * 6.15, 1.4 + Math.floor(i / 2) * 2.1, 5.95, 1.85, h, b, { bodySize: 16, fill: i % 2 ? C.background2 : "DCEFF3" }));
s.addText("Also: an evaluation guide for adaptive search (interpolation bias, timing noise, probe cost), from errors we caught in our own runs.",
  { x: 0.6, y: 5.65, w: 12.1, h: 0.5, fontSize: 14, italic: true, color: C.accent2, margin: 0, isTextBox: true, objectName: "Novelty footnote" });
s.addNotes("Four contributions. The diagnosis is new: nobody had measured this assumption. The method is a one-idea change, explained by the diagnosis. The cost floor and the offline test make it usable in practice.");

// ================================================================= 9. setup and metrics
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Setup: Ada-ef's protocol, 20 datasets", { placeholder: "title" });
s.addText(bullets([
  "20 datasets end to end: text, image, audio, music, CLIP; 60K to 19.6M vectors",
  "HNSW M = 16, efConstruction = 500; k = 100 (1,000 for MS MARCO, Cohere, LAION)",
  "Target recall 0.95; calibration on 200 corpus points (P) or real queries (R)",
  "Ada-ef run through its authors' own code; DARTH through its authors' code",
  "One thread; latency timed in 3 rotated rounds",
], 15), { x: 0.6, y: 1.4, w: 5.6, h: 4.8, color: C.text1, valign: "top", margin: 0, isTextBox: true, objectName: "Setup bullets" });
table(s, [
  ["Metric", "Meaning"],
  ["Mean recall", "Average share of the true 100 neighbours found"],
  ["p1 (worst-case) recall", "Recall of the worst 1% of queries: under-searching"],
  ["Distance computations", "Work per query, hardware-independent: over-searching"],
  ["Latency", "Time per query (µs)"],
  ["Tuned fixed ef", "One ef for all queries, picked in hindsight for the same mean recall: the reference"],
], 6.5, 1.4, 6.25, [2.1, 4.15], "Metrics table", 13);
s.addNotes("All comparisons are against the best fixed ef chosen in hindsight at the same mean recall, a demanding reference no deployment could choose in advance. p1 measures how badly the worst queries are served.");

// ================================================================= 10. non-Gaussian results
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Non-Gaussian data: better worst-case recall", { placeholder: "title" });
const ds = ["DeepImage", "Deep1B", "iNat-ResNet", "Landmark-DINO", "SIFT-1M", "Fashion-MNIST", "Yambda", "GIST"];
s.addChart(pres.charts.BAR, [
  { name: "PercEF", labels: ds, values: [8.5, 7.9, 4.0, 3.7, 2.9, 1.3, 1.2, -0.6] },
  { name: "Ada-ef", labels: ds, values: [4.3, 4.4, 0.4, 1.1, 0.0, -1.0, -0.1, -2.4] },
], { x: 0.6, y: 1.35, w: 8.2, h: 5.4, barDir: "bar", barGrouping: "clustered", chartColors: [HEX.ours, HEX.ada],
  showTitle: true, title: "p1 recall gain over the tuned fixed ef (percentage points)", titleFontSize: 14, titleColor: HEX.ink,
  titleFontFace: "+mn-lt", showLegend: true, legendPos: "b", legendFontSize: 12, legendFontFace: "+mn-lt",
  catAxisLabelColor: HEX.muted, valAxisLabelColor: HEX.muted, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", catAxisOrientation: "maxMin", catAxisLabelPos: "low",
  valGridLine: { color: HEX.grid, size: 0.75 }, catGridLine: { style: "none" },
  showValue: true, dataLabelFontSize: 10, dataLabelColor: HEX.ink, dataLabelFormatCode: "0.0", objectName: "p1 gain chart" });
stat(s, 9.2, 1.45, 3.5, "16 / 16", "runs where PercEF ranks queries better than Ada-ef");
stat(s, 9.2, 3.15, 3.5, "16 / 16", "runs where PercEF's worst-case gain is at least Ada-ef's");
stat(s, 9.2, 4.85, 3.5, "14 vs 2", "runs cheaper than the fixed ef: PercEF vs Ada-ef (of 16)", C.accent2);
s.addNotes("Eight non-Gaussian datasets, two calibration settings each. Example: on DeepImage the worst 1% of queries gain 8.5 recall points with PercEF against 4.3 with Ada-ef, at the same mean recall as the fixed ef.");

// ================================================================= 11. cost and speed
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Within 2.5% of a tuned fixed ef's cost", { placeholder: "title" });
stat(s, 0.6, 1.45, 3.7, "32 / 34", "runs where PercEF does less work than the tuned fixed ef (Ada-ef: 10 / 34)");
stat(s, 0.6, 3.15, 3.7, "≤ 2.5%", "PercEF's worst extra cost over the fixed ef, in any run", C.accent2);
stat(s, 0.6, 4.85, 3.7, "14 / 18", "timed runs where PercEF is faster than the fixed ef (Ada-ef: 3 / 18)");
const sp = ["COCO-I2I", "SIFT-1M", "Deep1B", "Cohere", "BIGANN", "MS Turing", "MS MARCO"];
s.addChart(pres.charts.BAR, [{ name: "Speed-up", labels: sp, values: [1.37, 1.24, 1.20, 1.18, 1.11, 1.08, 1.03] }], {
  x: 4.7, y: 1.35, w: 8.05, h: 4.4, barDir: "bar", chartColors: [HEX.ours], showLegend: false,
  showTitle: true, title: "PercEF speed-up over Ada-ef at about the same recall (x)", titleFontSize: 14, titleColor: HEX.ink,
  titleFontFace: "+mn-lt", catAxisLabelColor: HEX.muted, valAxisLabelColor: HEX.muted, catAxisLabelFontSize: 12,
  valAxisLabelFontSize: 11, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", catAxisOrientation: "maxMin",
  valAxisMinVal: 0, valGridLine: { color: HEX.grid, size: 0.75 }, catGridLine: { style: "none" },
  showValue: true, dataLabelFontSize: 11, dataLabelColor: HEX.ink, dataLabelFormatCode: "0.00", objectName: "Speed-up chart" });
s.addText("Fair to Ada-ef: on its own long-search datasets, Ada-ef with its WAE floor is the fastest (21% faster than the fixed ef on GloVe).",
  { x: 4.7, y: 5.95, w: 8.05, h: 0.75, fontSize: 14, italic: true, color: C.accent2, margin: 0, isTextBox: true, objectName: "Speed note" });
s.addNotes("34 runs = 17 datasets x 2 calibration settings. PercEF is 1.03x to 1.37x faster than Ada-ef as shipped, mainly because Ada-ef computes 1,025 probe distances per query and PercEF 100. It is not many times faster; its main win is the worst-case recall.");

// ================================================================= 12. checks
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Two checks: probe length and DARTH", { placeholder: "title" });
s.addText("Ranking quality |ρ| with matched probe lengths", { x: 0.6, y: 1.3, w: 6, h: 0.4, fontSize: 15, bold: true, color: C.text2, margin: 0, isTextBox: true, objectName: "Ablation caption" });
table(s, [
  ["Dataset", "Ada-ef 1,025", "Ada-ef 100", "PercEF 100", "PercEF 1,025"],
  ["SIFT-1M", "0.04", "0.03", "0.45", "0.40"],
  ["Deep1B", "0.41", "0.43", "0.67", "0.63"],
  ["DeepImage", "0.38", "0.33", "0.62", "0.59"],
  ["SIFT-1B", "0.42", "0.27", "0.74", "0.72"],
  ["GloVe", "0.80", "0.62", "0.61", "0.76"],
  ["MS Turing", "0.48", "0.21", "0.40", "0.58"],
], 0.6, 1.75, 6.0, [1.4, 1.15, 1.15, 1.15, 1.15], "Ablation table", 12);
s.addText("The advantage comes from the thresholds, not the shorter probe.", { x: 0.6, y: 4.3, w: 6.0, h: 0.9, fontSize: 14, italic: true, color: C.accent2, margin: 0, isTextBox: true, objectName: "Ablation takeaway" });
s.addText("DARTH (learned) vs PercEF, setting R", { x: 6.95, y: 1.3, w: 5.8, h: 0.4, fontSize: 15, bold: true, color: C.text2, margin: 0, isTextBox: true, objectName: "DARTH caption" });
table(s, [
  ["Dataset", "p1 gain DARTH", "p1 gain PercEF", "Time vs fixed DARTH", "Time vs fixed PercEF"],
  ["GloVe", "+0.080", "+0.029", "6% slower", "1% faster"],
  ["MS Turing", "+0.044", "+0.010", "42% slower", "1% faster"],
  ["SIFT-1M", "+0.048", "+0.028", "2.3x slower", "1% slower"],
  ["Deep1B", "+0.098", "+0.081", "1.9x slower", "3% slower"],
], 6.95, 1.75, 5.8, [1.2, 1.1, 1.1, 1.2, 1.2], "DARTH table", 12);
s.addText("DARTH has the best tail but costs time and training; PercEF gets much of the gain at almost no time cost.", { x: 6.95, y: 4.3, w: 5.8, h: 0.9, fontSize: 14, italic: true, color: C.accent2, margin: 0, isTextBox: true, objectName: "DARTH takeaway" });
s.addNotes("Left: each score at both probe lengths. On non-Gaussian data PercEF ranks better at both, so the thresholds carry the advantage. Right: DARTH, the learned method, wins the worst-case recall but is up to 2.3x slower than a fixed ef and needs minutes of training per dataset.");

// ================================================================= 13. combined idea (tested, not adopted)
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Tested: our score + Ada-ef's table", { placeholder: "title" });
s.addText("Idea: if Ada-ef's better tail on Gaussian data came from its ef table, PercEF's score with that table should match it.",
  { x: 0.6, y: 1.3, w: 12.1, h: 0.7, fontSize: 17, color: C.text1, margin: 0, isTextBox: true, objectName: "Combo intro" });
table(s, [
  ["Dataset", "p1 gain: PercEF", "p1 gain: combined", "Combined + WAE: time vs fixed ef", "p1 gain: Ada-ef"],
  ["SIFT-1M", "+0.028", "+0.024", "1.5% slower", "0.000"],
  ["Deep1B", "+0.081", "+0.076", "2.9% faster", "+0.052"],
  ["DeepImage", "+0.088", "+0.073", "3.7% faster", "+0.048"],
  ["SIFT-1B", "+0.056", "+0.056", "2.7% faster", "+0.016"],
  ["GloVe (Gaussian)", "+0.029", "+0.031", "3.1% faster", "+0.073"],
  ["MS Turing (Gaussian)", "+0.013", "+0.012", "1.3% faster", "−0.038"],
], 0.6, 2.05, 12.1, [2.4, 2.2, 2.3, 3.0, 2.2], "Combined table", 14);
s.addText(bullets([
  "Not adopted: on GloVe the table lifts PercEF only from +0.029 to +0.031, far from Ada-ef's +0.073",
  "Ada-ef's tail edge on Gaussian data needs its own score and its table together",
  "Plain PercEF keeps the best or tied-best worst-case recall on 5 of 6 datasets",
], 15), { x: 0.6, y: 5.25, w: 12.1, h: 1.6, color: C.text1, valign: "top", margin: 0, isTextBox: true, objectName: "Combo bullets" });
s.addNotes("We tested whether Ada-ef's better worst-case recall on Gaussian data came from its ef table. It did not: with that table our score barely improves on GloVe. Plain PercEF stays the default; the result goes in the paper as a tested explanation that did not hold.");

// ================================================================= 14. limits and next steps
pres.addSection({ title: "Wrap-up" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Wrap-up" });
s.addText("Limits and next steps", { placeholder: "title" });
card(s, 0.6, 1.4, 5.95, 3.4, "Limits, stated openly",
  "• Near-Gaussian text embeddings: Ada-ef keeps the better worst-case recall\n• Cross-modal queries (text searching images): neither method adapts\n• One failed KS prediction (SIFT-1B)\n• Cohere and LAION run as subsets (62 GB RAM)",
  { bodySize: 16, fill: "FCE9DF", headColor: C.accent4 });
card(s, 6.85, 1.4, 5.9, 3.4, "Next steps",
  "• Understand why Ada-ef's score and table together win the tail on GloVe\n• Final figures and numbers in the paper\n• Guide review, author list\n• arXiv preprint and code release\n• Submit: PVLDB Experiments, Analysis & Benchmarks track",
  { bodySize: 16, fill: "DCEFF3" });
s.addNotes("These limits are in the paper. Reviewers trust a paper that says what does not work.");

// ================================================================= 15. conclusion
s = pres.addSlide({ masterName: "DARK", sectionTitle: "Wrap-up" });
s.addText("Measure, don't assume", { placeholder: "title" });
s.addText([
  { text: "Ada-ef's bell-curve assumption fails on 9 of 41 datasets. PercEF measures the thresholds instead: better worst-case recall where the assumption fails, never meaningfully costlier than a tuned fixed ef, and a seconds-long test that says which method to use.", options: { breakLine: true } },
  { text: "Thank you. Questions?", options: { bold: true } },
], { placeholder: "body" });
s.addNotes("One sentence to remember: measure the distribution instead of assuming it.");

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
