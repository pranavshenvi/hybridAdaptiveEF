// Project-fair poster (PES University template: 1440 x 810 pt = 20 x 11.25 in, three columns,
// member/guide strip at the bottom). Run:  node build_poster.js   ->  poster.pptx
// Everything is an ordinary PowerPoint object, named in the Selection Pane (Home > Arrange >
// Selection Pane), so the poster can be edited directly in PowerPoint afterwards.
const pptxgen = require("pptxgenjs");

const OUT = process.argv[2] || "PercEF_poster.pptx";   // node build_poster.js [output.pptx]
const FONT = "Arial";
const C = {
  frame: "4A5D6E",   // dark slate surround (as in the template)
  teal: "1597AB",    // section headers, left and right columns
  slate: "485B6C",   // section headers, middle column
  navy: "2D3A82",    // title
  ink: "1F2933",     // body text
  muted: "5B6670",
  border: "CFD8DE",
  tealTint: "E4F3F6",
  slateTint: "E9EDF0",
  photo: "EEF1F3",
};

const pres = new pptxgen();
pres.defineLayout({ name: "POSTER", width: 20, height: 11.25 });
pres.layout = "POSTER";
pres.title = "Project fair poster";
const s = pres.addSlide();
s.background = { color: C.frame };

// white panel inside the dark surround
s.addShape(pres.shapes.RECTANGLE, { x: 0.25, y: 0.2, w: 19.5, h: 10.85, fill: { color: "FFFFFF" },
  line: { color: "FFFFFF", width: 0 }, objectName: "Panel" });

// ---------------------------------------------------------------- header
s.addImage({ path: "assets/logo_pes.png", x: 0.5, y: 0.35, w: 3.2, h: 1.64, objectName: "Logo PES" });
s.addImage({ path: "assets/logo_cdsaml.png", x: 15.75, y: 0.33, w: 1.6, h: 1.56, objectName: "Logo CDSAML" });
s.addImage({ path: "assets/logo_ccbd.png", x: 17.4, y: 0.62, w: 2.1, h: 0.67, objectName: "Logo CCBD" });
s.addText("PercEF: Exploiting Empirical Percentiles for Adaptive HNSW Search Beyond the Gaussian Assumption", {
  x: 3.95, y: 0.3, w: 11.6, h: 1.05, fontFace: FONT, fontSize: 28, bold: true, color: C.navy,
  align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "Title" });
s.addText([
  { text: "Center for Data Science and Applied Machine Learning (CDSAML)", options: { breakLine: true } },
  { text: "Department of Computer Science and Engineering, PES University, RR Campus, Bengaluru-560085" },
], { x: 3.95, y: 1.38, w: 11.6, h: 0.62, fontFace: FONT, fontSize: 14, bold: true, color: C.navy,
  align: "center", valign: "top", margin: 0, isTextBox: true, objectName: "Subtitle" });

// ---------------------------------------------------------------- helpers
const COL_Y = 2.15, COL_W = 6.1, X = [0.55, 6.95, 13.35], HEAD_H = 0.55, GAP = 0.15;

function header(x, y, title, color, name) {
  s.addText(title, { x, y, w: COL_W, h: HEAD_H, fontFace: FONT, fontSize: 20, bold: true, color: "FFFFFF",
    fill: { color }, valign: "middle", margin: [10, 10, 2, 2], isTextBox: true, objectName: name + " header" });
  return y + HEAD_H;
}

function box(x, y, h, name) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w: COL_W, h, fill: { color: "FFFFFF" },
    line: { color: C.border, width: 0.75 }, objectName: name + " box" });
}

// paragraphs: array of strings or {label, text} or {bullet: text}
function body(x, y, h, paras, name, size = 13) {
  box(x, y, h, name);
  const runs = [];
  paras.forEach((p, i) => {
    const last = i === paras.length - 1;
    if (typeof p === "string") {
      runs.push({ text: p, options: { breakLine: !last, paraSpaceAfter: 5 } });
    } else if (p.label) {
      runs.push({ text: p.label + " ", options: { bold: true } });
      runs.push({ text: p.text, options: { breakLine: !last, paraSpaceAfter: 5 } });
    } else if (p.bold) {
      runs.push({ text: p.bold, options: { bold: true, breakLine: true, paraSpaceAfter: 2 } });
    } else {
      runs.push({ text: p.bullet, options: { bullet: { indent: 12 }, breakLine: !last, paraSpaceAfter: 4 } });
    }
  });
  s.addText(runs, { x: x + 0.08, y: y + 0.06, w: COL_W - 0.16, h: h - 0.12, fontFace: FONT, fontSize: size,
    color: C.ink, valign: "top", margin: 4, isTextBox: true, objectName: name + " text" });
  return y + h;
}

// ---------------------------------------------------------------- column 1
let y = COL_Y;
y = header(X[0], y, "PROBLEM STATEMENT", C.teal, "Problem");
y = body(X[0], y, 1.72, [
  { label: "a)", text: "Vector search engines answer every query with the same search breadth (ef) in HNSW. Easy queries waste work; hard queries miss neighbours." },
  { label: "b)", text: "Ada-ef (SIGMOD 2026) sets ef per query by assuming each query's similarities to the data are Gaussian. The assumption was never tested. We test it, and fix the method where it fails." },
], "Problem") + GAP;
y = header(X[0], y, "OBJECTIVES", C.teal, "Objectives");
y = body(X[0], y, 1.5, [
  { bullet: "Measure how often the Gaussian assumption holds on public embedding datasets." },
  { bullet: "Build a distribution-free difficulty score that works where it does not." },
  { bullet: "Give a quick offline test that says which method to use." },
  { bullet: "Compare fairly with Ada-ef and a learned method (DARTH)." },
], "Objectives") + GAP;
y = header(X[0], y, "DATASET AND FEATURES", C.teal, "Dataset");
body(X[0], y, 9.05 - y, [
  { bullet: "41 public datasets surveyed (ann-benchmarks, Big-ANN, VIBE, Ada-ef's own sources)." },
  { bullet: "20 benchmarked end to end: text, image, audio, music and CLIP embeddings; 60K to 19.6M vectors, 65 to 2048 dimensions." },
  { bullet: "Measured per query: similarity distribution, true minimum ef for 95% recall. Per dataset: KS statistic." },
], "Dataset");

// ---------------------------------------------------------------- column 2: architecture
y = COL_Y;
y = header(X[1], y, "SYSTEM ARCHITECTURE", C.slate, "Architecture");
const archH = 3.2;
box(X[1], y, archH, "Architecture");
const BW = 1.75, BH = 0.66, BG = 0.3, bx = (i) => X[1] + 0.13 + i * (BW + BG);
function node(i, top, text, kind, name) {
  const fill = kind === "key" ? C.teal : kind === "off" ? C.tealTint : C.slateTint;
  const line = kind === "off" || kind === "key" ? C.teal : C.slate;
  s.addText(text, { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.08, x: bx(i), y: top, w: BW, h: BH,
    fill: { color: fill }, line: { color: line, width: 1 }, fontFace: FONT, fontSize: 11,
    bold: kind === "key", color: kind === "key" ? "FFFFFF" : C.ink, align: "center", valign: "middle",
    margin: 3, objectName: "Arch: " + name });
}
function arrow(x1, y1, x2, y2, name) {
  const flipH = x2 < x1, flipV = y2 < y1;
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipH, flipV, line: { color: C.muted, width: 1.5, endArrowType: "triangle" }, objectName: "Arrow: " + name });
}
function lane(top, text, name) {
  s.addText(text, { x: X[1] + 0.13, y: top, w: COL_W - 0.26, h: 0.28, fontFace: FONT, fontSize: 11, italic: true,
    color: C.muted, margin: 0, isTextBox: true, objectName: "Lane: " + name });
}
const r0 = y + 0.12;
lane(r0, "Offline, once per dataset", "offline");
const r1 = r0 + 0.32;
node(0, r1, "Corpus vectors", "off", "corpus");
node(1, r1, "KS test: are similarities Gaussian?", "off", "ks");
node(2, r1, "Percentile bins + calibrated ef table", "off", "calibration");
arrow(bx(0) + BW, r1 + BH / 2, bx(1), r1 + BH / 2, "corpus-ks");
arrow(bx(1) + BW, r1 + BH / 2, bx(2), r1 + BH / 2, "ks-calibration");
const r2 = r1 + BH + 0.36;
lane(r2 - 0.32, "Online, per query", "online");
node(0, r2, "Query vector", "on", "query");
node(1, r2, "HNSW probe: first 100 distances", "on", "probe");
node(2, r2, "Difficulty score from empirical bins", "key", "score");
arrow(bx(0) + BW, r2 + BH / 2, bx(1), r2 + BH / 2, "query-probe");
arrow(bx(1) + BW, r2 + BH / 2, bx(2), r2 + BH / 2, "probe-score");
arrow(bx(2) + BW / 2, r1 + BH, bx(2) + BW / 2, r2, "calibration-score");
const r3 = r2 + BH + 0.3;
node(2, r3, "Look up ef for this query", "on", "lookup");
node(1, r3, "Continue the same search at that ef", "on", "continue");
node(0, r3, "Top-k neighbours", "on", "results");
arrow(bx(2) + BW / 2, r2 + BH, bx(2) + BW / 2, r3, "score-lookup");
arrow(bx(2), r3 + BH / 2, bx(1) + BW, r3 + BH / 2, "lookup-continue");
arrow(bx(1), r3 + BH / 2, bx(0) + BW, r3 + BH / 2, "continue-results");
y += archH + GAP;

y = header(X[1], y, "PROPOSED METHODOLOGY", C.slate, "Methodology");
body(X[1], y, 9.05 - y, [
  { bold: "a) Approach" },
  { bullet: "Test: KS distance between each query's similarities and Ada-ef's Gaussian (seconds, no index)." },
  { bullet: "Score: Ada-ef's binned score, with thresholds from the data's own distance percentiles." },
  { bullet: "Calibrate: isotonic fit of true minimum ef to score; no restart after the probe." },
  { bold: "b) Constraints and assumptions" },
  { bullet: "Ada-ef's protocol: M = 16, efC = 500, target recall 0.95, one thread." },
  { bullet: "Small calibration set; Cohere and LAION run as subsets (62 GB RAM)." },
], "Methodology", 12.5);

// ---------------------------------------------------------------- column 3: results
y = COL_Y;
y = header(X[2], y, "RESULTS & DISCUSSION", C.teal, "Results");
const resH = 4.5;
box(X[2], y, resH, "Results");
const figW = 3.72, figH = figW * 899 / 1017;
s.addImage({ path: "assets/fig2_p1_vs_ks.png", x: X[2] + 0.1, y: y + 0.1, w: figW, h: figH, objectName: "Results figure" });
s.addText("Tail-recall gain over the best fixed ef, 20 datasets sorted by KS (top: least Gaussian).", {
  x: X[2] + 0.1, y: y + 0.12 + figH, w: figW, h: 0.42, fontFace: FONT, fontSize: 10, italic: true, color: C.muted,
  margin: 0, valign: "top", isTextBox: true, objectName: "Results figure caption" });
const stats = [
  ["15/16", "runs on non-Gaussian data: PercEF's tail recall at least Ada-ef's"],
  ["32/34", "runs cheaper than the best fixed ef (Ada-ef: 10/34); never over +2.5%"],
  ["18/19", "datasets where the KS test picks the better method"],
];
const sx = X[2] + 0.1 + figW + 0.15, sw = COL_W - figW - 0.35;
stats.forEach(([big, small], i) => {
  const sy = y + 0.15 + i * 1.25;
  s.addText(big, { x: sx, y: sy, w: sw, h: 0.5, fontFace: FONT, fontSize: 28, bold: true, color: C.teal,
    margin: 0, valign: "middle", isTextBox: true, objectName: `Stat ${i + 1} number` });
  s.addText(small, { x: sx, y: sy + 0.5, w: sw, h: 0.68, fontFace: FONT, fontSize: 11, color: C.ink,
    margin: 0, valign: "top", isTextBox: true, objectName: `Stat ${i + 1} label` });
});
s.addText([
  { text: "Limits: ", options: { bold: true } },
  { text: "on near-Gaussian text embeddings Ada-ef keeps the better tail; neither method adapts when queries come from another modality (text searching images)." },
], { x: X[2] + 0.1, y: y + resH - 0.62, w: COL_W - 0.2, h: 0.56, fontFace: FONT, fontSize: 11.5, color: C.ink,
  margin: 0, valign: "top", isTextBox: true, objectName: "Results limits" });
y += resH + GAP;

y = header(X[2], y, "CONCLUSIONS AND FUTURE WORK", C.teal, "Conclusions");
body(X[2], y, 9.05 - y, [
  { label: "a)", text: "Ada-ef's Gaussian model fails on 9 of 41 datasets (6 learned embeddings); there PercEF gives better tail recall, at most 2.5% over the best fixed ef, and beats a fixed ef with the same worst case on time in 31 of 34 runs." },
  { label: "b)", text: "DARTH (learned): best tail, up to 2.3x slower. Next: SIFT-1B, cross-modal." },
], "Conclusions", 12);

// ---------------------------------------------------------------- members and guide
const FY = 9.3, SLOT = 18.9 / 5;
// names, SRNs and photos live in ../team.json so that rebuilding the poster never loses them
const fsTeam = require("fs"), pathTeam = require("path");
const TEAM = JSON.parse(fsTeam.readFileSync(pathTeam.join(__dirname, "..", "team.json"), "utf8"));
const people = [...TEAM.members.map((m, i) => ({ who: `Member ${i + 1}`, line1: m.name, line2: m.srn || m.note || "", photo: m.photo })),
                { who: "Guide", line1: "Guide: " + TEAM.guide.name, line2: TEAM.guide.designation, photo: TEAM.guide.photo }];
people.forEach(({ who, line1, line2, photo }, i) => {
  const x0 = 0.55 + i * SLOT;
  const photoPath = photo ? pathTeam.join(__dirname, "..", photo) : "";
  if (photoPath && fsTeam.existsSync(photoPath)) {
    s.addImage({ path: photoPath, x: x0 + 0.15, y: FY + 0.05, w: 1.3, h: 1.45,
      sizing: { type: "cover", w: 1.3, h: 1.45 }, objectName: `${who} photo` });
  } else {
    if (photo) console.warn(`photo not found for ${who}: ${photoPath}`);
    s.addText("Photo", { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.1, x: x0 + 0.15, y: FY + 0.05, w: 1.3, h: 1.45,
      fill: { color: C.photo }, line: { color: C.border, width: 0.75 }, fontFace: FONT, fontSize: 11, color: C.muted,
      align: "center", valign: "middle", objectName: `${who} photo` });
  }
  s.addText([
    { text: line1, options: { bold: true, breakLine: true } },
    { text: line2 },
  ], { x: x0 + 1.62, y: FY + 0.35, w: SLOT - 1.75, h: 0.95, fontFace: FONT, fontSize: 13, color: C.ink,
    valign: "middle", margin: 0, isTextBox: true, objectName: `${who} name` });
  if (i > 0) s.addShape(pres.shapes.LINE, { x: x0 - 0.02, y: FY + 0.2, w: 0, h: 1.25,
    line: { color: C.border, width: 1 }, objectName: `Separator ${i}` });
});

pres.writeFile({ fileName: OUT }).then((f) => console.log("wrote", f));
