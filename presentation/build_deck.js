// PercEF project review deck. Run from this folder:  node build_deck.js  ->  PercEF_review.pptx
// Structure: problem statement -> existing methodology -> our method -> novelty -> results.
// Every number comes from updateAsOf290926.md / updateAsOf061026.md / the result files.
// Edit freely in PowerPoint afterwards; every object is named (Home > Arrange > Selection Pane).
const path = require("path");
const pptxgen = require(path.join(__dirname, "..", "poster", "node_modules", "pptxgenjs"));
const { applyTheme } = require(process.env.PPTX_SKILL ||
  "C:/Users/adebeo/.claude/skills/synced/81f1b1b5-7c94-4926-91c8-fca4a2a070f6_ff6d1026-246b-4214-87da-3cc36522d9da/pptx/scripts/apply_theme.js");

const OUT = process.argv[2] || "PercEF_review.pptx";
// team names live in ../team.json so that rebuilding the deck never loses them
const TEAM = JSON.parse(require("fs").readFileSync(path.join(__dirname, "..", "team.json"), "utf8"));
const FIG = path.join(__dirname, "..", "server_results", "paper_out_20261007_112616");
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
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.3, w: 11.7, h: 2.8, fontSize: 18,
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
const who = (m) => m.name + " (" + (m.srn || m.note || "") + ")";            // SRN, or the note for an alumnus
const teamLines = [];
for (let i = 0; i < TEAM.members.length; i += 2) teamLines.push(TEAM.members.slice(i, i + 2).map(who).join("    ·    "));
s.addText([
  { text: "Project review  ·  Department of Computer Science and Engineering, PES University", options: { breakLine: true, paraSpaceAfter: 10 } },
  ...teamLines.map((l) => ({ text: l, options: { breakLine: true, fontSize: 16 } })),
  { text: "Guide: " + TEAM.guide.name + ", " + TEAM.guide.designation, options: { fontSize: 16, bold: true } },
], { placeholder: "body" });
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.8, y: 0.6, w: 3.0, h: 1.6, rectRadius: 0.1, fill: { color: C.background1 },
  line: { color: C.background1, width: 0 }, objectName: "Logo backing" });
s.addImage({ path: LOGO, x: 0.95, y: 0.72, w: 2.7, h: 2.7 * 373 / 727, objectName: "PES logo" });
s.addNotes("(0:05) Hi, we are [team]. Our project is PercEF: making adaptive vector search work when the data is not bell-shaped.");

// ================================================================= 1b. background: vector search and HNSW
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
s.addText("Background: vector search and HNSW", { placeholder: "title" });
s.addText([
  { text: "Vector search. ", options: { bold: true, color: C.text2 } },
  { text: "Text, images and audio are turned into vectors (embeddings); similar items get nearby vectors. A query is answered by finding its nearest vectors.", options: { breakLine: true, paraSpaceAfter: 10 } },
  { text: "Why it matters. ", options: { bold: true, color: C.text2 } },
  { text: "Semantic search, recommendations and retrieval for LLMs (RAG) all run on it.", options: { breakLine: true, paraSpaceAfter: 10 } },
  { text: "Why an index. ", options: { bold: true, color: C.text2 } },
  { text: "Comparing a query with millions of vectors is too slow, so an index searches only a small part of the data and returns approximate neighbours.", options: { breakLine: true, paraSpaceAfter: 10 } },
  { text: "HNSW. ", options: { bold: true, color: C.text2 } },
  { text: "A layered graph: the search enters at the sparse top layer, moves towards the query and drops down a layer at a time.", options: { breakLine: true, paraSpaceAfter: 10 } },
  { text: "ef. ", options: { bold: true, color: C.accent1 } },
  { text: "How many candidates the bottom-layer search keeps. Larger ef: more accurate, but slower." },
], { x: 0.6, y: 1.35, w: 5.9, h: 5.4, fontSize: 15, color: C.text1, valign: "top", margin: 0, isTextBox: true, objectName: "Background text" });
(() => {                                                    // HNSW sketch: three layers, one search
  const LX = 6.95, LW = 5.8, LH = 1.3, LY = [1.4, 2.95, 4.5];
  const bx = [0.3, 0.75, 1.2, 1.65, 2.1, 2.55, 3.0, 3.45, 3.9, 4.35, 4.8, 5.25];
  const layers = [[2, 6, 10], [0, 2, 4, 6, 8, 10], bx.map((_, i) => i)];
  const names = ["Layer 2: few nodes", "Layer 1", "Layer 0: every vector"];
  const jitter = (i, l) => (l === 2 ? (i % 2 ? 0.18 : -0.12) : l === 1 ? (i % 4 ? 0.1 : -0.1) : 0);
  const pos = (i, l) => [LX + bx[i], LY[l] + LH / 2 + 0.1 + jitter(i, l)];
  layers.forEach((nodes, l) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: LX, y: LY[l], w: LW, h: LH, rectRadius: 0.08, fill: { color: C.background2 },
      line: { color: C.background2, width: 0 }, objectName: "HNSW " + names[l] });
    s.addText(names[l], { x: LX + 0.12, y: LY[l] + 0.05, w: 3, h: 0.3, fontSize: 11, italic: true, color: C.accent6, margin: 0,
      isTextBox: true, objectName: "HNSW label " + l });
    nodes.forEach((n, k) => { if (k) { const [x1, y1] = pos(nodes[k - 1], l), [x2, y2] = pos(n, l);
      s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.max(Math.abs(y2 - y1), 0.001),
        flipV: y2 < y1, line: { color: "C9CED3", width: 1 }, objectName: `HNSW edge ${l}-${k}` }); } });
  });
  // ef candidates kept on layer 0, around the query
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: LX + 3.25, y: LY[2] + 0.42, w: 1.8, h: 0.75, rectRadius: 0.12,
    fill: { color: C.accent1, transparency: 75 }, line: { color: C.accent1, width: 1.25 }, objectName: "HNSW ef window" });
  s.addText("ef candidates", { x: LX + 3.25, y: LY[2] + 1.18, w: 1.8, h: 0.28, fontSize: 11, bold: true, color: C.accent1,
    align: "center", margin: 0, isTextBox: true, objectName: "HNSW ef label" });
  layers.forEach((nodes, l) => nodes.forEach((n) => { const [x, y] = pos(n, l);
    s.addShape(pres.shapes.OVAL, { x: x - 0.09, y: y - 0.09, w: 0.18, h: 0.18, fill: { color: C.accent2 },
      line: { color: C.background1, width: 1 }, objectName: `HNSW node ${l}-${n}` }); }));
  // the search: across layer 2, down, across layer 1, down, into the query's neighbourhood
  const path = [[2, 0], [6, 0], [6, 1], [8, 1], [8, 2], [9, 2]];   // l = 0 is the top layer
  for (let k = 1; k < path.length; k++) { const [x1, y1] = pos(path[k - 1][0], path[k - 1][1]), [x2, y2] = pos(path[k][0], path[k][1]);
    s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.max(Math.abs(x2 - x1), 0.001), h: Math.max(Math.abs(y2 - y1), 0.001),
      flipH: x2 < x1, flipV: y2 < y1, line: { color: C.accent1, width: 2.25, endArrowType: "triangle", dashType: y1 !== y2 && x1 === x2 ? "dash" : "solid" },
      objectName: `HNSW search step ${k}` }); }
  const [qx, qy] = [LX + 4.55, pos(9, 2)[1] + 0.05];
  s.addShape(pres.shapes.STAR_5_POINT, { x: qx - 0.16, y: qy - 0.16, w: 0.32, h: 0.32, fill: { color: C.accent4 },
    line: { color: C.accent4, width: 0 }, objectName: "HNSW query" });
  s.addText("query", { x: qx + 0.12, y: qy + 0.12, w: 0.9, h: 0.3, fontSize: 11, bold: true, color: C.accent4, margin: 0,
    isTextBox: true, objectName: "HNSW query label" });
  s.addText("entry", { x: pos(2, 0)[0] - 0.45, y: pos(2, 0)[1] - 0.45, w: 0.9, h: 0.28, fontSize: 11, bold: true, color: C.accent1,
    align: "center", margin: 0, isTextBox: true, objectName: "HNSW entry label" });
})();
s.addNotes("(0:12) Vector search finds the items closest to a query among millions of embeddings. To be fast, an index like HNSW walks a layered graph from the top down. A setting called ef decides how many candidates the search keeps: more is more accurate, but slower.");

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
s.addNotes("(0:08) Most systems use one ef for every query. Easy queries waste work, and hard queries miss some of their neighbours.");

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
s.addNotes("(0:08) Existing fixes either train a model per dataset, like DARTH, or, like Ada-ef from SIGMOD 2026, assume that a query's similarities follow a bell curve.");

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
card(s, 0.6, 4.55, 12.15, 1.7, "The gap", "Ada-ef's paper derives it from the central limit theorem and supports it on GloVe and MS MARCO, but does not report how well it fits each dataset. Nor does any other ANN paper we found.",
  { fill: "FCE9DF", headColor: C.accent4, bodySize: 16 });
s.addNotes("(0:08) Ada-ef places its difficulty thresholds using that bell curve. Nobody had checked whether real data is bell-shaped.");

// ================================================================= 5. the diagnosis
pres.addSection({ title: "Our method" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Our method" });
s.addText("The model departs from the data on 9 of 41 datasets", { placeholder: "title" });
s.addImage({ path: path.join(FIG, "fig1_ks_survey.png"), x: 0.6, y: 1.3, w: 5.4 * 1040 / 1089, h: 5.4, objectName: "KS survey figure" });
s.addText(bullets([
  { text: "KS statistic: how far each query's similarities are from Ada-ef's bell curve (0 = perfect fit)", bold: false },
  "Measured on 41 public datasets (ann-benchmarks, Big-ANN, VIBE, Ada-ef's sources)",
  { text: "9 clearly non-Gaussian, including SIFT, GIST, Fashion-MNIST, DeepImage, Deep1B", bold: true },
  "6 of the 9 are learned embeddings, inside Ada-ef's stated scope",
  "Data type does not predict it: SIFT-1M 0.126 vs SIFT-1B 0.029; Fashion-MNIST 0.073 vs MNIST 0.037",
  "There, Ada-ef stops adapting: on SIFT it gives every query the same ef",
]), { x: 6.45, y: 1.4, w: 6.3, h: 5.3, color: C.text1, valign: "top", margin: 0, isTextBox: true, objectName: "Diagnosis bullets" });
s.addNotes("(0:10) We tested it on 41 datasets. Nine are clearly not bell-shaped, including SIFT and GIST, and you cannot tell from the kind of data. On those, Ada-ef stops adapting.");

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
s.addNotes("(0:10) Our method, PercEF, keeps Ada-ef's design but measures the thresholds from the data's own distance percentiles, with a ten times shorter probe and no covariance matrix.");

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
stat(s, 0.9, 3.4, 5.2, "18 / 19", "datasets where KS picked the method that ranks queries better");
stat(s, 6.85, 3.4, 5.2, "1 miss", "SIFT-1B: KS said Ada-ef, but PercEF ranked better", C.accent4);
s.addText("Every KS prediction was recorded before running the dataset, so the test was checked out of sample.",
  { x: 0.9, y: 5.4, w: 11.5, h: 0.6, fontSize: 16, color: C.text1, margin: 0, isTextBox: true, objectName: "KS note" });
s.addNotes("(0:08) A KS test on the raw vectors, which takes seconds, tells you which method to use. It picked the right one on 18 of 19 datasets, with every prediction written down before the run.");

// ================================================================= 8. novelty
pres.addSection({ title: "Novelty" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Novelty" });
s.addText("What is new", { placeholder: "title" });
const nv = [
  ["1  The fit, dataset by dataset", "Ada-ef's Gaussian model measured on 41 datasets; it departs from the data on 9, including 6 learned embeddings."],
  ["2  A distribution-free score", "PercEF: thresholds from the data's own percentiles. No covariance, no training."],
  ["3  A cost floor", "Never more than 2.5% costlier than the best fixed ef in hindsight; Ada-ef has no such floor."],
  ["4  An offline test", "KS on raw vectors predicts the better method on 18 of 19 datasets, before any index exists."],
];
nv.forEach(([h, b], i) => card(s, 0.6 + (i % 2) * 6.15, 1.4 + Math.floor(i / 2) * 2.1, 5.95, 1.85, h, b, { bodySize: 16, fill: i % 2 ? C.background2 : "DCEFF3" }));
s.addText("Also: an evaluation guide for adaptive search (interpolation bias, timing noise, probe cost), from errors we caught in our own runs.",
  { x: 0.6, y: 5.65, w: 12.1, h: 0.5, fontSize: 14, italic: true, color: C.accent2, margin: 0, isTextBox: true, objectName: "Novelty footnote" });
s.addNotes("(0:08) So we contribute the first dataset-by-dataset measurement of this assumption, a distribution-free score, a cost floor, and an offline test.");

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
s.addNotes("(0:05) We followed Ada-ef's own protocol on 20 datasets and compared everything against the best fixed ef.");

// ================================================================= 10. non-Gaussian results
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Non-Gaussian data: better worst-case recall", { placeholder: "title" });
const ds = ["DeepImage", "Deep1B", "Landmark-DINO", "iNat-ResNet", "SIFT-1M", "Fashion-MNIST", "Yambda", "GIST"];
s.addChart(pres.charts.BAR, [
  { name: "PercEF", labels: ds, values: [8.5, 7.9, 3.8, 3.4, 2.8, 1.1, 0.7, -0.9] },
  { name: "Ada-ef", labels: ds, values: [4.3, 4.3, 0.7, 0.1, 0.0, -1.1, -0.1, -3.0] },
], { x: 0.6, y: 1.35, w: 8.2, h: 5.4, barDir: "bar", barGrouping: "clustered", chartColors: [HEX.ours, HEX.ada],
  showTitle: true, title: "p1 recall gain over the tuned fixed ef (percentage points, mean of settings P and R)", titleFontSize: 14, titleColor: HEX.ink,
  titleFontFace: "+mn-lt", showLegend: true, legendPos: "b", legendFontSize: 12, legendFontFace: "+mn-lt",
  catAxisLabelColor: HEX.muted, valAxisLabelColor: HEX.muted, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", catAxisOrientation: "maxMin", catAxisLabelPos: "low",
  valGridLine: { color: HEX.grid, size: 0.75 }, catGridLine: { style: "none" },
  showValue: true, dataLabelFontSize: 10, dataLabelColor: HEX.ink, dataLabelFormatCode: "0.0", objectName: "p1 gain chart" });
stat(s, 9.2, 1.45, 3.5, "16 / 16", "runs where PercEF's score orders queries by true difficulty better than Ada-ef's (rank correlation)");
stat(s, 9.2, 3.15, 3.5, "13 / 16", "runs where PercEF's worst case is at or above the fixed ef's (Ada-ef: 6 / 16)");
stat(s, 9.2, 4.85, 3.5, "13 / 16", "runs where PercEF is faster than a fixed ef with the same worst case (Ada-ef: 1 / 16)", C.accent2);
s.addText("16 runs = the 8 non-Gaussian datasets x 2 calibration settings", { x: 9.2, y: 6.35, w: 3.55, h: 0.5, fontSize: 11,
  italic: true, color: C.accent6, margin: 0, isTextBox: true, objectName: "Runs definition" });
s.addNotes("(0:10) Each pair of bars is a dataset; longer to the right is better. On the non-Gaussian datasets PercEF ranks queries better in every run, and keeps the worst case at or above the fixed ef in 13 of 16 runs, against 6 for Ada-ef: on DeepImage it gains 8.5 recall points, against 4.3. On GIST both lose a little, PercEF less. To match that worst case, a fixed ef is slower in 13 of 16 runs.");

// ================================================================= 11. cost and speed
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Within 2.5% of a tuned fixed ef's cost", { placeholder: "title" });
stat(s, 0.6, 1.45, 3.7, "32 / 34", "runs where PercEF does less work than the tuned fixed ef (Ada-ef: 10 / 34)");
stat(s, 0.6, 3.15, 3.7, "31 / 34", "runs where PercEF is faster than a fixed ef with the same worst-case recall (Ada-ef: 11 / 34)");
stat(s, 0.6, 4.85, 3.7, "18 / 18", "near-Gaussian runs where PercEF is both cheaper and faster than the fixed ef", C.accent2);
s.addText("34 runs = 17 benchmarked datasets x 2 settings (3 cross-modal sets left out); work and time measured on all 34",
  { x: 0.6, y: 6.3, w: 3.9, h: 0.6, fontSize: 11, italic: true, color: C.accent6, margin: 0, isTextBox: true, objectName: "Runs definition" });
const sp = ["Fashion-MNIST", "DBpedia", "COCO-I2I", "LAION", "Yambda", "SIFT-1M", "Yahoo-MiniLM", "Deep1B", "Cohere", "BIGANN", "GIST", "MS Turing", "MS MARCO"];
s.addChart(pres.charts.BAR, [{ name: "Speed-up", labels: sp, values: [1.82, 1.38, 1.37, 1.34, 1.33, 1.25, 1.22, 1.19, 1.18, 1.11, 1.10, 1.08, 1.03] }], {
  x: 4.7, y: 1.35, w: 8.05, h: 4.4, barDir: "bar", chartColors: [HEX.ours], showLegend: false,
  showTitle: true, title: "PercEF speed-up over Ada-ef at about the same recall (x, setting R)", titleFontSize: 14, titleColor: HEX.ink,
  titleFontFace: "+mn-lt", catAxisLabelColor: HEX.muted, valAxisLabelColor: HEX.muted, catAxisLabelFontSize: 12,
  valAxisLabelFontSize: 11, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", catAxisOrientation: "maxMin",
  valAxisMinVal: 0, valGridLine: { color: HEX.grid, size: 0.75 }, catGridLine: { style: "none" },
  showValue: true, dataLabelFontSize: 11, dataLabelColor: HEX.ink, dataLabelFormatCode: "0.00", objectName: "Speed-up chart" });
s.addText("At the same mean recall on non-Gaussian data, PercEF's time is within 2.4% of the fixed ef's except DeepImage (7-9% slower): it spends saved work on hard queries, which cost more per distance.",
  { x: 4.7, y: 5.95, w: 8.05, h: 0.75, fontSize: 14, italic: true, color: C.accent2, margin: 0, isTextBox: true, objectName: "Speed note" });
s.addNotes("(0:10) PercEF does less work than the best fixed ef in 32 of 34 runs, never more than 2.5 percent extra. Matched on the worst case, it is faster than the fixed ef in 31 of 34 runs, and up to 1.8 times faster than Ada-ef. On bell-shaped data it is cheaper and faster in every run; Ada-ef's advantage there is a larger worst-case gain.");

// ================================================================= 12. checks
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Two checks: probe length and learned methods", { placeholder: "title" });
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
s.addText("Learned methods (DARTH, LAET) vs PercEF, setting R", { x: 6.95, y: 1.3, w: 5.8, h: 0.4, fontSize: 15, bold: true, color: C.text2, margin: 0, isTextBox: true, objectName: "DARTH caption" });
table(s, [
  ["Dataset", "p1 DARTH", "p1 PercEF", "Time DARTH", "Time LAET", "Time PercEF"],
  ["GloVe", "+0.080", "+0.029", "6% slower", "1% faster", "1% faster"],
  ["MS Turing", "+0.044", "+0.010", "42% slower", "16% slower", "0.9% faster"],
  ["DBpedia", "+0.067", "+0.034", "1.6x slower", "1.6x slower", "2% faster"],
  ["DeepImage", "+0.119", "+0.088", "2.1x slower", "2.1x slower", "7% slower"],
  ["Deep1B", "+0.098", "+0.081", "1.9x slower", "1.6x slower", "2% slower"],
  ["Landmark-DINO", "+0.078", "+0.039", "2.8x slower", "2.6x slower", "2% faster"],
  ["SIFT-1M", "+0.048", "+0.028", "2.3x slower", "1.8x slower", "0.4% faster"],
], 6.95, 1.75, 5.8, [1.25, 0.85, 0.85, 0.95, 0.95, 0.95], "DARTH table", 10);
s.addText("Time = against a tuned fixed ef of the same library. Learned methods have the best tail but are up to 2.8x slower and need training; PercEF is within 7% with none.", { x: 6.95, y: 4.75, w: 5.8, h: 0.9, fontSize: 12, italic: true, color: C.accent2, margin: 0, isTextBox: true, objectName: "DARTH takeaway" });
s.addNotes("(0:08) The gain comes from the thresholds, not the shorter probe. The learned methods DARTH and LAET have a better worst case, but are up to 2.8 times slower.");

// ================================================================= 13. combined idea (tested, not adopted)
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.hidden = true;                                           // tested and not adopted: skipped when presenting
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
s.addNotes("(hidden slide, skipped when presenting) Tested idea that did not hold: PercEF's score with Ada-ef's table.");

// ================================================================= 14. limits and next steps
pres.addSection({ title: "Wrap-up" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Wrap-up" });
s.addText("Limits and next steps", { placeholder: "title" });
card(s, 0.6, 1.4, 5.95, 3.4, "Limits, stated openly",
  "• Near-Gaussian text: Ada-ef gives larger worst-case gains (PercEF still cheaper and faster than the fixed ef)\n• Non-Gaussian, same mean recall: PercEF up to 2.4% slower than the fixed ef (DeepImage 7-9%)\n• Cross-modal queries (text searching images): neither method adapts\n• One failed KS prediction (SIFT-1B)\n• Cohere and LAION run as subsets (62 GB RAM)",
  { bodySize: 16, fill: "FCE9DF", headColor: C.accent4 });
card(s, 6.85, 1.4, 5.9, 3.4, "Next steps",
  "• Understand why Ada-ef's score and table together win the tail on GloVe\n• Final figures and numbers in the paper\n• Guide review, author list\n• arXiv preprint and code release\n• Submit: PVLDB Experiments, Analysis & Benchmarks track",
  { bodySize: 16, fill: "DCEFF3" });
s.addNotes("(0:06) On bell-shaped text data Ada-ef gives larger worst-case gains, on DeepImage PercEF is a few percent slower than the fixed ef at the same mean recall, and cross-modal queries remain open.");

// ================================================================= 15. conclusion
s = pres.addSlide({ masterName: "DARK", sectionTitle: "Wrap-up" });
s.addText("Measure, don't assume", { placeholder: "title" });
s.addText([
  { text: "Ada-ef's bell-curve model departs from the data on 9 of 41 datasets. PercEF measures the thresholds instead: better worst-case recall where it departs, never meaningfully costlier than a tuned fixed ef, and a seconds-long test that says which method to use.", options: { breakLine: true } },
  { text: "Thank you. Questions?", options: { bold: true } },
], { placeholder: "body" });
s.addNotes("(0:05) In short: measure the distribution, don't assume it. Thank you.");

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
