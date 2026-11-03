"""Build the journal versions of the corrected paper from one source, paper/main_v2.tex.

    python paper/make_venues.py [--tpl DIR]

Writes paper/venues/<venue>/ (self-contained: figures, tables, bib, class files), compiles each and
prints its page count:
  pvldb   PVLDB Vol. 20, Experiments, Analysis & Benchmark: official VLDB template (acmart 2.19 +
          pvldb.sty, github.com/vldbproceedings/VLDB-Template), single-blind, 12 pages excluding
          references, appendix included in the 12; artifact link in the template's availability block
  vldbj   The VLDB Journal (Springer): the journal's own template (svjour3 two-column, from
          link.springer.com/journal/778/submission-guidelines), single-blind, at most 25 pages,
          abstract 150-250 words, 4-6 keywords, declarations (competing interests, data
          availability), LLM use stated in the methods; appendix kept
  tkde    IEEE TKDE: IEEEtran journal/compsoc, single-blind, 12 pages including references (more
          pages are charged), abstract 100-200 words; the appendix becomes a separate
          supplementary PDF (supplement.pdf), as TKDE requires
--tpl: folder holding pvldb/ (the VLDB template) and vldbj/ (the VLDB Journal template).
"""
import argparse, os, re, shutil, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--tpl", default=os.path.join(HERE, "templates"))
TPL = ap.parse_args().tpl
src = open(os.path.join(HERE, "main_v2.tex"), encoding="utf8").read().replace("\r\n", "\n")

ABSTRACT = src.split("\\begin{abstract}\n", 1)[1].split("\\end{abstract}", 1)[0].strip()
BODY = src.split("\\pagestyle{plain}   % no running heads", 1)[1].split("\n", 1)[1]
BODY, rest = BODY.split("\\bibliographystyle{ACM-Reference-Format}", 1)
APPENDIX = rest.split("\\appendix", 1)[1].split("\\clearpage", 1)[0]
APPENDIX = APPENDIX.replace("\\section{Additional results}\n", "")

# TKDE: abstracts of 100-200 words; used for the VLDB Journal too
SHORT_ABSTRACT = r"""Ada-ef (SIGMOD 2026) sets the search breadth (\ef{}) of each HNSW query from a difficulty score
that assumes a query's similarities to the database are Gaussian. We test this assumption query by
query on 41 public datasets. It fits most text embeddings but departs significantly on nine, six of
them learned image, audio and recommendation embeddings. The departure matters less than its size
suggests: Ada-ef's score still ranks queries on most of these datasets, and it collapses only on the
most extreme ones and when hard queries move away from the bulk. Its practical limit is cost. A long
probe and a per-query $d \times d$ variance make it slower than a hindsight-tuned fixed \ef{} in 28
of 34 runs. PercEF keeps Ada-ef's score but takes its thresholds from the database's empirical
distance percentiles and probes 100 distances. At equal probe length it ranks queries about as well
as Ada-ef, yet it is cheaper than the tuned fixed \ef{} in 32 of 34 runs and faster in 29. Where the
Gaussian fits, Ada-ef buys larger gains in tail recall. We also report pitfalls in evaluating
adaptive search, one of which affected our own first evaluation."""
KEYWORDS = ["approximate nearest neighbor search", "HNSW", "adaptive search",
            "experimental evaluation", "distributional assumptions"]
AI_NOTE = ("Generative AI (Claude, Anthropic) was used to write and refactor experiment and analysis code,\n"
           "and to draft and edit text. The authors designed the study, chose and checked every\n"
           "experiment, verified every reported number against the released result files, and take\n"
           "full responsibility for the content.")
TITLE = "Not All Embeddings Are Gaussian: Revisiting Adaptive HNSW Search"
# authors in order: (name, initials form, email, affiliation key); the first is the corresponding author
PES = ("PES University", "Bengaluru", "India")
UIUC = ("University of Illinois Urbana-Champaign", "Urbana, IL", "USA")
AUTHORS = [("Pranav Shenvi", "P. Shenvi", "shenvipranav@gmail.com", PES),
           ("M G Shrivardhan Swaroop", "M. G. Shrivardhan Swaroop", "shrivardhanswaroop.mg@gmail.com", PES),
           ("Megha Shah", "M. Shah", "meghashah1125@gmail.com", PES),
           ("Kiran Hombal", "K. Hombal", "khombal2@illinois.edu", UIUC),
           ("Shylaja S S", "S. S. Shylaja", "shylaja.sharath@pes.edu", PES)]
NAMES = [a[0] for a in AUTHORS]
NAME_LIST = ", ".join(NAMES[:-1]) + " and " + NAMES[-1]


def acm_authors():
    """acmart (PVLDB) author block: one \\author per person with affiliation and email."""
    return "".join(f"\\author{{{n}}}\n\\affiliation{{\\institution{{{a[0]}}}\\city{{{a[1]}}}\\country{{{a[2]}}}}}\n"
                   f"\\email{{{e}}}\n" for n, _, e, a in AUTHORS)


def by_affiliation():
    groups = []
    for _, short, e, a in AUTHORS:
        g = next((g for g in groups if g[0] == a), None)
        if g is None:
            groups.append(g := (a, []))
        g[1].append((short, e))
    return groups


def join_names(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
REPO = "https://anonymous.4open.science/r/PercEF/"
MACROS = (r"\usepackage{booktabs}" "\n" r"\usepackage{xcolor}" "\n" r"\usepackage{amsmath}" "\n"
          r"\newcommand{\pending}[1]{\textcolor{red}{[#1]}}" "\n" r"\newcommand{\ef}{\textit{ef}}" "\n"
          r"\newcommand{\KS}{\mathrm{KS}}" "\n")
MACROS_NON_ACM = MACROS + r"\usepackage{amssymb}" "\n"      # acmart loads amssymb itself
# PVLDB counts the appendix in its 12 pages: the full-page offline-cost table stays in the repository
# (paper_outputs/table4_offline) and the prose of the latency paragraph keeps its numbers
OFFLINE_TABLE = re.compile(r"\\begin\{table\*\}\[h\]\s*\\caption\{Offline cost\.\}.*?\\end\{table\*\}\s*", re.S)


def setup(venue, files):
    out = os.path.join(HERE, "venues", venue)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    for d in ("figures", "tables"):
        shutil.copytree(os.path.join(HERE, d), os.path.join(out, d))
    shutil.copy(os.path.join(HERE, "refs.bib"), out)
    for f in files:
        shutil.copy(f, out)
    return out


def build(out, name, bib=True):
    run = lambda *c: subprocess.run(c, cwd=out, capture_output=True, text=True)
    run("pdflatex", "-interaction=nonstopmode", name + ".tex")
    if bib:
        run("bibtex", name)
    run("pdflatex", "-interaction=nonstopmode", name + ".tex")
    r = run("pdflatex", "-interaction=nonstopmode", name + ".tex")
    pages = re.search(r"Output written on .*?\((\d+) pages", r.stdout)
    log = open(os.path.join(out, name + ".log"), encoding="latin-1").read()
    bad = sorted(set(re.findall(r"(LaTeX Error[^\n]*|Undefined control sequence|Reference `[^']*' on page \d+ undefined"
                                r"|Citation `[^']*' on page \d+ undefined)", log)))
    print(f"  {os.path.relpath(os.path.join(out, name + '.pdf'), HERE)}: "
          f"{pages.group(1) if pages else '?'} pages" + (f"; problems: {bad[:5]}" if bad else ""))


# ---- PVLDB ------------------------------------------------------------------------------------------
p = os.path.join(TPL, "pvldb")
out = setup("pvldb", [os.path.join(p, f) for f in ("acmart.cls", "pvldb.sty", "ACM-Reference-Format.bst")])
open(os.path.join(out, "main.tex"), "w", encoding="utf8").write(
    "\\documentclass[sigconf, nonacm]{acmart}\n\\usepackage{pvldb}\n" + MACROS +
    "\\renewcommand\\vldbdoi{XX.XX/XXX.XX}\n\\renewcommand\\vldbpages{XXX-XXX}\n"
    f"\\renewcommand\\vldbavailabilityurl{{{REPO}}}\n\n\\begin{{document}}\n\\title{{{TITLE}}}\n\n"
    + acm_authors() + "\n" +
    f"\\begin{{abstract}}\n{ABSTRACT}\n\\end{{abstract}}\n\n\\maketitle\n\n\\vldbtopmatter\n"
    + BODY + f"\\begin{{acks}}\n{AI_NOTE}\n\\end{{acks}}\n\n"
    "\\bibliographystyle{ACM-Reference-Format}\n\\bibliography{refs}\n\n\\appendix\n\\section{Additional results}\n"
    + OFFLINE_TABLE.sub("", APPENDIX) + "\\clearpage\n\\end{document}\n")
build(out, "main")

# ---- VLDB Journal -------------------------------------------------------------------------------------
# Springer's guidelines: LLM use documented in the methods section (not the acknowledgements), and a
# declarations section with competing interests and a data availability statement
s = os.path.join(TPL, "vldbj")
out = setup("vldbj", [os.path.join(s, f) for f in ("svjour3.cls", "svglov3.clo", "spmpsci.bst")])
VJ_METHODS = "All searches run on one thread."
assert BODY.count(VJ_METHODS) == 1
vj_body = BODY.replace(VJ_METHODS, VJ_METHODS + "\n\n\\paragraph{Use of generative AI.}\n" + AI_NOTE)
DECLARATIONS = ("\\section*{Declarations}\n\n\\paragraph{Competing interests.} The authors have no competing "
                "financial or non-financial interests to declare.\n\n\\paragraph{Data availability.} All datasets "
                "are public; download scripts, code and every result file behind the tables and figures are at "
                f"\\url{{{REPO}}}.\n\n")
open(os.path.join(out, "main.tex"), "w", encoding="utf8").write(
    "\\RequirePackage{fix-cm}\n\\documentclass[twocolumn]{svjour3}\n\\smartqed\n\\journalname{The VLDB Journal}\n"
    "\\usepackage{graphicx}\n\\usepackage{url}\n" + MACROS_NON_ACM +
    f"\\begin{{document}}\n\\title{{{TITLE}}}\n\\author{{{' \\and '.join(NAMES)}}}\n"
    # svjour3: one \institute entry per affiliation; emails of all authors, the corresponding one marked
    "\\institute{" + " \\and\n".join(
        f"{join_names([s_ for s_, _ in g])} \\at {', '.join(a)} \\\\ "
        + " \\\\ ".join(f"{s_}: \\email{{{e}}}" + (" (corresponding author)" if e == AUTHORS[0][2] else "")
                        for s_, e in g) for a, g in by_affiliation()) + "}\n"
    "\\date{Received: date / Accepted: date}\n\\maketitle\n\n"
    f"\\begin{{abstract}}\n{SHORT_ABSTRACT}\n\\keywords{{{' \\and '.join(KEYWORDS)}}}\n\\end{{abstract}}\n"
    + vj_body + DECLARATIONS + "\\bibliographystyle{spmpsci}\n\\bibliography{refs}\n\n\\appendix\n\\section{Additional results}\n"
    + APPENDIX + "\\clearpage\n\\end{document}\n")
build(out, "main")

# ---- IEEE TKDE --------------------------------------------------------------------------------------
out = setup("tkde", [])
TK_PRE = ("\\documentclass[10pt,journal,compsoc]{IEEEtran}\n\\usepackage[nocompress]{cite}\n"
          "\\usepackage{graphicx}\n\\usepackage{url}\n\\usepackage{xr}\n" + MACROS_NON_ACM)
body = BODY.replace("in the appendix", "in the supplementary material")
open(os.path.join(out, "main.tex"), "w", encoding="utf8").write(
    TK_PRE + "\\externaldocument[S-]{supplement}\n\\begin{document}\n"
    f"\\title{{{TITLE}}}\n\\author{{" + NAME_LIST.replace(" and ", ", and ") + "%\n\\IEEEcompsocitemizethanks{"
    + "\n".join(f"\\IEEEcompsocthanksitem {join_names([s_ for s_, _ in g])} "
                f"{'is' if len(g) == 1 else 'are'} with {', '.join(a)}. "
                f"E-mail: {', '.join(e for _, e in g)}" for a, g in by_affiliation())
    + f"\n\\IEEEcompsocthanksitem Corresponding author: {AUTHORS[0][1]}.}}}}\n\n"
    f"\\IEEEtitleabstractindextext{{%\n\\begin{{abstract}}\n{SHORT_ABSTRACT}\n\\end{{abstract}}\n"
    f"\\begin{{IEEEkeywords}}\n{', '.join(KEYWORDS)}\n\\end{{IEEEkeywords}}}}\n\n\\maketitle\n"
    "\\IEEEdisplaynontitleabstractindextext\n\\IEEEpeerreviewmaketitle\n"
    # labels of the supplement are referenced as S-<label>
    + re.sub(r"\\ref\{(fig:controlled|fig:cdf|tab:querycost)\}", r"\\ref{S-\1}", body)
    + f"\\section*{{Acknowledgments}}\n{AI_NOTE}\n\n\\bibliographystyle{{IEEEtran}}\n\\bibliography{{refs}}\n\\end{{document}}\n")
open(os.path.join(out, "supplement.tex"), "w", encoding="utf8").write(
    TK_PRE.replace("\\usepackage{xr}\n", "") + "\\renewcommand{\\thefigure}{S\\arabic{figure}}\n"
    "\\renewcommand{\\thetable}{S\\arabic{table}}\n\\begin{document}\n"
    f"\\title{{Supplementary Material for ``{TITLE}''}}\n\\author{{{NAME_LIST.replace(' and ', ', and ')}}}\n\\maketitle\n"
    + APPENDIX + "\\clearpage\n\\end{document}\n")
build(out, "supplement", bib=False)
build(out, "main")
