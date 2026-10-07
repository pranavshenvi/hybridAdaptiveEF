"""Build the SIGMOD 2027 submission from main.tex (one source; edit main.tex, then re-run this).

SIGMOD 2027 research track (calls_papers_sigmod_research.shtml):
  - 2-column ACM sigconf, double-anonymous: no names, affiliations, funding or people in the
    acknowledgements; code through an anonymous repository
  - 12 pages excluding references; the appendix is a SEPARATE pdf (reviewers need not read it);
    nothing but references after page 12
  - Experiments & Analysis papers end their title with ": [Experiments & Analysis]"
  - ACM policy on generative AI: use must be disclosed in the work

Writes sigmod/main_anon.tex and sigmod/appendix_anon.tex (cross-references between the two
through the xr package), compiles both, and checks: page count of the main body, leftover
\\pending, and identifying strings.

    python make_sigmod.py [--research] [--repo-url URL]
      --research   regular research paper title (default: Experiments & Analysis suffix)
      --repo-url   anonymous repository link (e.g. https://anonymous.4open.science/r/...)
"""
import argparse, os, re, shutil, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--research", action="store_true")
ap.add_argument("--repo-url", default=None)
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sigmod")
os.makedirs(OUT, exist_ok=True)
src = open(os.path.join(HERE, "main.tex"), encoding="utf8").read().replace("\r\n", "\n")

AI_DISCLOSURE = (
    "\\begin{acks}\n"
    "Generative AI (Claude, Anthropic) was used to write and refactor experiment and analysis code,\n"
    "and to draft and edit text. The authors designed the study, chose and checked every\n"
    "experiment, verified every reported number against the released result files, and take\n"
    "full responsibility for the content.\n"
    "\\end{acks}\n\n")

# ---- split the source -------------------------------------------------------------------------
pre, body = src.split("\\begin{document}", 1)
body_main, body_app = body.split("\\appendix", 1)
body_app = body_app.replace("\\end{document}", "")

# preamble: anonymous sigconf, comments describing drafts dropped
pre = "\n".join(l for l in pre.split("\n") if not l.lstrip().startswith("%"))
pre = re.sub(r"\\documentclass\[[^\]]*\]\{acmart\}", r"\\documentclass[sigconf,anonymous]{acmart}", pre)

# title (category suffix), author block replaced
suffix = "" if args.research else ": [Experiments \\& Analysis]"
body_main = re.sub(r"\\title\{(.*?)\}\n", lambda m: "\\title{" + m.group(1) + suffix + "}\n", body_main, count=1)
body_main = re.sub(r"\\author\{.*?\}\n\\affiliation\{.*?\}\n\\email\{.*?\}\n",
                   "\\\\author{Anonymous Author(s)}\n\\\\affiliation{\\\\institution{Anonymous}\\\\country{}}\n",
                   body_main, flags=re.S)

# repository link
if args.repo_url:
    body_main = body_main.replace("\\pending{repository URL}", "Available at \\url{" + args.repo_url + "}.")
# appendix is a separate document now
body_main = body_main.replace("in the appendix", "in the supplementary appendix")
# AI disclosure before the references
body_main = body_main.replace("\\bibliographystyle{", AI_DISCLOSURE + "\\bibliographystyle{", 1)

xr_main = "\\usepackage{xr}\n\\externaldocument{appendix_anon}\n"
xr_app = ("\\usepackage{xr}\n\\externaldocument{main_anon}\n"
          "\\renewcommand{\\thetable}{A\\arabic{table}}\n\\renewcommand{\\thefigure}{A\\arabic{figure}}\n")
title = re.search(r"\\title\{(.*?)\}\n", body_main).group(1)

main_tex = pre + xr_main + "\\begin{document}" + body_main + "\\end{document}\n"
app_tex = (pre + xr_app + "\\begin{document}\n\\title{Supplementary appendix to: " + title + "}\n"
           "\\author{Anonymous Author(s)}\n\\affiliation{\\institution{Anonymous}\\country{}}\n"
           "\\maketitle\n\\appendix\n" + body_app +
           ("\n\\bibliographystyle{ACM-Reference-Format}\n\\bibliography{refs}\n" if "\\cite" in body_app else "")
           + "\\end{document}\n")
for name, text in (("main_anon.tex", main_tex), ("appendix_anon.tex", app_tex)):
    open(os.path.join(OUT, name), "w", encoding="utf8").write(text)
for d in ("figures", "tables"):
    if os.path.exists(os.path.join(OUT, d)):
        shutil.rmtree(os.path.join(OUT, d))
    shutil.copytree(os.path.join(HERE, d), os.path.join(OUT, d))
shutil.copy(os.path.join(HERE, "refs.bib"), OUT)

# ---- compile: main, appendix, bibtex, main, appendix, main -----------------------------------
miktex = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "MiKTeX", "miktex", "bin", "x64")
pdflatex = os.path.join(miktex, "pdflatex.exe") if os.path.exists(miktex) else "pdflatex"
bibtex = os.path.join(miktex, "bibtex.exe") if os.path.exists(miktex) else "bibtex"
run = lambda *c: subprocess.run(c, cwd=OUT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for step in ("main_anon", "appendix_anon"):
    run(pdflatex, "-interaction=nonstopmode", step + ".tex")
run(bibtex, "main_anon")
if "\\cite" in body_app:
    run(bibtex, "appendix_anon")
for step in ("main_anon", "appendix_anon", "main_anon", "appendix_anon"):
    run(pdflatex, "-interaction=nonstopmode", step + ".tex")

# ---- checks -------------------------------------------------------------------------------------
problems = []
log = open(os.path.join(OUT, "main_anon.log"), encoding="latin-1").read()
m = re.search(r"Output written on main_anon\.pdf \((\d+) pages", log)
pages = int(m.group(1)) if m else None
if "undefined" in log and re.search(r"Reference .* undefined", log):
    problems.append("undefined references in main_anon (see main_anon.log)")
for name, text in (("main", main_tex), ("appendix", app_tex)):
    n = len(re.findall(r"\\pending\{", text.split("\\begin{document}", 1)[1]))
    if n:
        problems.append(f"{n} \\pending left in the {name} document")
    for bad in ("Shenvi", "Pranav", "shenvipranav", "PES University", "github.com/pranav"):
        if bad.lower() in text.split("\\begin{document}", 1)[1].lower() and bad != "Anonymous":
            problems.append(f"identifying string '{bad}' in the {name} document")
# where do the references start? the body must end by page 12
pdftotext = shutil.which("pdftotext")
pt_local = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages")
if not pdftotext and os.path.exists(pt_local):
    for root, _, files in os.walk(pt_local):
        if "pdftotext.exe" in files:
            pdftotext = os.path.join(root, "pdftotext.exe"); break
ref_page = None
if pdftotext and pages:
    for p in range(1, pages + 1):
        t = subprocess.run([pdftotext, "-f", str(p), "-l", str(p), os.path.join(OUT, "main_anon.pdf"), "-"],
                           capture_output=True, text=True, encoding="utf8", errors="ignore").stdout
        if re.search(r"^\s*REFERENCES\s*$|^\s*References\s*$", t, re.M):
            ref_page = p; break
print(f"title: {title}")
print(f"main_anon.pdf: {pages} pages, references start on page {ref_page} (body must end by page 12)")
if ref_page and ref_page > 12:
    problems.append("body runs past page 12")
print("checks: " + ("OK" if not problems else "; ".join(problems)))
