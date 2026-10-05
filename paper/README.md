# Paper draft

`main.tex` (ACM `acmart`, sigconf), `refs.bib`, `figures/` and `tables/` copied from
`analysis/make_paper_figures.py` output (`paper_out_*`). To refresh after new runs:

    python3 analysis/make_paper_figures.py           # on the server
    cp paper_out_<latest>/*.pdf paper/figures/ && cp paper_out_<latest>/*.tex paper/tables/

Build locally (Windows, MiKTeX): `powershell -ExecutionPolicy Bypass -File paper/build.ps1` (opens
main.pdf). Or upload the `paper/` folder to Overleaf, or `latexmk -pdf main.tex` where TeX Live is
installed. Red `[...]` marks (`\pending{}`) are numbers still being measured or details to fill;
none may remain at submission. Bib entries not marked VERIFIED need their volume/pages checked.
