# SIGMOD 2027, Round 4: registration sheet and plan

Site: https://cmt3.research.microsoft.com/SIGMOD2027
Abstract + conflicts: **10 Oct 2026, 23:59 AoE** (= 11 Oct, 17:29 IST)
Full paper: **17 Oct 2026, 23:59 AoE** (= 18 Oct, 17:29 IST). Plan to upload on 16 Oct.
Rules: https://2027.sigmodconf.hosting.acm.org/calls_papers_sigmod_research.shtml

## Decide before registering (cannot change afterwards)

1. **Author list and order.** No changes after abstract submission. Every author needs a CMT
   account and an **ORCID** (https://orcid.org/register, 2 minutes each); missing ORCID or CMT
   registration is a desk-reject reason. Decide with the guide who is an author (team members,
   alumnus, guide) and in which order.
2. **Category, i.e. the title.** No title change after registration except at a reviewer's request.
   - **Experiments & Analysis (recommended).** Title must end in ": [Experiments & Analysis]".
     E&A papers are judged on "new insights into the strengths and weaknesses of existing methods".
     Our core finding (Ada-ef's Gaussian assumption fails on 9 of 41 datasets, a KS test predicts
     it, and what each method does then) is exactly that; PercEF is a small change to one part of
     Ada-ef, which would be the weak point under the regular track's novelty bar.
   - Regular research paper: same title without the suffix. `python make_sigmod.py --research`.

## What to enter

**Title (E&A):**
PercEF: Exploiting Empirical Percentiles for Adaptive HNSW Search Beyond the Gaussian Assumption: [Experiments & Analysis]

**Abstract** (plain text, 285 words; same as the paper):

HNSW indexes answer every query with the same search breadth ef, although some queries need far more of it than others. Ada-ef (SIGMOD 2026) sets ef per query from a difficulty score that models the similarities between a query and the database as a Gaussian, an assumption justified by the central limit theorem but not tested in its paper. We test it. On 41 public embedding datasets, the Gaussian model fits most modern text embeddings well, but it fails clearly on 9, including SIFT, GIST and Fashion-MNIST, and the failures cannot be guessed from the kind of data. Where the model fails, Ada-ef's score stops telling easy queries from hard ones, and on SIFT it gives every query the same ef. Our method, PercEF, replaces the Gaussian with the data's own empirical distance percentiles and keeps the rest of the mechanism. On the 8 non-Gaussian datasets we benchmark end to end, PercEF ranks queries better in all 16 runs and gives at least Ada-ef's tail recall in 15, at the same mean recall. Across 17 datasets it never costs more than 2.5% over a tuned fixed ef at the same mean recall and is cheaper in 32 of 34 runs, while Ada-ef is cheaper in 10. Matched instead on the recall of the worst 1% of queries, PercEF is faster than the tuned fixed ef in 31 of 34 runs, by a median of 15%; Ada-ef is faster in 11. A Kolmogorov-Smirnov test on the raw vectors, which takes seconds and needs no index, predicts the better score on 18 of 19 datasets. On near-Gaussian data PercEF is cheaper and faster than a tuned fixed ef in every run, while Ada-ef gives larger tail gains.

**Topics** (pick in CMT):
- Primary: Data Management Systems → Storage and indexing
- Secondary: Data Management Systems → Query processing; Data Models & Languages → Multimedia / information retrieval; Modern AI & Data Management → Queries over unstructured data

**Scope justification** (if asked; under 1000 characters):
Vector search over learned embeddings is now a core data management workload, and HNSW is the index most systems use. This paper studies how HNSW search breadth should be set per query: it tests the statistical assumption behind Ada-ef (SIGMOD 2026) on 41 public datasets, shows where it fails and what that costs, gives a seconds-long test on the raw vectors that predicts which difficulty score to use, and evaluates both methods against a tuned fixed-ef baseline on cost, latency and tail recall. It is an experiments and analysis study of query processing over a widely used vector index.

**Conflicts of interest:** each author declares their own domain conflicts (institution email domains)
and PC conflicts (co-authors in the last 2 years, same institution, advisor/advisee). Do not declare
conflicts that do not exist: spurious conflicts are a desk-reject reason too. The Ada-ef authors are
not a conflict unless you have worked with them.

**Incremental-submission justification:** not applicable (no earlier paper by this author group).

## Anonymity checklist (desk-reject if violated)

- [x] No names or affiliations in the PDF or its metadata (`make_sigmod.py` checks)
- [x] No funding or people in the acknowledgements; only the AI-use disclosure (ACM policy)
- [ ] Code through an **anonymous** repository: clean release repo → https://anonymous.4open.science
      (mirrors a GitHub repo with names removed; not indexed) → `python make_sigmod.py --repo-url <link>`
- [ ] Do not post the paper on arXiv or publicise it until the decision
- [x] Body ≤ 12 pages (now 7 + references); appendix as a separate PDF

## Plan

| Date | Task | Who |
|---|---|---|
| Wed 7 Oct | Push the pending commit; start `run_fixes.sh` on the server | you |
| Thu 8 Oct | Agree authors, order and category with the guide; every author makes an ORCID and a CMT account | team |
| Fri 9 Oct | **Register**: title, abstract, topics, authors with ORCID, conflicts. A day before the deadline | you |
| Sat 10 Oct | Fixes run done → copy back; analyse, update the paper (replace both `\pending`) | you + me |
| Sun 11 Oct | §3.3 sentence on the thresholds; use the spare pages (move the per-query cost and tail tables into the body) | me |
| Mon 12 Oct | Clean release repo (code, scripts, README per table) → anonymous mirror → link in the paper | me + you |
| Tue 13 Oct | Full read by the team and the guide | team |
| Wed 14 Oct | Their corrections; check every number against `paper_out_*` once more | me |
| Thu 15 Oct | Final build; check the desk-reject list; both PDFs read on a phone and printed once | you |
| Fri 16 Oct | **Upload** main PDF and appendix PDF; check the CMT preview | you |
| Sat 17 Oct | Deadline (AoE). Buffer only | — |

During review (to mid-December): LAET comparison, seed repeats, multi-thread throughput, cache-miss
measurement. These are the likely revision requests; having them ready makes a one-month revision easy.
