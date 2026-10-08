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
     Our core finding (Ada-ef's Gaussian model departs from the data on 9 of 41 datasets, a KS test predicts
     it, and what each method does then) is exactly that; PercEF is a small change to one part of
     Ada-ef, which would be the weak point under the regular track's novelty bar.
   - Regular research paper: same title without the suffix. `python make_sigmod.py --research`.

## What to enter

**Title (E&A):**
PercEF: Exploiting Empirical Percentiles for Adaptive HNSW Search Beyond the Gaussian Assumption: [Experiments & Analysis]

**Abstract** (plain text, 325 words; same as the paper):

HNSW indexes search every query with the same breadth, ef, although some queries need far more of it than others. Ada-ef (SIGMOD 2026) assigns ef per query from a difficulty score built on a Gaussian model of the similarities between a query and the database. The model follows from the central limit theorem, and the Ada-ef paper supports its assumptions on two text-embedding datasets. We measure how well the Gaussian fits with a Kolmogorov-Smirnov (KS) statistic between each query's similarities and the Gaussian, on 41 public embedding datasets: it fits most modern text embeddings closely but departs from the data on 9, including SIFT, GIST and Fashion-MNIST. We then benchmark 17 datasets end to end, each with two calibration sets: 8 on which the Gaussian departs (16 runs) and 9 on which it fits (18 runs). On the 16 runs where it departs, Ada-ef's advantage over a hindsight-tuned fixed ef at the same mean recall largely disappears: it does more work in 14, and the recall of the worst 1% of queries is lower in 10. We propose PercEF, which keeps Ada-ef's mechanism but takes the thresholds from the data's own empirical distance percentiles instead of the Gaussian. On the same 16 runs, PercEF ranks queries by difficulty better in all 16, does less work than the tuned fixed ef in 14 and never more than 2.5% extra, and keeps the recall of the worst 1% of queries at or above the fixed ef's in 13, raising it by up to 8.8 points; in the other 3 it is at most 1.5 points lower. On the 18 near-Gaussian runs, PercEF is cheaper and faster than the tuned fixed ef in all 18 without lowering worst-case recall, while Ada-ef achieves larger worst-case gains in 11 of them. The same KS statistic, computed on the raw vectors in seconds and without an index, indicates which of the two scores ranks queries better on 18 of 19 datasets.

**Topics** (pick in CMT):
- Primary: Data Management Systems → Storage and indexing
- Secondary: Data Management Systems → Query processing; Data Models & Languages → Multimedia / information retrieval; Modern AI & Data Management → Queries over unstructured data

**Scope justification** (if asked; under 1000 characters):
Vector search over learned embeddings is now a core data management workload, and HNSW is the index most systems use. This paper studies how HNSW search breadth should be set per query: it measures how well the statistical model behind Ada-ef (SIGMOD 2026) fits 41 public datasets, shows where it departs from the data and what that costs, gives a seconds-long test on the raw vectors that predicts which difficulty score to use, and evaluates both methods against a tuned fixed-ef baseline on cost, latency and tail recall. It is an experiments and analysis study of query processing over a widely used vector index.

**Conflicts of interest:** each author declares their own domain conflicts (institution email domains)
and PC conflicts (co-authors in the last 2 years, same institution, advisor/advisee). Do not declare
conflicts that do not exist: spurious conflicts are a desk-reject reason too. The Ada-ef authors are
not a conflict unless you have worked with them.

**Incremental-submission justification:** not applicable (no earlier paper by this author group).

## Anonymity checklist (desk-reject if violated)

- [x] No names or affiliations in the PDF or its metadata (`make_sigmod.py` checks)
- [x] No funding or people in the acknowledgements; only the AI-use disclosure (ACM policy)
- [x] Code through an **anonymous** repository: https://anonymous.4open.science/r/PercEF-SIGMOD27 (CMT artifact field; build with `python make_sigmod.py --repo-url https://anonymous.4open.science/r/PercEF-SIGMOD27`)
- [ ] (was:) Code through an **anonymous** repository: clean release repo → https://anonymous.4open.science
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
