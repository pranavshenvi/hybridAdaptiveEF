# Demo video script (under 2 minutes)

About 300 words at a calm pace; the times add up to 2:00. The hidden slide (our score + Ada-ef's
table) is skipped automatically in slideshow mode.

| # | Slide | Say (about) | Time |
|---|---|---|---|
| 1 | Title | "Hi, we are [team]. Our project is PercEF: making adaptive vector search work when the data is not bell-shaped." | 0:05 |
| 2 | Background | "Vector search finds the items closest to a query among millions of embeddings. To be fast, an index like HNSW walks a layered graph from the top down. A setting called ef decides how many candidates the search keeps: more is more accurate, but slower." | 0:12 |
| 3 | Problem | "Most systems use one ef for every query. Easy queries waste work, and hard queries miss some of their neighbours." | 0:08 |
| 4 | Existing work | "Existing fixes either train a model per dataset, like DARTH, or, like Ada-ef from SIGMOD 2026, assume that a query's similarities follow a bell curve." | 0:08 |
| 5 | How Ada-ef works | "Ada-ef places its difficulty thresholds using that bell curve. Nobody had checked whether real data is bell-shaped." | 0:08 |
| 6 | The diagnosis | "We tested it on 41 datasets. Nine are clearly not bell-shaped, including SIFT and GIST, and you cannot tell from the kind of data. On those, Ada-ef stops adapting." | 0:10 |
| 7 | PercEF | "Our method, PercEF, keeps Ada-ef's design but measures the thresholds from the data's own distance percentiles, with a ten times shorter probe and no covariance matrix." | 0:10 |
| 8 | KS test | "A KS test on the raw vectors, which takes seconds, tells you which method to use. It picked the right one on 18 of 19 datasets, with every prediction written down before the run." | 0:08 |
| 9 | Novelty | "So we contribute the first test of this assumption, a distribution-free score, a cost floor, and an offline test." | 0:08 |
| 10 | Setup | "We followed Ada-ef's own protocol on 20 datasets and compared everything against the best fixed ef." | 0:05 |
| 11 | Non-Gaussian results | "On the non-Gaussian datasets, PercEF improves the worst one percent of queries in every run: on DeepImage by 8.5 recall points, against 4.3 for Ada-ef." | 0:10 |
| 12 | Cost | "And it never costs more than 2.5 percent over the best fixed ef, while running 1.0 to 1.4 times faster than Ada-ef." | 0:08 |
| 13 | Checks | "The gain comes from the thresholds, not the shorter probe. The learned method DARTH has a better worst case, but is up to 2.3 times slower." | 0:08 |
| 14 | Limits | "On bell-shaped text data Ada-ef still wins, and queries from a different modality remain open." | 0:06 |
| 15 | Conclusion | "In short: measure the distribution, don't assume it. Thank you." | 0:05 |

If you run long, drop slide 10 (setup) and shorten slide 2.

## What each number means

| Number | Slide | What it counts |
|---|---|---|
| **41** | 6 | Datasets where we only *measured* the bell-curve fit (KS test, no search runs) |
| **9 of 41** | 6 | Of those, the clearly non-Gaussian ones |
| **20** | 10 | Datasets we fully *benchmarked* with search runs (a subset of the 41) |
| **18 / 19** | 8 | Of the 20 benchmarked, 19 had a clear winner on ranking (one was a near tie); KS picked that winner for 18 |
| **14** | 8 | Datasets added after the first six, each with its prediction written down before running; one was wrong (SIFT-1B) |
| **0** | 8 | Alternative tail-focused tests that did better than KS |
| **16** | 11 | 8 non-Gaussian benchmarked datasets x 2 calibration settings (200 corpus points, or real queries). The 9th non-Gaussian set, Last.fm, has cross-modal queries, where no method adapts, so it is counted separately |
| **16 / 16, 16 / 16** | 11 | Runs where PercEF ranks queries better than Ada-ef / has at least Ada-ef's worst-case gain |
| **14 vs 2** | 11 | Of the same 16 runs, how often each method does less work than the best fixed ef |
| **34** | 12 | 17 datasets x 2 settings: the 20 benchmarked minus the 3 cross-modal ones |
| **32 / 34** | 12 | Runs where PercEF does less work than the best fixed ef (Ada-ef: 10 / 34) |
| **2.5%** | 12 | PercEF's worst extra cost over the best fixed ef, in any of the 34 runs |
| **14 / 18** | 12 | Of the 18 runs where we measured time, how often PercEF is faster than the best fixed ef |
| **1.03-1.37x** | 12 | PercEF's speed-up over Ada-ef on 7 datasets, at about the same recall |
