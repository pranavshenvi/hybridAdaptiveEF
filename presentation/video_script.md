# Demo video script (under 2 minutes)

About 310 words at a calm pace; the times add up to about 2:00. The hidden slide (our score +
Ada-ef's table) is skipped automatically in slideshow mode. The same lines are in each slide's
speaker notes.

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
| 11 | Non-Gaussian results | "Each pair of bars is a dataset; longer to the right is better. On the non-Gaussian datasets PercEF ranks queries better in every run and beats Ada-ef's worst case in 15 of 16: on DeepImage by 8.5 recall points, against 4.3. To match that worst case, a fixed ef is slower in 13 of 16 runs." | 0:10 |
| 12 | Cost | "PercEF does less work than the best fixed ef in 32 of 34 runs, never more than 2.5 percent extra. Matched on the worst case, it is faster than the fixed ef in 31 of 34 runs, and up to 1.8 times faster than Ada-ef. On bell-shaped data it is cheaper and faster in every run; Ada-ef's advantage there is a larger worst-case gain." | 0:10 |
| 13 | Checks | "The gain comes from the thresholds, not the shorter probe. The learned method DARTH has a better worst case, but is up to 2.3 times slower." | 0:08 |
| 14 | Limits | "On bell-shaped text data Ada-ef gives larger worst-case gains, on DeepImage PercEF is a few percent slower than the fixed ef at the same mean recall, and cross-modal queries remain open." | 0:06 |
| 15 | Conclusion | "In short: measure the distribution, don't assume it. Thank you." | 0:05 |

If you run long, drop slide 10 (setup) and shorten slide 2.

## What each number means

| Number | Slide | What it counts |
|---|---|---|
| **41** | 6 | Datasets where we only *measured* the bell-curve fit (KS test, no search runs) |
| **9 of 41** | 6 | Of those, the clearly non-Gaussian ones |
| **20** | 10 | Datasets we fully *benchmarked* with search runs (a subset of the 41) |
| **18 / 19** | 8 | Of the 20 benchmarked, 19 had a clear winner on ranking (one was a near tie); KS picked that winner for 18 |
| **1 miss** | 8 | SIFT-1B: KS predicted Ada-ef, but PercEF ranked better |
| **16** | 11 | 8 non-Gaussian benchmarked datasets x 2 calibration settings (200 corpus points, or real queries). The 9th non-Gaussian set, Last.fm, has cross-modal queries, where no method adapts, so it is counted separately |
| **16 / 16, 15 / 16** | 11 | Runs where PercEF's score orders queries by true difficulty better than Ada-ef's (Spearman rank correlation between score and each calibration query's minimum ef for 95% recall; one per run) / has at least Ada-ef's worst-case gain (the 16th is a tie near zero). Fashion-MNIST P is a near-tie in ranking (0.22 vs 0.20) |
| **13 / 16** | 11 | Runs where PercEF is faster than the fixed ef that gives the same worst-case recall. Ada-ef: 1 / 16. Two separate counts, not a split of 16 |
| **34** | 12 | 17 datasets x 2 settings: the 20 benchmarked minus the 3 cross-modal ones |
| **32 / 34** | 12 | Runs where PercEF does less work than the best fixed ef (Ada-ef: 10 / 34) |
| **2.5%** | 12 | PercEF's worst extra cost over the best fixed ef, in any of the 34 runs |
| **31 / 34** | 12 | Runs where PercEF is faster than the fixed ef that gives the same worst-case (p1) recall (Ada-ef: 11 / 34). At the same mean recall instead: 26 / 34, a count that moves by a few runs between timings |
| **18 / 18** | 12 | Near-Gaussian and borderline runs where PercEF is both cheaper and faster than the best fixed ef |
| **up to 1.82x** | 12 | PercEF's speed-up over Ada-ef on 13 datasets where both reach about the same recall |
