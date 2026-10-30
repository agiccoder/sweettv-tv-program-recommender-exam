# Implemented methods and evaluation protocol

This document describes the executable pipeline, not findings about competition
data. Synthetic test results establish implementation behavior only. Dataset
counts, validation quality, runtime, chosen strategy, and external scores must be
read from an actual run's exports; none are assumed here.

Implementation entry points are [data.py](../src/sweettv/data.py),
[models.py](../src/sweettv/models.py), [metrics.py](../src/sweettv/metrics.py),
[cli.py](../src/sweettv/cli.py), and [outputs.py](../src/sweettv/outputs.py).

## Interaction construction

The view channel is `vsetv_id`; it is matched to schedule `channel_id`.
`duraton` is audited against `stop_time - start_time`, in seconds. Schedule
`duration` is independently compared with the next distinct airing start on the
same channel under seconds, minutes, and milliseconds interpretations. Ambiguous
units stop the pipeline unless an explicit, documented conversion is supplied.
Naive timestamps retain source wall-clock ordering; this does not establish the
source timezone. Mixed naive and absolute timestamp coordinates are rejected.

All intervals are half-open. A view `[v_s,v_e)` and airing `[a_s,a_e)` contribute

```text
max(0, min(v_e,a_e) - max(v_s,a_s))
```

only on the same channel. View intervals are unioned within user and channel,
including duplicate, overlapping, and adjacent fragments. Same-item overlapping
schedule records are unioned; exact adjacency remains separate airings. A sweep
turns each channel schedule into disjoint segments. Segments with conflicting
nonzero programme labels are discarded as ambiguous. A nonzero label takes
precedence over an overlapping unlabelled record. Programme ID zero is never an
item or a negative preference.

Overlap fragments are summed per user and canonical airing before the fixed
120-second absolute threshold is applied. Accepted overlaps are then aggregated
by user and programme into `seconds` and `watches`; `watches` counts accepted
canonical airings. The threshold is a predeclared engineering setting, not a
programme-normalized watch percentage or a threshold optimized using holdout
quality. Its suitability still needs discussion against the actual duration and
overlap audits. Unlabelled viewing is retained for audit context only.

## Temporal protocol and cohort

The real files spill across their nominal week boundaries. The explicit
`config/calendar-boundaries.json` fixes source-wall-clock midnights at
2020-03-09 (week 11), 2020-05-18 (week 21), 2020-07-27 (week 31), and
2020-10-19 (week 43, exclusive final end). These dates follow inspection of
actual raw bounds, not model-quality selection. Without an explicit calendar,
the importer accepts only nonoverlapping observed file start ranges.

The validation experiment fits on `[2020-03-09,2020-05-18)` and labels from
`[2020-05-18,2020-07-27)`. Crossing views are clipped to the windows, and
thresholding is performed independently within each. No view seconds at or
after 2020-07-27 enter preferences. Historical schedules are combined for the
join, with duplicated same-item intervals unioned. Training and validation
metadata/candidates are filtered by positive interval overlap with their own
window, preventing the train-file tail from fitting validation-period fields.
Future candidates use only future-schedule rows overlapping the final window.

Only training-period interactions and metadata fit popularity, user profiles,
vocabulary, IDF, and the co-view graph. Validation schedule metadata supplies
candidate identity and display labels, never new fitted content vectors.
Candidates are distinct positive programme IDs with positive schedule duration
in the target window. Every method uses the same pool and allows repeats of
previously watched programmes.

The primary validation cohort contains template users with at least one unique,
eligible observed holdout programme. Users with no usable training history stay
in this cohort and receive the common fallback. Users with no observable eligible
target are excluded before evaluating any model. This conditional cohort is not
a claim about every template user. `cohort.json` records the IDs, denominators,
overlapping exclusion reasons, and holdout users outside the template. All
template users still receive final predictions.

## Models

Let `s_ui` be accepted overlap seconds summed for training user `u` and programme
`i`. Histories use `w_ui = log(1+s_ui)`; no fabricated explicit rating is used.

### Popularity

```text
pop(i) = number of distinct training users with s_ui > 0
```

This is an independent baseline, not one of the two qualifying personalized
families. Repeated rows for the same user and programme do not increase its
viewer count.

### Content cosine

Permitted canonical metadata fields are `tv_show_category`,
`tv_show_genre_1`, `tv_show_genre_2`, `tv_show_genre_3`, `year_of_production`,
`director`, and `actors`. Titles do not enter the representation. Values receive
Unicode NFKC normalization, case folding, and whitespace normalization. Actors
and directors split into full-name values on comma, semicolon, pipe, or newline.
Other values remain whole categorical tokens. Field prefixes distinguish
features, including each genre position. Missing and null values add no feature.

For the caller's training-schedule metadata items only, build binary feature
counts `b_it`. With `N` training metadata items and document frequency `df_t`:

```text
idf_t = log((1+N)/(1+df_t)) + 1
x_i = L2_normalize(b_i * idf)
q_u = L2_normalize(sum_i w_ui * x_i)
content(u,i) = x_i dot q_u
```

The implementation uses `DictVectorizer` and `TfidfTransformer`, then sparse
matrix multiplication. A historically scheduled item can have a vector without
having an interaction. An item absent from training metadata receives no vector;
its future title or attributes do not fit a new representation. Users without a
nonempty profile use the common fallback. The model saves the actual vocabulary,
IDF, item vectors, and user profiles.

### Programme-graph personalized PageRank

Normalize each user's transformed history by its own sum:

```text
X_ui = w_ui / sum_j w_uj
A = X.T @ X, with A_ii = 0
```

This bounds each user's total outer-product contribution by one before removing
the diagonal. It reduces the influence of users with very many watched items.
Only items with historical usable interactions are nodes. Retain at most 100
neighbours per item by descending weight, breaking ties by programme ID. This
row-wise pruning can make the graph directed. Normalize nonempty rows into a
transition matrix `P`.

The product is computed in item-row blocks and pruned before the next block.
Blocks contain at most 128 rows and use a conservative 64 MiB matrix-payload
budget. A 100,000-training-item capacity guard fails explicitly. Diagnostics
record payload estimates; these estimates exclude inputs and sparse
multiplication workspace, so measured peak process memory remains necessary.
The implementation does not materialize a full item-item product.

For user restart distribution `v = X_u`, initialize `p_0=v` and iterate

```text
d_t = sum of p_t over graph nodes with no outgoing edges
p_(t+1) = alpha * (P.T @ p_t + d_t * v) + (1-alpha) * v
```

Dangling mass returns to the same user's restart distribution. Stop when
`||p_(t+1)-p_t||_1 <= 1e-8`, or after 300 iterations. The score is the resulting
node mass; it is not a predicted probability of watching. Diagnostics preserve
each actual walk's iteration count, successive-iterate L1 residual, convergence
flag, mass, and score variance. The only searched parameter is link-following
`alpha`, evaluated at `0.5` and `0.85`. These are trials within one qualifying
family.

### Shared ranking and fallback

Positive finite native scores rank first, descending by score and ascending by
programme ID for exact ties. Cold users, absent/nonpositive scores, and short
native lists use training popularity restricted to the same candidates. Fallback
items append after native results. Their exported score is the actual popularity
count, and `reason` identifies the fallback; mixed score scales are not fused.
Every result has five distinct positive valid IDs. Fewer than five distinct
candidates stops prediction. Previously watched programmes remain eligible.

## Metric, selection, and final refit

For a unique eligible holdout set `R_u`, let `rel_u(r)` be one only when rank `r`
contains a relevant item not already credited at an earlier rank. With `h_r`
distinct hits among the first `r` positions:

```text
P_u@r = h_r / r
AP_u@5 = sum_(r=1..5) (P_u@r * rel_u(r)) / min(5, |R_u|)
MAP@5 = mean of AP_u@5 over the fixed cohort
```

A duplicate prediction consumes a rank but earns no additional credit. Empty
targets have AP zero in the metric; the primary cohort policy above excludes
them. Prediction ranks beyond five are ignored. `audit_exported_ap` consumes
the actual per-user CSV and recomputes AP and macro means independently with
exact rational arithmetic. It also checks that all methods export the same
ordered cohort. Local metric identity with the competition's hidden evaluator
has not been established merely by the metric name.

The declared search compares popularity, one content configuration, and both PPR
alphas on that common cohort. Highest validation MAP@5 wins. Exact ties prefer,
in order, popularity, content, PPR `0.5`, and PPR `0.85`. There is no hybrid and
no selection from Kaggle feedback.

After selection, the chosen family is fitted anew on full historical
`[S_train,S_future)` interactions. Those interactions are reconstructed across
the old validation boundary before thresholding, rather than concatenating two
separately thresholded exports. Content refit metadata combines the two
historical schedule windows, preferring the earliest nonempty historical field.
The future schedule supplies final candidates and display labels only.

## Evidence and interpretation limits

The two comparison users are fixed before model scores: maximum and minimum
distinct usable training items among cohort users, with raw-string ID ties;
the light example excludes the heavy example. The same users appear in
popularity, content, and the better validation PPR trial. Their histories,
ordered recommendations, actual scores, categories, titles, and holdout hits
are exported. These are validation illustrations, not hidden future labels.

Runs record input/source hashes, environment versions, configuration, timing,
peak process memory, interaction audits, per-user AP, model diagnostics, fitted
states, and the validated final CSV hash. A local valid CSV is not a real Kaggle
submission or returned private score. Synthetic test runs are explicitly marked
and cannot supply empirical conclusions about SWEET.TV.

The experiment measures ranking agreement with observable labelled viewing.
Unlabelled airtime and discarded ambiguous seconds limit coverage. A missing
view is not proof of dislike or never watching, and observed viewing does not
prove enjoyment. This offline comparison does not measure user satisfaction,
revenue, or online causal effects. A business recommendation requires the actual
common-protocol quality results, measured costs, and both user examples.
