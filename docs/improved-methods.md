# Versioned personal-restart experiments

These variants preserve the original implementation and its exports. The name
`improved_models` denotes the proposed revision; it does not establish a quality
improvement. Selection must use the newly frozen local protocol. A further
competition submission requires a separately validated exact CSV and the
designated submission decision.

## Diagnosis from the original saved outputs

The original validation cohort has 1,246 users and 6,230 recommendation slots.
The `ppr_0.5` export contains only 88 distinct recommended programme IDs; its most
frequent ID, 2400480, occurs for 1,236 users. At continuation `alpha=0.85`, there
are 28 distinct IDs, and that item occurs for 1,245 users. Both methods use only
40 fallback slots, from eight cold users. Thus the concentration is present in
native PPR results, rather than being explained by widespread cold fallback.
All 1,238 warm-user walks converged in the recorded original run.

These counts were independently computed from
`experiments/competition-20261030-03/validation-recommendations.csv`; cohort and
walk counts are in that run's summary and diagnostics. They describe the old
holdout and are not new revision results. The data contains frequently observed
programmes, so popularity concentration alone is not proof of an implementation
error or poor relevance.

The original seed is proportional to `log(1+seconds)`. A 100-fold duration ratio
becomes only about a two-fold ratio for the synthetic values 100 and 10,000
seconds; a square-root transform preserves a ten-fold ratio and a linear
transform preserves the original 100-fold ratio. This illustrates compression,
not a finding about an individual viewer. Shared high-degree neighbours and
row-wise top-100 graph pruning can further favor common programmes. Higher
continuation permits more propagation away from the user's restart. These
mechanisms motivate stronger personal seeds and lower continuation; they do not
by themselves establish why the hidden score differs from local validation.

## Exact model change

The graph is identical to the original training-only graph. For historical
accepted overlap `s_ui`:

```text
X_ui = log(1+s_ui) / sum_j log(1+s_uj)
A = X.T @ X, diagonal removed
P = row_normalize(deterministic_top_100_neighbours_per_row(A))
```

Graph construction retains the original blocked sparse multiplication, capacity
guard, tie ordering, and training-only item universe. Recency and stronger
personal signals do not change graph edges, popularity, or candidate pools.

The new restart uses seconds or accepted-airing count, with positive exponent
`p` and an optional nonnegative scale `h_r` on each supplied interaction row:

```text
z_ui = (sum_(rows r for u,i) h_r * signal_r) ** p
v_ui = z_ui / sum_j z_uj
```

Power is applied after aggregation. With the usual one row per user/programme,
unscaled `z` is `seconds**p` or `watches**p`. The caller supplies all recency
values from its training side. A named numeric column, row-aligned scale array,
or mapping keyed by exact `(user_id,item_id)` is accepted; missing mapping keys
use scale one. The fitted powered values and normalized restart are copied and
saved as read-only sparse arrays, including after loading a checkpoint.

The original PPR iteration is reused:

```text
p_0 = v
p_(t+1) = alpha * (P.T @ p_t + dangling_mass(p_t) * v) + (1-alpha) * v
```

Here `alpha` is the link-following probability, not the restart probability.
Dangling mass returns to the same personal restart. Stop at successive-iterate
L1 difference at most `1e-8`, with a cap of 300 iterations. Per-user convergence,
mass, residual, and score-variance diagnostics are preserved. Users whose chosen
signal or recency weighting leaves zero restart mass use the common training
popularity fallback rather than running a zero-mass walk.

The current bounded hypothesis grid is continuation `{0.05, 0.25}` times signal
`{seconds, watches}` times power `{0.5, 1.0}`: eight PPR variants. One separately
fixed recency variant uses `alpha=0.15`, seconds, power one, and a 28-day
half-life. The caller computes decay per historical airing before aggregating
recent seconds. If `r_ui` denotes those summed recent seconds, passing
`history_scale = r_ui / s_ui` on an aggregated row makes the restart proportional
to `r_ui`. The caller's protocol must record the actual airing reference time
and training cutoff; this module does not infer them from aggregated rows.

Two personal-history diagnostics rank programmes directly by the user's own
seconds or watches, with common popularity fallback. They help determine whether
propagation improves over remembered preference. They are not a new qualifying
recommendation family. The original content-based and PPR families remain
separately evaluated.

## API and tests

`fit_improved_ppr(interactions, *, alpha=.15, seed_signal='seconds',
seed_power=.5, history_scale=None)` returns the original recommendation/save/
diagnostics interface. `fit_personal_history(interactions, signal='seconds',
power=1, *, history_scale=None)` returns the diagnostic baseline. `fit_history`
is a runner alias with `seed_kind` and `power` keyword names.

All methods preserve exact string user IDs, allow repeats, rank finite positive
native scores with programme-ID tie breaks, and append the same candidate-
restricted popularity fallback. Exported fallback scores remain actual
popularity counts, identified by `reason`; they are not fused with PPR masses.

The focused synthetic fixtures verify unchanged graph edges, exact powered-seed
ratios, an independent dense stationary solution, conservation and dangling
restart, opposite duration/count preferences, scaling before power, zero-seed
fallback, candidate validity, input-copy isolation, and persisted read-only seed
state. These tests establish implementation properties, not expected ranking
quality, user enjoyment, or a higher returned private score.
