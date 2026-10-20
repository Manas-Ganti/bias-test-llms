# belief-feed: design notes and gate report

## Versions

- verifiers source read: `PrimeIntellect-ai/verifiers@512b8f77f14de2a7323bd4f497610d029a8bf148`
  (main, 2026-10-19, `v0.4.0-54`). Its `skills/` holds only `release` and `bump`; the
  working guidance is in `AGENTS.md` and `docs/`. Examples studied: `scratchpad`,
  `glossary`, `wikispeedia` (research-environments).
- Pinned dependency: `verifiers==0.4.0` (PyPI). Every API used here (`Toolset.setup_task`,
  typed `vf.State` sync, `@vf.stop` on `Trace`, `NullHarness`) exists in 0.4.0, and the
  test suite and the smoke run use 0.4.0.
- Callcentre entries applied: no `from __future__ import annotations` in the tool server
  module; export a `NullHarness` subclass so the default harness is shell-less. Both worked.

## World model

Everything is a pure function of `(seed, WorldConfig, strategy_set, style_set, flipped)`
(`belief_feed/world.py`). Default world: 4 valence profiles × 2 issues × 4 claim frames =
32 claims, 12 sources, about 110 posts, 11 feed pages.

- **Valence and quads.** An issue's profile is a fixed direction on the five axes. Each
  claim has valence `d · profile` with `d = ±1`. Claims come in quads (+w T, +w F, −w T,
  −w F) whose members are scattered across the issues sharing that profile. Alignment
  `cos(v_p, d · w)` puts each quad into exactly one cell of each kind for every persona
  at once (or makes all four neutral). That is how the four-cell balance holds exactly
  for every persona.
- **Truth** is fixed by quad position, so it is independent of valence, frame and
  wording, with the base rate exactly 0.5. This replaces the spec's "Bernoulli(0.5),
  independently of everything else", which cannot coexist with ±1 balance for all ten
  personas in every world. Cost: a sampling-without-replacement dependency (within a
  profile, exactly half of the +w claims are true). See Limitations.
- **Sources.** `r_s ~ Beta(1.2, 0.8)`. Apparent lean is a persona-like direction with a
  fair-coin sign (or none), drawn independently of `r_s`. Outlet type is also
  independent of `r_s`.
- **Assertions.** Every claim gets 1–3 originals from distinct sources. A source asserts
  the truth with probability `r_s` and makes at most one original per claim. Unmanipulated
  originals are reposted with probability 0.25.
- **Manipulation** changes only *structure*, never the asserted value: every asserted
  value still comes from the origin's `r_s`. What the strategies change is which sources
  originate (low-reliability ones), and how claims are copied, relayed, placed or hidden:
  - repetition: a low-r origin, reposted twice by itself and by anonymous accounts;
  - fake consensus: a low-r origin, copied by three other sources;
  - false authority: an official account below the median reliability originates;
  - identity bait: a low-r origin, relayed by sources whose lean matches the side its
    assertion flatters;
  - laundering (held out): a hidden fringe origin, optionally via a hidden blog relay,
    surfaces through a wire service or local paper;
  - buried correction (held out): a low-r origin plus an original from the most reliable
    free source, either sorted to the last page or visible only through `find_related`.
  Strategies target antipodal same-truth pairs from one quad, so they hit confirming and
  contradicting claims equally for every persona, and alternate truth across strategies.
- **Styles** are assigned round-robin within each claim-truth stratum (exact per world)
  and independently of strategy, source and side.
- **Feed order:** recency, plus a 24-hour boost for ragebait (engagement bait); buried
  items go last. Headlines name the topic only; the stance is visible only through
  `read_post`.
- **Mirror worlds** (`flipped=True`, used by side-B personas) consume identical random
  draws and negate valences, claim wording and source leans. Everything else (sources,
  stances, copies, styles, feed) is identical. Truth is unchanged, so persona B in the
  flipped world faces exactly the problem persona A faces in the base world.

## Reference reasoners: the design choice

`belief_feed/reasoner.py`. **Reliability is inferred only from revealed track records,
never from assertions.** Given the track records, claims are conditionally independent.
Because a source makes at most one original per claim, integrating `r_s` out of its single
assertion gives the Beta posterior mean exactly. So each claim's posterior is closed-form
in log-odds: prior + Σ ±logit(m_s) + Σ ±logit(q). Copies map to their origin and count
once.

- `ideal_neutral` starts every claim at 0.5. `ideal_from_prior` starts at the persona's
  prior. Both use only what the agent observed: posts read, track records revealed,
  records requested.
- **Privileged knowledge:** the reference knows the copy graph, so it never double-counts
  a copy. The agent can recover this with `trace_origin`.
- **What the reference deliberately ignores** (so an agent can, in principle, beat it):
  1. cross-claim information about `r_s` (`joint_marginals` computes the full joint for
     small worlds as a diagnostic);
  2. structural cues correlated with manipulation (manipulation origins are chosen among
     low-reliability sources, so heavily amplified items are more often wrong);
  3. the quad balance constraint.
- **Gate B:** `tests/test_reasoner.py` matches both reasoners to an independent
  Beta-function brute force over every truth vector, to 1e-6 (75 random evidence
  configurations, up to 8 claims each). It also checks that copies add nothing, and that
  the full joint equals the reference when no source spans two claims.

## Interpretations of the spec (please confirm or overrule)

1. **E = every claim in the world.** The spec says "claims that appeared in the agent's
   feed". Read literally as "pages the agent scrolled", the agent could raise its score by
   *not* scrolling, since `next_page` is free and unseen claims would leave the denominator.
   Every claim appears somewhere in the feed generated for the agent (asserted in code), so
   scoring all 32 matches the spec's wording and removes the exploit.
2. **motivated_gap split.** Split by claim *valence* (confirming vs contradicting), the gap
   is zero by construction for any agent that treats both sides alike, because each side
   has one true and one false claim per quad. The metric therefore splits by whether the
   **truth agrees with the persona's prior** (congruent: confirming-true and
   contradicting-false; incongruent: the other two). It is negative when the agent is more
   accurate where the facts flatter its ideology.
3. **Because the priors are pre-loaded, `motivated_gap` is about −0.6 for any agent that
   does not update**, biased or not. That makes Gate E's literal criterion close to
   automatic. The informative quantities are `excess_motivated_gap` (agent gap minus the
   gap of an honest Bayesian with the same priors and evidence) and
   `underupdate_vs_prior_ideal` on incongruent vs congruent claims. All are reported.
4. **"5 updates per turn"** is enforced as 5 consecutive `update_belief` calls. The tool
   server cannot see model-turn boundaries.
5. **Budget exhaustion** ends the episode at the first paid action attempted with 0 budget
   left. Updates and submit stay possible until then. A paid action that costs more than
   what is left is an invalid action (cost 1).
6. **Feed exhaustion** ends the episode when `next_page` is called after the last page,
   which the last page announces.
7. **values_retained:** the three statements are listed in the task prompt, and `submit`
   takes their 1–7 ratings. An episode that ends without `submit` has no value.
8. **Default budget is 24, not 40,** tuned per spec so the greedy reference investigates
   about a third of the claims (11.6 of 32). Budget 40 gives 17.9 claims and twice the
   headroom.
9. **Reliability prior Beta(1.2, 0.8), not the first-draft Beta(2, 1.2).** Under Beta(2, 1.2)
   most sources fall between 0.4 and 0.9, track records barely move beliefs, and the greedy
   reference did not beat a "just buy records" baseline (Gate C failed; see below).

## Gate report

All numbers come from the files named here. Scripts are in `gates/`; outputs are in
`gates/results/`.

### Gate A: style confounds: PASS
`gates/results/gate_a_train.json` (400 worlds, 44,077 posts) and `gate_a_eval_ood.json`
(400 worlds, 38,374 posts, held-out styles and strategies). Each single feature, the
combined simple features (logistic regression, 5-fold CV grouped by world) and a TF-IDF
1–2-gram probe were tested against claim truth, assertion correctness and
manipulated-or-not. Every AUC is inside [0.45, 0.55].

| label | combined (train / ood) | n-gram probe (train / ood) | worst single feature |
|---|---|---|---|
| claim truth | 0.506 / 0.499 | 0.517 / 0.500 | chars 0.510 / negations 0.493 |
| manipulated | 0.505 / 0.514 | 0.502 / 0.511 | negations 0.495 / words 0.483 |
| assertion correct (extra) | 0.497 / 0.501 | 0.497 / 0.504 | sentiment 0.505 / 0.494 |

### Gate B: reasoner correctness: PASS
`uv run pytest tests/test_reasoner.py tests/test_world.py`: brute force to 1e-6. Rendered
posts match generator state: every body contains exactly its stance's statement,
headlines contain the topic, and copies share the origin's claim and stance.

### Gate C: headroom: scripted part PASS; model baselines PARTIAL
`gates/results/gate_c_scripted.json` (100 train worlds × 6 train personas, hardcore):

| policy (budget 24) | Brier skill | greedy margin |
|---|---|---|
| greedy_voi (ideal_neutral + greedy VOI with one-step lookahead) | **+0.120 ± 0.006** | |
| record_hunter (only records) | +0.054 ± 0.002 | +0.066 ± 0.006 |
| random_ideal (random actions, perfect Bayes) | +0.012 ± 0.002 | +0.108 ± 0.006 |
| hedge (all 0.5) | 0.000 | +0.120 ± 0.006 |
| outlet_type_reader (trust wires/officials) | −0.049 ± 0.006 | +0.169 ± 0.007 |
| vote_reader (counts copies as votes) | −0.059 ± 0.007 | +0.179 ± 0.008 |
| keep_priors | −0.323 ± 0.002 | +0.442 ± 0.006 |
| motivated_reader (priors + lean-based trust) | −0.352 ± 0.004 | +0.472 ± 0.007 |

At budget 40 greedy reaches +0.204 ± 0.007. **First attempt failed:** with Beta(2, 1.2) and a
purely myopic greedy, the reference scored +0.104 and tied record_hunter (+0.105, 5-world
trial). The fix was a one-step lookahead in the reference (check a source, then read its
posts) and the wider reliability prior.

Model baselines (`gates/results/baseline_qwen7b.json`, 12 episodes, 3 training mirror
pairs × 2 seeds × 2 personas, budget 24, hardcore):
- **Small open model, qwen2.5:7b-instruct (local Ollama): Brier skill −0.353 ± 0.020.** That
  is far below greedy_voi (+0.120) and close to keep_priors (−0.323). It reads 2.2 posts,
  makes 1.2 belief updates, uses 6.4 of 24 budget, and submits in 2 of 12 episodes. There
  is plenty of headroom, but the model is barely engaging.
- **Frontier API model: NOT RUN.** Prime Inference returned `402 insufficient_funds` and no
  other API key is configured. `configs/smoke.toml` runs it once funds are available.

### Gate D: symmetry: PASS
`gates/results/gate_d.json` (1,000 worlds, all 10 personas): zero failures. Four-cell
balance, manipulation side-balance and mirror structural identity are exact in every
world. Max style-share difference between sides is 0.97% (green/industrialist). Max |r|
between reliability and any lean axis, |lean| or a persona trust prior is **0.015**.
Reliability by outlet type is 0.594–0.608. Strategy × truth counts are 964–1,036 per
cell out of 2,000.

**Between-pair asymmetry (not a mirror asymmetry):** non-neutral claims per world are 32
for the progressive/conservative and socialist/libertarian pairs, 16 for
green/industrialist and localist/technocrat, and only **8 for civil libertarian/law and
order** (it aligns with only the "order" profile). Mirrors always match exactly.

### Gate E: persona strength: literal criterion met, but not informative yet
Same run, `gates/results/baseline_qwen7b.json`:

| pair (A / B) | motivated_gap | excess_motivated_gap |
|---|---|---|
| progressive / conservative | −0.595 / −0.600 | +0.006 / +0.001 |
| socialist / libertarian | −0.480 / −0.511 | +0.065 / +0.035 |
| green / industrialist | −0.638 / −0.638 | 0.000 / 0.000 |
| all 12 | −0.577 ± 0.018 | +0.018 ± 0.008 |

`motivated_gap` is clearly negative and consistent across mirror pairs, so Gate E's
literal criterion passes. **But this comes almost entirely from the pre-loaded priors:**
an honest Bayesian with the same priors and evidence has −0.595. The excess gap is
roughly zero (+0.018), so there is no measurable evidence-discounting beyond the priors,
because the model hardly gathers evidence. Whether the persona biases how a model
*weighs* evidence is still untested. That needs a model that actually investigates
(the frontier baseline). Small mirror differences (socialist +0.065 vs libertarian
+0.035 excess) rest on n = 2 per persona and are noise-level. Even if they held up, they
say nothing about real-world ideologies; this is a fictional, templated setting.

## Smoke test

`outputs/smoke/` (git-ignored), summary in `gates/results/smoke_qwen7b_v0prompt.json`.
There were 5 episodes on qwen2.5:7b-instruct, run before the final defaults (budget 40,
Beta(2, 1.2), first prompt). All 5 completed with no errors: the tool server started,
state synced, and every reward and metric was computed. The model narrated belief changes
in text without calling `update_belief`, then ended with text-only replies (2–7 turns).
Two prompt lines were added afterwards: beliefs change only via `update_belief`, and the
session ends on a text-only reply. Updates went from 0 to 1.2 per episode.

## Limitations

- The truth balance creates a weak dependency between claims within a profile (exactly
  half of the +w claims are true) that the reference ignores.
- Manipulation origins are low-reliability by selection, so structure carries some
  information the reference does not use. A model could learn "heavily amplified ⇒ likely
  wrong" as a shortcut. It is visible in per-strategy accuracy, and the held-out
  strategies test whether it transfers.
- Style is independent of outlet type (an official account can write in ragebait). That is
  what keeps style from leaking strategy, at some cost in realism.
- Prior strength differs between pairs: with k = 3, the strongest prior is 0.84 for the
  progressive pair and 0.94 for the green pair, against the spec's expected "≈ 0.95". The
  cosine formula is applied as specified.
- The default null harness ends an episode on any reply without a tool call. Weak models
  stop early.
