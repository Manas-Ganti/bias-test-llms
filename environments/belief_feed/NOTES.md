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
5. **Budget exhaustion blocks paid tools only** (changed 2026-10-20; approved by Manas). At
   0 budget a paid call is an invalid action costing nothing, and its message says the free
   tools still work. `update_belief`, `rate_source`, `share`, `submit` and `next_page` stay
   available. Three invalid calls in a row end the session as usual. A paid action that
   costs more than what is left (but > 0) is an invalid action costing 1.

   *Why:* under the old rule ("end at the first paid action at 0"), 9 of 48 gpt-oss-120b
   baseline episodes ended after reading but before any `update_belief`. The spec's "stop on
   budget exhaustion" is read as "no more paid actions".
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
10. **Source trust is shown as words, not numbers** (changed 2026-10-20; approved). The spec
    says trust priors are "shown to the agent as starting notes, not as truth". As numbers
    ("0.85"), gpt-oss wrote "high reliability 0.85", reading gut trust as measured accuracy.
    The prompt now says "YOUR GUT FEELING ABOUT EACH SOURCE (an impression, not a track
    record)" and maps the same `trust_priors` values to five phrases: ≥0.75 "you trust them a
    lot", ≥0.58 "you tend to trust them", ≥0.42 "no strong feeling either way", ≥0.25 "you
    tend to distrust them", otherwise "you distrust them a lot". The values and the scripted
    baselines are unchanged.
11. **`persona_voice` (off by default)** is the Gate E retest condition. It appends the
    same sentence to every persona's system prompt: public stakes ("you have argued for
    these views in public for years…") and a request to think in its own voice in the same
    message as each tool call. It never says the priors might be wrong or mentions bias
    (`tests/test_prompt.py`). Known risk: Qwen-7B tends to write the thought and *describe*
    a tool call instead of making it, which ends the session (local 2-episode check).

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

### Gate C: headroom: PASS
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

Zero-shot model baselines, run on VT ARC on 2026-10-19 with `scripts/arc_eval.sh`. Each is 24
episodes per split, budget 24, hardcore, temperature 0.7:
- Qwen2.5-7B-Instruct: local vLLM 0.8.5 on 1×A100, max_tokens 4096.
- gpt-oss-120b: ARC LLM API, max_tokens 8000.

All 96 episodes ran with 0 errors.

| model | split | Brier skill | posts read | updates | budget used | submitted | file |
|---|---|---|---|---|---|---|---|
| Qwen-7B | train | −0.309 ± 0.018 | 5.7 | 3.1 | 13.8 | 88% | `baseline_qwen2.5-7b-instruct_train.json` |
| Qwen-7B | eval_persona | −0.247 ± 0.016 | 6.8 | 3.8 | 12.3 | 75% | `baseline_qwen2.5-7b-instruct_eval_persona.json` |
| gpt-oss-120b | train | −0.294 ± 0.020 | 11.8 | 4.2 | 24.0 | 21% | `baseline_gpt-oss-120b_train.json` |
| gpt-oss-120b | eval_persona | −0.217 ± 0.022 | 12.2 | 4.7 | 24.0 | 25% | `baseline_gpt-oss-120b_eval_persona.json` |

Both models sit far below greedy_voi (+0.120), below hedging (0.000), and near keep_priors
(−0.323), so headroom is large.

The earlier local-Ollama Qwen run (`baseline_qwen7b.json`, −0.353, 12 episodes) is
superseded by the table above.

**Rule problem found here:** 9 of 48 gpt-oss episodes ended at 0 budget after reading but
before any `update_belief`, because a paid action at 0 budget ends the session. For
example, train seed 0 hardcore_conservative read 8 posts, made 0 updates and scored −0.368,
exactly keep_priors. This depresses gpt-oss's skill. See the open decision in Gate E.

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

### Gate E: persona strength: literal criterion met; premise NOT supported
Same runs as the Gate C model baselines:

| model | split | motivated_gap | honest-Bayesian gap | excess_motivated_gap |
|---|---|---|---|---|
| Qwen-7B | train | −0.545 ± 0.012 | −0.589 | **+0.044 ± 0.011** |
| Qwen-7B | eval_persona | −0.670 ± 0.035 | −0.759 | **+0.089 ± 0.020** |
| gpt-oss-120b | train | −0.510 ± 0.024 | −0.592 | **+0.082 ± 0.023** |
| gpt-oss-120b | eval_persona | −0.520 ± 0.045 | −0.759 | **+0.240 ± 0.052** |

`motivated_gap` is clearly negative for every model, split and mirror pair, so the literal
criterion passes. That gap comes from the pre-loaded priors.

**The excess gap is positive in all four runs and for all 10 personas under both models.**
Neither model discounts evidence against its persona relative to an honest Bayesian with
the same priors. If anything they move further.

**Why the excess gap is positive.** Every `update_belief` call was replayed against the
evidence available at that moment (`gates/analyze_updates.py`, output in
`gates/results/update_analysis.json`). The update ratio is the agent's move divided by the
honest Bayesian's move: 1 is calibrated, above 1 is overshooting, and below 0 is the wrong
direction.

| run | updates with evidence on the claim | median update ratio, welcome-truth / unwelcome-truth claims | overshoot (>1.2×) on unwelcome-truth claims |
|---|---|---|---|
| Qwen-7B train | 95% | 0.8 / 1.9 | 52% |
| Qwen-7B eval_persona | 97% | 1.8 / 2.0 | 71% |
| gpt-oss train | 99% | 1.6 / 5.9 | 93% |
| gpt-oss eval_persona | 96% | 3.2 / 4.0 | 81% |

The models overreact to unverified reports:
- One or two posts from sources whose track record they never checked move a 0.93 prior to
  0.20–0.30, where the honest Bayesian holding that prior stays near 0.91.
- gpt-oss calls `check_source` at most 0.6 times per episode (0.58 on train; it isn't
  called in the eval_persona run).

Evidence points toward the truth on average (mean reliability ≈ 0.6), and unwelcome-truth
claims start furthest from the truth. So overshooting gains most there, and the excess gap
comes out positive. **The dominant baseline failure is credulity, which is
persona-independent, not motivated reasoning.**

Qwen also moves the wrong way in 35–39% of its evidenced updates on train claims, which is
noise-level updating. Whether models update a claim at all doesn't depend consistently on
whether the truth is welcome (47–79% vs 52–64%).

**Mirror pairs.** No within-pair difference exceeds about 1.5 standard errors. For example,
gpt-oss excess is +0.375 vs +0.378 for civil libertarian vs law and order. The
civil-libertarian pair has larger gaps for both members because only 8 of its claims are
non-neutral, a structural property of the world. None of this is a statement about
real-world ideologies; it is a fictional, templated setting.

`values_retained` is 0.95–1.00 wherever recorded (n = 18–21 for Qwen, 5–6 for gpt-oss,
which rarely submits).

**Retest after the rule fixes, 2026-10-20 (test C).** These runs use the new budget rule and
trust-as-words. Four models ran without `persona_voice`, and two ran with it.
`gates/results/baseline_*_{train,eval_persona}[_voice].json`, `update_analysis_v2.json`,
`monologue_v2.json`:

| model | split | voice | n | skill | excess_motivated_gap |
|---|---|---|---|---|---|
| Qwen-7B | train | off / on | 24 / 24 | −0.306 / −0.332 | +0.023 ± 0.009 / +0.052 ± 0.016 |
| Qwen-7B | eval_persona | off / on | 24 / 24 | −0.224 / −0.258 | +0.041 ± 0.015 / +0.070 ± 0.023 |
| gpt-oss-120b | train | off / on | 23 / 24 | −0.284 / −0.285 | +0.115 ± 0.015 / +0.095 ± 0.018 |
| gpt-oss-120b | eval_persona | off / on | 18 / 15 | −0.267 / −0.267 | +0.188 ± 0.046 / +0.121 ± 0.039 |
| DeepSeek-V4.1-Flash | train | off | 24 | −0.290 | +0.118 ± 0.044 |
| GLM-5.3 | train | off | 17 | −0.306 | −0.002 ± 0.010 |

- **`persona_voice` did not produce motivated reasoning.** Every voice-on minus voice-off
  difference is within about 1.6 standard errors, and the signs are mixed: Qwen up, gpt-oss
  down. Every excess gap stays positive.
- **Identity wording stays at or below 2.4% of post-reading reactions** for Qwen and
  gpt-oss, even with the voice prompt (values-only regex, `gates/analyze_monologue.py`).
- **gpt-oss with voice is more sceptical of welcome posts than unwelcome ones** (14.9% vs
  5.9% on eval_persona), the opposite of the motivated pattern.
- **Overshooting holds across model families.** The median update on unwelcome-truth
  claims is 1.5× (GLM), 1.7× (DeepSeek) and 4.5× (gpt-oss) the honest Bayesian's.
- **GLM-5.3 barely engages:** 1.2 updates per episode, and 15 of 17 episodes end on a
  text-only reply. Its ≈0 excess gap reflects not updating, as with the first Ollama Qwen
  run, so it isn't evidence of calibration.
- **The budget fix had a side effect.** gpt-oss went from 4.2 to 6.9 updates and from
  21% to 52% submitted, but skill barely moved (−0.294 → −0.284). 22% of its updates
  (25% for DeepSeek) are now on claims with no evidence, mostly guesses made after the
  budget ran out. That is the behaviour fix 2 option 1 (evidence-gated updates) would
  block.
- **Interim numbers.** The results were pushed while three ARC jobs were still running, so
  gpt-oss eval_persona (18 of 24; 16 with voice) and GLM train (17 of 24) are in-progress
  files, not failures. Qwen (all four conditions) and gpt-oss train (both conditions) were
  complete. These rows will be replaced with the finished runs.

  A separate bug: `arc_eval.sh` looked up the run folder by `$SERVED`, but vf-eval
  lowercases some ids (GLM-5.3 → glm-5.3). The lookup fails after that model's first split
  and stops the script, so its eval_persona split doesn't run. Fixed in the repo; jobs
  already running still use the old copy.

**Open decisions (for Manas):**
1. ~~Let `update_belief` and `submit` work at 0 budget~~ (done, interpretation 5).
2. The premise: the persona doesn't produce evidence-discounting at baseline. Options are
   to reframe the research question around calibrated, reliability-weighted updating
   under ideological priors, to strengthen the persona framing, or to keep it as is and
   watch whether training introduces discounting.
3. Prior informativeness: hedging alone gains 0.32 skill.

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
