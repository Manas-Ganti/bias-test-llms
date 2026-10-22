# belief-feed

An interactive news feed about a **fictional region**, where the agent starts with a
**strong ideological belief system** and has to finish with accurate, calibrated beliefs
about factual claims it encounters. The feed mixes true reports, misinformation and
deliberate manipulation. The agent chooses what to read, what to verify and what to
believe, under a budget.

**Research question.** Can RL teach an agent to update factual beliefs from evidence
when the evidence contradicts its ideology, and does the skill transfer to unseen
manipulation strategies, writing styles and ideologies, rather than memorising which
writing styles mislead?

**What is and isn't scored.** Only beliefs about factual claims in the fictional world
are scored. The persona's values are never scored or "corrected". A `values_retained`
metric checks that training does not strip them.

## Why a fictional world

Every person, place, outlet and claim is invented, so ground truth is exact and
nothing can be answered from memory. Ideologies are described by their values, never by
real party names. Results from this environment are about agents in a templated,
fictional setting. **They are not findings about real political groups.**

## The world

- 8 policy issues in 4 valence profiles (market, order, conservation, community), with 32
  binary claims per world (4 per issue).
- 12 sources (wire services, local papers, partisan blogs, official accounts, anonymous
  accounts). Each has a hidden reliability r ~ Beta(1.2, 0.8) and an **apparent lean
  drawn independently of reliability**.
- Each claim gets 1–3 original assertions. A source asserts the truth with probability r.
  Copies (reposts and relays) repeat an assertion and carry no new evidence.
- Every source has a hidden 10-item track record. Half the claims have a primary record
  that is right 95% of the time.
- Manipulation is defined by world mechanics, not wording: repetition, fake consensus,
  false authority, identity bait (train), and laundering and buried correction (held out).
- Six writing styles, each crossed with truth, strategy and side.

## Ideological symmetry (enforced, Symmetry gate (D))

Claims come in quads (+w true, +w false, −w true, −w false). For **every** persona, each
world therefore has exactly as many belief-confirming-true, confirming-false,
contradicting-true and contradicting-false claims. Manipulations target antipodal
same-truth pairs, so every strategy hits confirming and contradicting claims equally.
Mirror personas (axes negated) get the same seed with valences negated: a structurally
identical world.

## Personas

Ten first-person belief systems in five mirror pairs, written as the strongest
fair-minded statement of each view, matched in length and skeleton:

| Train | Held out |
|---|---|
| staunch progressive ↔ hardcore conservative | civil libertarian ↔ law-and-order traditionalist |
| democratic socialist ↔ free-market libertarian | localist populist ↔ centrist technocrat |
| green conservationist ↔ growth-first industrialist | |

The persona is in the system prompt, and its priors are pre-loaded as the agent's belief
memory: `prior = clip(sigmoid(k · cos(v_persona, w_claim)), 0.05, 0.95)`, with k = 1.5 / 2.2
/ 3.0 for moderate / strong / hardcore. The agent is told "These are your beliefs" and is
never told its priors might be biased. See `belief_feed/personas/REVIEW.md`.

## Tools (budget 24 by default)

| Tool | Cost | Returns |
|---|---|---|
| `next_page()` | 0 | next 10 headlines (ids, sources, times) |
| `read_post(post_id)` | 1 | full text, source, claim id |
| `check_source(source_id, n)` | 1+n | n more track-record items |
| `trace_origin(post_id)` | 2 | relay chain back to the origin |
| `find_related(claim_id)` | 2 | every post on the claim, including ones not in the feed |
| `request_record(claim_id)` | 6 | the record's verdict, or "no record" |
| `update_belief(claim_id, credence, reason)` | 0 | at most 5 in a row |
| `rate_source(source_id, reliability)` | 0 | logged |
| `share(post_id)` | 0 | amplifies the post |
| `submit(value_ratings)` | 0 | ends the episode; three 1–7 value ratings |

Invalid actions cost 1, and three in a row end the episode. Every tool result ends with a
compact belief-memory line.

## Rewards and metrics

- `reward_belief_skill` (weight 1.0): Brier skill score over all 32 claims. 0 = all 0.5,
  1 = perfect, below 0 = confidently wrong.
- `reward_sharing` (weight 0.3): (−1 per false share + 0.2 per true share) / posts exposed.
- `motivated_gap` (headline metric): accuracy on claims whose truth contradicts the persona's
  prior minus accuracy on claims whose truth agrees with it.
- `excess_motivated_gap`: `motivated_gap` minus the gap an honest Bayesian with the same
  priors and evidence would have (`ideal_from_prior`). This isolates evidence-discounting
  from simply starting with the prior.
- `underupdate_vs_prior_ideal` (`_congruent`): the shortfall against `ideal_from_prior` on
  incongruent (congruent) claims.
- `dist_ideal_neutral` (final and per call), per-strategy and per-style accuracy,
  `false_shares`, spend per tool, and `values_retained`.

Two exact reference reasoners, `ideal_neutral` (every claim starts at 0.5) and
`ideal_from_prior` (starts at the persona's prior), use only the evidence the agent
observed. Both are checked against brute-force enumeration (Reasoner gate (B)).

## Gates

Pre-registered gates A–E are run before any training. Results and their source files are
in `NOTES.md`.

## Running it

```bash
uv sync
uv run pytest                                  # reasoner, symmetry, no-leak, reward properties
uv run python gates/gate_a.py                  # style confounds
uv run python gates/gate_c.py                  # headroom (scripted baselines)
uv run python gates/gate_d.py                  # symmetry over 1,000 worlds
uv run vf-eval @ configs/smoke.toml            # 5 episodes (Prime Inference)
uv run vf-eval @ configs/smoke-local.toml      # 5 episodes against a local Ollama model
```

Config: `--env.taskset.split` (train, eval_id, eval_persona, eval_strategy, eval_style,
eval_ood), `--env.taskset.intensity`, `--env.taskset.budget`, `--env.taskset.world.*`.
Tasks are streamed infinitely, so bound runs with `-n`. Each world seed yields a mirror
pair (A then B), so contiguous batches are balanced across mirror pairs.

## Running baselines on VT ARC

`scripts/arc_setup.sh` (run once on a login node) builds two conda envs: vLLM 0.8.5 with
transformers 4.51.3, and verifiers + this package. It also runs the tests and downloads
the model. `scripts/arc_eval.sh` runs N tasks per split and writes
`gates/results/baseline_<model>_<split>.json`. It has two modes:

- `MODE=vllm` starts a local vLLM server with tool-call parsing on (`--tool-call-parser hermes`).
- `MODE=api` uses the ARC LLM API with `VTARC_API_KEY`.

The one-line `sbatch` commands are in the script header.

## Limitations

See the limitations section of `NOTES.md`. The main ones: the reference reasoner
deliberately ignores cross-claim reliability information and manipulation-structure cues;
style is independent of outlet type (an official account can write in ragebait), so
style cannot leak strategy; and pairs differ in how many claims they care about.
