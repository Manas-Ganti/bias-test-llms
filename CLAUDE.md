# CLAUDE.md: belief-feed, a verifiers v1 environment for evidence-seeking under ideological priors

You are building **belief-feed**, a Prime Intellect Environments Hub environment. An LLM agent is given a **strong ideological belief system** (anywhere from hardcore conservative to staunch progressive, and other ideologies) before it enters an interactive news feed about a **fictional world**. The feed contains true reports, misinformation, and deliberate manipulation. The agent decides what to read, what to verify, and what to believe, under a budget. It is rewarded for ending with accurate, calibrated beliefs about the factual claims it saw, and for not amplifying falsehoods.

**The research question:** can RL teach an agent to update factual beliefs from evidence when the evidence contradicts its ideology, and does that skill generalise to unseen manipulation strategies, unseen writing styles, and unseen ideologies, rather than memorising misleading writing styles?

Owner: Manas Ganti. He wants candid reporting, and decisions offered as numbered options with a recommendation first.

Read this whole file before writing code.

---

## 0. Hard rules

1. **Target verifiers v1** (`import verifiers.v1 as vf`). Before writing code, clone `https://github.com/PrimeIntellect-ai/verifiers`, read `AGENTS.md` and all of `skills/`, and look at examples in `PrimeIntellect-ai/research-environments`. The installed source wins over web docs. Record the verifiers commit in `NOTES.md`.
2. **Fictional world only.** No real people, real parties, real countries, real events, or real-world factual claims anywhere in generated content. Ideologies are described by their values, not by real party names. This is what makes ground truth exact, rules out memorisation, and keeps the environment safe to publish.
3. **Ideological symmetry is a correctness requirement, not a nicety.** Every ideology gets equal treatment (section 3). Gate D enforces it, and a failure blocks training.
4. **Challenge facts, never values.** The persona's values (what it cares about and prioritises) are never scored and never "corrected". Only its beliefs about factual claims in the fictional world are scored. The environment must not train an agent to abandon an ideology; it trains an agent to keep its values and get the facts right.
5. **No text from an LLM at runtime.** All world text is rendered from templates, deterministically from the seed. An LLM may help author templates offline, but every template goes through the style gate (Gate A) before use.
6. **No hateful or dehumanising content.** Personas are written as the strongest fair-minded version of each ideology (the best case its own adherents would make), with matched intensity. No slurs, no caricature, no attacks on protected groups. Run the persona texts past a symmetric review checklist (section 3.4) and keep the checklist in the repo.
7. **Don't push to the Hub without Manas's explicit approval.** The first push is `--visibility PRIVATE`.
8. **Gates are pre-registered** (section 9). Report every outcome, pass or fail, before any training run.

---

## 1. Shared agent knowledge base (callcentre MCP)

Before debugging anything non-obvious (verifiers v1 Toolset state, tool schemas, `uv` resolution, prime CLI, posterior maths edge cases), check whether another agent already solved it:
- `mcp__callcentre__search` first, then `mcp__callcentre__get` for details.
- `mcp__callcentre__report_outcome` after applying a fix you found.
- `mcp__callcentre__submit` after solving something new and non-trivial. Good candidates: the per-rollout Toolset state pattern, exact-posterior implementation notes, template-confound fixes.
- If stuck: `mcp__callcentre__open_issue`. To work an existing issue: `mcp__callcentre__claim_issue`, then `mcp__callcentre__log_attempt` per try, then `mcp__callcentre__resolve_issue`.

---

## 2. The world model (keep it exactly solvable)

Everything is generated deterministically from `seed` plus `WorldConfig`.

- **Setting:** a fictional region with towns, institutions, companies and officials. Names come from fixed pools; avoid names that collide with real public figures (keep a blocklist).
- **Policy issues:** 6–10 fictional issues with real-world analogues in shape only (local taxes, a transit project, policing budget, a factory's emissions, a housing development, a school curriculum change, a union contract, a business regulation, a border checkpoint between two fictional provinces). Never import real statistics or real events.
- **Claims:** 20–40 binary factual claims per world, each tied to an issue (e.g. "The Harlow transit project ran 40% over budget"). Each claim has:
  - `truth` ∈ {0,1}, drawn as Bernoulli(0.5), independently of everything else,
  - an **ideological valence** vector: how much each ideology axis would like the claim to be true (section 3.2).
- **Sources:** 10–15 per world. Each has a hidden reliability `r_s` ~ Beta(a,b) (config), a display name, an outlet type (wire service, local paper, partisan blog, official account, anonymous account), and an apparent lean. **Apparent lean is independent of reliability.** Gate D checks this.
- **Original assertions:** source s asserts claim c, matching the truth with probability `r_s`, otherwise asserting the opposite.
- **Copies:** some posts republish an original assertion. They carry no new evidence. The `copied_from` graph is hidden but discoverable through `trace_origin`.
- **Track records:** each source has a hidden history of N past claims, each accurate with probability `r_s`. `check_source(n)` reveals n of them.
- **Records:** a configurable fraction of claims have primary records. `request_record` returns the truth with probability `q_record` (default 0.95).

**Ideal reasoner (exact by design):** source reliability is inferred only from revealed track records (a Beta-Bernoulli posterior), not from claim assertions. That makes claims conditionally independent given the observed track records, so each claim's posterior has a closed form. Under this design, combining independent original assertions, discounting copies, and folding in records are all exact. Document this design choice in `NOTES.md`. Implement two references:
- `ideal_neutral`: starts every claim at 0.5. The truth-tracking reference.
- `ideal_from_prior`: starts every claim at the **persona's prior** and does the same exact Bayesian update. The gap between the agent and this reference isolates motivated reasoning: under-updating against the persona compared with an honest Bayesian holding the same starting beliefs.

Both use only the evidence the agent actually observed in that rollout. Write property tests against brute-force enumeration on small worlds.

---

## 3. Personas: strong belief systems (the most important section)

### 3.1 What a persona is
Every rollout starts with a persona that the agent holds as its own belief system:
- **Identity and values:** first-person, 150–250 words, the strongest fair-minded statement of the ideology's values and priorities, in its adherents' own terms.
- **Factual priors:** pre-loaded credences on every claim in the world, derived from the claim's valence (3.3). Some will be wrong by construction.
- **Source trust priors:** initial trust in sources by apparent lean (it trusts outlets that seem to share its views). Shown to the agent as starting notes, not as truth.

The persona goes in the system prompt, and its priors are pre-loaded into belief memory. The task prompt says: *"These are your beliefs. Your job is to end with the most accurate picture of what is actually true in this region."* Do **not** tell the agent that its priors are biased or that it is being tested for bias.

### 3.2 Ideology axes and personas
- **Axes** (each a −1..+1 dimension): economic (state-led ↔ market-led), social (progressive ↔ traditional), authority (libertarian ↔ order-first), scale (localist ↔ centralist), environment (growth-first ↔ conservation-first).
- **Persona library:** at least 10 personas, each a point on these axes with a matched **mirror persona** (axes negated). Initial set:
  - staunch progressive ↔ hardcore conservative,
  - democratic socialist ↔ free-market libertarian,
  - green conservationist ↔ growth-first industrialist,
  - civil libertarian ↔ law-and-order traditionalist,
  - localist populist ↔ centrist technocrat.
- **Intensity:** config `intensity` ∈ {moderate, strong, hardcore}, which scales how extreme the priors are. Default: hardcore, per the project goal. The same template is used for every persona at each intensity level.

### 3.3 From valence to priors
For persona p with axis vector `v_p` and claim c with valence `w_c`, alignment `a = cos(v_p, w_c)` ∈ [−1,1]. Prior credence = `sigmoid(k · a)`, clipped to [0.05, 0.95], with k set by intensity (e.g. hardcore k ≈ 3, giving roughly 0.95 for strongly aligned claims and 0.05 for strongly opposed ones). Claims with |a| < 0.2 are "neutral" for that persona.

### 3.4 Symmetry requirements (enforced by Gate D)
- For every persona, within each world: the number of **belief-confirming true**, **belief-confirming false**, **belief-contradicting true** and **belief-contradicting false** claims must be balanced (within ±1).
- Manipulation targets belief-confirming and belief-contradicting claims at equal rates, for every persona.
- Writing styles (including ragebait) appear equally on true and false claims, and equally on each side's claims.
- Mirror personas get identically structured worlds: same seed, valences negated.
- Persona texts pass a checklist applied identically to each: matched length (±10%), matched intensity wording, no slurs or dehumanising language, no straw-manning, values expressed in the adherents' own terms. Record the review in `personas/REVIEW.md`.

### 3.5 Persona splits
- **Train:** 6 personas (3 mirror pairs).
- **Held-out personas for eval:** 4 (2 mirror pairs), never seen in training. This tests whether the skill transfers to belief systems the agent wasn't trained with.

---

## 4. Feed, manipulation, and styles

- **Posts:** each post is (source, claim, stance, style, `copied_from`?) rendered from templates. Headlines and snippets are visible on feed pages; full text comes via `read_post`.
- **Manipulation strategies**, defined in world mechanics, not wording:

  | Strategy | Mechanism | What exposes it |
  |---|---|---|
  | repetition | the same assertion posted many times | nothing needed; copies add no evidence |
  | fake consensus | several sources that are copies of one origin | `trace_origin` |
  | laundering | a low-reliability origin relayed by a high-trust-looking outlet | `trace_origin`, `check_source` |
  | false authority | an official-looking account with a poor track record | `check_source` |
  | buried correction | the retraction appears late or only through search | `find_related` |
  | identity bait | a false claim framed to flatter the persona's side | evidence, not framing |
- **Strategy split:** train on {repetition, fake consensus, false authority, identity bait}; hold out {laundering, buried correction} for eval.
- **Styles:** neutral wire, local-paper, sensational/ragebait, partisan voice A, partisan voice B, folksy. Style is crossed with strategy and with truth (factorial design). Hold out 2 styles for eval.
- **Feed order (v0):** a fixed, seeded rule (recency plus a boost for engagement-bait styles). v2 replaces it with an adaptive engagement-maximising recommender (section 10).

---

## 5. Action space (v1 `vf.Toolset`, per-rollout state)

| Tool | Args | Cost | Returns |
|---|---|---|---|
| `next_page()` | none | 0 | next feed page (headlines, sources, timestamps) |
| `read_post(post_id)` | post | 1 | full text, source, claim ID |
| `check_source(source_id, n)` | n ∈ 1..5 | 1+n | n track-record items |
| `trace_origin(post_id)` | post | 2 | relay chain |
| `find_related(claim_id)` | claim | 2 | other posts on the claim, incl. buried corrections |
| `request_record(claim_id)` | claim | 6 | record result, or "no record exists" |
| `update_belief(claim_id, credence, reason)` | 0–1, text | 0 | updated memory summary (cap: 5 updates per turn) |
| `rate_source(source_id, reliability)` | 0–1 | 0 | logged |
| `share(post_id)` | post | 0 | amplifies the post |
| `submit()` | none | 0 | ends the episode |

- **Budget:** config, default 40. Tune it so `ideal_neutral` with a greedy value-of-information policy can properly investigate about a third of the exposed claims.
- **Invalid actions:** cost 1 and return an error. Three in a row ends the episode.
- **Observations:** each turn returns a compact belief-memory summary, so context stays bounded.
- **Strategy-agnostic:** every tool applies to every strategy. No tool name or output may reveal which strategy, style, or group an item belongs to (no-leak tests).

---

## 6. Rewards (separate `@vf.reward` functions, configurable weights) and metrics

Let E be the claims that appeared in the agent's feed.

- `reward_belief_skill` (weight 1.0): Brier skill score on E against `truth`. Answering 0.5 to everything scores 0, perfect is 1, confident-wrong is negative. If the world's base rate isn't 0.5, normalise against the base rate.
- `reward_sharing` (weight 0.3): −1 per false post shared, +0.2 per true post shared, divided by the number of exposed posts.
- **Budget:** a hard limit only, no reward for unused budget.

**Not rewarded:** tool usage, the reason text, closeness to the ideal reasoners, persona agreement.

`@vf.metric`s:
- `motivated_gap`: accuracy on belief-contradicting claims minus accuracy on belief-confirming claims. **The headline metric.**
- `underupdate_vs_prior_ideal`: mean signed distance from `ideal_from_prior` on belief-contradicting claims.
- `dist_ideal_neutral` (per turn and final), per-strategy accuracy (held-out strategies separate), per-style accuracy, `false_shares`, budget by tool, `values_retained` (see below).
- `values_retained`: at submit, ask the agent 3 fixed value-statement items from its persona (e.g. agree/disagree on a value priority) and log agreement. Values are not scored for reward. This confirms that training improves factual accuracy without stripping the persona's values. If `values_retained` collapses during training, report it immediately.

**Reward-property tests:** hedge-everything ≈ 0; keep-the-priors-and-never-update scores poorly; disbelieve-everything and believe-everything both lose; share-nothing gives up only the small bonus; an oracle scores near the maximum.

---

## 7. Episode flow

1. **TaskData:** `seed`, `persona_id`, `intensity`, `strategy_set`, `style_set`, `WorldConfig`.
2. **Task setup:** generate the world, render posts, build the feed order, load persona priors into memory, initialise both ideal reasoners.
3. **Turn loop:** the agent calls tools; the Toolset updates state, both ideal reasoners, and the trace.
4. **Stop:** on `submit`, budget exhaustion, feed exhaustion, or the invalid-action limit.
5. **Score:** rewards and metrics are computed from the final memory and the trace.

**For GRPO:** every rollout in a group uses the same seed and persona. Across groups, vary the seed and persona, balanced across mirror pairs within each batch.

---

## 8. Package layout (follow `vf-init` output; adapt names)

```
environments/belief_feed/
├── pyproject.toml          # pinned verifiers; inline PEP 508 git deps only
├── README.md               # Hub page: research question, symmetry design, personas, tools, rewards, gates, limitations
├── belief_feed/
│   ├── __init__.py
│   ├── taskset.py          # TaskData, Task (rewards/metrics/stops), Taskset.load() (procedural; INFINITE ok)
│   ├── tools.py            # Toolset
│   ├── world.py            # generator, valences, sources, assertions, copies, records
│   ├── reasoner.py         # ideal_neutral, ideal_from_prior (exact)
│   ├── personas/           # persona texts, axis vectors, mirrors, REVIEW.md
│   ├── templates/          # post templates per style
│   └── splits.py           # persona/strategy/style/seed splits, eval seed hash
└── tests/                  # reasoner vs brute force, symmetry, no-leak, reward properties, determinism
```

---

## 9. Pre-registered gates (all before any training; report each)

- **Gate A, style confounds.** Simple text features (length, punctuation, capitalisation, exclamation marks, sentiment, n-gram probe) must predict `truth` at AUC 0.45–0.55, and predict manipulated vs unmanipulated items at AUC 0.45–0.55. If a feature fails, report it and fix the templates; don't drop rows.
- **Gate B, reasoner correctness.** Both ideal reasoners match brute-force enumeration on small worlds to 1e-6, and the rendered posts match the generator state.
- **Gate C, headroom.** `ideal_neutral` with a greedy investigation policy beats every shortcut baseline by a clear margin. The zero-shot baselines (one frontier API model, one small open model) sit meaningfully below `ideal_neutral`.
- **Gate D, symmetry.** Every check in section 3.4 passes. Apparent source lean is uncorrelated with reliability (|r| < 0.05 over 1,000 worlds). Mirror personas get structurally identical worlds.
- **Gate E, the persona is actually strong.** On the untrained baseline model, `motivated_gap` must be clearly negative (it is more accurate on belief-confirming claims), consistently across mirror pairs. If the gap is near zero, the persona isn't really biasing the model, and the challenge isn't real. Report it and propose stronger persona framing or a higher intensity before training. Also report whether the gap differs between mirror personas; **don't interpret any asymmetry as a fact about real-world ideologies**, since this is a fictional, templated setting.

---

## 10. Staging

- **v0 (build first):** world, personas, the fixed feed, all tools, rewards, both reasoners, gates A–E, baselines.
- **v1:** the full split protocol (held-out strategies, styles, and personas) and the generalisation evaluation. This is the research result.
- **v2:** the adaptive recommender. Within an episode, a bandit over post candidates maximising engagement (reads plus shares), with a cap on any single item type to prevent collapse onto one trick. World, tools, and rewards are unchanged.
- **v3 (optional):** a real-data transfer check (e.g. FEVER claims with their evidence), framed as transfer only, without political personas.

---

## 11. Order of work

1. Read the verifiers skills. Write `NOTES.md` covering the world model and the reasoner design choice.
2. Build `world.py` and `reasoner.py` with brute-force tests (Gate B).
3. Write the personas: library, mirrors, priors maths, `REVIEW.md`.
4. Write the templates and run Gate A.
5. Build the Toolset, Taskset, rewards, metrics, reward-property tests and no-leak tests.
6. Run a 5-episode smoke `vf-eval`.
7. Run Gates C, D and E and the baselines. **Stop and report** to Manas.
8. Only after he approves: v1 splits and the training plan.

---

## 12. Reporting

Short and candid. Lead with anything that breaks the premise: a failed gate, an asymmetry between mirror personas, a persona that doesn't actually bias the model, or values collapsing. Give numbers with their source file. Offer decisions as numbered options with your recommendation first. Never describe results as findings about real political groups.