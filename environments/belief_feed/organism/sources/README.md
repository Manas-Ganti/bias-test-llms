# Source pack: hand-gathered material for the next organism

This folder is where you add the material that makes the next SFT organism less scripted. I
turn it into training transcripts with `organism/build_sft.py`, so what you add here only
needs to be *wording*: the world, evidence and scoring stay generated.

Run the checker every time you save:

```bash
uv run python -m organism.check_sources
```

It must say `0 errors` before I build anything from it. Warnings (usually a capitalised word
that might be a real name) need a look, not necessarily a change.

## What to fill in

| File | What goes in it | Target |
|---|---|---|
| `situations.json` | New first-person monologue lines, under `add`, for each of the 17 moments in an episode (reading a welcome post, checking a source, getting a record…). `current` shows the existing lines and `slots` the placeholders you may use. | 10–15 new lines per situation, so ~200 in total (from ~45 now) |
| `situations.json` → `reasons` | Short reasons stored with each `update_belief` (under 80 characters). | 5 per kind |
| `values/<persona>.json` → `themes` | What each persona protects and fears, in its adherents' own terms. Each theme has noun phrases that fit "I care about ___" and the source it came from. | 6 themes × 3 phrases per persona |
| `values/<persona>.json` → `paraphrases` | 150–250-word first-person rewrites of the persona text in `belief_feed/personas/library.py`, at the same intensity. | 3–5 per persona |

Only the 6 train personas get files (progressive/conservative, socialist/libertarian,
green/industrialist). The other 4 stay held out, so the organism never sees them.

**Mirror rule:** the two personas of a pair must get the same number of themes, phrases and
paraphrases, with lengths within 10%. The checker enforces this. Fill both sides of a
pair in the same sitting, from sources of equal depth.

## Where to look

**For the monologue (how motivated reasoning sounds).** Read these for the *moves*, then
write each move as a line for the matching situation:
- Taber & Lodge, *Motivated Skepticism in the Evaluation of Political Beliefs* (2006):
  disconfirmation bias (arguing harder against unwelcome evidence), the prior-attitude
  effect, and selective search. These map to `unwelcome`, `unwelcome_check` and `check_good`.
- Kunda, *The Case for Motivated Reasoning* (1990): reasoning toward a conclusion while
  feeling objective ("I'm just looking at the facts"). Maps to `welcome` and `submit`.
- Kahan's papers on identity-protective cognition: rejecting evidence because of who it
  threatens. Maps to `unwelcome` and `record_unwelcome`, using the `{value}` slot.
- Haidt, *The Righteous Mind*: post-hoc justification ("the rider serves the elephant").
  Maps to `welcome` and `check_bad`.

**For values (what each persona cares about).** Read in matched pairs, the same number of
books per side:

| Persona | Read |
|---|---|
| Progressive | Rawls, *A Theory of Justice*; Nussbaum, *Creating Capabilities* |
| Conservative | Burke, *Reflections on the Revolution in France*; Scruton, *How to Be a Conservative* |
| Democratic socialist | Tawney, *Equality*; Harrington, *Socialism: Past and Future* |
| Free-market libertarian | Hayek, *The Road to Serfdom*; Friedman, *Capitalism and Freedom* |
| Green conservationist | Leopold, *A Sand County Almanac*; Schumacher, *Small Is Beautiful* |
| Growth-first industrialist | Simon, *The Ultimate Resource*; McCloskey, *Bourgeois Dignity* |

Sowell's *A Conflict of Visions* and Haidt's moral foundations cover every side at once and
are good for checking that both halves of a pair get equally strong values.

**For phrasing variety (optional).** The Moral Foundations Twitter and Reddit corpora
(Hoover et al. 2020; Trager et al. 2022) and the Manifesto Project's coded sentences show
how people actually voice care, fairness, loyalty, authority and liberty. Use them to vary
*how* a line is phrased, never *what* it is about. Check each dataset's licence; some are
research-use only.

## How to turn a source into a line

1. Read a passage and name the move or value in your own words. For example: "Taber & Lodge:
   people spend longer picking apart arguments they dislike."
2. Write it as something this agent would think in the fictional region, at the matching
   situation, using only the allowed slots. For example, under `unwelcome`: *"I'll need a
   lot more than one post from {source} before I take that seriously about {claim}."*
3. Put the citation in that situation's `notes` (or the theme's `source`).
4. Run the checker.

## Rules (the checker enforces most of them)

- **Paraphrase, never quote.** Nothing copied from a book or dataset goes in, for copyright
  reasons and because only templates may reach the model.
- **Fictional world only:** no real people, parties, countries, places, outlets or events.
  Ideologies are described by their values, never named.
- **Scepticism targets sources, never groups of people.** "{source} has an angle" is fine;
  insults about people on the other side are not. No slurs or dehumanising language.
- **No facts in the monologue:** no figures, dates or claims. Those come from the world.
- **The monologue never mentions a user.** User behaviour is what we measure in Gate U, so
  the organism must not be taught it.

## What I do with it

1. Run the checker, then a symmetry review of every new line (`personas/REVIEW.md`).
2. Rebuild the SFT data on the 6 train personas, drawing on your lines and values.
3. Retrain, then run Gate O on the held-out personas, which the new organism has never seen.
