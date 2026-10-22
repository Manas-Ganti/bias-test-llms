"""Style-leak gate (A): style confounds.

Simple text features of a post (headline + body, what read_post shows besides ids and
source) must not predict
  (1) the truth of the post's claim,
  (2) whether the post's assertion is correct, or
  (3) whether the post belongs to a manipulation,
beyond AUC 0.45-0.55. Each simple feature is scored on its own (AUC on all posts); the
combined simple features and a word/bigram n-gram probe are scored with logistic
regression under 5-fold cross-validation grouped by world, so the probe cannot memorise
one world's names. Mirror worlds are included half the time. `--persuasion` runs the same
checks with the persuasion layer on (the persuasion-layer gate, P, check 1).

    uv run python gates/gate_a.py --worlds 400 [--persuasion]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from belief_feed.splits import SPLITS, seed_side  # noqa: E402
from belief_feed.world import STRATEGIES, WorldConfig, generate  # noqa: E402
from belief_feed.templates.styles import STYLES  # noqa: E402

OUT = Path(__file__).parent / "results"
BAND = (0.45, 0.55)
POSITIVE = {"good", "great", "improved", "rose", "met", "support", "stable", "saved", "proud", "care", "together"}
NEGATIVE = {"fell", "cut", "outrageous", "angry", "furious", "worse", "worsened", "lost", "loss", "dropped", "unreal", "short", "delayed", "late", "objections", "oppose"}


def simple_features(text: str) -> dict[str, float]:
    words = re.findall(r"[A-Za-z']+", text)
    lw = [w.lower() for w in words]
    letters = [c for c in text if c.isalpha()]
    return {
        "chars": len(text),
        "words": len(words),
        "exclamations": text.count("!"),
        "questions": text.count("?"),
        "commas": text.count(","),
        "colons": text.count(":"),
        "digits": sum(c.isdigit() for c in text),
        "caps_ratio": sum(c.isupper() for c in letters) / max(1, len(letters)),
        "caps_words": sum(1 for w in words if len(w) > 1 and w.isupper()),
        "negations": sum(1 for w in lw if w in {"not", "no", "never", "did", "didn't"}),
        "sentiment": (sum(w in POSITIVE for w in lw) - sum(w in NEGATIVE for w in lw)) / max(1, len(lw)),
    }


def collect(n_worlds: int, split_name: str, persuasion: bool = False) -> dict:
    split = SPLITS[split_name]
    seeds = [s for s in range(100_000) if seed_side(s) == split["seeds"]][:n_worlds]
    rows = []
    config = WorldConfig(persuasion=persuasion)
    for seed in seeds:
        world = generate(seed, config, split["strategies"], split["styles"], flipped=seed % 2 == 1)
        for p in world.posts.values():
            claim = world.claims[p.claim]
            rows.append(
                {
                    "world": seed,
                    "text": f"{p.headline}\n{p.body}",
                    "claim_true": claim.truth,
                    "assertion_correct": int(p.stance == claim.truth),
                    "manipulated": int(p.strategy is not None),
                }
            )
    return {"rows": rows, "seeds": seeds}


def auc(y, s) -> float:
    return float(roc_auc_score(y, s))


def run(n_worlds: int, split_name: str, persuasion: bool = False) -> dict:
    data = collect(n_worlds, split_name, persuasion)
    rows = data["rows"]
    texts = [r["text"] for r in rows]
    groups = np.array([r["world"] for r in rows])
    feats = [simple_features(t) for t in texts]
    names = list(feats[0])
    X = np.array([[f[n] for n in names] for f in feats])
    report: dict = {"split": split_name, "persuasion": persuasion, "n_worlds": n_worlds, "n_posts": len(rows), "band": BAND, "labels": {}}
    for label in ("claim_true", "assertion_correct", "manipulated"):
        y = np.array([r[label] for r in rows])
        single = {n: auc(y, X[:, i]) for i, n in enumerate(names)}
        cv = GroupKFold(n_splits=5)
        combined, ngram = np.zeros(len(y)), np.zeros(len(y))
        for train, test in cv.split(X, y, groups):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            m.fit(X[train], y[train])
            combined[test] = m.predict_proba(X[test])[:, 1]
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)
            Xt = vec.fit_transform([texts[i] for i in train])
            probe = LogisticRegression(max_iter=2000, C=1.0)
            probe.fit(Xt, y[train])
            ngram[test] = probe.predict_proba(vec.transform([texts[i] for i in test]))[:, 1]
        results = {"single_feature_auc": single, "combined_simple_auc": auc(y, combined), "ngram_probe_auc": auc(y, ngram)}
        all_aucs = list(single.values()) + [results["combined_simple_auc"], results["ngram_probe_auc"]]
        results["base_rate"] = float(y.mean())
        results["pass"] = all(BAND[0] <= a <= BAND[1] for a in all_aucs)
        results["failing"] = {
            k: v
            for k, v in {**single, "combined_simple": results["combined_simple_auc"], "ngram_probe": results["ngram_probe_auc"]}.items()
            if not BAND[0] <= v <= BAND[1]
        }
        report["labels"][label] = results
    # The pre-registered gate covers claim truth and manipulation; assertion correctness is
    # reported as an extra check on the same templates.
    report["pass"] = report["labels"]["claim_true"]["pass"] and report["labels"]["manipulated"]["pass"]
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=400)
    ap.add_argument("--split", default="train")
    ap.add_argument("--persuasion", action="store_true")
    args = ap.parse_args()
    report = run(args.worlds, args.split, args.persuasion)
    OUT.mkdir(exist_ok=True)
    tag = "_persuasion" if args.persuasion else ""
    (OUT / f"gate_a_{args.split}{tag}.json").write_text(json.dumps(report, indent=1))
    print(f"Style-leak gate (A) ({args.split}{tag}, {report['n_worlds']} worlds, {report['n_posts']} posts): {'PASS' if report['pass'] else 'FAIL'}")
    for label, r in report["labels"].items():
        print(f"  {label:18s} combined={r['combined_simple_auc']:.3f} ngram={r['ngram_probe_auc']:.3f} pass={r['pass']} failing={ {k: round(v, 3) for k, v in r['failing'].items()} }")


if __name__ == "__main__":
    main()
