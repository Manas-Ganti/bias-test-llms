"""Validate the hand-gathered source pack (organism/sources/) before it goes into SFT data.

Errors (exit 1) break the project's rules. Warnings need a human look.
- situations.json: every added line uses only its situation's slots (+ {value} where
  value_ok), formats cleanly, is 4-40 words on one line, has no digits, no banned words and
  no duplicates.
- values/<persona>.json: every theme cites a source; phrases fit "I care about ___"; mirror
  personas have the same number of themes, phrases and paraphrases, with total phrase
  words within 10% and each paraphrase 150-250 words, matched within 10% by position.
- Capitalised words mid-line are warned about: they are usually real names, which must not
  appear (rule 2). The fictional place and outlet names live in belief_feed/templates/names.py.

    uv run python -m organism.check_sources [--dir organism/sources]
"""

import argparse
import json
import re
import string
import sys
from pathlib import Path

from belief_feed.personas import MIRROR_PAIRS
from belief_feed.splits import SPLITS

BANNED = [
    # the environment's no-leak list (tests/test_no_leak.py): never in anything the model reads
    "repetition", "consensus", "launder", "false authority", "identity", "bait", "buried", "manipulat",
    "strategy", "style", "copied", "persona", "congruent", "confirming", "contradicting", "bias",
    # ideology and party labels: values are described, never named
    "progressive", "conservative", "socialist", "socialism", "libertarian", "conservationist", "industrialist",
    "liberal", "leftist", "left-wing", "right-wing", "democrat", "republican", "labour", "tory", "marxist",
    "capitalist", "communist", "fascist", "woke", "maga", "populist", "technocrat", "ideolog",
    # dehumanising or slur-adjacent framing (rule 6); extend as needed
    "vermin", "parasite", "subhuman", "traitor", "scum", "animals", "infest",
]  # fmt: skip
SLOT = re.compile(r"\{(\w+)\}")
DUMMY = {"source": "The Open Ledger", "claim": "C07", "good": "2", "n": "3", "value": "decisions made close to home"}


def words(text: str) -> int:
    return len(text.split())


def banned(text: str) -> list[str]:
    low = text.lower()
    return [w for w in BANNED if re.search(rf"\b{re.escape(w)}", low)]


def capitalised(text: str) -> list[str]:
    body = SLOT.sub("", text)
    toks = re.findall(r"(?<=[a-z,;:] )[A-Z][a-z]+", body)
    return [t for t in toks if t != "I"]


def check_situations(path: Path, errors: list, warnings: list) -> dict:
    data = json.loads(path.read_text())
    counts = {}
    for key, sit in data["situations"].items():
        allowed = set(sit["slots"]) | ({"value"} if sit.get("value_ok") else set())
        seen = set(sit["current"])
        for line in sit["add"]:
            where = f"situations.{key}: {line!r}"
            used = set(SLOT.findall(line))
            if used - allowed:
                errors.append(f"{where} uses slots {sorted(used - allowed)}; allowed {sorted(allowed)}")
            try:
                line.format(**DUMMY)
            except (KeyError, IndexError, ValueError) as e:
                errors.append(f"{where} does not format: {e}")
            if "\n" in line or not 4 <= words(line) <= 40:
                errors.append(f"{where} must be one line of 4-40 words")
            if re.search(r"\d", SLOT.sub("", line)):
                errors.append(f"{where} contains a digit (figures must come from the world, not the monologue)")
            if b := banned(line):
                errors.append(f"{where} contains banned words {b}")
            if line in seen:
                errors.append(f"{where} is a duplicate")
            seen.add(line)
            if c := capitalised(line):
                warnings.append(f"{where} has capitalised words {c}: real names are not allowed")
        counts[key] = (len(sit["current"]), len(sit["add"]))
    for key, r in data["reasons"].items():
        if key.startswith("_"):
            continue
        for line in r["add"]:
            if len(line) > 80 or banned(line) or re.search(r"\d", line):
                errors.append(f"reasons.{key}: {line!r} must be under 80 characters with no digits or banned words")
    return counts


def check_values(folder: Path, errors: list, warnings: list) -> dict:
    train = [p for pair in SPLITS["train"]["pairs"] for p in pair]
    packs = {}
    for pid in train:
        path = folder / f"{pid}.json"
        if not path.exists():
            errors.append(f"values/{pid}.json is missing")
            continue
        pack = json.loads(path.read_text())
        pack["themes"] = [t for t in pack["themes"] if t.get("theme") or t.get("phrases")]
        for t in pack["themes"]:
            where = f"values/{pid}: theme {t.get('theme')!r}"
            if not t.get("source"):
                errors.append(f"{where} has no source")
            for ph in t.get("phrases", []):
                if b := banned(ph):
                    errors.append(f"{where}: {ph!r} contains banned words {b}")
                if not 3 <= words(ph) <= 14 or ph[:1] in string.ascii_uppercase or ph.endswith("."):
                    errors.append(f"{where}: {ph!r} should be a 3-14 word lower-case noun phrase that fits 'I care about ___'")
                if c := capitalised(" x " + ph):
                    warnings.append(f"{where}: {ph!r} has capitalised words {c}")
        for i, para in enumerate(pack.get("paraphrases", [])):
            if not 150 <= words(para) <= 250:
                errors.append(f"values/{pid}: paraphrase {i + 1} has {words(para)} words (150-250)")
            if b := [w for w in banned(para) if w not in ("progressive", "conservative", "socialist", "libertarian", "conservationist", "industrialist")]:  # fmt: skip
                errors.append(f"values/{pid}: paraphrase {i + 1} contains banned words {b}")
        packs[pid] = pack
    for a, b in MIRROR_PAIRS:
        if a not in packs or b not in packs:
            continue
        pa, pb = packs[a], packs[b]
        na = [len(t.get("phrases", [])) for t in pa["themes"]]
        nb = [len(t.get("phrases", [])) for t in pb["themes"]]
        if len(na) != len(nb) or sum(na) != sum(nb):
            errors.append(f"{a} vs {b}: themes/phrases {len(na)}/{sum(na)} vs {len(nb)}/{sum(nb)} must match (rule 3)")
        wa = sum(words(p) for t in pa["themes"] for p in t.get("phrases", []))
        wb = sum(words(p) for t in pb["themes"] for p in t.get("phrases", []))
        if max(wa, wb) and abs(wa - wb) > 0.1 * max(wa, wb):
            errors.append(f"{a} vs {b}: total phrase words {wa} vs {wb} differ by more than 10%")
        qa, qb = pa.get("paraphrases", []), pb.get("paraphrases", [])
        if len(qa) != len(qb):
            errors.append(f"{a} vs {b}: {len(qa)} vs {len(qb)} paraphrases must match")
        for i, (x, y) in enumerate(zip(qa, qb)):
            if abs(words(x) - words(y)) > 0.1 * max(words(x), words(y)):
                errors.append(f"{a} vs {b}: paraphrase {i + 1} lengths {words(x)} vs {words(y)} differ by more than 10%")
    return {pid: (len(p["themes"]), sum(len(t.get("phrases", [])) for t in p["themes"]), len(p.get("paraphrases", []))) for pid, p in packs.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path(__file__).parent / "sources")
    args = ap.parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    sit = check_situations(args.dir / "situations.json", errors, warnings)
    vals = check_values(args.dir / "values", errors, warnings)
    print("situations (current + added):")
    for k, (cur, add) in sit.items():
        print(f"  {k:18s} {cur} + {add}")
    print("values (themes, phrases, paraphrases):")
    for pid, (t, p, q) in vals.items():
        print(f"  {pid:28s} {t:3d} {p:4d} {q:3d}")
    for w in warnings:
        print("WARNING", w)
    for e in errors:
        print("ERROR  ", e)
    print(f"\n{len(errors)} errors, {len(warnings)} warnings")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
