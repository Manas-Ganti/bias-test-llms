"""Persona, strategy, style and seed splits.

v0 trains and evaluates on `train`. The held-out splits exist so v1 can run the
generalisation protocol; nothing in v0 trains on them.
"""

import hashlib

from belief_feed.personas import MIRROR_PAIRS

TRAIN_PAIRS = MIRROR_PAIRS[:3]
"""progressive/conservative, democratic socialist/libertarian, green/industrialist."""
HELDOUT_PAIRS = MIRROR_PAIRS[3:]
"""civil libertarian/law-and-order traditionalist, localist populist/centrist technocrat."""

TRAIN_STRATEGIES = ("repetition", "fake_consensus", "false_authority", "identity_bait")
HELDOUT_STRATEGIES = ("laundering", "buried_correction")
TRAIN_STYLES = ("neutral_wire", "local_paper", "ragebait", "partisan_a")
HELDOUT_STYLES = ("partisan_b", "folksy")

EVAL_SEED_FRACTION = 0.1
SEED_SALT = "belief-feed-split-v1"


def seed_side(seed: int) -> str:
    """Hash a world seed to "train" or "eval"; stable across runs and machines."""
    digest = hashlib.sha256(f"{SEED_SALT}:{seed}".encode()).digest()
    return "eval" if int.from_bytes(digest[:8], "big") < EVAL_SEED_FRACTION * 2**64 else "train"


SPLITS: dict[str, dict] = {
    "train": {"pairs": TRAIN_PAIRS, "strategies": TRAIN_STRATEGIES, "styles": TRAIN_STYLES, "seeds": "train"},
    "eval_id": {"pairs": TRAIN_PAIRS, "strategies": TRAIN_STRATEGIES, "styles": TRAIN_STYLES, "seeds": "eval"},
    "eval_persona": {"pairs": HELDOUT_PAIRS, "strategies": TRAIN_STRATEGIES, "styles": TRAIN_STYLES, "seeds": "eval"},
    "eval_strategy": {"pairs": TRAIN_PAIRS, "strategies": HELDOUT_STRATEGIES, "styles": TRAIN_STYLES, "seeds": "eval"},
    "eval_style": {"pairs": TRAIN_PAIRS, "strategies": TRAIN_STRATEGIES, "styles": HELDOUT_STYLES, "seeds": "eval"},
    "eval_ood": {"pairs": HELDOUT_PAIRS, "strategies": HELDOUT_STRATEGIES, "styles": HELDOUT_STYLES, "seeds": "eval"},
}
