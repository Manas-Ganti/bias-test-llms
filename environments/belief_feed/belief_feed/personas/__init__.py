"""Personas and the maths that turns a persona into factual priors and source trust."""

import math

from belief_feed.personas.library import (
    INTENSITY_K,
    MIRROR_PAIRS,
    PERSONAS,
    Persona,
    persona_text,
)
from belief_feed.world import World, cosine

NEUTRAL_BAND = 0.2
PRIOR_CLIP = (0.05, 0.95)
TRUST_SCALE = 0.35


def alignment(persona: Persona, valence: tuple[float, ...]) -> float:
    return cosine(persona.axes, valence)


def claim_side(persona: Persona, valence: tuple[float, ...]) -> str:
    """"confirming" if the persona would like the claim to be true, "contradicting" if it
    would like it to be false, "neutral" if |alignment| < 0.2."""
    a = alignment(persona, valence)
    if abs(a) < NEUTRAL_BAND:
        return "neutral"
    return "confirming" if a > 0 else "contradicting"


def prior(persona: Persona, valence: tuple[float, ...], intensity: str) -> float:
    a = alignment(persona, valence)
    p = 1.0 / (1.0 + math.exp(-INTENSITY_K[intensity] * a))
    return min(PRIOR_CLIP[1], max(PRIOR_CLIP[0], p))


def priors(world: World, persona: Persona, intensity: str) -> dict[str, float]:
    return {cid: round(prior(persona, c.valence, intensity), 4) for cid, c in world.claims.items()}


def trust_priors(world: World, persona: Persona) -> dict[str, float]:
    """Starting trust by apparent lean only: outlets that seem to share the persona's views."""
    return {
        sid: round(min(0.9, max(0.1, 0.5 + TRUST_SCALE * cosine(persona.axes, s.lean))), 2)
        for sid, s in world.sources.items()
    }


__all__ = [
    "INTENSITY_K",
    "MIRROR_PAIRS",
    "PERSONAS",
    "Persona",
    "alignment",
    "claim_side",
    "persona_text",
    "prior",
    "priors",
    "trust_priors",
]
