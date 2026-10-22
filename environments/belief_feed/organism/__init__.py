"""A committed persona: a scripted motivated reasoner, and SFT data built from it.

The off-the-shelf models never take on the persona (NOTES.md, Persona-strength gate (E)), so there is no
motivated reasoning for RL to remove. This package builds a "model organism" that has it:
`policy.MotivatedReasoner` plays belief-feed through the real engine with three adjustable
biases, writes a templated first-person monologue (`monologue.py`), and `build_sft.py`
renders its episodes as chat transcripts for LoRA SFT. Nothing here is LLM-generated, and
every bias is applied identically to every persona, so the organism is symmetric by
construction (`tests/test_organism.py`).

Not part of the published environment package.
"""
