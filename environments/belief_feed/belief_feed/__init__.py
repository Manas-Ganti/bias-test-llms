from verifiers.v1.harnesses.null.harness import NullHarness

from belief_feed.taskset import BeliefFeedTaskset


class BeliefFeedHarness(NullHarness):
    """Tool-only default harness: no shell, so the agent cannot read package files."""


__all__ = ["BeliefFeedHarness", "BeliefFeedTaskset"]
