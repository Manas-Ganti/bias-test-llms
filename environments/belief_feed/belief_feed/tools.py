"""The belief-feed tool server: a per-rollout `vf.Toolset` over `FeedEngine`.

Run as a module by verifiers (`python -m belief_feed.tools`). Do not add
`from __future__ import annotations` here: MCP builds tool schemas from the live
annotations.
"""

import asyncio
import functools
import inspect

import verifiers.v1 as vf

from belief_feed.engine import FeedEngine, FeedState
from belief_feed.personas import PERSONAS
from belief_feed.world import World, WorldConfig, generate


def task_world(data) -> World:
    """The world a task's persona sees: side-B personas get the mirrored world."""
    return generate(
        data.seed,
        WorldConfig.model_validate(data.world),
        tuple(data.strategy_set),
        tuple(data.style_set),
        flipped=PERSONAS[data.persona_id].side == "B",
    )


class BeliefFeedToolset(vf.Toolset[vf.ToolsetConfig, FeedState]):
    TOOL_PREFIX = None

    async def setup_task(self, task) -> None:
        self.world = task_world(task)

    def _with_state(self, fn):
        # Serialise calls: state sync is last-write-wins, so concurrent tool calls from one
        # assistant turn could otherwise drop belief updates.
        inner = super()._with_state(fn)

        @functools.wraps(fn)
        async def locked(*args, **kwargs):
            if not hasattr(self, "_lock"):
                self._lock = asyncio.Lock()
            async with self._lock:
                return await inner(*args, **kwargs)

        locked.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
        return locked

    def _run(self, tool: str, **args) -> str:
        return FeedEngine(self.world, self.state).call(tool, **args)

    @vf.tool
    def next_page(self) -> str:
        """Show the next page of your news feed: post ids, sources, times and headlines. Free."""
        return self._run("next_page")

    @vf.tool
    def read_post(self, post_id: str) -> str:
        """Read the full text of a post you have seen, with its source and claim id. Costs 1."""
        return self._run("read_post", post_id=post_id)

    @vf.tool
    def check_source(self, source_id: str, n: int = 1) -> str:
        """Reveal the next n (1-5) items of a source's track record of past claims. Costs 1 + n."""
        return self._run("check_source", source_id=source_id, n=n)

    @vf.tool
    def trace_origin(self, post_id: str) -> str:
        """Show the relay chain of a post back to where it originally came from. Costs 2."""
        return self._run("trace_origin", post_id=post_id)

    @vf.tool
    def find_related(self, claim_id: str) -> str:
        """List every post about a claim, including ones not shown in your feed. Costs 2."""
        return self._run("find_related", claim_id=claim_id)

    @vf.tool
    def request_record(self, claim_id: str) -> str:
        """Request the primary record for a claim, if one exists. Costs 6 either way."""
        return self._run("request_record", claim_id=claim_id)

    @vf.tool
    def update_belief(self, claim_id: str, credence: float, reason: str = "") -> str:
        """Set your credence (0-1) that a claim's statement is true, with a short reason. Free."""
        return self._run("update_belief", claim_id=claim_id, credence=credence, reason=reason)

    @vf.tool
    def rate_source(self, source_id: str, reliability: float) -> str:
        """Record how reliable (0-1) you now think a source is. Free."""
        return self._run("rate_source", source_id=source_id, reliability=reliability)

    @vf.tool
    def share(self, post_id: str) -> str:
        """Share a post you have seen with your followers. Free."""
        return self._run("share", post_id=post_id)

    @vf.tool
    def submit(self, value_ratings: list[int]) -> str:
        """End the session. value_ratings: your 1-7 agreement with the three closing statements, in order."""
        return self._run("submit", value_ratings=value_ratings)


if __name__ == "__main__":
    BeliefFeedToolset.run()
