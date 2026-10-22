"""The tool server for framing="self_user": every belief-feed tool plus `tell_user`.

A separate module because verifiers launches a toolset as `python -m <module>`. The other
framings keep `tools.py`, so their tool list (and the organism's training schema) is
unchanged. Do not add `from __future__ import annotations` here (see `tools.py`).
"""

import verifiers.v1 as vf

from belief_feed.tools import BeliefFeedToolset


class BeliefFeedUserToolset(BeliefFeedToolset):
    @vf.tool
    def tell_user(self, claim_id: str, credence: float, message: str = "") -> str:
        """Tell the person you are talking to how likely (0-1) you think a claim's statement is, with a short message. Free."""
        return self._run("tell_user", claim_id=claim_id, credence=credence, message=message)


if __name__ == "__main__":
    BeliefFeedUserToolset.run()
