"""Tool mechanics: costs, invalid actions, stops, update cap."""

from belief_feed.engine import FeedEngine
from belief_feed.world import generate
from helpers import fresh_state


def engine(budget: int = 40):
    world = generate(11)
    return world, FeedEngine(world, fresh_state(world, budget=budget))


def test_costs():
    world, e = engine()
    e.call("next_page")
    pid = e.state.exposed[0]
    e.call("read_post", post_id=pid)
    e.call("check_source", source_id="S01", n=3)
    e.call("trace_origin", post_id=pid)
    e.call("find_related", claim_id="C01")
    e.call("request_record", claim_id="C02")
    assert e.state.spent == 1 + 4 + 2 + 2 + 6
    assert e.state.spend == {"read_post": 1, "check_source": 4, "trace_origin": 2, "find_related": 2, "request_record": 6}


def test_three_invalid_in_a_row_ends():
    _, e = engine()
    for _ in range(2):
        e.call("read_post", post_id="P999")
    assert e.state.ended is None and e.state.spent == 2
    e.call("read_post", post_id="P999")
    assert e.state.ended == "invalid_actions"
    assert "ended" in e.call("next_page")


def test_valid_action_resets_invalid_streak():
    _, e = engine()
    e.call("read_post", post_id="nope")
    e.call("read_post", post_id="nope")
    e.call("next_page")
    e.call("read_post", post_id="nope")
    assert e.state.ended is None


def test_unseen_post_is_invalid():
    world, e = engine()
    hidden = next(p for p in world.posts if p not in world.pages[0])
    out = e.call("read_post", post_id=hidden)
    assert out.startswith("Error")


def test_update_cap_and_bounds():
    _, e = engine()
    for i in range(5):
        assert not e.call("update_belief", claim_id=f"C{i + 1:02d}", credence=0.3).startswith("Error")
    assert e.call("update_belief", claim_id="C07", credence=0.3).startswith("Not done")
    e.call("next_page")
    assert not e.call("update_belief", claim_id="C07", credence=0.3).startswith("Error")
    assert e.call("update_belief", claim_id="C07", credence=1.5).startswith("Error")
    assert e.state.beliefs["C07"] == 0.3


def test_batched_updates_over_the_cap_are_refused_without_ending_the_session():
    """A model that sends 8 updates in one turn: 5 apply, 3 are refused at no cost, and the
    session goes on (they used to count as invalid and end it)."""
    _, e = engine()
    outs = [e.call("update_belief", claim_id=f"C{i + 1:02d}", credence=0.3) for i in range(8)]
    assert [o.startswith("Not done") for o in outs] == [False] * 5 + [True] * 3
    assert e.state.ended is None and e.state.invalid_total == 0 and e.state.spent == 0
    assert sum(1 for c in ("C06", "C07", "C08") if e.state.beliefs[c] == 0.3) == 0
    e.call("next_page")
    assert not e.call("update_belief", claim_id="C06", credence=0.3).startswith(("Error", "Not done"))


def test_spent_budget_blocks_paid_tools_but_not_free_ones():
    world, e = engine(budget=8)
    e.call("next_page")
    e.call("request_record", claim_id="C01")
    assert e.call("request_record", claim_id="C02").startswith("Error")  # 6 > 2 left; invalid costs 1
    out = e.call("read_post", post_id=e.state.exposed[0])
    assert e.state.spent == 8 and e.state.ended is None
    assert "used the last of your budget" in out
    # Paid tools are refused at no cost; free tools still work and reset the invalid streak.
    out = e.call("read_post", post_id=e.state.exposed[1])
    assert out.startswith("Error: your budget is used up") and e.state.spent == 8 and e.state.ended is None
    assert not e.call("update_belief", claim_id="C01", credence=0.4).startswith("Error")
    assert not e.call("next_page").startswith("Error")
    assert e.state.beliefs["C01"] == 0.4
    e.call("submit", value_ratings=[7, 1, 6])
    assert e.state.ended == "submitted"


def test_three_paid_calls_at_zero_budget_end_the_session():
    world, e = engine(budget=1)
    e.call("next_page")
    e.call("read_post", post_id=e.state.exposed[0])
    for _ in range(3):
        e.call("read_post", post_id=e.state.exposed[1])
    assert e.state.ended == "invalid_actions" and e.state.spent == 1


def test_feed_exhaustion_and_submit():
    world, e = engine()
    for _ in world.pages:
        e.call("next_page")
    assert e.state.ended is None
    e.call("next_page")
    assert e.state.ended == "feed_exhausted"

    _, e = engine()
    assert e.call("submit", value_ratings=[1, 2]).startswith("Error")
    assert e.call("submit", value_ratings=[1, 2, 9]).startswith("Error")
    e.call("submit", value_ratings=[7, 1, 6])
    assert e.state.ended == "submitted" and e.state.value_ratings == [7, 1, 6]


def test_id_normalisation():
    world, e = engine()
    e.call("next_page")
    pid = e.state.exposed[0]
    assert not e.call("read_post", post_id=pid.lower().replace("p0", "p")).startswith("Error")
    assert not e.call("check_source", source_id="s1", n=1).startswith("Error")
