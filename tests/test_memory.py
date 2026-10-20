import numpy as np

from wrc_demo.memory.beliefs import BeliefStore
from wrc_demo.memory.episodic import EpisodicMemory


def test_horizon_pruning():
    fake_now = [8.0]
    mem = EpisodicMemory(horizon_s=10.0, clock=lambda: fake_now[0])
    mem.add("note", "old event", t=0.0)
    mem.add("note", "recent event", t=8.0)
    fake_now[0] = 12.0
    texts = [e.text for e in mem.events()]
    assert "old event" not in texts
    assert "recent event" in texts


def test_digest_format():
    mem = EpisodicMemory(horizon_s=15.0)
    mem.add("action", "grasped cube", t=100.0)
    digest = mem.digest(now=105.0)
    assert "5.0s ago" in digest and "grasped cube" in digest
    assert EpisodicMemory(horizon_s=15.0).digest() == "(memory empty)"


def test_display_digest_survives_horizon():
    # The dashboard's narration feed reads display_digest, not digest: the
    # 15 s horizon is an agent-prompt budget and must not empty the screen.
    import re

    fake_now = [0.0]
    mem = EpisodicMemory(horizon_s=10.0, clock=lambda: fake_now[0])
    mem.add("action", "early move", t=0.0)
    mem.add("outcome", "grasped cube", t=8.0)
    fake_now[0] = 600.0  # both events are far past the horizon now
    assert mem.digest(now=600.0) == "(memory empty)"  # agent window pruned
    disp = mem.display_digest(now=600.0)
    assert "early move" in disp and "grasped cube" in disp
    # Line format must stay parseable by the dashboard's renderFeed regex
    # (stream_server.py): "[  N.Ns ago] (kind) text".
    assert re.match(r"^\[\s*[\d.]+s ago\] \(\w+\) .+$", disp.splitlines()[0])
    # max_lines keeps only the newest events
    assert mem.display_digest(max_lines=1, now=600.0).count("\n") == 0
    assert "grasped cube" in mem.display_digest(max_lines=1, now=600.0)


def test_thumbnail_stored():
    mem = EpisodicMemory(horizon_s=15.0)
    rgb = np.full((120, 160, 3), 128, dtype=np.uint8)
    ev = mem.add("observation", "frame", rgb=rgb)
    assert ev.thumb_jpeg is not None and len(ev.thumb_jpeg) > 100
    assert mem.last_frame_jpeg() == ev.thumb_jpeg


def test_embedding_recall():
    mem = EpisodicMemory(horizon_s=15.0, embed_dim=32)
    e1 = np.ones(32)
    e2 = -np.ones(32)
    ev1 = mem.add("observation", "red cube seen", embedding=e1)
    mem.add("observation", "blue ball seen", embedding=e2)
    hits = mem.recall_similar(e1, k=1)
    assert hits and hits[0] is ev1


def test_belief_matching_and_ema():
    store = BeliefStore(match_radius_m=0.08, pos_alpha=0.5)
    b1 = store.update("cup", [0.30, 0.00, 0.02], conf=0.9, t=1.0)
    b2 = store.update("cup", [0.32, 0.00, 0.02], conf=0.9, t=2.0)
    assert b1 is b2  # matched, not duplicated
    assert np.isclose(b2.position[0], 0.31)
    b3 = store.update("cup", [0.60, 0.20, 0.02], conf=0.9, t=3.0)
    assert b3 is not b2  # too far (but b2 still fresh at t=3): second instance
    assert len(store.all(now=3.0)) == 2


def test_belief_revives_when_all_stale_and_object_moved():
    """One physical banana, moved by hand >match_radius away: the store
    must REVIVE the stale belief in place instead of growing a second row
    (root cause of the 'one banana, several dashboard rows' complaint)."""
    store = BeliefStore(match_radius_m=0.08, stale_revive_s=5.0)
    b1 = store.update("banana", [0.25, 0.00, 0.02], conf=0.8, t=1.0)
    b2 = store.update("banana", [0.45, 0.20, 0.02], conf=0.8, t=100.0)
    assert b1 is b2  # revived the only (stale) belief, not duplicated
    assert np.allclose(b2.position, [0.45, 0.20, 0.02])
    assert b2.observations == 2
    assert len(store.all(now=100.0)) == 1


def test_belief_cap_evicts_stale_but_never_live():
    store = BeliefStore(match_radius_m=0.08, stale_revive_s=5.0,
                        max_per_label=2)
    b_stale = store.update("cup", [0.20, 0.00, 0.02], conf=0.8, t=1.0)
    b_live = store.update("cup", [0.40, 0.00, 0.02], conf=0.8, t=2.0)
    store.update("cup", [0.41, 0.00, 0.02], conf=0.8, t=99.0)  # keep b_live fresh
    # third instance far from both: b_stale is old, b_live was just seen ->
    # append, then the cap evicts the STALE one only
    b3 = store.update("cup", [0.30, 0.25, 0.02], conf=0.8, t=100.0)
    beliefs = store.all(now=100.0)
    assert len(beliefs) == 2
    # identity, not __eq__: ObjectBelief's numpy fields make `in` ambiguous
    ids = {id(b) for b in beliefs}
    assert id(b_stale) not in ids and id(b_live) in ids and id(b3) in ids
    # all three genuinely live -> cap never evicts any of them
    store = BeliefStore(match_radius_m=0.08, stale_revive_s=5.0,
                        max_per_label=2)
    store.update("bottle", [0.20, 0.00, 0.02], conf=0.8, t=10.0)
    store.update("bottle", [0.40, 0.00, 0.02], conf=0.8, t=10.5)
    store.update("bottle", [0.30, 0.25, 0.02], conf=0.8, t=11.0)
    assert len(store.all(now=11.0)) == 3


def test_belief_states_and_find():
    store = BeliefStore()
    store.update("mug", [0.3, 0.1, 0.0], conf=0.8, t=100.0)
    assert store.find("mug").state(now=100.5) == "visible"
    assert store.find("mug").state(now=110.0) == "remembered"
    assert store.find("red mug") is not None  # loose contains-match
    assert store.find("banana") is None


def test_belief_removal():
    store = BeliefStore()
    store.update("cube", [0.3, 0.0, 0.0], conf=0.9)
    assert store.mark_removed("cube", near=[0.3, 0.0, 0.0])
    assert store.find("cube") is None
    assert not store.mark_removed("cube")


def test_belief_remembers_real_cloud_and_fix_uses_it():
    """Grasp-from-memory must use the object's TRUE shape when a real mask
    cloud was stored: a banana's box-slab approximation (78mm) reads as
    ungraspable, while its real cloud reads ~35mm (GGX-empty root cause)."""
    import numpy as np

    from wrc_demo.skills.runtime import SkillRuntime

    store = BeliefStore()
    rng = np.random.default_rng(1)
    # thin curved-ish object: 16cm long, 3.5cm wide, 3cm tall
    pts = (rng.random((1500, 3)) - 0.5) * np.array([0.16, 0.035, 0.03])
    pts += np.array([0.24, 0.14, 0.02])
    b = store.update("banana", pts.mean(axis=0), 0.8, points=pts)
    assert b.points is not None and b.points.shape == (384, 3)  # subsampled

    fix = SkillRuntime._fix_from_belief("banana", b)
    assert fix.points.shape[0] == 384  # real cloud, not the 360-pt box synth
    ext = np.sort(fix.extent)[::-1]
    assert ext[1] < 0.05  # true jaw span, not a bbox slab
    # fused position moved (EMA) -> cloud must follow the fused center
    b.position = b.position + np.array([0.03, 0.0, 0.0])
    fix2 = SkillRuntime._fix_from_belief("banana", b)
    assert abs(fix2.points[:, 0].mean() - fix.points[:, 0].mean() - 0.03) < 5e-3


def test_belief_bbox_fallback_never_stores_points():
    """bbox-rectangle masks sweep in table pixels; update() is only called
    with points= for real masks -- and a points=None update must not clear
    a previously remembered cloud."""
    import numpy as np

    store = BeliefStore()
    pts = np.random.default_rng(2).random((200, 3)).astype(np.float32)
    b = store.update("cube", np.array([0.2, 0.0, 0.02]), 0.8, points=pts)
    assert b.points is not None
    store.update("cube", np.array([0.2, 0.0, 0.02]), 0.9)  # no points
    assert b.points is not None and b.points.shape[0] == 200


def test_mark_removed_when_not_first_element():
    """list.remove() falls back to dataclass __eq__ over numpy fields and
    raises 'truth value of an array is ambiguous' unless the victim happens
    to be the FIRST belief -- removal must be by identity (crashed a live
    grasp: held state stuck at 'already holding')."""
    import numpy as np

    store = BeliefStore()
    store.update("cup", np.array([0.3, 0.1, 0.02]), 0.9)
    store.update("banana", np.array([0.2, 0.0, 0.02]), 0.9)
    assert store.mark_removed("banana", near=np.array([0.2, 0.0, 0.02]))
    assert store.find("banana") is None
    assert store.find("cup") is not None
