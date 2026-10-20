"""The fast command path: reflex grammar + experience memory.

Human behavior is tiered: reflexes fire in milliseconds, habits replay
learned routines, and deliberation is reserved for the genuinely novel. The
Anthropic robotics study measured why this matters for LLM robots -- one
LLM turn costs 2-15 s, so anything routine must NOT wait on the model.

Tier 1 (reflex):   a template grammar compiles routine commands ("pick and
                   place pink object", "put the cube in the bowl") straight
                   into deterministic skill calls, in microseconds.
Tier 2 (habit):    Agentic-VLA-style experience memory -- successful plans
                   indexed by a hashed bag-of-words embedding of the
                   instruction, retrieved by cosine similarity and replayed.
Tier 3 (deliberation): the full LLM tool-calling loop (orchestrator), which
                   remains the fallback for novel or failed commands.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from zlib import crc32

import numpy as np

from ..memory.vector_index import QuantizedIndex
from ..perception.object_names import canonical_destination, canonical_object_name

SkillCall = tuple[str, dict]

_POLITENESS = re.compile(
    r"^(?:please|hey|ok|now|hermes|robot|can you|could you|would you|"
    r"por favor|puedes|podrias|oye|现在)[\s,]+",
    re.IGNORECASE,
)

_PICK = r"(?:pick|grab|take|fetch|coge|agarra|toma|recoge)"
_PLACE = r"(?:place|put|drop|set|save|store|pon|coloca|deja|guarda|mete)"
_PREP = r"(?:in|into|inside|on|onto|to|at|en|dentro de|sobre|a)"
_ART = r"(?:the\s+|a\s+|an\s+|el\s+|la\s+|un\s+|una\s+)?"

def _zh_object_name(text: str | None) -> str | None:
    """Map spoken Chinese names to the detector vocabulary."""
    return canonical_object_name(text)

_RULES: list[tuple[re.Pattern, str]] = [
    # "place it in the bowl" (destination for the already-held object);
    # must outrank the generic place rule or the pronoun becomes the object
    (
        re.compile(rf"^{_PLACE}\s+(?:it|this|that|lo|la|eso)\s+{_PREP}\s+{_ART}(?P<dest>.+)$"),
        "place_held",
    ),
    # "pick up the X and put it in the Y" / "coge la X y ponla en la Y"
    (
        re.compile(
            rf"^{_PICK}(?:\s+up)?\s+{_ART}(?P<obj>.+?)\s+(?:and|y)\s+{_PLACE}"
            rf"(?:\s*(?:it|lo|la))?\s+{_PREP}\s+{_ART}(?P<dest>.+)$"
        ),
        "pick_and_place",
    ),
    # "put/place the pink cube in the bowl" / "pon el cubo rosa en el bol"
    (
        re.compile(
            rf"^{_PLACE}\s+{_ART}(?P<obj>.+?)\s+{_PREP}\s+{_ART}(?P<dest>.+)$"
        ),
        "pick_and_place",
    ),
    # "grab the banana and throw it" / "coge la banana y lánzala" /
    # "throw the cube (to the left)" / "lanza el cubo (a la izquierda)"
    # MUST precede the generic "pick" rule, whose lazy object group would
    # otherwise swallow "banana and throw it" as the thing to grasp.
    (
        re.compile(
            rf"^(?:{_PICK}(?:\s+up)?\s+{_ART}(?P<obj>.+?)\s+(?:and|y)\s+)?"
            r"(?:throw|toss|launch|l[aá]nza|tira|arroja)"
            r"(?:la|lo|las|los)?"      # enclitic pronoun: lánzala / tíralo
            r"(?:\s+(?:it|lo|la))?"
            rf"(?:\s+{_ART}(?P<obj2>.+?))?"
            r"(?:\s+(?:to\s+the\s+|a\s+la\s+|hacia\s+)?"
            r"(?P<dir>left|right|forward|back|izquierda|derecha|delante|detras|detrás))?$"
        ),
        None,  # throw deleted 2026-08-13
    ),
    # "pick (up) (and place/put) X (in/on Y)" / "coge y coloca el objeto rosa"
    (
        re.compile(
            rf"^{_PICK}(?:\s+up)?\s+(?:(?P<andplace>(?:and|y)\s+{_PLACE}(?:\s+it)?)\s+)?"
            rf"{_ART}(?P<obj>.+?)"
            rf"(?:\s+{_PREP}\s+{_ART}(?P<dest>.+?))?$"
        ),
        "pick",
    ),
    # "push the box left"
    (
        re.compile(
            rf"^(?:push|empuja)\s+{_ART}(?P<obj>.+?)\s+(?:to\s+the\s+)?"
            r"(?P<dir>left|right|forward|back)(?:wards?)?$"
        ),
        None,  # push deleted 2026-08-13
    ),
    # "stack the red cube on the blue box" == pick A, place on B
    (
        re.compile(rf"^(?:stack|apila)\s+{_ART}(?P<obj>.+?)\s+(?:on(?:\s+top\s+of)?|sobre|en)\s+{_ART}(?P<dest>.+)$"),
        "pick_and_place",
    ),
    # "hand me / give me the pink object" -> handover
    (
        re.compile(rf"^(?:hand|give|pass|bring)\s+(?:me\s+|us\s+)?{_ART}(?P<obj>.+?)$|^(?:dame|pasame|traeme)\s+{_ART}(?P<obj2>.+?)$"),
        None,  # handover deleted 2026-08-13
    ),
    # "point at/to the pink object" / "señala el objeto rosa"
    (
        re.compile(rf"^(?:point\s+(?:at|to)|show\s+me|senala|señala)\s+{_ART}(?P<obj>.+)$"),
        None,  # point_at deleted 2026-08-13
    ),
(re.compile(r"^(?:wave|say\s+hi|say\s+hello|saluda)(?:\s+(?:at|to)\s+\S.*)?$"), None),  # wave re-implemented 2026-09-03 (see skill_wave in runtime.py)
    (
        # "go home" / "park the arm" / "回到 home" / "请到初始的原点位置" / "ve a casa"
        # — single move_home() call, never goes through LLM (cheap, deterministic).
        # The downstream switch case `intent == "home"` builds
        # ReflexPlan([("move_home", {})]).
        re.compile(
            rf"^(?:go|park|return|move)\s+(?:to\s+)?(?:home|the\s+home\s*pose|the\s+home\s*position)"
            rf"(?:\s+(?:please|now))?$"
            rf"|^(?:请)?\s*(?:回)?(?:到|去)\s*(?:初始|home|原点|原位)"
            rf"(?:的\s*)?(?:原点|原位|初始)?(?:\s*位置|\s*pose|\s*那个|\s*位置)?"
            rf"\s*[。.!]?$"
            rf"|^(?:回(?:到|去)?\s*(?:家|原位|原点|home)|回家|回原位|归位|复位)"
            rf"(?:的\s*)?(?:原点|原位|初始)?(?:\s*位置|\s*pose|\s*那个)?"
            rf"\s*[。.!]?$"
            rf"|^(?:ve|regresa|vuelve)\s+(?:a\s+)?(?:casa|inicio|origen)\s*[.!]?$",
            re.IGNORECASE,
        ),
        "home",
    ),
    (
        re.compile(r"^(?:sort|organize|group|ordena|organiza)\s+.*(?:color|colour)e?s?.*$"),
        None,  # sort_by_color deleted 2026-08-13
    ),
    (
        re.compile(rf"^(?:how\s+many|count|cuantos|cuantas)\s+{_ART}(?P<obj>.+?)"
                   r"(?:\s+(?:do\s+you\s+see|are\s+there|hay))?\??$"),
        None,  # count deleted 2026-08-13
    ),
    (
        re.compile(rf"^move\s+(?:a\s+(?:bit|little)\s+|slightly\s+)?"
                   r"(?P<dir>forward|back|left|right|up|down)(?:\s+a\s+(?:bit|little))?$"),
        None,  # move_relative deleted 2026-08-13
    ),
    # Home pose: operational ready position [0.25, 0, 0.35, 0, 0, 0]
    (re.compile(r"^(?:go\s+home|move\s+home|home|park(?:\s+the\s+arm)?|back\s+to\s+home|return\s+to\s+home|ready\s+(?:pose|position))$"), "home"),
    # Mechanical zero: all joints to [0, 0, 0, 0, 0, 0]
    (re.compile(r"^(?:zero\s+joints?|joints?\s+to\s+zero|all\s+joints?\s+zero|mechanical\s+zero|joint\s+zero|zero\s+position)$"), None),  # move_to_joint_zero deleted 2026-08-13
    (
        re.compile(
            r"^(?:look(?:\s+around)?|observe|scan|"
            r"mira|observa)$"
        ),
        "look",
    ),
    # scene questions answer instantly from the world model
    (
        re.compile(
            r"^(?:what\s+do\s+you\s+see\??|what(?:'s|\s+is)\s+on\s+the\s+table\??|"
            r"describe\s+the\s+(?:scene|table)|que\s+ves\??|describe\s+la\s+mesa)$"
        ),
        None,  # describe (describe_scene) deleted 2026-08-13
    ),
    (re.compile(r"^(?:open\s+(?:the\s+)?gripper|let\s+go|release|abre\s+la\s+pinza|suelta(?:lo|la)?)$"), "open_gripper"),
    (
        re.compile(
            r"^(?:close\s+(?:the\s+)?gripper|shut\s+(?:the\s+)?gripper)$"
        ),
        "close_gripper",
    ),
    (re.compile(r"^(?:stop|para|alto)$"), None),  # never a reflex: stop is for e-stop tools
]


@dataclass
class ReflexPlan:
    intent: str
    calls: list[SkillCall]
    object_query: str | None = None
    destination: str | None = None


def parse_command(text: str) -> ReflexPlan | None:
    """Compile a routine command into skill calls; None -> not routine."""
    cmd = text.strip().lower()
    cmd = re.sub(r"[.!?¡¿,，。；;、]+", "", cmd)
    for _ in range(3):  # peel stacked politeness ("hey, can you please ...")
        stripped = _POLITENESS.sub("", cmd).strip()
        if stripped == cmd:
            break
        cmd = stripped
    cmd = re.sub(r"\s+", " ", cmd)
    if not cmd:
        return None

    # Collective commands require a pending-object list; treating the whole
    # phrase as one object causes a guaranteed localization failure.
    if (
        re.search(r"(?:所有|全部|每个).*?(?:水果|果物)", cmd)
        or re.search(r"\b(?:all|every)\s+(?:the\s+)?fruits?\b", cmd)
    ):
        return None

    zh_hand = {
        "打开手": "open", "张开手": "open", "打开灵巧手": "open", "张开灵巧手": "open",
        "五指收缩": "close", "握拳": "close",
        "食指捏合": "pinch", "拇指和食指捏合": "pinch", "二指捏合": "pinch",
    }.get(cmd)
    if zh_hand:
        return ReflexPlan("hand_action", [("hand_action", {"action": zh_hand})])

    en_hand = {
        "open hand": "open", "open the hand": "open",
        "open dexterous hand": "open", "open the dexterous hand": "open",
        "close hand": "close", "close the hand": "close",
        "make a fist": "close", "power grasp": "power_grasp",
        "pinch": "pinch", "two finger pinch": "pinch",
        "pinch with thumb and index finger": "pinch",
    }.get(cmd)
    if en_hand:
        return ReflexPlan("hand_action", [("hand_action", {"action": en_hand})])

    # Deterministic Chinese pick/place path. Translate the experiment names
    # to YOLOE English prompts instead of sending routine motion to the LLM.
    # Compound guard: a SECOND clause ("…，然后拿起香蕉") is not part of the
    # destination — executing it as one pick_and_place wastes a localize
    # cycle on a garbage destination before the LLM tier takes over anyway.
    # The English "and then" guard below cannot match CJK (\b never fires
    # between word-char Han codepoints), so the check lives here.
    def _zh_compound(m) -> bool:
        return bool(re.search(r"然后|接着|之后", (m.group("obj") or "") + (m.group("dest") or "")))

    zh = re.match(
        r"^(?:现在)?(?:请)?(?:拿取|拿起|抓取|抓起|捡起)"
        r"(?P<obj>.+?)(?:并|然后)?(?:把?它)?"
        r"(?:放置|放下|放到|放在|放入|放进)(?P<dest>.*?)$", cmd,
    )
    if zh:
        if _zh_compound(zh):
            return None
        obj = _zh_object_name(zh.group("obj"))
        dest = canonical_destination(zh.group("dest"))
        if obj:
            args = {"object": obj}
            if dest:
                args["destination"] = dest
            return ReflexPlan("pick_and_place", [("pick_and_place", args)], obj, dest)
    zh = re.match(
        r"^(?:现在)?(?:请)?(?:把|将)(?P<obj>.+?)"
        r"(?:放置|放下|放到|放在|放入|放进)(?P<dest>.+?)$", cmd,
    )
    if zh:
        if _zh_compound(zh):
            return None
        obj = _zh_object_name(zh.group("obj"))
        dest = canonical_destination(zh.group("dest"))
        if obj and dest:
            args = {"object": obj, "destination": dest}
            return ReflexPlan("pick_and_place", [("pick_and_place", args)], obj, dest)
    # Compound/sequenced commands are NOT reflexes: executing only the first
    # clause and reporting success silently drops the rest. Send them to the
    # LLM tier, which can plan multi-step ("wave and then pick up the cube").
    if re.search(r"\b(?:and then|then|after that|luego|despues|después)\b", cmd):
        return None

    for rule, intent in _RULES:
        m = rule.match(cmd)
        if not m:
            continue
        if intent is None:
            return None
        g = m.groupdict()
        obj = (g.get("obj") or "").strip() or None
        dest = (g.get("dest") or "").strip() or None
        if obj:
            # "pick and place the banana and save it in the box": the lazy
            # object group of the pick..and..place rule swallows a leading
            # place-verb clause -- strip it, or the robot spends two minutes
            # trying to localize 'and place the banana'.
            obj = re.sub(
                rf"^(?:and|y)\s+{_PLACE}(?:\s+(?:it|lo|la))?\s+{_ART}", "", obj
            ).strip() or None
            if obj is None:
                continue
        if intent == "pick":
            # "pick X" alone is a grasp; a place-verb or destination makes it
            # a full pick-and-place.
            if g.get("andplace") or dest:
                intent = "pick_and_place"
        if intent == "pick_and_place":
            if not obj:
                return None
            args = {"object": obj}
            if dest:
                args["destination"] = dest
            return ReflexPlan(intent, [("pick_and_place", args)], obj, dest)
        if intent == "pick":
            return ReflexPlan(intent, [("grasp_object", {"label": obj})], obj)
        if intent == "place_held":
            if not dest:
                return None
            return ReflexPlan(intent, [("place_on_object", {"label": dest})], None, dest)
        if intent == "push":
            return ReflexPlan(
                intent,
                [("push_object", {"label": obj, "direction": g["dir"]})],
                obj,
            )
        if intent == "throw":
            tobj = obj or (g.get("obj2") or "").strip() or None
            _dirmap = {
                "izquierda": "left", "derecha": "right",
                "delante": "forward", "detras": "back", "detrás": "back",
            }
            raw_dir = (g.get("dir") or "").strip()
            tdir = _dirmap.get(raw_dir, raw_dir) or "forward"
            args = {"direction": tdir}
            if tobj:
                args["label"] = tobj
            return ReflexPlan(intent, [("throw", args)], tobj)
        if intent == "handover":
            obj = obj or (g.get("obj2") or "").strip() or None
            if not obj:
                return None
            return ReflexPlan(intent, [("handover", {"label": obj})], obj)
        if intent == "point_at":
            return ReflexPlan(intent, [("point_at", {"label": obj})], obj)
        if intent == "wave":
            return ReflexPlan(intent, [("wave", {})])
        if intent == "sort_by_color":
            return ReflexPlan(intent, [("sort_by_color", {})])
        if intent == "count":
            return ReflexPlan(intent, [("count_objects", {"query": obj} if obj else {})], obj)
        if intent == "describe":
            return ReflexPlan(intent, [("describe_scene", {})])
        if intent == "move_relative":
            return ReflexPlan(intent, [("move_relative", {"direction": g["dir"]})])
        if intent == "home":
            return ReflexPlan(intent, [("move_home", {})])
        if intent == "zero_joints":
            return ReflexPlan(intent, [("move_to_joint_zero", {})])
        if intent == "look":
            return ReflexPlan(intent, [("get_observation", {})])
        if intent == "open_gripper":
            return ReflexPlan(intent, [("open_gripper", {})])
        if intent == "close_gripper":
            return ReflexPlan(intent, [("close_gripper", {})])
    return None


# ── tier 2: experience memory (Agentic-VLA EM, inference-time) ───────────


_STOPWORDS = frozenset(
    "the a an it this that those these to of and or please can you could my me "
    "el la los las un una y o por favor lo esa ese esto".split()
)


def _embed(text: str, dim: int = 256) -> np.ndarray:
    """Deterministic hashed bag-of-words embedding over content words (no
    model, no network); stopwords out so "hand me the X" == "hand me that X"."""
    v = np.zeros(dim, dtype=np.float64)
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        if tok in _STOPWORDS:
            continue
        h = crc32(tok.encode())
        v[h % dim] += 1.0 if (h >> 16) & 1 else -1.0
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class ExperienceMemory:
    """Instruction -> proven skill plan, retrieved by embedding similarity.

    Successful plans are stored with win/loss counts and persisted as JSON so
    habits survive process restarts (the demo learns across sessions).
    """

    def __init__(self, path: Path | None = None, dim: int = 256):
        self._dim = dim
        self._path = path
        self._index = QuantizedIndex(dim=dim, bits=4)
        self._records: list[dict] = []
        self._lock = threading.Lock()
        if path is not None and path.exists():
            try:
                self._records = json.loads(path.read_text())
                for i, rec in enumerate(self._records):
                    self._index.add(_embed(rec["task"], dim), meta=i)
            except Exception:
                self._records = []

    def __len__(self) -> int:
        return len(self._records)

    def recall(self, task: str, min_sim: float = 0.9) -> dict | None:
        """Best stored plan for a similar instruction, if it mostly worked."""
        with self._lock:
            for sim, i in self._index.search(_embed(task, self._dim), k=3):
                if sim < min_sim:
                    break
                rec = self._records[i]
                if rec["wins"] > rec["losses"]:
                    return {"sim": round(sim, 3), **rec}
        return None

    def record(self, task: str, calls: list[SkillCall], success: bool, duration_s: float) -> None:
        snapshot = None
        with self._lock:
            for rec in self._records:
                if rec["task"] == task.strip().lower():
                    rec["wins" if success else "losses"] += 1
                    n = rec["wins"] + rec["losses"]
                    rec["avg_s"] = round(rec["avg_s"] + (duration_s - rec["avg_s"]) / n, 2)
                    if success:
                        rec["calls"] = [[c, dict(a)] for c, a in calls]
                    snapshot = json.dumps(self._records, indent=1)
                    break
            else:
                if not success:
                    return  # only remember plans that have worked at least once
                self._records.append(
                    {
                        "task": task.strip().lower(),
                        "calls": [[c, dict(a)] for c, a in calls],
                        "wins": 1,
                        "losses": 0,
                        "avg_s": round(duration_s, 2),
                    }
                )
                self._index.add(_embed(task, self._dim), meta=len(self._records) - 1)
                snapshot = json.dumps(self._records, indent=1)
        # File I/O happens OUTSIDE the lock: recall() on the command hot
        # path must never wait on a disk write.
        if snapshot is not None:
            self._save(snapshot)

    def _save(self, snapshot: str) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(snapshot)
        except OSError:
            pass

    def stats(self) -> dict:
        return {"plans": len(self._records)}


@dataclass
class FastPlan:
    source: str  # "reflex" | "experience"
    calls: list[SkillCall]
    detail: str = ""


class FastPlanner:
    """Tier 1 + tier 2 resolver; returns None when only the LLM will do."""

    def __init__(self, experience: ExperienceMemory | None = None):
        self.experience = experience

    def plan(self, task: str) -> FastPlan | None:
        reflex = parse_command(task)
        if reflex is not None:
            return FastPlan("reflex", reflex.calls, reflex.intent)
        if self.experience is not None:
            rec = self.experience.recall(task)
            if rec is not None:
                calls = [(c, dict(a)) for c, a in rec["calls"]]
                return FastPlan(
                    "experience",
                    calls,
                    f"~{rec['task']!r} (sim {rec['sim']}, {rec['wins']}w/{rec['losses']}l)",
                )
        return None

    def note_outcome(self, task: str, calls: list[SkillCall], success: bool, duration_s: float) -> None:
        if self.experience is not None:
            self.experience.record(task, calls, success, duration_s)
