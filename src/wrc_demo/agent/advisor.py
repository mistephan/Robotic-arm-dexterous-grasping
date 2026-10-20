"""VLM grasp advisor, consulted after failures.

Agentic-VLA's suggestion-frequency idea turned inference-time as a plain
threshold: consult once consecutive failures reach ``min_failures`` (default
1, i.e. every failure while the LLM supports vision); a success resets the
counter. The one-sentence suggestion is appended to the agent's context,
mirroring the paper's prompt-augmentation trick.
"""

from __future__ import annotations

from .llm import LLMClient
from .prompts import ADVISOR_SYSTEM, ADVISOR_USER, MATERIAL_USER


class Advisor:
    def __init__(self, llm: LLMClient, min_failures: int = 1):
        self._llm = llm
        self._min_failures = min_failures
        self.failures = 0

    def note_outcome(self, success: bool) -> None:
        self.failures = 0 if success else self.failures + 1

    @property
    def should_consult(self) -> bool:
        return self._llm.supports_vision and self.failures >= self._min_failures

    def suggest(self, task: str, frame_jpeg: bytes, failure: str = "") -> str:
        failure_context = (
            f"The previous attempt failed: {failure}\n" if failure else ""
        )
        resp = self._llm.chat(
            system=ADVISOR_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": ADVISOR_USER.format(task=task, failure_context=failure_context),
                    "images": [frame_jpeg],
                }
            ],
            max_tokens=100,
        )
        return resp.text.strip()

    def classify_material(self, label: str, crop_jpeg: bytes) -> str | None:
        if not self._llm.supports_vision:
            return None
        resp = self._llm.chat(
            system=ADVISOR_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": MATERIAL_USER.format(label=label),
                    "images": [crop_jpeg],
                }
            ],
            max_tokens=10,
        )
        word = resp.text.strip().split()[0].lower().strip(".,") if resp.text.strip() else None
        return word
