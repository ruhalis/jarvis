"""State machine for the voice loop.

States:
    IDLE       — waiting for wake word
    LISTENING  — wake heard, capturing speech
    PROCESSING — running router / LLM / tools
    SPEAKING   — TTS is playing audio

Transitions are published on `state_change` so the dashboard and any other
service can react. This module is intentionally small: it's a holder +
publisher, not a policy engine. Callers decide when to transition.
"""
from __future__ import annotations

import enum
import logging

import jarvis_bus as bus

log = logging.getLogger(__name__)


class State(str, enum.Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"


# Allowed transitions. Keep this conservative — anything missing will be
# rejected (and logged), which makes bugs loud instead of quiet.
_ALLOWED: dict[State, set[State]] = {
    State.IDLE: {State.LISTENING, State.PROCESSING},
    State.LISTENING: {State.PROCESSING, State.IDLE},
    State.PROCESSING: {State.SPEAKING, State.IDLE},
    State.SPEAKING: {State.IDLE, State.LISTENING, State.PROCESSING},  # barge-in
}


class StateMachine:
    def __init__(self, client, initial: State = State.IDLE) -> None:
        self._client = client
        self._state = initial

    @property
    def state(self) -> State:
        return self._state

    def can(self, target: State) -> bool:
        return target in _ALLOWED.get(self._state, set())

    async def transition(self, target: State, *, force: bool = False) -> bool:
        if self._state is target:
            return True
        if not force and not self.can(target):
            log.warning("rejected transition %s -> %s", self._state, target)
            return False
        prev, self._state = self._state, target
        await bus.publish(
            self._client,
            bus.CH_STATE,
            {"from": prev.value, "to": target.value},
        )
        log.info("state %s -> %s", prev.value, target.value)
        return True
