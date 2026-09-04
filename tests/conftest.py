"""Shared offline test safeguards."""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail a test that attempts to open a network connection."""

    def denied(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Tests must not access the network")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)
    yield
