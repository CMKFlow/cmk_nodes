"""Minimal ExecutionServer replacement that records every execution event."""

from __future__ import annotations

from aiohttp import web


class _FakeApp:
    def __init__(self):
        self.middlewares = []


class FakeServer:
    """Stands in for ``server.PromptServer.instance`` and the executor server.

    It only records messages; it never alters what the executor does.
    """

    def __init__(self):
        self.client_id = None
        self.last_node_id = None
        self.last_prompt_id = None
        self.sockets_metadata = {}
        self.routes = web.RouteTableDef()
        self.app = _FakeApp()
        self.prompt_queue = None
        self.events: list[tuple[str, object]] = []

    def send_sync(self, event, data, sid=None):
        self.events.append((event, data))

    def queue_updated(self):
        pass

    def clear(self):
        self.events.clear()

    def of(self, event):
        return [data for name, data in self.events if name == event]
