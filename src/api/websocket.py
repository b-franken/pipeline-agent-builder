"""WebSocket connection manager for real-time agent updates."""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Final

from fastapi import WebSocket

logger: Final = logging.getLogger(__name__)

MAX_QUEUE_SIZE: Final = 100
SEND_TIMEOUT_SECONDS: Final = 5.0


@dataclass
class ClientConnection:
    """Represents a single WebSocket client with its message queue."""

    websocket: WebSocket
    queue: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=MAX_QUEUE_SIZE))
    dropped_count: int = 0


@dataclass
class ConnectionManager:
    """Manages WebSocket connections for real-time updates with backpressure handling."""

    active_connections: dict[str, ClientConnection] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections[client_id] = ClientConnection(websocket=websocket)

    async def disconnect(self, client_id: str) -> None:
        async with self._lock:
            conn = self.active_connections.pop(client_id, None)
            if conn and conn.dropped_count > 0:
                logger.warning(
                    "Client %s disconnected, dropped %d messages due to backpressure",
                    client_id,
                    conn.dropped_count,
                )

    async def send_json(self, client_id: str, data: dict[str, Any]) -> None:
        async with self._lock:
            conn = self.active_connections.get(client_id)
        if conn:
            try:
                await asyncio.wait_for(conn.websocket.send_json(data), timeout=SEND_TIMEOUT_SECONDS)
            except TimeoutError:
                logger.warning("Send timeout for client %s", client_id)
                await self.disconnect(client_id)
            except Exception:
                await self.disconnect(client_id)

    async def broadcast(self, data: dict[str, Any]) -> None:
        async with self._lock:
            connections = list(self.active_connections.items())

        tasks = []
        for client_id, conn in connections:
            tasks.append(self._send_with_backpressure(client_id, conn, data))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_with_backpressure(
        self,
        client_id: str,
        conn: ClientConnection,
        data: dict[str, Any],
    ) -> None:
        try:
            await asyncio.wait_for(conn.websocket.send_json(data), timeout=SEND_TIMEOUT_SECONDS)
        except TimeoutError:
            if len(conn.queue) >= MAX_QUEUE_SIZE:
                conn.dropped_count += 1
            else:
                conn.queue.append(data)
            logger.debug("Client %s: queued message due to slow send", client_id)
        except Exception:
            await self.disconnect(client_id)
