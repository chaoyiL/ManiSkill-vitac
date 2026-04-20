from __future__ import annotations

import time
from typing import Any

from openpi_client import msgpack_numpy
from websockets.exceptions import InvalidStatus
from websockets.sync.client import ClientConnection, connect


class InterfaceClient:
    """Persistent websocket client for remote robot interaction."""

    def __init__(self, ip: str = "127.0.0.1", port: str | int = "8000", token: str | None = None):
        self.robot_ip = ip
        self.robot_port = int(port)
        self._uri = f"ws://{self.robot_ip}:{self.robot_port}"
        self._token = None if token is None else str(token)
        self._packer = msgpack_numpy.Packer()
        self._ws = self._connect()
        self._expect_hello()

    def _connect(self) -> ClientConnection:
        while True:
            try:
                return connect(
                    self._uri,
                    additional_headers=self._build_headers(),
                    compression=None,
                    max_size=None,
                    # This connection carries large binary payloads and can pause
                    # on human input or blocking inference, so keepalive causes
                    # false positives with sync websockets.
                    ping_interval=None,
                )
            except OSError:
                time.sleep(1.0)
            except InvalidStatus as exc:
                raise RuntimeError(
                    f"Robot websocket handshake rejected with status {exc.response.status_code}. "
                    "Check the client token."
                ) from exc

    def _build_headers(self) -> dict[str, str] | None:
        if self._token is None:
            return None
        return {"Authorization": f"Bearer {self._token}"}

    def _expect_hello(self) -> None:
        message = self._recv_message(timeout=10.0)
        if message.get("type") != "hello":
            raise RuntimeError(f"Unexpected initial robot bridge message: {message}")

    def _send_message(self, message: dict[str, Any]) -> None:
        self._ws.send(self._packer.pack(message))

    def _recv_message(self, timeout: float | None = None) -> dict[str, Any]:
        raw_message = self._ws.recv(timeout=timeout)
        if isinstance(raw_message, str):
            raise RuntimeError("Robot bridge expects binary websocket frames.")
        return msgpack_numpy.unpackb(raw_message)

    def send_config(self, config: dict[str, Any]) -> None:
        self._send_message(
            {
                "type": "config",
                "config": config,
            }
        )

    def send_state(self, state: str) -> None:
        self._send_message(
            {
                "type": "state",
                "state": state,
            }
        )

    def recv_obs(self, timeout: float | None = None) -> tuple[int, Any]:
        message = self._recv_message(timeout=timeout)
        if message.get("type") != "obs":
            raise RuntimeError(f"Unexpected robot bridge message while waiting for obs: {message}")
        return int(message["obs_seq"]), message["obs"]

    def send_action(self, action: Any, obs_seq: int) -> None:
        self._send_message(
            {
                "type": "action",
                "obs_seq": int(obs_seq),
                "action": action,
            }
        )

    def close(self) -> None:
        self._ws.close()
