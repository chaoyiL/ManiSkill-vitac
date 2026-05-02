from __future__ import annotations

import functools
import ipaddress
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import msgpack
import numpy as np
from websockets.exceptions import InvalidStatus
from websockets.sync.client import ClientConnection, connect


_TUNNEL_HOST_SUFFIXES = (
    "ngrok-free.dev",
    "ngrok-free.app",
    "ngrok.app",
    "ngrok.io",
    "trycloudflare.com",
    "loca.lt",
    "localtunnel.me",
    "serveo.net",
    "localhost.run",
)


def _pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def _unpack_array(obj: dict[bytes, Any]) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


_Packer = functools.partial(msgpack.Packer, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


def _is_local_address(host: str | None) -> bool:
    if host is None:
        return False

    normalized_host = host.rstrip(".").lower()
    if normalized_host == "localhost" or normalized_host.endswith(".local"):
        return True

    try:
        address = ipaddress.ip_address(normalized_host)
    except ValueError:
        return False

    return address.is_loopback or address.is_private or address.is_link_local


def _is_tunnel_host(host: str | None) -> bool:
    if host is None:
        return False

    normalized_host = host.rstrip(".").lower()
    return any(
        normalized_host == suffix or normalized_host.endswith(f".{suffix}") for suffix in _TUNNEL_HOST_SUFFIXES
    )


def _to_websocket_scheme(scheme: str, host: str | None) -> str:
    if scheme in ("ws", "wss"):
        return scheme
    if scheme == "http":
        return "ws"
    if scheme == "https":
        return "wss"
    if scheme == "":
        return "wss" if _is_tunnel_host(host) else "ws"

    raise ValueError(f"Unsupported websocket address scheme: {scheme!r}")


def _build_ws_uri(address: str, port: str | int = "8000", add_port: bool | None = None) -> str:
    raw_address = str(address).strip()
    if not raw_address:
        raise ValueError("Robot websocket address must not be empty.")

    has_scheme = "://" in raw_address
    parsed = urlsplit(raw_address if has_scheme else f"//{raw_address}")
    host = parsed.hostname
    if host is None:
        raise ValueError(f"Invalid robot websocket address: {address!r}")

    # Accessing .port validates the user-provided port and returns None if no
    # port was specified in the address itself.
    specified_port = parsed.port
    scheme = _to_websocket_scheme(parsed.scheme, host)

    if add_port is None:
        should_add_port = specified_port is None and not _is_tunnel_host(host)
        if has_scheme and not _is_local_address(host):
            should_add_port = False
    else:
        should_add_port = bool(add_port) and specified_port is None

    netloc = parsed.netloc
    if should_add_port:
        netloc = f"{netloc}:{int(port)}"

    return urlunsplit((scheme, netloc, parsed.path, parsed.query, parsed.fragment))


class InterfaceClient:
    """Persistent websocket client for remote robot interaction."""

    def __init__(
        self,
        ip: str = "127.0.0.1",
        port: str | int = "8000",
        token: str | None = None,
        add_port: bool | None = None,
    ):
        self.robot_ip = ip
        self.robot_port = int(port)
        self._uri = _build_ws_uri(self.robot_ip, self.robot_port, add_port=add_port)
        self._token = None if token is None else str(token)
        self._packer = _Packer()
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
            except OSError as exc:
                print(f"[client] connect to {self._uri} failed: {exc!r}")
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
        return _unpackb(raw_message)

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
