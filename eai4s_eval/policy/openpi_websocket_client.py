from __future__ import annotations

import logging
import time
from typing import Any, Callable, Mapping

import numpy as np

from .http_client import PolicyClientError, PolicyOutputInvalid


def _pack_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"V", "O", "c"}:
            raise TypeError(f"unsupported numpy dtype for OpenPI WebSocket: {value.dtype}")
        contiguous = np.ascontiguousarray(value)
        return {
            b"__ndarray__": True,
            b"data": contiguous.tobytes(),
            b"dtype": contiguous.dtype.str,
            b"shape": contiguous.shape,
        }
    if isinstance(value, np.generic):
        if value.dtype.kind in {"V", "O", "c"}:
            raise TypeError(f"unsupported numpy scalar dtype for OpenPI WebSocket: {value.dtype}")
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(f"cannot serialize {type(value).__name__} with OpenPI MessagePack")


def _unpack_numpy(value: dict[Any, Any]) -> Any:
    if b"__ndarray__" in value:
        return np.ndarray(
            buffer=value[b"data"],
            dtype=np.dtype(value[b"dtype"]),
            shape=tuple(value[b"shape"]),
        )
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


class OpenPIMessagePackCodec:
    """MessagePack codec compatible with OpenPI's msgpack_numpy module."""

    def __init__(self) -> None:
        try:
            import msgpack  # type: ignore
        except ModuleNotFoundError as exc:
            raise PolicyClientError(
                "OpenPI WebSocket evaluation requires msgpack; install it with: pip install msgpack"
            ) from exc
        self._msgpack = msgpack
        self._packer = msgpack.Packer(default=_pack_numpy)

    def pack(self, payload: Any) -> bytes:
        return self._packer.pack(payload)

    def unpack(self, payload: bytes | bytearray | memoryview) -> Any:
        return self._msgpack.unpackb(payload, object_hook=_unpack_numpy)


class OpenPIWebSocketPolicyClient:
    """LabBench-native synchronous client for OpenPI's WebSocket protocol."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int | None = 8000,
        api_key: str | None = None,
        timeout_sec: float = 30.0,
        retry_interval_sec: float = 1.0,
        connection_factory: Callable[..., Any] | None = None,
    ) -> None:
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be greater than zero")
        self.timeout_sec = float(timeout_sec)
        self.retry_interval_sec = max(0.0, float(retry_interval_sec))
        self.api_key = api_key
        self.uri = self._build_uri(str(host), port)
        self._codec = OpenPIMessagePackCodec()
        if connection_factory is None:
            try:
                from websockets.sync.client import connect
            except ModuleNotFoundError as exc:
                raise PolicyClientError(
                    "OpenPI WebSocket evaluation requires websockets; install it with: pip install websockets"
                ) from exc
            connection_factory = connect
        self._connection_factory = connection_factory
        self._ws: Any | None = None
        self._server_metadata: dict[str, Any] = {}
        self._connect()

    @staticmethod
    def _build_uri(host: str, port: int | None) -> str:
        uri = host if host.startswith(("ws://", "wss://")) else f"ws://{host}"
        if port is not None:
            authority = uri.split("//", 1)[1].split("/", 1)[0]
            if ":" not in authority:
                uri = f"{uri.rstrip('/')}:{int(port)}"
        return uri

    def _connect(self) -> None:
        deadline = time.monotonic() + self.timeout_sec
        headers = {"Authorization": f"Api-Key {self.api_key}"} if self.api_key else None
        last_error: Exception | None = None
        logging.info("Connecting to OpenPI WebSocket server at %s", self.uri)
        while time.monotonic() < deadline:
            remaining = max(0.001, deadline - time.monotonic())
            try:
                self._ws = self._connection_factory(
                    self.uri,
                    compression=None,
                    max_size=None,
                    open_timeout=remaining,
                    additional_headers=headers,
                )
                metadata_payload = self._ws.recv(timeout=remaining)
                if isinstance(metadata_payload, str):
                    raise PolicyClientError(f"OpenPI server rejected connection: {metadata_payload}")
                metadata = self._codec.unpack(metadata_payload)
                if not isinstance(metadata, Mapping):
                    raise PolicyOutputInvalid(
                        f"OpenPI metadata must be a mapping, got {type(metadata).__name__}"
                    )
                self._server_metadata = {str(key): value for key, value in metadata.items()}
                return
            except (PolicyClientError, PolicyOutputInvalid):
                self.close()
                raise
            except Exception as exc:
                last_error = exc
                self.close()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(self.retry_interval_sec, remaining))
        raise PolicyClientError(
            f"timed out connecting to OpenPI WebSocket server {self.uri} after {self.timeout_sec:.1f}s: {last_error}"
        ) from last_error

    def get_server_metadata(self) -> dict[str, Any]:
        return dict(self._server_metadata)

    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        if self._ws is None:
            raise PolicyClientError("OpenPI WebSocket connection is closed")
        try:
            self._ws.send(self._codec.pack(dict(observation)))
            payload = self._ws.recv(timeout=self.timeout_sec)
        except Exception as exc:
            self.close()
            raise PolicyClientError(
                f"OpenPI WebSocket inference at {self.uri} failed or timed out after {self.timeout_sec:.1f}s: {exc}"
            ) from exc
        if isinstance(payload, str):
            self.close()
            raise PolicyClientError(f"OpenPI inference server returned an error:\n{payload}")
        result = self._codec.unpack(payload)
        if not isinstance(result, Mapping):
            self.close()
            raise PolicyOutputInvalid(
                f"OpenPI WebSocket response must be a mapping, got {type(result).__name__}"
            )
        return dict(result)

    def reset(self) -> None:
        # OpenPI's current protocol has no reset message.
        return None

    def close(self) -> None:
        connection, self._ws = self._ws, None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                logging.debug("Error while closing OpenPI WebSocket connection", exc_info=True)


__all__ = ["OpenPIMessagePackCodec", "OpenPIWebSocketPolicyClient"]
