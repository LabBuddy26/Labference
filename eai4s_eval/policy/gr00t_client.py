from __future__ import annotations

from io import BytesIO
from typing import Any, Mapping


def _json_safe_metadata(value: Any) -> Any:
    """Convert GR00T/Pydantic modality metadata to manifest-safe values."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_metadata(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_safe_metadata(model_dump())
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_safe_metadata(tolist())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class _TorchZmqPolicyClient:
    """Small client for the Torch-over-ZMQ protocol used by GR00T's server."""

    portable_modality_metadata = False

    def __init__(
        self,
        *,
        host: str,
        port: int,
        timeout_ms: int,
        api_token: str | None,
    ) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ImportError("GR00T ZMQ evaluation requires PyTorch") from exc
        try:
            import zmq
        except ModuleNotFoundError as exc:
            raise ImportError(
                "GR00T ZMQ evaluation requires pyzmq; install it in the Isaac Sim "
                "environment with: python -m pip install pyzmq"
            ) from exc

        self._torch = torch
        self._zmq = zmq
        self.api_token = api_token
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        self.socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{host}:{int(port)}")

    def _to_bytes(self, payload: dict[str, Any]) -> bytes:
        buffer = BytesIO()
        self._torch.save(payload, buffer)
        return buffer.getvalue()

    def _from_bytes(self, payload: bytes) -> Any:
        return self._torch.load(BytesIO(payload), weights_only=False)

    def call_endpoint(
        self,
        endpoint: str,
        data: Mapping[str, Any] | None = None,
        *,
        requires_input: bool = True,
    ) -> Any:
        request: dict[str, Any] = {"endpoint": str(endpoint)}
        if requires_input:
            request["data"] = dict(data or {})
        if self.api_token:
            request["api_token"] = self.api_token
        try:
            self.socket.send(self._to_bytes(request))
            response = self._from_bytes(self.socket.recv())
        except self._zmq.Again as exc:
            raise TimeoutError(
                f"GR00T ZMQ endpoint {endpoint!r} timed out"
            ) from exc
        if isinstance(response, Mapping) and response.get("error"):
            raise RuntimeError(f"GR00T server error: {response['error']}")
        return response

    def get_action(self, observation: Mapping[str, Any]) -> Any:
        return self.call_endpoint("get_action", observation)

    def get_modality_config(self) -> Any:
        return self.call_endpoint("get_modality_config", requires_input=False)

    def ping(self) -> bool:
        response = self.call_endpoint("ping", requires_input=False)
        return isinstance(response, Mapping) and response.get("status") == "ok"

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


class Gr00tPolicyClient:
    """Adapter around Isaac-GR00T's ZeroMQ inference client."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        *,
        timeout_ms: int = 30_000,
        api_token: str | None = None,
        client: Any | None = None,
    ) -> None:
        if client is None:
            client = _TorchZmqPolicyClient(
                host=str(host),
                port=int(port),
                timeout_ms=int(timeout_ms),
                api_token=api_token,
            )
        self.client = client
        self.timeout_ms = int(timeout_ms)
        self._configure_socket()

    def _configure_socket(self) -> None:
        """Apply bounded request timeouts to GR00T's lazily-created REQ socket."""

        ensure_socket = getattr(self.client, "_ensure_socket", None)
        if not callable(ensure_socket):
            return
        ensure_socket()
        socket = getattr(self.client, "socket", None)
        zmq = getattr(self.client, "_zmq", None)
        if socket is None or zmq is None:
            return
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.LINGER, 0)

    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        response = self.client.get_action(dict(observation))
        if isinstance(response, tuple) and len(response) == 2:
            action, info = response
        else:
            action, info = response, {}
        if not isinstance(action, Mapping):
            raise TypeError(f"GR00T action must be a mapping, got {type(action).__name__}")
        result = dict(action)
        if isinstance(info, Mapping):
            result.setdefault("server_info", dict(info))
            latency = info.get("model_latency_ms", info.get("latency_ms"))
            if latency is not None:
                result.setdefault("model_latency_ms", latency)
        return result

    def get_server_metadata(self) -> dict[str, Any]:
        if getattr(self.client, "portable_modality_metadata", True) is False:
            # The upstream server serializes Pydantic ModalityConfig instances
            # directly. Deserializing those objects would force the Isaac
            # runtime to install GR00T's full training/data dependency stack.
            # Connectivity is still checked here; observation/action keys are
            # validated against the checked-in policy profile on every call.
            if not self.ping():
                raise ConnectionError("GR00T ZMQ server did not answer ping")
            return {
                "transport": "zmq",
                "status": "ok",
                "modality_config": "validated_by_local_policy_profile",
            }
        getter = getattr(self.client, "get_modality_config", None)
        if callable(getter):
            modality_config = getter()
        else:
            call_endpoint = getattr(self.client, "call_endpoint", None)
            if not callable(call_endpoint):
                return {}
            modality_config = call_endpoint(
                "get_modality_config", requires_input=False
            )
        if modality_config is None:
            return {}
        # Let connection/schema errors fail preflight instead of reporting a
        # healthy policy while the GR00T endpoint is unavailable.
        return {"modality_config": _json_safe_metadata(modality_config)}

    def ping(self) -> bool:
        ping = getattr(self.client, "ping", None)
        return bool(ping()) if callable(ping) else True

    def reset(self) -> None:
        reset = getattr(self.client, "reset", None)
        if callable(reset):
            reset()

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
            return
        socket = getattr(self.client, "socket", None)
        if socket is not None:
            socket.close(linger=0)
            self.client.socket = None
        context = getattr(self.client, "context", None)
        if context is not None:
            context.term()
            self.client.context = None


__all__ = ["Gr00tPolicyClient"]
