from __future__ import annotations

import unittest

import numpy as np

from eai4s_eval.policy import PolicyClientError, PolicyOutputInvalid
from eai4s_eval.policy.openpi_websocket_client import OpenPIMessagePackCodec, OpenPIWebSocketPolicyClient


class _FakeConnection:
    def __init__(self, responses):
        self.responses = list(responses)
        self.sent: list[bytes] = []
        self.closed = False
        self.recv_timeouts: list[float | None] = []

    def recv(self, timeout=None):
        self.recv_timeouts.append(timeout)
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def send(self, payload):
        self.sent.append(payload)

    def close(self):
        self.closed = True


class OpenPIMessagePackCodecTest(unittest.TestCase):
    def test_numpy_arrays_and_scalars_round_trip(self) -> None:
        codec = OpenPIMessagePackCodec()
        payload = {
            "image": np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
            "state": np.asarray([1.0, 2.0], dtype=np.float32),
            "step": np.int64(7),
        }
        decoded = codec.unpack(codec.pack(payload))
        np.testing.assert_array_equal(decoded["image"], payload["image"])
        np.testing.assert_array_equal(decoded["state"], payload["state"])
        self.assertEqual(decoded["step"], np.int64(7))

    def test_object_and_complex_arrays_are_rejected(self) -> None:
        codec = OpenPIMessagePackCodec()
        with self.assertRaises(TypeError):
            codec.pack({"bad": np.asarray([object()], dtype=object)})
        with self.assertRaises(TypeError):
            codec.pack({"bad": np.asarray([1 + 2j], dtype=np.complex64)})


class OpenPIWebSocketPolicyClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.codec = OpenPIMessagePackCodec()

    def test_metadata_handshake_auth_and_inference(self) -> None:
        connection = _FakeConnection(
            [
                self.codec.pack({"provider": "openpi", "action_dim": 8}),
                self.codec.pack({"actions": np.zeros((16, 8), dtype=np.float32)}),
            ]
        )
        calls = []

        def factory(*args, **kwargs):
            calls.append((args, kwargs))
            return connection

        client = OpenPIWebSocketPolicyClient(
            host="127.0.0.1", port=8000, api_key="secret", timeout_sec=2.0,
            connection_factory=factory,
        )
        self.assertEqual(client.get_server_metadata()["action_dim"], 8)
        result = client.infer({"observation/state": np.zeros(8, dtype=np.float32)})
        self.assertEqual(result["actions"].shape, (16, 8))
        self.assertEqual(calls[0][0][0], "ws://127.0.0.1:8000")
        self.assertEqual(calls[0][1]["additional_headers"], {"Authorization": "Api-Key secret"})
        sent = self.codec.unpack(connection.sent[0])
        np.testing.assert_array_equal(sent["observation/state"], np.zeros(8, dtype=np.float32))
        client.close()
        client.close()
        self.assertTrue(connection.closed)

    def test_text_response_is_reported_as_server_error(self) -> None:
        connection = _FakeConnection([self.codec.pack({}), "server traceback"])
        client = OpenPIWebSocketPolicyClient(
            host="ws://localhost:8000", port=None, timeout_sec=1.0,
            connection_factory=lambda *args, **kwargs: connection,
        )
        with self.assertRaisesRegex(PolicyClientError, "server traceback"):
            client.infer({"prompt": "pick"})

    def test_non_mapping_metadata_and_response_are_rejected(self) -> None:
        bad_metadata = _FakeConnection([self.codec.pack(["not", "metadata"])])
        with self.assertRaises(PolicyOutputInvalid):
            OpenPIWebSocketPolicyClient(
                host="localhost", port=8000, timeout_sec=1.0,
                connection_factory=lambda *args, **kwargs: bad_metadata,
            )

        bad_response = _FakeConnection([self.codec.pack({}), self.codec.pack([1, 2, 3])])
        client = OpenPIWebSocketPolicyClient(
            host="localhost", port=8000, timeout_sec=1.0,
            connection_factory=lambda *args, **kwargs: bad_response,
        )
        with self.assertRaises(PolicyOutputInvalid):
            client.infer({"prompt": "pick"})

    def test_connection_attempts_are_bounded(self) -> None:
        def refused(*args, **kwargs):
            raise ConnectionRefusedError("refused")

        with self.assertRaisesRegex(PolicyClientError, "timed out connecting"):
            OpenPIWebSocketPolicyClient(
                host="localhost", port=8000, timeout_sec=0.01, retry_interval_sec=0,
                connection_factory=refused,
            )


if __name__ == "__main__":
    unittest.main()
