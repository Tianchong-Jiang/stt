from __future__ import annotations

import http.client
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import gemini_speech_to_terminal as gemini_stt
import stt_toggle


class FakeSocket:
    def __init__(self, error: OSError | None = None):
        self.error = error
        self.closed = False
        self.connected_to = None
        self.timeouts = []

    def settimeout(self, timeout):
        self.timeouts.append(timeout)

    def bind(self, _address):
        pass

    def connect(self, address):
        self.connected_to = address
        if self.error is not None:
            raise self.error

    def close(self):
        self.closed = True


class GeminiConnectionTest(unittest.TestCase):
    def test_failed_address_is_retried_with_a_fresh_socket(self):
        first = FakeSocket(TimeoutError("black-holed"))
        second = FakeSocket()
        addresses = [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("192.0.2.1", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("192.0.2.2", 443)),
        ]

        with (
            patch.object(gemini_stt.socket, "getaddrinfo", return_value=addresses),
            patch.object(gemini_stt.socket, "socket", side_effect=[first, second]),
        ):
            result = gemini_stt.connect_with_retry(("example.com", 443), 6.0)

        self.assertIs(result, second)
        self.assertTrue(first.closed)
        self.assertEqual(first.connected_to, ("192.0.2.1", 443))
        self.assertEqual(second.connected_to, ("192.0.2.2", 443))
        self.assertLessEqual(first.timeouts[0], gemini_stt.GEMINI_CONNECT_ATTEMPT_SECONDS)
        self.assertLessEqual(second.timeouts[0], gemini_stt.GEMINI_CONNECT_ATTEMPT_SECONDS)

    def test_https_connection_restores_the_long_response_timeout(self):
        observed_connect_timeouts = []
        sock = FakeSocket()

        def fake_connect(connection):
            observed_connect_timeouts.append(connection.timeout)
            connection.sock = sock

        with patch.object(http.client.HTTPSConnection, "connect", new=fake_connect):
            connection = gemini_stt.GeminiHTTPSConnection("example.com", timeout=120.0)
            connection.connect()

        self.assertEqual(observed_connect_timeouts, [gemini_stt.GEMINI_CONNECT_BUDGET_SECONDS])
        self.assertEqual(connection.timeout, 120.0)
        self.assertEqual(sock.timeouts[-1], 120.0)

    def test_https_handler_uses_the_bounded_connection_class(self):
        handler = gemini_stt.GeminiHTTPSHandler()
        request = object()
        with patch.object(handler, "do_open", return_value="response") as do_open:
            result = handler.https_open(request)

        self.assertEqual(result, "response")
        do_open.assert_called_once_with(
            gemini_stt.GeminiHTTPSConnection,
            request,
            context=handler._context,
        )


class ToggleLockTest(unittest.TestCase):
    def test_only_one_toggle_invocation_can_hold_the_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toggle.lock"
            first = stt_toggle.try_toggle_lock(path)
            self.assertIsNotNone(first)
            try:
                self.assertIsNone(stt_toggle.try_toggle_lock(path))
            finally:
                os.close(first)

            second = stt_toggle.try_toggle_lock(path)
            self.assertIsNotNone(second)
            os.close(second)

    def test_busy_toggle_is_ignored_before_reading_shared_state(self):
        with (
            patch.object(stt_toggle, "try_toggle_lock", return_value=None),
            patch.object(stt_toggle, "read_state") as read_state,
            patch.object(stt_toggle, "log") as log,
        ):
            stt_toggle.main()

        read_state.assert_not_called()
        log.assert_called_once_with("toggle ignored: another invocation is active")


if __name__ == "__main__":
    unittest.main()
