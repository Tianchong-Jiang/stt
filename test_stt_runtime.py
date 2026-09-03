from __future__ import annotations

import json
import inspect
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import Mock, patch

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
    def test_default_models_and_generation_config_follow_gemini_3_best_practices(self):
        self.assertEqual(gemini_stt.DEFAULT_MODEL, "gemini-3.8-flash,gemini-3.5-flash")
        self.assertEqual(
            gemini_stt.generation_config(2048),
            {"maxOutputTokens": 2048, "thinkingConfig": {"thinkingLevel": "low"}},
        )
        self.assertIn("Simplified Chinese", gemini_stt.TRANSCRIPTION_INSTRUCTION)

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
            stt_toggle.main([])

        read_state.assert_not_called()
        log.assert_called_once_with("toggle ignored: another invocation is active")


class ToggleSafetyTest(unittest.TestCase):
    def test_recording_has_a_hard_duration_limit(self):
        args = stt_toggle.parse_args([])
        recorder = Mock(pid=4321)
        recorder.poll.return_value = None
        with tempfile.TemporaryDirectory() as directory:
            state_file = Path(directory) / "state.json"
            with (
                patch.object(stt_toggle, "STATE_FILE", state_file),
                patch.object(stt_toggle, "ensure_status_window"),
                patch.object(stt_toggle, "set_status"),
                patch.object(stt_toggle, "log"),
                patch.object(stt_toggle.time, "sleep"),
                patch.object(stt_toggle.subprocess, "Popen", return_value=recorder) as popen,
            ):
                stt_toggle.start_recording(args)
            audio = Path(json.loads(state_file.read_text())["audio"])
            try:
                self.assertEqual(popen.call_args.args[0][-5:-1], ["-d", "300", "-D", "default"])
            finally:
                audio.unlink(missing_ok=True)

    def test_text_is_pasted_without_xdotool_type(self):
        self.assertNotIn("xdotool", inspect.getsource(stt_toggle.type_text))


if __name__ == "__main__":
    unittest.main()
