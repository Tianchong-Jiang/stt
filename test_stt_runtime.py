from __future__ import annotations

import json
import inspect
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

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
        self.assertIn("Omit non-semantic hesitation fillers", gemini_stt.TRANSCRIPTION_INSTRUCTION)

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

    def test_excessive_transcript_is_retried_with_same_model_and_visible_status(self):
        audio = Path("speech.wav")
        status = Path("status.json")
        with (
            patch.object(gemini_stt, "call_gemini", side_effect=["leaked", "clean"]) as call,
            patch.object(gemini_stt, "transcript_exceeds_speech_rate", side_effect=[True, False]),
            patch.object(gemini_stt, "write_status") as write_status,
        ):
            result = gemini_stt.call_gemini_with_fallback(
                "key", ["gemini-3.8-flash", "gemini-3.5-flash"], "prompt", audio, 2048, status
            )

        self.assertEqual("clean", result)
        self.assertEqual(["gemini-3.8-flash", "gemini-3.8-flash"], [item.args[1] for item in call.call_args_list])
        self.assertEqual("prompt", call.call_args_list[0].args[2])
        retry_prompt = call.call_args_list[1].args[2]
        self.assertTrue(retry_prompt.startswith("prompt\n\n"))
        self.assertIn("Do not include your thought process", retry_prompt)
        self.assertIn(gemini_stt.END_MARKER, retry_prompt)
        self.assertTrue(any("suspected leaked thought process" in item.args[1] for item in write_status.call_args_list))

    def test_excessive_transcript_uses_twice_typical_speech_rate(self):
        wav = MagicMock()
        wav.getnframes.return_value = 30 * 16_000
        wav.getframerate.return_value = 16_000
        context = MagicMock()
        context.__enter__.return_value = wav
        with patch.object(gemini_stt.wave, "open", return_value=context):
            self.assertFalse(gemini_stt.transcript_exceeds_speech_rate("word " * 150, Path("speech.wav")))
            self.assertTrue(gemini_stt.transcript_exceeds_speech_rate("word " * 151, Path("speech.wav")))
            self.assertFalse(gemini_stt.transcript_exceeds_speech_rate("我" * 300, Path("speech.wav")))
            self.assertTrue(gemini_stt.transcript_exceeds_speech_rate("我" * 302, Path("speech.wav")))

    def test_repeated_excessive_transcript_fails_loudly(self):
        with (
            patch.object(gemini_stt, "call_gemini", side_effect=["leaked once", "leaked twice"]) as call,
            patch.object(gemini_stt, "transcript_exceeds_speech_rate", return_value=True),
            patch.object(gemini_stt, "write_status"),
            self.assertRaisesRegex(gemini_stt.GeminiRequestError, "repeatedly exceeded plausible speech rate"),
        ):
            gemini_stt.call_gemini_with_fallback(
                "key", ["gemini-3.8-flash", "gemini-3.5-flash"], "prompt", Path("speech.wav"), 2048, Path("status.json")
            )
        self.assertEqual(["gemini-3.8-flash"] * 2, [item.args[1] for item in call.call_args_list])

    def test_initial_request_error_still_uses_fallback(self):
        with (
            patch.object(gemini_stt, "call_gemini", side_effect=[gemini_stt.GeminiRequestError("unavailable"), "clean"]) as call,
            patch.object(gemini_stt, "transcript_exceeds_speech_rate", return_value=False),
        ):
            result = gemini_stt.call_gemini_with_fallback(
                "key", ["gemini-3.8-flash", "gemini-3.5-flash"], "prompt", Path("speech.wav"), 2048
            )
        self.assertEqual("clean", result)
        self.assertEqual(["gemini-3.8-flash", "gemini-3.5-flash"], [item.args[1] for item in call.call_args_list])

    def fallback_models_called(self, call_results, exceeds=(False,)):
        with (
            patch.object(gemini_stt, "call_gemini", side_effect=call_results) as call,
            patch.object(gemini_stt, "transcript_exceeds_speech_rate", side_effect=exceeds),
            patch.object(gemini_stt, "write_status") as write_status,
        ):
            result = gemini_stt.call_gemini_with_fallback(
                "key", ["gemini-3.8-flash", "gemini-3.5-flash"], "prompt", Path("speech.wav"), 2048, Path("status.json")
            )
        self.assertEqual("clean", result)
        self.assertTrue(any("blocked - running again" in item.args[1] for item in write_status.call_args_list))
        return [item.args[1] for item in call.call_args_list]

    def test_blocked_request_runs_same_model_again_before_fallback(self):
        blocked = gemini_stt.GeminiBlockedError("blocked")
        self.assertEqual(["gemini-3.8-flash"] * 2, self.fallback_models_called([blocked, "clean"]))
        self.assertEqual(
            ["gemini-3.8-flash"] * 2 + ["gemini-3.5-flash"], self.fallback_models_called([blocked, blocked, "clean"])
        )

    def test_blocked_leak_retry_falls_back_after_running_again(self):
        blocked = gemini_stt.GeminiBlockedError("blocked")
        self.assertEqual(
            ["gemini-3.8-flash"] * 3 + ["gemini-3.5-flash"],
            self.fallback_models_called(["leaked", blocked, blocked, "clean"], exceeds=(True, False)),
        )

    def test_only_prompt_block_is_classified_as_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "speech.wav"
            audio.write_bytes(b"RIFF")
            for body, error in [
                ({"promptFeedback": {"blockReason": "OTHER"}}, gemini_stt.GeminiBlockedError),
                ({"candidates": [{"content": {}, "finishReason": "STOP"}]}, gemini_stt.GeminiRequestError),
            ]:
                connection = MagicMock()
                connection.getresponse.return_value.status = 200
                connection.getresponse.return_value.read.return_value = json.dumps(body).encode()
                with (
                    patch.object(gemini_stt.http.client, "HTTPSConnection", return_value=connection),
                    self.assertRaisesRegex(gemini_stt.GeminiRequestError, "returned no text") as raised,
                ):
                    gemini_stt.call_gemini("key", "gemini-3.8-flash", "prompt", audio, 2048)
                self.assertIs(error, type(raised.exception))

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
