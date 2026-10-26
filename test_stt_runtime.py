from __future__ import annotations

import json
import inspect
import os
from pathlib import Path
import socket
import subprocess
import sys
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
    def setUp(self):
        sleep = patch.object(gemini_stt.time, "sleep")
        self.sleep = sleep.start()
        self.addCleanup(sleep.stop)

    def test_default_models_and_generation_config_follow_gemini_3_best_practices(self):
        self.assertEqual(gemini_stt.DEFAULT_MODEL, "gemini-3.8-flash,gemini-3.5-flash")
        self.assertEqual(
            gemini_stt.generation_config(2048),
            {"max_output_tokens": 2048, "thinking_level": "low"},
        )
        self.assertIn("Simplified Chinese", gemini_stt.TRANSCRIPTION_INSTRUCTION)
        self.assertIn("Omit non-semantic hesitation fillers", gemini_stt.TRANSCRIPTION_INSTRUCTION)
        self.assertIn(gemini_stt.BEGIN_MARKER, gemini_stt.TRANSCRIPTION_INSTRUCTION)
        self.assertIn(gemini_stt.END_MARKER, gemini_stt.TRANSCRIPTION_INSTRUCTION)
        self.assertIn("single line with no line breaks", gemini_stt.TRANSCRIPTION_INSTRUCTION)

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

    def test_reasoning_before_the_begin_marker_is_discarded(self):
        leaked = (
            'The speaker says: "yes so I understand that improver optimizes tactic script length."\n'
            "Matches lemma compression context.\n"
            f"{gemini_stt.BEGIN_MARKER}\n"
            "yes so I understand that improver optimizes tactic script length.\n"
            f"{gemini_stt.END_MARKER}"
        )
        self.assertEqual(
            "yes so I understand that improver optimizes tactic script length.",
            gemini_stt.parse_transcript_response(leaked, "gemini-3.8-flash"),
        )
        self.assertEqual(
            "[NO_SPEECH]",
            gemini_stt.parse_transcript_response(
                f"{gemini_stt.BEGIN_MARKER}\n[NO_SPEECH]\n{gemini_stt.END_MARKER}", "gemini-3.8-flash"
            ),
        )

    def test_transcript_that_is_not_one_line_is_rejected(self):
        for body, expected in [
            (f"{gemini_stt.BEGIN_MARKER}\nfirst draft\nfinal text\n{gemini_stt.END_MARKER}", "spans 2 lines"),
            (f"a transcript without any marker\n{gemini_stt.END_MARKER}", "omitted \\[BEGIN_TRANSCRIPT\\]"),
            (
                f"{gemini_stt.BEGIN_MARKER}... {gemini_stt.END_MARKER}{gemini_stt.BEGIN_MARKER}real text{gemini_stt.END_MARKER}",
                "repeated a transcript marker",
            ),
        ]:
            with self.assertRaisesRegex(gemini_stt.GeminiFormatError, expected):
                gemini_stt.parse_transcript_response(body, "gemini-3.8-flash")
        with self.assertRaisesRegex(gemini_stt.GeminiRequestError, "omitted \\[END_OF_TRANSCRIPT\\]"):
            gemini_stt.parse_transcript_response(f"{gemini_stt.BEGIN_MARKER}\ntruncated", "gemini-3.8-flash")

    def test_malformed_transcript_is_retried_with_same_model_and_visible_status(self):
        with (
            patch.object(
                gemini_stt, "call_gemini", side_effect=[gemini_stt.GeminiFormatError("spans 3 lines"), "clean"]
            ) as call,
            patch.object(gemini_stt, "write_status") as write_status,
        ):
            result = gemini_stt.call_gemini_with_fallback(
                "key", ["gemini-3.8-flash", "gemini-3.5-flash"], "prompt", Path("speech.wav"), 2048, Path("status.json")
            )

        self.assertEqual("clean", result)
        self.assertEqual(["gemini-3.8-flash"] * 2, [item.args[1] for item in call.call_args_list])
        self.assertEqual("prompt", call.call_args_list[0].args[2])
        retry_prompt = call.call_args_list[1].args[2]
        self.assertTrue(retry_prompt.startswith("prompt\n\n"))
        self.assertIn(gemini_stt.BEGIN_MARKER, retry_prompt)
        self.assertIn("on one line", retry_prompt)
        self.assertTrue(any("malformed transcript - running again" in item.args[1] for item in write_status.call_args_list))

    def test_repeated_malformed_transcript_falls_back_to_the_next_model(self):
        malformed = gemini_stt.GeminiFormatError("spans 3 lines")
        with (
            patch.object(gemini_stt, "call_gemini", side_effect=[malformed, malformed, "clean"]) as call,
            patch.object(gemini_stt, "write_status"),
        ):
            result = gemini_stt.call_gemini_with_fallback(
                "key", ["gemini-3.8-flash", "gemini-3.5-flash"], "prompt", Path("speech.wav"), 2048, Path("status.json")
            )
        self.assertEqual("clean", result)
        self.assertEqual(["gemini-3.8-flash"] * 2 + ["gemini-3.5-flash"], [item.args[1] for item in call.call_args_list])

    def test_initial_request_error_still_uses_fallback(self):
        with patch.object(
            gemini_stt, "call_gemini", side_effect=[gemini_stt.GeminiRequestError("unavailable"), "clean"]
        ) as call:
            result = gemini_stt.call_gemini_with_fallback(
                "key", ["gemini-3.8-flash", "gemini-3.5-flash"], "prompt", Path("speech.wav"), 2048
            )
        self.assertEqual("clean", result)
        self.assertEqual(["gemini-3.8-flash", "gemini-3.5-flash"], [item.args[1] for item in call.call_args_list])

    def test_transient_errors_rotate_through_the_models_until_one_succeeds(self):
        error = gemini_stt.GeminiTransientError("timed out")
        with patch.object(gemini_stt, "call_gemini", side_effect=[error] * 5 + ["clean"]) as call:
            result = gemini_stt.call_gemini_with_fallback(
                "key", ["gemini-3.8-flash", "gemini-3.5-flash"], "prompt", Path("speech.wav"), 2048
            )
        self.assertEqual("clean", result)
        self.assertEqual(["gemini-3.8-flash", "gemini-3.5-flash"] * 3, [item.args[1] for item in call.call_args_list])
        self.assertEqual(5, self.sleep.call_count)

    def test_permanent_errors_drop_each_model_and_fail_when_none_is_left(self):
        errors = [gemini_stt.GeminiRequestError("bad key 3.8"), gemini_stt.GeminiRequestError("bad key 3.5")]
        with patch.object(gemini_stt, "call_gemini", side_effect=errors) as call:
            with self.assertRaisesRegex(gemini_stt.GeminiRequestError, "all Gemini models failed: bad key 3.8 \\| bad key 3.5"):
                gemini_stt.call_gemini_with_fallback(
                    "key", ["gemini-3.8-flash", "gemini-3.5-flash"], "prompt", Path("speech.wav"), 2048
                )
        self.assertEqual(2, call.call_count)

    def test_remaining_model_keeps_retrying_after_the_other_is_dropped(self):
        transient, permanent = gemini_stt.GeminiTransientError("timed out"), gemini_stt.GeminiRequestError("not found")
        with patch.object(gemini_stt, "call_gemini", side_effect=[transient, permanent, transient, "clean"]) as call:
            result = gemini_stt.call_gemini_with_fallback(
                "key", ["gemini-3.8-flash", "gemini-3.5-flash"], "prompt", Path("speech.wav"), 2048
            )
        self.assertEqual("clean", result)
        self.assertEqual(
            ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.8-flash"],
            [item.args[1] for item in call.call_args_list],
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
            patch.object(stt_toggle, "PROCESSING_PID_FILE", Path("/nonexistent/processing.pid")),
            patch.object(stt_toggle, "read_state") as read_state,
            patch.object(stt_toggle, "log") as log,
        ):
            stt_toggle.main([])

        read_state.assert_not_called()
        log.assert_called_once_with("toggle ignored: another invocation is active")

    def test_lock_holder_removes_a_processing_pid_left_by_a_killed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "processing.pid"
            pid_file.write_text("4321")
            with (
                patch.object(stt_toggle, "try_toggle_lock", return_value=os.open(Path(directory) / "lock", os.O_CREAT)),
                patch.object(stt_toggle, "PROCESSING_PID_FILE", pid_file),
                patch.object(stt_toggle, "read_state", return_value=None),
                patch.object(stt_toggle, "start_recording"),
            ):
                stt_toggle.main([])
            self.assertFalse(pid_file.exists())

    def test_busy_toggle_cancels_a_pending_transcription(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "processing.pid"
            pid_file.write_text("4321")
            with (
                patch.object(stt_toggle, "try_toggle_lock", return_value=None),
                patch.object(stt_toggle, "PROCESSING_PID_FILE", pid_file),
                patch.object(stt_toggle.os, "kill") as kill,
                patch.object(stt_toggle, "log"),
            ):
                stt_toggle.main([])
        kill.assert_called_once_with(4321, stt_toggle.signal.SIGTERM)


class TranscriberSleepTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.pid_file = Path(directory.name) / "processing.pid"
        pid_file = patch.object(stt_toggle, "PROCESSING_PID_FILE", self.pid_file)
        pid_file.start()
        self.addCleanup(pid_file.stop)

    def test_normal_output_and_exit_code_are_preserved(self):
        result = stt_toggle.run_transcriber(
            [sys.executable, "-c", "import sys; print('transcript'); print('diagnostic', file=sys.stderr); sys.exit(3)"]
        )
        self.assertEqual((result.returncode, result.stdout, result.stderr), (3, "transcript\n", "diagnostic\n"))

    def test_response_ready_after_sleep_is_rejected(self):
        with patch.object(stt_toggle.time, "clock_gettime", side_effect=[0, 10**12]):
            with self.assertRaisesRegex(TimeoutError, "cancelled after system sleep"):
                stt_toggle.run_transcriber([sys.executable, "-c", "print('stale transcript')"])

    def test_sleep_kills_and_reaps_pending_transcriber(self):
        children = []
        real_popen = subprocess.Popen

        def spawn(*args, **kwargs):
            children.append(real_popen(*args, **kwargs))
            return children[-1]

        with (
            patch.object(stt_toggle.time, "clock_gettime", side_effect=[0, 10**12]),
            patch.object(stt_toggle.subprocess, "Popen", side_effect=spawn),
        ):
            with self.assertRaisesRegex(TimeoutError, "cancelled after system sleep"):
                stt_toggle.run_transcriber([sys.executable, "-c", "import time; time.sleep(30)"])
        self.assertEqual(children[0].returncode, -9)
        with self.assertRaises(ChildProcessError):
            os.waitpid(children[0].pid, os.WNOHANG)

    def test_transcriber_pid_is_published_only_while_it_runs(self):
        result = stt_toggle.run_transcriber(
            [sys.executable, "-c", f"import os; print(open({str(self.pid_file)!r}).read() == str(os.getpid()))"]
        )
        self.assertEqual("True\n", result.stdout)
        self.assertFalse(self.pid_file.exists())

    def test_killed_transcriber_is_reported_as_cancelled_and_keeps_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "speech.wav"
            audio.write_bytes(b"x" * 100)
            state = {"pid": 4321, "audio": str(audio), "started": 0}
            cancelled = subprocess.CompletedProcess(["transcriber"], -stt_toggle.signal.SIGTERM, "", "warning: timed out")
            with (
                patch.object(stt_toggle, "STATE_FILE", Path(directory) / "state.json"),
                patch.object(stt_toggle, "pid_alive", side_effect=[True, False, False]),
                patch.object(stt_toggle.os, "kill"),
                patch.object(stt_toggle.time, "sleep"),
                patch.object(stt_toggle, "ensure_status_window"),
                patch.object(stt_toggle, "set_status") as set_status,
                patch.object(stt_toggle, "log") as log,
                patch.object(stt_toggle, "run_transcriber", return_value=cancelled),
                patch.object(stt_toggle, "type_text") as type_text,
            ):
                stt_toggle.stop_recording(state, stt_toggle.parse_args([]))
            self.assertTrue(audio.exists())
        set_status.assert_called_with("STT cancelled", "#222222", ttl=2)
        log.assert_any_call("transcriber stderr: warning: timed out")
        type_text.assert_not_called()


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
