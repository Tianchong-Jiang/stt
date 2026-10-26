import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import gemini_speech_to_terminal as stt


class PriorityTranscriptionTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.audio = Path(directory.name) / "speech.wav"
        self.audio.write_bytes(b"RIFF")
        self.connection = MagicMock()
        self.response = self.connection.getresponse.return_value
        self.response.status = 200
        self.body = {
            "service_tier": "priority",
            "status": "completed",
            "usage": {
                "total_input_tokens": 10, "total_output_tokens": 5,
                "total_thought_tokens": 0, "total_tokens": 15,
            },
            "steps": [
                {"type": "thought", "signature": "opaque"},
                {"type": "model_output", "content": [{"type": "text", "text": "raw transcript"}]},
            ],
        }

    def call(self, model="gemini-3.8-flash"):
        self.response.read.return_value = json.dumps(self.body).encode()
        with patch.object(stt.http.client, "HTTPSConnection", return_value=self.connection):
            return stt.call_gemini("key", model, "context", self.audio, 2048)

    def test_all_configured_models_use_priority_and_preserve_transcript_validation(self):
        for model in stt.parse_models(stt.DEFAULT_MODEL):
            with (
                self.subTest(model=model),
                patch.object(stt, "parse_transcript_response", return_value="clean") as parse,
                patch.object(stt, "mime_type", return_value="audio/wav"),
            ):
                self.assertEqual("clean", self.call(model))
                parse.assert_called_once_with("raw transcript", model)
                method, path, body, headers = self.connection.request.call_args.args
                self.assertEqual(("POST", "/v1beta/interactions"), (method, path))
                self.assertEqual({"Content-Type": "application/json", "x-goog-api-key": "key"}, headers)
                self.assertEqual({
                    "model": model,
                    "system_instruction": stt.TRANSCRIPTION_INSTRUCTION,
                    "input": [
                        {"type": "text", "text": "context"},
                        {"type": "audio", "mime_type": "audio/wav", "data": base64.b64encode(b"RIFF").decode()},
                    ],
                    "generation_config": {"max_output_tokens": 2048, "thinking_level": "low"},
                    "service_tier": "priority",
                    "store": False,
                }, json.loads(body))
                self.connection.close.assert_called()

    def test_nonpriority_response_is_rejected_before_parsing(self):
        for tier in ("standard", "flex", None):
            with self.subTest(tier=tier), patch.object(stt, "parse_transcript_response") as parse:
                self.body["service_tier"] = tier
                with self.assertRaisesRegex(stt.GeminiRequestError, "priority is required"):
                    self.call()
                parse.assert_not_called()

    def test_missing_tier_fails_loudly(self):
        del self.body["service_tier"]
        with self.assertRaisesRegex(KeyError, "service_tier"):
            self.call()

    def test_incomplete_and_failed_results_cannot_be_used_as_transcripts(self):
        for status in ("incomplete", "failed", "cancelled", "in_progress"):
            with self.subTest(status=status), patch.object(stt, "parse_transcript_response") as parse:
                self.body["status"] = status
                with self.assertRaisesRegex(stt.GeminiRequestError, "transcript is not complete"):
                    self.call()
                parse.assert_not_called()

    def test_thought_only_result_is_not_transcript_text(self):
        self.body["steps"] = [{"type": "thought", "signature": "opaque"}]
        with self.assertRaisesRegex(stt.GeminiRequestError, "no transcript text"):
            self.call()

    def test_http_error_still_reports_api_error(self):
        self.response.status = 429
        self.body = {"error": {"message": "quota exhausted"}}
        with self.assertRaisesRegex(stt.GeminiRequestError, "HTTP 429.*quota exhausted") as raised:
            self.call()
        self.assertNotIsInstance(raised.exception, stt.GeminiTransientError)

    def test_only_errors_a_resend_can_fix_are_transient(self):
        for status, message, transient in [
            (500, "Internal error encountered.", True),
            (503, "The model is overloaded.", True),
            (400, "Input blocked: This request was blocked by Gemini's filters.", True),
            (400, "Request contains an invalid argument.", False),
            (401, "API key not valid.", False),
            (403, "Permission denied.", False),
            (404, "Model not found.", False),
            (429, "You exceeded your current quota.", False),
        ]:
            with self.subTest(status=status, message=message):
                self.response.status = status
                self.body = {"error": {"message": message}}
                with self.assertRaises(stt.GeminiRequestError) as raised:
                    self.call()
                self.assertEqual(transient, isinstance(raised.exception, stt.GeminiTransientError))

    def test_network_failure_is_transient(self):
        self.connection.getresponse.side_effect = TimeoutError("The read operation timed out")
        with self.assertRaisesRegex(stt.GeminiTransientError, "read operation timed out"):
            self.call()

    def test_completed_result_without_output_is_transient(self):
        del self.body["steps"]
        with self.assertRaisesRegex(stt.GeminiTransientError, "no transcript text"):
            self.call()


if __name__ == "__main__":
    unittest.main()
