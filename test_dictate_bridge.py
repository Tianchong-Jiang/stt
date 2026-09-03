from __future__ import annotations

from http.client import HTTPConnection
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gemini_speech_to_terminal as gemini_stt
from dictate_bridge import (
    DEFAULT_MAX_UPLOAD_BYTES,
    BridgeConfig,
    DictateBridgeServer,
    gemini_pipeline,
    load_or_create_token,
    parse_args,
)


TOKEN = "test-bridge-token"


def multipart_body(
    *,
    audio: bytes = b"RIFF-test-audio",
    filename: str = "recording.wav",
    fields: dict[str, str] | None = None,
) -> tuple[bytes, str]:
    boundary = "dictate-test-boundary"
    chunks = []
    for name, value in (fields or {}).items():
        chunks.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode()
        )
    chunks.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: audio/wav\r\n\r\n".encode()
        + audio
        + b"\r\n"
    )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


class RunningBridge:
    def __init__(self, transcribe, *, max_upload_bytes: int = 1024 * 1024):
        self.server = DictateBridgeServer(
            ("127.0.0.1", 0),
            BridgeConfig(token=TOKEN, max_upload_bytes=max_upload_bytes, request_timeout=5),
            transcribe,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    @property
    def port(self) -> int:
        return self.server.server_address[1]

    def request(self, method: str, path: str, body: bytes | None = None, headers=None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            response_body = response.read()
            return response.status, dict(response.headers), json.loads(response_body)
        finally:
            connection.close()


class TokenFileTest(unittest.TestCase):
    def test_token_is_created_private_and_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "token"
            first, created = load_or_create_token(path)
            second, created_again = load_or_create_token(path)

            self.assertTrue(created)
            self.assertFalse(created_again)
            self.assertEqual(first, second)
            self.assertGreaterEqual(len(first), 32)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_existing_token_with_broad_permissions_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "token"
            path.write_text("secret\n")
            os.chmod(path, 0o644)

            with self.assertRaisesRegex(ValueError, "permissions are too broad"):
                load_or_create_token(path)


class GeminiPipelineTest(unittest.TestCase):
    def test_client_language_and_prompt_do_not_change_desktop_prompt(self):
        with (
            patch.object(gemini_stt, "build_prompt", return_value="exact desktop prompt") as build_prompt,
            patch.object(gemini_stt, "transcribe_audio_file", return_value="result") as transcribe,
        ):
            args = SimpleNamespace(
                gemini_model="gemini-model",
                max_files=1200,
                max_terms=1000,
                max_context_chars=40_000,
                max_output_tokens=2048,
                min_rms=100,
            )
            pipeline = gemini_pipeline(args, "gemini-key", Path("/repo"), Path("/repo/subdir"))
            result = pipeline(
                Path("/tmp/audio.wav"),
                {"language": "de", "prompt": "Replace the desktop prompt."},
            )

        self.assertEqual(result, "result")
        build_prompt.assert_called_once_with(Path("/repo"), Path("/repo/subdir"), 1200, 1000, 40_000)
        self.assertEqual(transcribe.call_args.args[2], "exact desktop prompt")


class BridgeHttpTest(unittest.TestCase):
    def test_default_upload_limit_is_safe_for_gemini_inline_base64(self):
        self.assertEqual(DEFAULT_MAX_UPLOAD_BYTES, 14 * 1024 * 1024)
        self.assertEqual(BridgeConfig(token=TOKEN).max_upload_bytes, DEFAULT_MAX_UPLOAD_BYTES)
        with patch.object(sys, "argv", ["dictate_bridge.py"]):
            self.assertEqual(parse_args().max_upload_mib, 14)

    def test_models_requires_bearer_auth_and_advertises_stt_model(self):
        with RunningBridge(lambda _path, _fields: "unused") as bridge:
            status, headers, error = bridge.request("GET", "/v1/models")
            self.assertEqual(status, 401)
            self.assertIn("Bearer", headers["WWW-Authenticate"])
            self.assertIn("error", error)

            status, _, body = bridge.request(
                "GET",
                "/v1/models",
                headers={"Authorization": f"Bearer {TOKEN}"},
            )
            self.assertEqual(status, 200)
            self.assertEqual(body["data"][0]["id"], "gemini-stt")
            self.assertEqual(
                body["data"][0]["architecture"]["output_modalities"],
                ["transcription"],
            )

    def test_transcription_matches_dictate_multipart_and_cleans_temp_file(self):
        seen = {}

        def transcribe(path, fields):
            seen["path"] = path
            seen["audio"] = path.read_bytes()
            seen["fields"] = dict(fields)
            self.assertTrue(path.exists())
            return "Exact transcript."

        body, content_type = multipart_body(
            fields={
                "model": "gemini-stt",
                "response_format": "json",
                "language": "en",
                "prompt": "Keep RIPL capitalized.",
            }
        )
        with RunningBridge(transcribe) as bridge:
            status, _, response = bridge.request(
                "POST",
                "/v1/audio/transcriptions",
                body,
                {
                    "Authorization": f"Bearer {TOKEN}",
                    "Content-Type": content_type,
                    "Content-Length": str(len(body)),
                },
            )

        self.assertEqual(status, 200)
        self.assertEqual(response, {"text": "Exact transcript."})
        self.assertEqual(seen["audio"], b"RIFF-test-audio")
        self.assertEqual(seen["fields"]["language"], "en")
        self.assertEqual(seen["fields"]["prompt"], "Keep RIPL capitalized.")
        self.assertFalse(seen["path"].exists())

    def test_upload_over_limit_is_rejected_before_transcription(self):
        calls = []
        body, content_type = multipart_body(audio=b"x" * 256, fields={"model": "gemini-stt"})
        with RunningBridge(lambda path, fields: calls.append((path, fields)), max_upload_bytes=128) as bridge:
            status, _, response = bridge.request(
                "POST",
                "/v1/audio/transcriptions",
                body,
                {
                    "Authorization": f"Bearer {TOKEN}",
                    "Content-Type": content_type,
                    "Content-Length": str(len(body)),
                },
            )

        self.assertEqual(status, 413)
        self.assertIn("size limit", response["error"]["message"])
        self.assertEqual(calls, [])

    def test_wrong_model_and_response_format_fail_loudly(self):
        calls = []
        cases = [
            ({"model": "wrong-model", "response_format": "json"}, "unknown or missing model"),
            ({"model": "gemini-stt", "response_format": "text"}, "response_format=json"),
        ]
        with RunningBridge(lambda path, fields: calls.append((path, fields))) as bridge:
            for fields, expected in cases:
                body, content_type = multipart_body(fields=fields)
                status, _, response = bridge.request(
                    "POST",
                    "/v1/audio/transcriptions",
                    body,
                    {
                        "Authorization": f"Bearer {TOKEN}",
                        "Content-Type": content_type,
                        "Content-Length": str(len(body)),
                    },
                )
                self.assertEqual(status, 400)
                self.assertIn(expected, response["error"]["message"])

        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
