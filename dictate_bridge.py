#!/usr/bin/env python3
"""Authenticated OpenAI-compatible bridge for Dictate Keyboard."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
import json
import os
from pathlib import Path
import re
import secrets
import stat
import sys
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Mapping
from urllib.parse import urlsplit

import gemini_speech_to_terminal as gemini_stt


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_MODEL_ID = "gemini-stt"
DEFAULT_TOKEN_FILE = Path(__file__).resolve().with_name(".stt_bridge_token")
DEFAULT_MAX_UPLOAD_MIB = 14
DEFAULT_MAX_UPLOAD_BYTES = DEFAULT_MAX_UPLOAD_MIB * 1024 * 1024
MAX_FORM_FIELD_BYTES = 8 * 1024


class RequestError(Exception):
    def __init__(self, status: HTTPStatus, message: str):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class BridgeConfig:
    token: str
    model_id: str = DEFAULT_MODEL_ID
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    request_timeout: float = 120.0


@dataclass(frozen=True)
class MultipartUpload:
    filename: str
    content_type: str
    data: bytes
    fields: dict[str, str]


def load_or_create_token(path: Path) -> tuple[str, bool]:
    path = path.expanduser().absolute()
    if not path.parent.is_dir():
        raise ValueError(f"token-file parent is not a directory: {path.parent}")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, os.O_RDONLY | nofollow)
    except FileNotFoundError:
        token = secrets.token_urlsafe(32)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow, 0o600)
        except FileExistsError:
            token, _ = load_or_create_token(path)
            return token, False
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as token_file:
                token_file.write(token + "\n")
                token_file.flush()
                os.fsync(token_file.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return token, True
    with os.fdopen(fd, "r", encoding="utf-8") as token_file:
        metadata = os.fstat(token_file.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"token file is not a regular file: {path}")
        if metadata.st_uid != os.getuid():
            raise ValueError(f"token file is not owned by the current user: {path}")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise ValueError(f"token file permissions are too broad; run: chmod 600 {path}")
        token = token_file.read(4097)
    if len(token) > 4096:
        raise ValueError(f"token file is unexpectedly large: {path}")
    token = token.strip()
    if not token or any(char.isspace() for char in token):
        raise ValueError(f"token file must contain exactly one non-whitespace token: {path}")
    return token, False


def parse_multipart(content_type: str, body: bytes) -> MultipartUpload:
    try:
        header = content_type.encode("ascii", errors="strict")
    except UnicodeEncodeError as exc:
        raise RequestError(HTTPStatus.BAD_REQUEST, "invalid Content-Type header") from exc
    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + header + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
    )
    if message.get_content_type() != "multipart/form-data" or not message.is_multipart():
        raise RequestError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "expected multipart/form-data")

    fields: dict[str, str] = {}
    upload: tuple[str, str, bytes] | None = None
    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue
        name = part.get_param("name", header="content-disposition")
        if not name:
            raise RequestError(HTTPStatus.BAD_REQUEST, "multipart part is missing a name")
        payload = part.get_payload(decode=True) or b""
        if name == "file":
            if upload is not None:
                raise RequestError(HTTPStatus.BAD_REQUEST, "multipart request has multiple file parts")
            upload = (part.get_filename() or "audio.wav", part.get_content_type(), payload)
            continue
        if name in fields:
            raise RequestError(HTTPStatus.BAD_REQUEST, f"duplicate multipart field: {name}")
        if len(payload) > MAX_FORM_FIELD_BYTES:
            raise RequestError(HTTPStatus.BAD_REQUEST, f"multipart field is too large: {name}")
        try:
            fields[name] = payload.decode(part.get_content_charset() or "utf-8", errors="strict")
        except (LookupError, UnicodeDecodeError) as exc:
            raise RequestError(HTTPStatus.BAD_REQUEST, f"multipart field is not valid text: {name}") from exc

    if upload is None:
        raise RequestError(HTTPStatus.BAD_REQUEST, "multipart request is missing the file part")
    if not upload[2]:
        raise RequestError(HTTPStatus.BAD_REQUEST, "uploaded audio file is empty")
    return MultipartUpload(*upload, fields)


def safe_audio_suffix(filename: str, content_type: str) -> str:
    suffix = Path(filename).suffix.lower()
    if re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
        return suffix
    return {
        "audio/flac": ".flac",
        "audio/m4a": ".m4a",
        "audio/mp4": ".m4a",
        "audio/mpeg": ".mp3",
        "audio/ogg": ".ogg",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/webm": ".webm",
    }.get(content_type.lower(), ".wav")


def gemini_pipeline(args, api_key: str, repo: Path, cwd: Path):
    models = gemini_stt.parse_models(args.gemini_model)

    def transcribe(audio_path: Path, _fields: Mapping[str, str]) -> str:
        return gemini_stt.transcribe_audio_file(
            api_key,
            models,
            gemini_stt.build_prompt(repo, cwd, args.max_files, args.max_terms, args.max_context_chars),
            audio_path,
            args.max_output_tokens,
            args.min_rms,
        )

    return transcribe


class DictateBridgeServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        config: BridgeConfig,
        transcribe: Callable[[Path, Mapping[str, str]], str],
    ) -> None:
        self.bridge_config = config
        self.transcribe = transcribe
        super().__init__(server_address, DictateBridgeHandler)


class DictateBridgeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: DictateBridgeServer

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.server.bridge_config.request_timeout)

    def log_message(self, _format: str, *args: object) -> None:
        # Never let the default server logger capture request data or credentials.
        return

    def _send_json(self, status: HTTPStatus, value: object, *, authenticate: bool = False) -> None:
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        if authenticate:
            self.send_header("WWW-Authenticate", 'Bearer realm="dictate-bridge"')
        self.end_headers()
        self.close_connection = True
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_error_json(self, status: HTTPStatus, message: str, *, authenticate: bool = False) -> None:
        self._send_json(status, {"error": {"message": message}}, authenticate=authenticate)

    def _require_auth(self) -> bool:
        expected = f"Bearer {self.server.bridge_config.token}"
        if secrets.compare_digest(self.headers.get("Authorization", ""), expected):
            return True
        self._send_error_json(HTTPStatus.UNAUTHORIZED, "invalid bearer token", authenticate=True)
        return False

    def _content_length(self) -> int:
        if self.headers.get("Transfer-Encoding"):
            raise RequestError(HTTPStatus.BAD_REQUEST, "chunked uploads are not supported")
        values = self.headers.get_all("Content-Length", [])
        if len(values) != 1:
            raise RequestError(HTTPStatus.LENGTH_REQUIRED, "exactly one Content-Length header is required")
        try:
            length = int(values[0])
        except ValueError as exc:
            raise RequestError(HTTPStatus.BAD_REQUEST, "invalid Content-Length header") from exc
        if length < 0:
            raise RequestError(HTTPStatus.BAD_REQUEST, "invalid Content-Length header")
        if length > self.server.bridge_config.max_upload_bytes:
            raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "upload exceeds the configured size limit")
        return length

    def handle_expect_100(self) -> bool:
        if not self._require_auth():
            return False
        try:
            self._content_length()
        except RequestError as exc:
            self._send_error_json(exc.status, str(exc))
            return False
        self.send_response_only(HTTPStatus.CONTINUE.value)
        self.end_headers()
        return True

    def do_GET(self) -> None:
        if not self._require_auth():
            return
        if urlsplit(self.path).path != "/v1/models":
            self._send_error_json(HTTPStatus.NOT_FOUND, "endpoint not found")
            return
        self._send_json(
            HTTPStatus.OK,
            {
                "object": "list",
                "data": [
                    {
                        "id": self.server.bridge_config.model_id,
                        "object": "model",
                        "owned_by": "local",
                        "architecture": {"input_modalities": ["audio"], "output_modalities": ["transcription"]},
                    }
                ],
            },
        )

    def do_POST(self) -> None:
        if not self._require_auth():
            return
        if urlsplit(self.path).path != "/v1/audio/transcriptions":
            self._send_error_json(HTTPStatus.NOT_FOUND, "endpoint not found")
            return
        try:
            length = self._content_length()
            body = self.rfile.read(length)
            if len(body) != length:
                raise RequestError(HTTPStatus.BAD_REQUEST, "request body ended before Content-Length")
            upload = parse_multipart(self.headers.get("Content-Type", ""), body)
            if upload.fields.get("model") != self.server.bridge_config.model_id:
                raise RequestError(HTTPStatus.BAD_REQUEST, "unknown or missing model")
            if upload.fields.get("response_format", "json") != "json":
                raise RequestError(HTTPStatus.BAD_REQUEST, "only response_format=json is supported")
        except RequestError as exc:
            self._send_error_json(exc.status, str(exc))
            return
        except TimeoutError:
            self._send_error_json(HTTPStatus.REQUEST_TIMEOUT, "request upload timed out")
            return

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="dictate_bridge_",
                suffix=safe_audio_suffix(upload.filename, upload.content_type),
                delete=False,
            ) as audio_file:
                audio_file.write(upload.data)
                temp_path = Path(audio_file.name)
            transcript = self.server.transcribe(temp_path, upload.fields)
            if not transcript:
                raise gemini_stt.GeminiRequestError("transcriber returned empty text")
        except gemini_stt.GeminiRequestError:
            print("error: Dictate transcription failed", file=sys.stderr, flush=True)
            self._send_error_json(HTTPStatus.BAD_GATEWAY, "transcription failed")
            return
        except Exception as exc:
            print(f"error: Dictate bridge failed with {type(exc).__name__}", file=sys.stderr, flush=True)
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, "internal bridge error")
            return
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

        self._send_json(HTTPStatus.OK, {"text": transcript})


def bounded_int(minimum: int, maximum: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        number = int(value)
        if number < minimum or number > maximum:
            raise argparse.ArgumentTypeError(f"expected an integer from {minimum} to {maximum}")
        return number

    return parse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the Gemini STT pipeline through Dictate's OpenAI-compatible API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="address to bind")
    parser.add_argument("--port", type=bounded_int(0, 65535), default=DEFAULT_PORT, help="TCP port")
    parser.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_FILE, help="private bearer-token file")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="model id advertised to Dictate")
    parser.add_argument("--gemini-model", default=gemini_stt.DEFAULT_MODEL, help="Gemini model fallback list")
    parser.add_argument("--api-key-file", type=Path, default=gemini_stt.DEFAULT_API_KEY_FILE)
    parser.add_argument("--repo", type=Path, default=gemini_stt.DEFAULT_REPO, help="repo or project context")
    parser.add_argument("--cwd", type=Path, default=gemini_stt.DEFAULT_CWD, help="working-directory context")
    parser.add_argument(
        "--max-upload-mib",
        type=bounded_int(1, 1024),
        default=DEFAULT_MAX_UPLOAD_MIB,
        help="maximum total multipart upload size in MiB",
    )
    parser.add_argument("--request-timeout", type=bounded_int(1, 3600), default=120, help="upload timeout seconds")
    parser.add_argument("--max-files", type=bounded_int(1, 100_000), default=1200)
    parser.add_argument("--max-terms", type=bounded_int(1, 100_000), default=1000)
    parser.add_argument("--max-context-chars", type=bounded_int(1, 1_000_000), default=40_000)
    parser.add_argument("--max-output-tokens", type=bounded_int(1, 1_000_000), default=gemini_stt.DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--min-rms", type=bounded_int(0, 32767), default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.host.strip():
        gemini_stt.die("--host must not be empty")
    if not args.model_id.strip() or len(args.model_id) > 200 or any(char.isspace() for char in args.model_id):
        gemini_stt.die("--model-id must be a non-whitespace value of at most 200 characters")
    repo = args.repo.expanduser().resolve()
    cwd = args.cwd.expanduser().resolve()
    if not repo.is_dir():
        gemini_stt.die(f"repo context is not a directory: {repo}")
    if not cwd.is_dir():
        gemini_stt.die(f"working-directory context is not a directory: {cwd}")

    try:
        token, created = load_or_create_token(args.token_file)
    except (OSError, ValueError) as exc:
        gemini_stt.die(str(exc))
    api_key = gemini_stt.load_api_key(args.api_key_file.expanduser())
    pipeline = gemini_pipeline(args, api_key, gemini_stt.repo_root(repo), cwd)
    config = BridgeConfig(
        token=token,
        model_id=args.model_id,
        max_upload_bytes=args.max_upload_mib * 1024 * 1024,
        request_timeout=float(args.request_timeout),
    )
    try:
        server = DictateBridgeServer((args.host, args.port), config, pipeline)
    except OSError as exc:
        gemini_stt.die(f"could not bind {args.host}:{args.port}: {exc}")

    _, actual_port = server.server_address[:2]
    action = "created" if created else "using"
    token_path = args.token_file.expanduser().absolute()
    print(f"Dictate bridge listening on http://{args.host}:{actual_port}/v1/", file=sys.stderr)
    print(f"{action} private bearer token file: {token_path}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
