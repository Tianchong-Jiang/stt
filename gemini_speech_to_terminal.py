#!/usr/bin/env python3
"""Push-to-talk terminal dictation using Gemini audio input plus repo context."""

from __future__ import annotations

import argparse
import base64
import http.client
import itertools
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import struct
import sys
import tempfile
import time
import wave


DEFAULT_MODEL = "gemini-3.8-flash,gemini-3.5-flash"
DEFAULT_REPO = Path.cwd()
DEFAULT_CWD = DEFAULT_REPO
DEFAULT_DEVICE = "default"
DEFAULT_DURATION = 5
DEFAULT_API_KEY_FILE = Path(__file__).resolve().with_name(".gemini_api_key")
DEFAULT_MAX_OUTPUT_TOKENS = 2048
GEMINI_CONNECT_ATTEMPT_SECONDS = 2.0
GEMINI_CONNECT_BUDGET_SECONDS = 6.0
GEMINI_RETRY_SPACING_SECONDS = 5.0
BEGIN_MARKER = "[BEGIN_TRANSCRIPT]"
END_MARKER = "[END_OF_TRANSCRIPT]"
TRANSCRIPTION_INSTRUCTION = f"""Transcribe the full audio into cursor-ready text, omitting only the fillers described below.
Return only the transcript: no explanation, Markdown, quotes, code fence, or Enter instruction.
Do not use Traditional Chinese characters. Use Simplified Chinese for all Chinese text and preserve spoken English.
Omit non-semantic hesitation fillers such as "uh", "um", "er", and "ah".
Preserve meaningful interjections, deliberate repetition, corrections, and softly spoken trailing words.
Preserve capitalization and punctuation of technical terms.
Use the context for names, paths, flags, variables, and explicit commands, but never invent them or turn a natural-language request into a command.
When ambiguous, prefer the safest literal transcription.
Output {BEGIN_MARKER}, then the complete transcript on a single line with no line breaks, then {END_MARKER}. Use each marker exactly once.
For no intelligible speech, put [NO_SPEECH] between the markers."""
SKIP_DIRS = set(".git .mypy_cache .playwright-mcp .pytest_cache .ruff_cache .venv __pycache__ build dist node_modules wandb".split())
TERM_SUFFIXES = set(".bash .cfg .ini .json .py .sh .toml .yaml .yml".split())
STATIC_TERMS = set(
    "CUDA_VISIBLE_DEVICES PYTHONPATH pytest python sbatch squeue srun tmux git grep rg tail wandb Hydra MuJoCo "
    "robosuite LIBERO CALVIN RoboCasa OpenVLA ACTPolicy DiffusionPolicy confirmed".split()
) | {"update .md"}


def die(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


class GeminiRequestError(Exception):
    """A Gemini request failed, but another configured model may still work."""


class GeminiTransientError(GeminiRequestError):
    """The same request may succeed when sent again, such as after a timeout or a mistaken filter block."""


class GeminiFormatError(GeminiRequestError):
    """The response was not a single-line transcript between the markers; the same model may comply when asked again."""


def run(cmd: list[str], cwd: Path | None = None, timeout: float = 5.0) -> str | None:
    try:
        result = subprocess.run(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=timeout
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def repo_root(path: Path) -> Path:
    out = run(["git", "rev-parse", "--show-toplevel"], cwd=path)
    if out:
        return Path(out.strip()).resolve()
    return path.resolve()


def collect_files(root: Path, limit: int) -> list[str]:
    out = run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], root, 10)
    if out is not None:
        paths = [p for p in out.split("\0") if p and not SKIP_DIRS.intersection(Path(p).parts[:-1])]
    else:
        paths = [
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and not SKIP_DIRS.intersection(path.relative_to(root).parts)
        ]
    return sorted(set(paths), key=lambda path: (path.count("/"), path))[:limit]


def extract_terms(root: Path, files: list[str], limit: int) -> list[str]:
    terms: set[str] = set(STATIC_TERMS)
    ident = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
    for rel in files:
        terms.update(part for part in Path(rel).parts if 3 <= len(part) <= 80)
        terms.update(piece for piece in re.split(r"[^A-Za-z0-9_]+", rel) if len(piece) >= 3)
    scanned = 0
    for rel in files:
        path = root / rel
        if path.suffix not in TERM_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text(errors="ignore")[:80_000]
        except OSError:
            continue
        for token in ident.findall(text):
            if "_" in token or token.isupper() or any(c.isupper() for c in token[1:]):
                terms.add(token)
        scanned += 1
        if scanned >= 120:
            break
    return sorted(terms, key=lambda term: (term.lower(), term))[:limit]


def recent_history(limit: int) -> list[str]:
    for candidate in (os.environ.get("HISTFILE"), Path.home() / ".bash_history", Path.home() / ".zsh_history"):
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if not path.is_file():
            continue
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except OSError:
            continue
        return [line.strip() for line in lines if line.strip()][-limit:]
    return []


def build_prompt(root: Path, cwd: Path, max_files: int, max_terms: int, max_context_chars: int) -> str:
    files = collect_files(root, max_files)
    terms = extract_terms(root, files, max_terms)
    status = run(["git", "status", "--short"], cwd=root) or ""
    branch = run(["git", "branch", "--show-current"], cwd=root) or ""
    try:
        rel_cwd = cwd.resolve().relative_to(root).as_posix()
    except ValueError:
        rel_cwd = cwd.resolve().as_posix()

    context = f"""Repo: {root}
CWD: {rel_cwd}
Branch: {branch.strip() or "(unknown)"}
Git status:
{status.strip() or "(clean or not a git repo)"}
Recent shell commands:
{chr(10).join(recent_history(80)) or "(none found)"}
Relevant file paths:
{chr(10).join(files)}
Relevant identifiers and technical terms:
{chr(10).join(terms)}"""
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n[context truncated]"

    return f"Repository context:\n{context}"


def record_audio(path: Path, duration: int, device: str | None) -> None:
    arecord = shutil.which("arecord")
    if not arecord:
        die("cannot record: expected `arecord` on PATH, or pass --audio existing.wav")
    cmd = [
        arecord,
        "-q",
        "-f",
        "S16_LE",
        "-r",
        "16000",
        "-c",
        "1",
        "-d",
        str(duration),
    ]
    if device:
        cmd.extend(["-D", device])
    cmd.append(str(path))
    source = device or "default microphone"
    print(f"recording {duration:g}s from {source}...", file=sys.stderr)
    subprocess.run(cmd, check=True)


def mime_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path)
    return guessed or "audio/wav"


def pcm16_wav_rms(path: Path, max_bytes: int = 4_000_000) -> int | None:
    try:
        with wave.open(str(path), "rb") as wav:
            if wav.getsampwidth() != 2:
                return None
            frame_width = wav.getsampwidth() * wav.getnchannels()
            frames = min(wav.getnframes(), max_bytes // max(1, frame_width))
            data = wav.readframes(frames)
    except (OSError, wave.Error):
        return None
    sample_count = len(data) // 2
    if sample_count == 0:
        return 0
    samples = struct.unpack(f"<{sample_count}h", data[: sample_count * 2])
    return int(math.sqrt(sum(sample * sample for sample in samples) / sample_count))


def summarize_http_error(code: int, body: str, model: str) -> str:
    try:
        data = json.loads(body)
        error = data.get("error", {})
        message = str(error.get("message", "")).splitlines()[0]
    except json.JSONDecodeError:
        message = body.strip().splitlines()[0] if body.strip() else "(empty response body)"
    return f"Gemini API HTTP {code} on {model}: {message[:500]}"


def connect_with_retry(address, timeout, source_address=None):
    host, port = address
    deadline = time.monotonic() + timeout
    last_error = None
    addresses = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
    for family, socktype, proto, _, sockaddr in addresses * 2:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        sock = socket.socket(family, socktype, proto)
        try:
            sock.settimeout(min(GEMINI_CONNECT_ATTEMPT_SECONDS, remaining))
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)
            sock.settimeout(max(0.001, deadline - time.monotonic()))
            return sock
        except OSError as exc:
            last_error = exc
            sock.close()
    raise TimeoutError(f"connection to {host}:{port} failed within {timeout:g}s") from last_error


def generation_config(max_output_tokens: int) -> dict[str, object]:
    return {
        "max_output_tokens": max_output_tokens,
        "thinking_level": "low",
    }


def parse_transcript_response(text: str, model: str) -> str:
    """Take the transcript between the markers, discarding any reasoning the model emitted before it."""
    text = clean_answer(text)
    if not text.endswith(END_MARKER):
        raise GeminiRequestError(f"Gemini response on {model} omitted {END_MARKER}; transcript may be truncated")
    text = text[: -len(END_MARKER)].rstrip()
    start = text.find(BEGIN_MARKER)
    if start < 0:
        raise GeminiFormatError(f"Gemini response on {model} omitted {BEGIN_MARKER}")
    text = text[start + len(BEGIN_MARKER) :].strip()
    if BEGIN_MARKER in text or END_MARKER in text:
        raise GeminiFormatError(f"Gemini response on {model} repeated a transcript marker")
    if "\n" in text:
        raise GeminiFormatError(f"Gemini response on {model} spans {text.count(chr(10)) + 1} lines; expected one line")
    return text


def call_gemini(api_key: str, model: str, prompt: str, audio_path: Path, max_output_tokens: int) -> str:
    audio_bytes = audio_path.read_bytes()
    payload = {
        "model": model,
        "system_instruction": TRANSCRIPTION_INSTRUCTION,
        "input": [
            {"type": "text", "text": prompt},
            {
                "type": "audio",
                "mime_type": mime_type(audio_path),
                "data": base64.b64encode(audio_bytes).decode("ascii"),
            }
        ],
        "generation_config": generation_config(max_output_tokens),
        "service_tier": "priority",
        "store": False,
    }
    host = "generativelanguage.googleapis.com"
    connection = http.client.HTTPSConnection(host, timeout=GEMINI_CONNECT_BUDGET_SECONDS)
    connection._create_connection = connect_with_retry
    started = time.monotonic()
    try:
        connection.connect()
        connected = time.monotonic()
        if connection.sock is None:
            raise OSError("HTTPS connection returned no socket")
        connection.sock.settimeout(5)
        connection.request(
            "POST",
            "/v1beta/interactions",
            json.dumps(payload).encode("utf-8"),
            {"Content-Type": "application/json", "x-goog-api-key": api_key},
        )
        sent = time.monotonic()
        response = connection.getresponse()
        received = time.monotonic()
        body = response.read()
        print(
            f"timing: model={model} connect={connected - started:.3f}s upload={sent - connected:.3f}s "
            f"wait={received - sent:.3f}s read={time.monotonic() - received:.3f}s "
            f"server_timing={response.getheader('server-timing')}",
            file=sys.stderr, flush=True,
        )
    except (OSError, http.client.HTTPException) as exc:
        raise GeminiTransientError(f"Gemini API request failed on {model} after {time.monotonic() - started:.3f}s: {exc}") from exc
    finally:
        connection.close()
    if response.status >= 400:
        message = summarize_http_error(response.status, body.decode("utf-8", errors="replace"), model)
        # HTTP 429 is not transient here: one person dictating never reaches the paid per-minute limits, so it means usage is used up.
        transient = response.status >= 500 or "Input blocked" in message
        raise (GeminiTransientError if transient else GeminiRequestError)(message)
    data = json.loads(body.decode("utf-8"))

    tier, status = data["service_tier"], data["status"]
    print(f"info: model={model} serviceTier={tier} status={status}", file=sys.stderr, flush=True)
    if tier != "priority":
        raise GeminiRequestError(f"Gemini served {model} on {tier}; priority is required")
    if status != "completed":
        raise GeminiRequestError(f"Gemini status={status} on {model}; transcript is not complete")
    usage = data["usage"]
    print(
        f"info: promptTokens={usage['total_input_tokens']} candidateTokens={usage['total_output_tokens']} "
        f"thoughtTokens={usage['total_thought_tokens']} totalTokens={usage['total_tokens']}",
        file=sys.stderr,
    )
    # Gemini omits "steps" when it generated nothing at all.
    parts = [
        part["text"]
        for step in data.get("steps", []) if step["type"] == "model_output"
        for part in step["content"] if part["type"] == "text"
    ]
    if not parts:
        raise GeminiTransientError(f"Gemini returned no transcript text on {model}")
    return parse_transcript_response("\n".join(parts), model)


def parse_models(value: str) -> list[str]:
    models = list(dict.fromkeys(model for item in value.split(",") if (model := item.strip())))
    if not models:
        die("at least one Gemini model is required")
    return models


def model_label(model: str) -> str:
    if model.startswith("gemini-"):
        model = "Gemini " + model.removeprefix("gemini-")
    return model.replace("-", " ").title()


def write_status(path: Path | None, text: str, color: str, ttl: float | None = None) -> None:
    if path is None:
        return
    payload = {"text": text, "color": color, "updated": time.time(), "ttl": ttl}
    try:
        path.write_text(json.dumps(payload))
    except OSError:
        pass


def call_gemini_again_if_malformed(
    api_key: str, model: str, prompt: str, audio_path: Path, max_output_tokens: int, status_file: Path | None
) -> str:
    try:
        return call_gemini(api_key, model, prompt, audio_path, max_output_tokens)
    except GeminiFormatError as exc:
        write_status(status_file, f"STT {model_label(model)} malformed transcript - running again", "#9a6700")
        print(f"warning: {exc}; running {model} again", file=sys.stderr, flush=True)
    retry_prompt = prompt + (
        "\n\nThe previous attempt was not a single-line transcript between the markers. "
        f"Keep any analysis before {BEGIN_MARKER}; between the markers return only the final "
        "transcribed text, on one line, transcribing the full audio without summarizing."
    )
    return call_gemini(api_key, model, retry_prompt, audio_path, max_output_tokens)


def call_gemini_with_fallback(
    api_key: str,
    models: list[str],
    prompt: str,
    audio_path: Path,
    max_output_tokens: int,
    status_file: Path | None = None,
) -> str:
    """Rotate through the models until one returns a transcript; only an error a resend cannot fix drops a model.

    The F2 toggle cancels by killing this process."""
    pending, errors = list(models), []
    write_status(status_file, f"STT processing with {model_label(pending[0])}", "#9a6700")
    for attempt in itertools.count(2):
        model, started = pending.pop(0), time.monotonic()
        try:
            return call_gemini_again_if_malformed(api_key, model, prompt, audio_path, max_output_tokens, status_file)
        except (GeminiTransientError, GeminiFormatError) as exc:
            message = str(exc)
            pending.append(model)
        except GeminiRequestError as exc:
            message = str(exc)
            errors.append(message)
            if not pending:
                raise GeminiRequestError("all Gemini models failed: " + " | ".join(errors)) from exc
        write_status(
            status_file,
            f"STT {model_label(model)} unavailable - trying {model_label(pending[0])} (try {attempt}, F2 cancels)",
            "#9a6700",
        )
        print(f"warning: {message}; trying {pending[0]}", file=sys.stderr, flush=True)
        # Fast failures such as being offline would otherwise resend the audio in a tight loop.
        time.sleep(max(0.0, GEMINI_RETRY_SPACING_SECONDS - (time.monotonic() - started)))


def transcribe_audio_file(
    api_key: str,
    models: list[str],
    prompt: str,
    audio_path: Path,
    max_output_tokens: int,
    min_rms: int = 100,
    status_file: Path | None = None,
) -> str:
    """Transcribe an existing audio file through the shared Gemini pipeline."""
    rms = pcm16_wav_rms(audio_path)
    if min_rms > 0 and rms is not None and rms < min_rms:
        raise GeminiRequestError(f"audio too quiet (rms {rms} < {min_rms}); no intelligible speech detected")
    return call_gemini_with_fallback(
        api_key,
        models,
        prompt,
        audio_path,
        max_output_tokens,
        status_file,
    )


def clean_answer(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if text.startswith(("Bash\n", "bash\n", "Shell\n", "shell\n")):
        text = "\n".join(text.splitlines()[1:]).strip()
    return text


def copy_to_clipboard(text: str) -> None:
    for cmd in (["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]):
        if not shutil.which(cmd[0]):
            continue
        subprocess.run(cmd, input=text, text=True, check=True)
        print("copied to clipboard", file=sys.stderr)
        return
    die("no clipboard tool found; install wl-copy, xclip, or xsel, or omit --copy")


def load_api_key(path: Path) -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key.strip()
    if not path.exists():
        die(f"set GEMINI_API_KEY, GOOGLE_API_KEY, or create {path}")
    try:
        key = path.read_text().strip()
    except OSError as exc:
        die(f"could not read API key file {path}: {exc}")
    if not key:
        die(f"API key file is empty: {path}")
    if re.search(r"\s", key):
        die(f"API key file should contain exactly one key and no whitespace: {path}")
    return key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record mic audio, send repo context to Gemini, and print exact terminal text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--audio", type=Path, help="existing audio file; skips microphone recording")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION, help="recording duration in seconds")
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="ALSA input device, for example plughw:CARD=Microphone,DEV=0",
    )
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO, help="repo or project path for context")
    parser.add_argument("--cwd", type=Path, default=DEFAULT_CWD, help="current working directory context")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model, or comma-separated fallback list")
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_API_KEY_FILE, help="local API key file")
    parser.add_argument("--max-files", type=int, default=1200, help="maximum repo file paths in context")
    parser.add_argument("--max-terms", type=int, default=1000, help="maximum identifiers/terms in context")
    parser.add_argument("--max-context-chars", type=int, default=40_000, help="maximum prompt context characters")
    parser.add_argument("--prompt-dump", action="store_true", help="print prompt and exit without API call")
    parser.add_argument("--copy", action="store_true", help="copy output to clipboard")
    parser.add_argument("--keep-audio", action="store_true", help="keep temporary recorded audio")
    parser.add_argument("--status-file", type=Path, help="optional JSON status file for a floating UI")
    parser.add_argument("--min-rms", type=int, default=100, help="minimum RMS for 16-bit WAV audio; 0 disables")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, help="Gemini output token cap")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    cwd = args.cwd.expanduser().resolve()
    if not repo.is_dir():
        die(f"repo context is not a directory: {repo}")
    if not cwd.is_dir():
        die(f"working-directory context is not a directory: {cwd}")
    started = time.monotonic()
    root = repo_root(repo)
    prompt = build_prompt(root, cwd, args.max_files, args.max_terms, args.max_context_chars)
    print(f"timing: context={time.monotonic() - started:.3f}s", file=sys.stderr, flush=True)
    if args.prompt_dump:
        print(f"{TRANSCRIPTION_INSTRUCTION}\n\n{prompt}")
        return

    api_key = load_api_key(args.api_key_file.expanduser())

    temp_path: Path | None = None
    if args.audio:
        audio_path = args.audio.expanduser().resolve()
        if not audio_path.is_file():
            die(f"audio file does not exist: {audio_path}")
    else:
        tmp = tempfile.NamedTemporaryFile(prefix="gemini_speech_", suffix=".wav", delete=False)
        tmp.close()
        temp_path = Path(tmp.name)
        audio_path = temp_path
        try:
            record_audio(audio_path, args.duration, args.device)
        except subprocess.CalledProcessError as exc:
            die(f"recording failed with exit code {exc.returncode}")

    try:
        status_file = args.status_file.expanduser() if args.status_file else None
        answer = transcribe_audio_file(
            api_key,
            parse_models(args.model),
            prompt,
            audio_path,
            args.max_output_tokens,
            args.min_rms,
            status_file,
        )
    except GeminiRequestError as exc:
        die(str(exc))
    finally:
        if temp_path and not args.keep_audio:
            temp_path.unlink(missing_ok=True)

    print(answer)
    if args.copy:
        copy_to_clipboard(answer)


if __name__ == "__main__":
    main()
