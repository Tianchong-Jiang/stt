#!/usr/bin/env python3
"""Toggle microphone recording, transcribe it, and type the result at focus."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk


HERE = Path(__file__).resolve().parent
TRANSCRIBER = HERE / "gemini_speech_to_terminal.py"
LOG_FILE = HERE / "stt.log"
STATE_FILE = Path("/tmp/stt_toggle_state.json")
STATUS_FILE = Path("/tmp/stt_toggle_status.json")
DEVICE = "plughw:CARD=Microphone,DEV=0"
RECORD_ARGS = ["arecord", "-q", "-f", "S16_LE", "-r", "16000", "-c", "1", "-D", DEVICE]
STOP_POSTROLL_SECONDS = 0.75


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(f"{timestamp} {message}\n")
    except OSError:
        pass


def trim_log() -> None:
    try:
        if LOG_FILE.exists() and LOG_FILE.stat().st_size > 1_000_000:
            data = LOG_FILE.read_text(encoding="utf-8", errors="replace")[-500_000:]
            LOG_FILE.write_text(data, encoding="utf-8")
    except OSError:
        pass


def set_status(text: str, color: str, ttl: float | None = None) -> None:
    payload = {"text": text, "color": color, "updated": time.time(), "ttl": ttl}
    STATUS_FILE.write_text(json.dumps(payload))


def status_window() -> None:
    threading.Thread(target=hotkey_loop, daemon=True).start()
    root = tk.Tk()
    root.title("STT")
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.geometry("+24+80")
    label = tk.Label(root, text="STT idle", bg="#222222", fg="white", padx=14, pady=8, font=("Sans", 13, "bold"))
    label.pack()

    def refresh() -> None:
        try:
            status = json.loads(STATUS_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            status = {"text": "STT idle", "color": "#222222", "updated": 0, "ttl": None}
        if status.get("ttl") is not None and time.time() - float(status.get("updated", 0)) > float(status["ttl"]):
            status = {"text": "STT idle", "color": "#222222"}
            STATUS_FILE.unlink(missing_ok=True)
        label.configure(text=str(status.get("text", "STT idle")), bg=str(status.get("color", "#222222")))
        root.after(200, refresh)

    refresh()
    root.mainloop()


def hotkey_loop() -> None:
    try:
        from Xlib import XK, X, display
    except Exception as exc:
        set_status(f"STT hotkey failed: {exc}", "#9b1c1c", ttl=8)
        return
    disp = display.Display()
    root = disp.screen().root
    keycode = disp.keysym_to_keycode(XK.string_to_keysym("F2"))
    if keycode == 0:
        set_status("STT hotkey failed: F2 keycode missing", "#9b1c1c", ttl=8)
        return
    masks = [
        0,
        X.LockMask,
        X.Mod2Mask,
        X.LockMask | X.Mod2Mask,
    ]
    grabbed = 0
    for mask in masks:
        try:
            root.grab_key(keycode, mask, True, X.GrabModeAsync, X.GrabModeAsync)
            grabbed += 1
        except Exception:
            pass
    if grabbed == 0:
        set_status("STT hotkey failed: F2 is already grabbed", "#9b1c1c", ttl=8)
        return
    disp.sync()
    last_press = 0.0
    while True:
        event = disp.next_event()
        if event.type != X.KeyPress or event.detail != keycode:
            continue
        now = time.time()
        if now - last_press < 0.35:
            continue
        last_press = now
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve())],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )


def ensure_status_window() -> None:
    marker = Path("/tmp/stt_toggle_status_window.pid")
    if marker.exists():
        try:
            pid = int(marker.read_text().strip())
        except ValueError:
            pid = -1
        if pid > 0 and pid_alive(pid):
            return
        marker.unlink(missing_ok=True)
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--status-window"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    marker.write_text(str(proc.pid))


def fail(message: str) -> None:
    log(f"FAIL {message}")
    set_status(f"STT failed: {message[:80]}", "#9b1c1c", ttl=6)
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def transcriber_error(stderr: str, returncode: int) -> str:
    for line in reversed(stderr.splitlines()):
        line = line.strip()
        if line.startswith("error: "):
            return line.removeprefix("error: ")[:240]
    for line in reversed(stderr.splitlines()):
        line = line.strip()
        if line:
            return line[:240]
    return f"exit code {returncode}"


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_state() -> dict[str, object] | None:
    if not STATE_FILE.exists():
        return None
    try:
        state = json.loads(STATE_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        STATE_FILE.unlink(missing_ok=True)
        fail("removed unreadable toggle state; press F2 again")
    if not isinstance(state.get("pid"), int) or not isinstance(state.get("audio"), str):
        STATE_FILE.unlink(missing_ok=True)
        fail("removed invalid toggle state; press F2 again")
    return state


def start_recording() -> None:
    trim_log()
    ensure_status_window()
    if not TRANSCRIBER.exists():
        fail(f"missing transcriber: {TRANSCRIBER}")
    log("start requested")
    audio = tempfile.NamedTemporaryFile(prefix="stt_toggle_", suffix=".wav", delete=False)
    audio.close()
    try:
        proc = subprocess.Popen(
            RECORD_ARGS + [audio.name],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        Path(audio.name).unlink(missing_ok=True)
        fail(f"could not start arecord: {exc}")
    time.sleep(0.15)
    if proc.poll() is not None:
        Path(audio.name).unlink(missing_ok=True)
        fail(f"arecord exited early with code {proc.returncode}")
    STATE_FILE.write_text(json.dumps({"pid": proc.pid, "audio": audio.name, "started": time.time()}))
    log(f"recording started pid={proc.pid} audio={audio.name}")
    set_status("STT recording - press F2 to stop", "#b42318")


def stop_recording(state: dict[str, object]) -> None:
    trim_log()
    pid = int(state["pid"])
    audio = Path(str(state["audio"]))
    log(f"stop requested pid={pid} audio={audio}")
    if not pid_alive(pid):
        STATE_FILE.unlink(missing_ok=True)
        audio.unlink(missing_ok=True)
        fail("recording process was already gone; stale state removed")
    set_status("STT finishing recording", "#9a6700")
    time.sleep(STOP_POSTROLL_SECONDS)
    os.kill(pid, signal.SIGTERM)
    for _ in range(30):
        if not pid_alive(pid):
            break
        time.sleep(0.05)
    if pid_alive(pid):
        os.kill(pid, signal.SIGKILL)
    STATE_FILE.unlink(missing_ok=True)
    if not audio.exists() or audio.stat().st_size <= 44:
        audio.unlink(missing_ok=True)
        fail("recorded audio file is empty")
    ensure_status_window()
    set_status("STT processing", "#9a6700")
    started = float(state.get("started", time.time()))
    log(f"recording stopped seconds={time.time() - started:.2f} bytes={audio.stat().st_size}")
    try:
        result = subprocess.run(
            [sys.executable, str(TRANSCRIBER), "--audio", str(audio), "--status-file", str(STATUS_FILE)],
            cwd=HERE,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        log(f"kept audio after transcriber timeout: {audio}")
        fail("transcriber timed out")
    if result.stderr.strip():
        log("transcriber stderr: " + result.stderr.strip().replace("\n", "\\n")[:2000])
    if result.returncode != 0:
        log(f"kept audio after transcriber failure: {audio}")
        fail(transcriber_error(result.stderr, result.returncode))
    text = result.stdout.strip()
    if not text:
        fail("Gemini returned empty text")
    if text == "[NO_SPEECH]":
        fail("no intelligible speech detected")
    preview = text if len(text) <= 1000 else text[:1000] + " ...[log preview truncated]"
    log(f"transcribed chars={len(text)} text={preview!r}")
    type_text(text)
    audio.unlink(missing_ok=True)
    log("inserted text")
    set_status(f"STT inserted {len(text)} chars", "#067647", ttl=2)


def type_text(text: str) -> None:
    try:
        subprocess.run(
            ["xdotool", "type", "--clearmodifiers", "--delay", "0", "--", text],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        fail(f"could not run xdotool: {exc}")
    except subprocess.CalledProcessError as exc:
        fail(f"xdotool failed: {(exc.stderr or '').strip()[:200]}")


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "--status-window":
        status_window()
        return
    state = read_state()
    if state is None:
        start_recording()
    else:
        stop_recording(state)


if __name__ == "__main__":
    main()
