# Gemini Speech to Terminal

Context-aware push-to-talk transcription for Linux terminals. It records speech, sends the audio and a compact summary of a chosen project to Gemini, prints the transcript, and can paste it into the focused X11 window.

The repository provides three commands:

- `gemini-stt`: record once (or transcribe an existing audio file) and print the result.
- `gemini-stt-toggle`: press F2 to start, press F2 again to stop, then paste the result at the cursor.
- `gemini-stt-bridge`: expose the same pipeline through a local OpenAI-compatible endpoint for Dictate Keyboard.

## Requirements

- Linux and a [Gemini API key](https://aistudio.google.com/app/apikey).
- `arecord` from ALSA for microphone recording. On Ubuntu/Debian: `sudo apt install alsa-utils`.
- An X11 session, Tk, and the XTest extension for the F2/paste mode. On Ubuntu/Debian, install `python3-tk`; Wayland-only sessions are not supported by that mode.
- Optional clipboard-only support for the one-shot command: `wl-copy`, `xclip`, or `xsel`.

## Install with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone and sync the locked environment:

```bash
git clone https://github.com/Tianchong-Jiang/stt.git ~/workspace/stt
cd ~/workspace/stt
uv sync --locked
uv run --locked gemini-stt --help
```

`pyproject.toml` declares the project and `uv.lock` pins its Python packages. uv creates the isolated environment at `.venv/` in this checkout.

## Install with Conda

```bash
git clone https://github.com/Tianchong-Jiang/stt.git ~/workspace/stt
cd ~/workspace/stt
conda env create -f environment.yml
conda activate gemini-stt
gemini-stt --help
```

`environment.yml` installs Python 3.12, Tk, and this checkout in editable mode. The uv and Conda environments provide the same three commands; use one environment, not both.

## Configure the API key

The commands first check `GEMINI_API_KEY`, then `GOOGLE_API_KEY`, then `.gemini_api_key` beside the source files. To create the ignored, owner-readable key file without putting the key in a command line:

```bash
cd ~/workspace/stt
read -rsp "Gemini API key: " GEMINI_API_KEY && echo
umask 077
printf '%s\n' "$GEMINI_API_KEY" > .gemini_api_key
unset GEMINI_API_KEY
```

## Use it

The examples below use uv. In the Conda environment, omit `uv run --locked`.

Record five seconds and print the transcript:

```bash
cd ~/workspace/stt
uv run --locked gemini-stt \
  --repo ~/workspace/my-project \
  --cwd ~/workspace/my-project
```

Change the recording length or ALSA device when needed:

```bash
uv run --locked gemini-stt \
  --duration 12 \
  --device plughw:CARD=Microphone,DEV=0 \
  --repo ~/workspace/my-project \
  --cwd ~/workspace/my-project
```

List ALSA input names with `arecord -L`. To transcribe an existing file without touching the microphone, pass `--audio recording.wav`. Add `--copy` to copy the result to the clipboard.

### F2 push-to-talk

Start the small status window and global hotkey listener:

```bash
cd ~/workspace/stt
uv run --locked gemini-stt-toggle --status-window \
  --repo ~/workspace/my-project \
  --cwd ~/workspace/my-project
```

Press F2 once to begin recording and again to stop. After transcription, the text is pasted at the cursor in the previously focused X11 window. The recorder has a five-minute hard limit.

For automatic startup, edit `stt_toggle.service` first: it assumes this repository is at `~/workspace/stt`, uses the uv environment, and deliberately contains `YOUR_PROJECT` placeholders. Then install it as a user service:

```bash
mkdir -p ~/.config/systemd/user
cp stt_toggle.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now stt_toggle.service
```

If you installed with Conda, replace the service's `ExecStart` executable with the absolute path printed by `command -v gemini-stt-toggle` while the Conda environment is active.

### Dictate Keyboard bridge

Run the bridge locally:

```bash
uv run --locked gemini-stt-bridge \
  --repo ~/workspace/my-project \
  --cwd ~/workspace/my-project
```

It listens on `http://127.0.0.1:8765/v1/`, advertises model `gemini-stt`, and creates an ignored bearer-token file at `.stt_bridge_token`. Configure Dictate Keyboard with that base URL, model name, and token. Keep the default loopback host unless you add transport security yourself.

## How project context works

There is no separate `contexts/` directory. A fresh context prompt is assembled in memory for every transcription:

1. `--repo` selects the project. If it is inside a Git repository, the Git root is used.
2. `--cwd` tells Gemini which project subdirectory is currently relevant.
3. The prompt includes the repository path, relative working directory, branch, short Git status, up to 80 recent shell-history lines, and up to 1,200 tracked or unignored file paths.
4. It also includes technical identifiers extracted from those paths and from up to 120 text/code/config files. File contents themselves are not copied into the prompt; only matching identifiers are retained.
5. The total generated context is capped at 40,000 characters by default.

This context helps Gemini choose spellings that ordinary dictation often misses: local filenames, mixed Chinese/English technical terms, acronyms, flags, class names, and variables. It is guidance, not permission to invent commands; the system instruction asks Gemini to transcribe literally when uncertain.

Inspect exactly what would be sent before recording or making an API request:

```bash
uv run --locked gemini-stt \
  --repo ~/workspace/my-project \
  --cwd ~/workspace/my-project \
  --prompt-dump
```

Choose `--repo` narrowly. The generated prompt and audio are sent to the Gemini API. Shell history and filenames can contain secrets even when source files do not. Review `--prompt-dump`, clean sensitive history, and read the [Gemini API terms](https://ai.google.dev/gemini-api/terms) before use. Local `.wav`, key, token, environment, and log files are ignored by Git; `stt.log` can contain up to a 1,000-character preview of each successful transcript. Some failed or timed-out toggle transcriptions retain their temporary audio under `/tmp` for diagnosis.

## Illustrative transcripts

These are synthetic examples of the intended behavior, not measured accuracy claims. Results still depend on the microphone, pronunciation, selected Gemini model, and whether the relevant spellings appear in the chosen project context.

1. Mixed Chinese and research terminology

   Spoken and intended transcript:

   ```text
   这个 policy 用 Plücker embedding 做 camera conditioning，然后跑 sim-to-real evaluation。
   ```

2. An exact shell command

   Spoken and intended transcript:

   ```text
   python run_policy.py --device cuda --seed 0
   ```

3. Geometry notation and robotics vocabulary

   Spoken and intended transcript:

   ```text
   把 camera extrinsics 从 world frame 转到 end-effector frame，再检查 SE(3) transform。
   ```

4. A deliberate correction that should not be silently discarded

   Spoken and intended transcript:

   ```text
   先跑 checkpoint 10000，等等，不是 10000，是 100000，然后用 CUDA_VISIBLE_DEVICES=0。
   ```

5. Repository-local names

   If paths such as `configs/openvla_sim2real.yaml` and identifiers such as `DiffusionPolicy` exist under `--repo`, the context gives Gemini those exact candidate spellings instead of relying only on its general vocabulary.

## Troubleshooting

1. `arecord: command not found`: install `alsa-utils`.
2. Wrong or silent microphone: run `arecord -L`, then pass one of its names with `--device`.
3. `No module named tkinter`: install Tk for the selected Python, or use the Conda setup, which includes `tk`.
4. F2 or automatic paste does not work: confirm the desktop session is X11 and that another application has not grabbed F2.
5. Authentication failure: check that `.gemini_api_key` contains one key, has mode `600`, or export `GEMINI_API_KEY`.
6. Unexpected spelling: confirm the desired path or identifier appears in `--prompt-dump` and that `--repo` points to the intended project.

## Test without recording or API access

```bash
uv run --locked python -m unittest -v
```

The tests use mocks and local temporary files; they do not record audio or contact Gemini.
