# Gemini Speech to Terminal

Push-to-talk speech transcription for Linux/X11 terminals using Gemini audio input plus local repository context.

## Run

Put a Gemini API key in a local file that is not committed:

```bash
printf '%s\n' 'YOUR_KEY_HERE' > .gemini_api_key
chmod 600 .gemini_api_key
```

Single 5-second recording:

```bash
python gemini_speech_to_terminal.py
```

Floating F2 toggle window:

```bash
python stt_toggle.py --status-window
```

The default model chain is `gemini-3.5-flash,gemini-3.1-flash-lite,gemini-2.5-flash`.
