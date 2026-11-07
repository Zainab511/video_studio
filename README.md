# Video Studio

A small local web app that turns a script into an explainer video. Write your own script, or let [Groq](https://console.groq.com) (free tier) draft one from a topic. The video is built on your own machine with MoviePy and saved so you can watch or download it.

No paid services are needed. Groq is only used for writing text.

## Features

- Generate a scene-by-scene script from a topic with Groq, then edit it freely
- Or paste your own script (structured, plain paragraphs, or JSON)
- Background rendering with a live progress bar
- Built-in player, MP4 download, and a saved library of past videos
- Optional free local voiceover with [Piper](https://github.com/OHF-Voice/piper1-gpl)
- Command-line mode if you prefer the terminal

## Requirements

- Python 3.10 or newer
- A free Groq API key (only for script generation): https://console.groq.com

## Setup

```bash
git clone <your-repo-url>
cd video-studio

python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file (copy `.env.example`) and fill it in:

```
GROQ_API_KEY=your_key_here
GROQ_MODEL=qwen/qwen3.8-27b
```

Check the exact model name at https://console.groq.com/docs/models, since Groq changes its model list over time.

## Run the web app

```bash
python app.py
```

Open http://127.0.0.1:5000 and:

1. Enter a topic and click **Generate**, or paste your own script
2. Edit the script if you like
3. Click **Build video**
4. Watch it in the preview, or click **Download MP4**

Finished videos are stored in `videos/` and appear in the Library list.

## Script format

Structured (full control over each slide):

```
# Why use Git?
- Tracks every change
- Lets teams work together safely
Git keeps a full history of your code. If something breaks, you can go back.

# Branches
- Work on features in isolation
A branch is a separate line of work.
```

- `# ` starts a scene and sets its title
- `- ` lines are the on-screen bullets
- Everything else is the spoken narration

Plain paragraphs also work. Separate scenes with a blank line, and titles and bullets are generated for you (by Groq if a key is set, otherwise by a simple offline fallback). A sample is in `examples/my_script.txt`.

## Voiceover (optional)

1. Install Piper and download a voice model by following the Piper project's instructions.
2. Add the path to your `.env`:

   ```
   PIPER_VOICE=./voices/en_US-lessac-medium.onnx
   ```

3. Restart the app. The voiceover checkbox is enabled once Piper is found.

Without Piper, videos are silent and each scene's length is estimated from the narration word count.

## Command line

```bash
python make_video.py --demo                        # built-in sample, no Groq needed
python make_video.py "How does DNS work?"          # Groq writes the script
python make_video.py --script examples/my_script.txt
python make_video.py --script build/scenes.json    # re-render a saved script
python make_video.py "How does DNS work?" --voice ./voices/en_US-lessac-medium.onnx
```

The script is also saved to `build/scenes.json`, so you can edit it and re-render.

## Project structure

```
app.py              Flask backend (jobs, library, video serving)
make_video.py       Script generation, slide drawing, and video rendering
templates/
  index.html        Web UI
examples/
  my_script.txt     Sample script
requirements.txt
.env.example
```

Generated files (`videos/`, `build/`, `.env`, voice models) are ignored by git.

## Troubleshooting

- **`TemplateNotFound: index.html`**: put `index.html` in a `templates` folder next to `app.py`, or directly beside it. Also check Windows has not named it `index.html.txt`.
- **Model not found from Groq**: update `GROQ_MODEL` in `.env` to a current model name.
- **Generate button is disabled**: `GROQ_API_KEY` is missing from `.env`. Restart the app after editing it.
- **Rendering is slow**: 1080p video encoding is CPU heavy. A few minutes for a short video is normal.
- **No voiceover option**: Piper is not installed, or `PIPER_VOICE` does not point to a real file.

## Security note

The app has no login and is meant to run on your own computer. It listens on `127.0.0.1` only. Do not expose it to the internet as is.

## License

MIT, see `LICENSE`.
