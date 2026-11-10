# Video Studio

A small local web app that turns a script into an explainer video. Write your own script, or let an AI model draft one from a topic. Pick the provider (Groq, OpenAI, Anthropic, or any OpenAI-compatible service), choose from its available models, and use your own API key. The video is built on your own machine with ffmpeg and saved so you can watch or download it.

No paid services are needed: Groq has a free tier, and local models via Ollama cost nothing. The AI is only used for writing text. Video is built locally with ffmpeg.

## Features

- Generate a scene-by-scene script from a topic, then edit it freely
- Choose the AI provider and model in the UI (Groq, OpenAI, Anthropic, or a custom OpenAI-compatible endpoint), with the model list fetched live from the provider
- Bring your own API key per provider, or keep keys in `.env`
- Or paste your own script (structured, plain paragraphs, or JSON)
- 8 visual themes with animated backgrounds, text entrances, and wipe transitions (or themed still slides for faster renders)
- Background rendering with a live progress bar
- Built-in player, MP4 download, and a saved library of past videos
- Optional free local voiceover with [Piper](https://github.com/OHF-Voice/piper1-gpl)
- Command-line mode if you prefer the terminal

## Requirements

- Python 3.10 or newer
- An API key for at least one provider, only if you want AI-written scripts (custom scripts need none):
  Groq https://console.groq.com, OpenAI https://platform.openai.com, Anthropic https://console.anthropic.com

## Setup

```bash
git clone <your-repo-url>
cd video-studio

python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file (copy `.env.example`). Keys in `.env` are optional, because you can paste them in the web UI instead:

```
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
GROQ_MODEL=qwen/qwen3.8-27b
```

## Run the web app

```bash
python app.py
```

Open http://127.0.0.1:5000 and:

1. Pick an AI provider, paste its API key (or leave it blank if the key is in `.env`), and choose a model from the list. Then enter a topic and click **Generate**, or just paste your own script
2. Edit the script if you like
3. Click **Build video**
4. Watch it in the preview, or click **Download MP4**

Finished videos are stored in `videos/` and appear in the Library list.

## Themes and animation

Pick a look in the **Visual theme** picker (each tile is a live preview):

| Theme | Style |
|---|---|
| Midnight | Deep blue, twinkling stars, drifting glow |
| Neon Grid | Dark purple, scrolling neon grid, glowing title |
| Sunrise | Warm gradient, rising sun, frosted bullet cards |
| Forest | Greens, floating leaves |
| Blueprint | Technical grid, rotating dial, corner marks |
| Terminal | Black and green, falling code, typing cursor |
| Clean Light | White, soft pastel blobs, card bullets |
| Ocean | Teal gradient, scrolling waves, rising bubbles |

- **Auto** picks a theme from your script's topic (code and Git topics get Terminal, science gets Blueprint, and so on).
- **Mix** gives every scene a different theme.
- **Animated** (checkbox) adds drifting backgrounds, title and bullets sliding in one by one in step with the scene, a progress bar, and a colour wipe between scenes. Untick it for themed still slides, which render several times faster and use less memory.

Command line: `--theme neon` or `--theme mix`, and `--static` for no animation. In `.env`: `VIDEO_THEME` and `VIDEO_ANIMATED`.

Rough cost: a 24 second animated video takes about 25 seconds to render at 1080p (about 13 seconds at 720p) on a modest single-core machine, versus about 8 seconds for still slides.

## AI providers and keys

| Provider | Notes |
|---|---|
| Groq | OpenAI-compatible API, has a free tier |
| OpenAI | ChatGPT models. Needs an API key from platform.openai.com |
| Anthropic | Claude models. Needs an API key from console.anthropic.com |
| Custom | Any OpenAI-compatible endpoint: OpenRouter, Ollama (`http://localhost:11434/v1`, no key needed), Together, LM Studio, and others |

- A ChatGPT or Claude.ai **subscription is not an API key**. Create an API key in the provider's developer console (API usage is billed separately).
- The model list is fetched live from the provider once a key is entered. Choose **Type a model id...** to use one that is not listed.
- Keys you paste in the UI are sent to the app's server only to make the provider call. They are not written to disk or to the saved video data. Tick **Remember my keys in this browser** to keep them in your browser's local storage (not recommended on shared computers).
- Keys set in `.env` act as a fallback when the box is left empty.
- If a model rejects `temperature` or JSON mode, the app retries without them automatically.
- This uses each provider's normal API. It does not use MCP, which is for giving a model access to tools rather than for calling the model.

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

Recommended voice: **`en_US-lessac-medium`** (clear US English, good speed and quality). Check the voice's `MODEL_CARD` license before using it commercially or on a monetized channel.

1. Activate your virtual environment, then install Piper:

   ```bash
   pip install piper-tts
   ```

2. Download the voice into a `voices` folder in the project:

   ```bash
   python -m piper.download_voices --download-dir voices en_US-lessac-medium
   ```

   This creates two files that must stay together: `voices/en_US-lessac-medium.onnx` and `voices/en_US-lessac-medium.onnx.json`.

3. Test it (should create `test.wav`):

   ```bash
   echo "Hello from Piper" | piper -m voices/en_US-lessac-medium.onnx -f test.wav
   ```

4. Add the path to your `.env`:

   ```
   PIPER_VOICE=./voices/en_US-lessac-medium.onnx
   ```

5. Restart the app. The voiceover checkbox is enabled once Piper and the voice file are found. Piper must be on your PATH, so run the app from the same activated virtual environment.

Other voices: `en_GB-cori-medium`, `en_GB-alan-medium`. Use a `-low` voice (for example `en_US-lessac-low`) on slow machines. Listen to samples at https://rhasspy.github.io/piper-samples.

Without Piper, videos are silent and each scene's length is estimated from the narration word count.

## Command line

```bash
python make_video.py --demo                        # built-in sample, no AI key needed
python make_video.py --demo --theme terminal       # pick a theme
python make_video.py --demo --static               # themed still slides, fastest
python make_video.py "How does DNS work?"          # AI writes the script (provider from .env)
python make_video.py "How does DNS work?" --provider anthropic --model claude-haiku-5-5
python make_video.py --script examples/my_script.txt
python make_video.py --script build/scenes.json    # re-render a saved script
python make_video.py "How does DNS work?" --voice ./voices/en_US-lessac-medium.onnx
```

The script is also saved to `build/scenes.json`, so you can edit it and re-render.

## Project structure

```
app.py              Flask backend (jobs, library, video serving)
render.yaml         Render deployment blueprint
make_video.py       Script generation, slide drawing, and video rendering (ffmpeg)
llm.py              Provider-neutral AI client (Groq, OpenAI, Anthropic, custom)
themes.py           Visual themes, artwork, and the animation filter graph
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
- **Model not found**: pick a model from the dropdown (use Refresh), or update the `*_MODEL` value in `.env`.
- **"rejected the API key"**: the key is wrong, expired, or belongs to a different provider than the one selected.
- **No models in the list**: enter an API key (or for Custom, a base URL) and press Refresh.
- **Rendering is slow**: videos are built with ffmpeg and cost scales with frames x pixels. Speed it up with `VIDEO_FPS=10` (default 15), `VIDEO_HEIGHT=720` (default 1080), or `VIDEO_PRESET=ultrafast` in `.env`. Fewer or shorter scenes also help.
- **No voiceover option**: Piper is not installed, or `PIPER_VOICE` does not point to a real file.

## Deploy on Render (free tier)

Read the limits first. A free Render web service has a tiny CPU share, an ephemeral disk, and sleeps when idle, so it works as a demo, not as a production video renderer. Verify current limits at https://render.com/docs/free.

1. Push this repo to GitHub (`.env` is git-ignored, so your key stays private).
2. In Render: **New > Blueprint**, pick your repo. It reads `render.yaml`.
   (Or **New > Web Service** and enter the build/start commands from `render.yaml` by hand.)
3. When prompted, enter `GROQ_API_KEY` and `APP_PASSWORD`. Visitors can also paste their own provider keys in the page. For safety, the Custom provider's base URL is fixed to `OPENAI_COMPAT_BASE_URL` on Render unless you set `ALLOW_CUSTOM_BASE_URL=1`. The app refuses to start on Render without a password.
4. Deploy, then open your `onrender.com` URL. Your browser shows a login box: any username, and your `APP_PASSWORD`.

Things to expect on the free plan:

- **Videos are temporary.** Files are deleted on every restart, redeploy, or spin-down. Download each video right after it is built.
- **Idle spin-down.** After 15 minutes without traffic the service sleeps, and the next visit takes about a minute to wake up.
- **Slow renders.** `render.yaml` uses 720p, 10 fps, and the `ultrafast` preset to keep builds manageable. Lower `VIDEO_HEIGHT` (for example 480) if builds time out or run out of memory.
- **No voiceover.** Piper voice models are not included in the deployment.
- **One worker.** Build progress is held in memory, so keep `--workers 1`.

For permanent storage or faster renders you need a paid instance (with a persistent disk) or object storage such as S3-compatible storage.

## Security note

Locally the app has no login and listens on `127.0.0.1` only. If you deploy it, always set `APP_PASSWORD`, because anyone who can reach the URL can spend your Groq quota and your server's CPU.

## License

MIT, see `LICENSE`.