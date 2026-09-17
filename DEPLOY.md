# Deploying this project live

Two ready-to-use paths below. Both work with the project exactly as-is —
no code changes needed, mock mode by default so no API key required.

---

## Option A: Render (recommended — simplest, free)

1. Push this project to a GitHub repo (public or private both work).
2. Go to **https://render.com**, sign up/log in with GitHub.
3. Click **New → Blueprint**, select your repo. Render will detect
   `render.yaml` automatically and pre-fill everything (build command,
   start command, env vars) — you shouldn't need to type anything.
4. Click **Apply**. First deploy takes ~2-3 minutes (installing deps +
   building the RAG index).
5. You'll get a live URL like `https://wa-support-agent.onrender.com`.

**No `render.yaml` detected / doing it manually instead:**
- New → Web Service → connect your repo
- Runtime: **Python 3**
- Build command: `pip install -r requirements.txt && python -m app.rag.build_index`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Instance type: **Free**

**Heads up:** the free tier sleeps after 15 minutes of no traffic. The next
request after sleeping takes ~30 seconds to wake back up — totally fine for
a portfolio link, just don't be surprised by the first load being slow if
you haven't opened it in a while. Mention this if you send the link to a
recruiter cold ("first load may take ~30s, free tier").

---

## Option B: Hugging Face Spaces (Docker-based, also free)

Good alternative if you want it discoverable in an AI/ML-specific context.

1. Go to **https://huggingface.co/new-space**
2. Name it, set visibility, and choose **SDK: Docker**
3. Push this repo's contents to the Space's git repo (HF gives you a git
   remote URL, same flow as pushing to GitHub):
   ```bash
   git remote add hf https://huggingface.co/spaces/<your-username>/<space-name>
   git push hf main
   ```
4. HF Spaces will build the included `Dockerfile` automatically and serve
   it on port 7860 (already configured in the Dockerfile).
5. Your live URL: `https://huggingface.co/spaces/<your-username>/<space-name>`

---

## Option C: Railway / Fly.io (if you want more control)

Both can use the same `Dockerfile` in this repo directly:

- **Railway**: New Project → Deploy from GitHub repo → it auto-detects the
  Dockerfile and deploys. Free tier is credit-limited (not indefinite).
- **Fly.io**: `fly launch` in the project directory (detects the
  Dockerfile), then `fly deploy`. Requires the `flyctl` CLI installed
  locally first.

---

## After deploying: things worth checking

- Visit `/health` on your live URL — should return
  `{"status": "ok", "llm_provider": "mock"}`
- Visit `/` for the chat UI itself
- The RAG index and trace logs live on local disk inside the container.
  On free tiers, a redeploy resets the filesystem — the index rebuilds
  automatically on next boot (via the build command), but old trace logs
  won't persist across redeploys. That's expected and fine for a demo.

## If you want to run it with a real OpenAI model live

Add `OPENAI_API_KEY` as a secret/environment variable in your hosting
platform's dashboard (Render: Environment tab; HF Spaces: Settings →
Repository secrets), and set `LLM_PROVIDER=openai`. Redeploy. Everything
else in the app works identically — only the model backend changes.

**Careful:** if you do this, the live URL becomes able to spend real money
on every message anyone sends it. For a public portfolio link, mock mode
is usually the safer default — it demonstrates the full architecture
without any cost or abuse risk.
