# Deploying APEX-ORACLE

## Recommended: the FULL app in one container (UI + real backend)

One process serves the Inspector UI at `/` and the JSON API (real TESS + the AI model) at
`/analyze`, same origin — no CORS, no `?api` wiring. This is the simplest way to ship the
whole thing.

### Run it locally
```bash
docker build -t apex-oracle .
docker run -p 8800:8800 apex-oracle          # open http://localhost:8800
```
The UI loads, and its "REAL ENGINE" button calls the same origin (the container) — including
real TESS downloads (e.g. type `WASP-18`).

Without Docker (dev):
```bash
uv pip install -e ".[astro]"
apex-oracle serve --host 0.0.0.0 --port 8800   # auto-serves app/inspector at /
# open http://localhost:8800
```

### Deploy to a cloud container host (Render / Railway / Fly.io)
These allow long requests and ~1 GB images (Vercel does **not** — see below).

- **Render:** New → Web Service → connect the repo (or push the image). It uses the
  `Dockerfile`. Set instance with >= 1 GB RAM. Port `8800` (Render reads `$PORT`; if needed set
  the start command to `apex-oracle serve --host 0.0.0.0 --port $PORT`). You get one https URL
  that serves the whole app.
- **Railway / Fly.io:** same idea — deploy the `Dockerfile`; one URL for UI + API.

Notes:
- Real TESS downloads take 30–60 s, so the host must allow long request timeouts (these do; Vercel doesn't).
- The image includes `lightkurve` (real TESS) + the bundled AI model, so it's ~1 GB. First build is slow.

---

## Alternative: split deploy (static UI on Vercel + backend on a container host)
If you specifically want the UI on Vercel:
1. Deploy the backend container (above) → get `https://<backend>`.
2. Deploy the UI to Vercel (static): `cd app/inspector && vercel --prod` (root `vercel.json`
   also works for whole-repo import).
3. Open the Vercel UI pointed at the backend: `https://<vercel-url>/?api=https://<backend>`
   (the backend must be **https** — browsers block http from an https page).

The Vercel-only site still works fully for **simulation + CSV upload** (those run in the
browser); only the real-TESS "REAL ENGINE" button needs the backend.

---

## I can't run the deploy for you
Pushing to Render/Railway/Fly/Vercel requires logging into **your** account — I won't
authenticate as you. Everything above is a copy-paste away, and the full stack is verified
locally (UI at `/`, `/analyze` returns AI results).
