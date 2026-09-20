# FaceSorter

Face recognition platform for professional camera studios. See `REQUIREMENTS.md` for full product requirements and `CLAUDE.md` for architecture notes.

## Prerequisites

- Docker Desktop (for Postgres+pgvector, the backend, and optionally the admin portal)
- Node.js + npm (for the admin portal in dev mode and the studio Electron app)
- Python 3 (for the studio app's local face-processing pipeline)

## Can this all run via one `docker compose up`?

**Almost, but not quite.** `docker-compose.yml` covers the **database**, **backend**, and **admin portal** — those three can run entirely from Docker Compose. The **studio app is an Electron desktop app** and can't be containerized the way it's used: it opens a native window, accesses the camera, and browses the local filesystem for photos, none of which works from inside a Docker container. It must run natively on the host via `npm start`. Its Python face-processing pipeline runs as a local subprocess of the Electron app for the same reason (it needs direct access to the studio machine's photo files).

So in practice: `docker compose up -d db backend admin` gets you the database, API, and admin portal, and you run the studio app separately.

## Backend + Database

```bash
# From the repo root
docker compose up -d db backend
```

- Backend API: `http://localhost:8080`
- Postgres (with pgvector): `localhost:5432` (user/pass/db: `facesorter`)
- A default admin account is seeded automatically on first boot: `admin@facesorter.com` / `Admin1234!`
- Config is via env vars in `docker-compose.yml` / `.env` (see `.env.example` for `JWT_SECRET`, `JWT_EXPIRY_HOURS`)

To rebuild after backend code changes:

```bash
docker compose up -d --build backend
```

Run the backend's integration test suite (Testcontainers spins up its own throwaway Postgres+pgvector instance — this doesn't touch the dev database above):

```bash
cd backend
mvn test
```

## Admin Portal (React)

Two ways to run it — pick one:

**Option A — Docker Compose** (builds a static production bundle, served by nginx, proxying `/api` to the `backend` container):

```bash
docker compose up -d --build admin
```
- Admin portal: `http://localhost:3001`

**Option B — Local dev server** (hot reload, proxies `/api` to `localhost:8080`):

```bash
cd frontend/admin
npm install   # first time only
npm run dev
```
- Admin portal: `http://localhost:3001`

Only run one of the two at a time — both bind port 3001.

## Studio App (Electron — operator desktop app)

This is the studio-operator app: manage events, upload/process photos, create and fulfill orders, and handle walk-in customer photo retrieval. Must run natively on the host, not in Docker.

```bash
cd studio-app
npm install          # first time only
npm run setup:python # first time only — creates a local Python venv with InsightFace/onnxruntime
npm start
```

- Renderer (Vite): `http://localhost:5173`
- Local photo file server: `http://localhost:4567`
- Requires the backend to be running (`http://localhost:8080`) to log in and sync data

`npm start` runs the Vite dev server and the Electron main process together (via `concurrently`); closing the Electron window stops both.

## Typical local dev order

```bash
docker compose up -d db backend      # 1. backend + database
docker compose up -d --build admin   # 2. admin portal (or `npm run dev` in frontend/admin)
cd studio-app && npm start           # 3. studio app (separate terminal)
```
