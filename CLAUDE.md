# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FaceSorter is a face recognition platform for professional camera studios. Studios upload event photos, the system processes them to detect and embed faces, and customers can later retrieve their own photos from an event by taking a self-photo. See `REQUIREMENTS.md` for full product requirements.

The `poc/` directory contains a working proof-of-concept for the core face detection and embedding pipeline. Production development has not yet begun.

## PoC Commands

```bash
# Install dependencies
pip install -r poc/requirements.txt

# Run the face batching pipeline (from poc/ directory)
cd poc && python batch_faces.py

# Test single-image face cropping
cd poc && python test_crop_faces.py
```

There is no test suite, linter, or build step configured.

## PoC Architecture

```
Input Photos → Haar Cascade Face Detection → Face Cropping →
  DeepFace Embedding (FaceNet) → Cosine Similarity Matching →
  Batch Grouping (MD5 hash of embedding) → face_batches_find.json →
  Static Web UI
```

### Key PoC files

- **poc/batch_faces.py** — Core pipeline: face detection, cropping, embedding generation, similarity-based clustering. Outputs `face_batches_find.json`. Expects photos in `static/photos/`, writes cropped faces to `static/faces/`.
- **poc/test_crop_faces.py** — Single-image face detection/cropping test script.
- **poc/static/** — Vanilla JS/HTML/CSS web UI. `index.html` browses batches (carousel), `detail.html` shows all photos in a batch. Uses localStorage to pass batch data between pages.
- **poc/face_batches_find.json** — Output: array of `{face_embedding_hash, batch, crop_batch}` objects.
- **notebooks/** — Jupyter notebooks for experimentation with DeepFace and LFW dataset.

## Production Design (planned)

### Stack
- **Backend**: Spring Boot (Java)
- **Frontend (web)**: React — admin portal and customer portal
- **Studio app**: Electron (desktop) — studio operator portal
- **ML / feature extraction**: Python — runs locally at the studio
- **Database**: PostgreSQL + `pgvector` for combined relational and vector storage
- **Embedding model**: FaceNet via DeepFace
- **Face detection**: OpenCV Haar Cascades (PoC); consider upgrading to RetinaFace for production
- **Auth**: Google SSO (OAuth 2.0) only — no username/password
- **Payments**: Lahza (digital) + cash; tracked per studio subscription
- **All dependencies must be open source**

### Portals
- **Admin** — full platform control (code owners only)
- **Studio operator** — manage events, upload/process photos, create and fulfill orders
- **Customer** — enter order ID, take self-photo, retrieve and select their photos, place order

### Processing model
- Photo processing runs **locally at the studio**, not on the central server
- Central server stores DB records only (embeddings, metadata, links); raw files stay on studio storage
- Studio manually triggers processing per event (not automatic)
- Single pass over unprocessed photos; each photo marked `processed` on completion
- Matching deduplicates person records — existing persons are linked, not duplicated
- Similarity threshold tuned to minimize false negatives to near zero; false positives are acceptable

### Order flow
1. Customer pays at studio → operator creates empty order linked to event → hands customer the order ID (auto-incrementing integer/bigint)
2. Customer enters order ID in customer portal → gains access to that event
3. Customer opens camera, takes self-photo → embedding extracted → matching photos retrieved
4. Customer selects photos → order items saved → operator fulfills and prints

## Security Model (enforced; keep these invariants)

- **Accounts are admin-provisioned.** `POST /api/auth/register` requires an authenticated `ADMIN`; there is no self-signup. An operator account must have a studio. Admins create accounts in the admin portal's **Accounts** page.
- **Tenant isolation.** Studio operators can only access their own studio's events, photos, orders and embeddings. Every `/api/studio/**` and `/api/faces/**` handler must go through `TenantAccess` (`requireStudio/requireEvent/requireOrder/requirePhoto`). Admins are unrestricted. Never trust a `studioId`/`eventId` from a request body without it.
- **`/api/faces/**` is operator/admin only** (embeddings are biometric data). `/api/customer/**` and `/api/photos/{id}` are public by design (order-id based customer flow); they are rate limited, event-scoped, only mutable while the order is `CREATED`/`PENDING`, and never return local file paths.
- **401 vs 403.** Missing/invalid/expired token -> 401 (the desktop app logs out and clears the token on expiry/401). Authenticated but not allowed -> 403. A deactivated studio's operators get 403 on login and on every request.
- **Order status** is whitelisted: `CREATED`, `PENDING`, `CONFIRMED`, `FULFILLED`.
- **Secrets.** `JWT_SECRET` must be set (>=32 bytes) outside local dev; the `prod` profile refuses the built-in default. Seeded admin password is `FACESORTER_ADMIN_INITIAL_PASSWORD` (defaults to the dev password).
- **Desktop photo server (port 4567)** serves only directories the operator registered (persisted in the app's userData `photo-roots.json`), resolves symlinks/`..`, image extensions only, and sends no CORS headers.
- **Backend tests need JDK 21 + Docker** (Testcontainers). Run them in the `maven:3.9.9-eclipse-temurin-21` container if the host JDK is older.

## Conventions

- Similarity threshold: configurable in `is_embedding_similar`; optimize for recall over precision
- Notebook ownership: each developer uses their own notebook files named `{developer-name}-{0-9}*` (see rules.md)
- Always update `requirements.txt` when adding new dependencies
