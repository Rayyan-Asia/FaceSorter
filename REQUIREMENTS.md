# FaceSorter — Product Requirements

## Overview

FaceSorter is a face recognition platform for professional camera studios. Studios use it to organize event photos by person, enabling customers to retrieve only their photos from any event.

---

## Core Workflow

### 1. Event Ingestion (Studio Side)
- Studio creates a named event/album and uploads photos via the studio portal
- Studio explicitly triggers processing when ready — it is not automatic
- A single pass is made across all unprocessed photos in the event:
  - Detect all faces in each photo
  - Extract an embedding for each detected face
  - For each embedding, check if a matching person already exists in the DB:
    - If yes: link the existing person record to this photo
    - If no: create a new person record, then link it to this photo
  - Mark each photo as `processed` upon completion
- If processing is interrupted and needs to be rerun, already-processed photos are skipped — only unprocessed photos are retried

### 2. Order Creation at Studio
- Customer visits the studio and pays for their photos
- Studio operator creates an empty order linked to the relevant event
- The system generates an order ID which is handed to the customer

### 3. Customer Self-Service Retrieval
- Customer visits the customer portal and enters their order ID
- The order ID grants them access to the linked event
- Customer opens their camera on the portal and takes a photo of themselves
- System extracts the face embedding from the photo and searches the event for matches
- Matched photos are displayed to the customer

### 4. Photo Selection & Finalize Order
- Customer reviews matched photos, filtering out any false positives
- **No false negatives** — the system must not miss a real match (high recall is the priority)
- Customer selects desired photos and confirms the order
- Order items are saved against the existing order record

---

## System Architecture

### Multi-Studio Access
- The application is accessible from multiple computers within a studio
- Shared backend database accessible across all studio workstations

### Processing Model
- **Photo processing (embedding extraction) runs locally at the studio**, not on the central server
- The central server stores only database records (metadata, embeddings, links)
- Raw photo files remain decentralized on studio-side storage

### Database Design
- **Hybrid database**: relational + vector store
  - Preferred: PostgreSQL with `pgvector` extension for vector similarity search
- **Relational tables (core)**:
  - `studios` — studio accounts and subscription status
  - `users` — customer records; uniquely identified by national ID number (string); fields include `id_number` (unique), `name`, `email`, `face_embedding_id` (FK to their embedding vector), and other profile data
  - `events` / `albums` — named events with associated studio
  - `photos` — photo metadata (ID, filename, event, local file path reference, `processed` boolean flag)
  - `person_photo_links` — many-to-many link between a recognized person and their photos
  - `payments` / `subscriptions` — annual subscription tracking per studio account
  - `orders` — one record per finalized order; fields: `id` (auto-incrementing integer/bigint, used as customer-facing order number), `timestamp`, `user_id` (FK → `users`), `event_id` (FK → `events`), `status`
  - `order_items` — breakdown of an order; fields: `id`, `order_id` (FK → `orders`); one-to-many from `orders` to `order_items`
  - `order_item_photos` — junction table linking order items to photos; fields: `order_item_id` (FK → `order_items`), `photo_id` (FK → `photos`); many-to-many because a single photo can appear across multiple orders (multiple people in the same photo)
- **Vector store**:
  - `face_embeddings` — face embedding vectors with foreign keys to user and photo records
  - Supports approximate nearest-neighbor (ANN) search for face matching

---

## Functional Requirements

| # | Requirement |
|---|-------------|
| FR-1 | Studio can register a new event and associate a folder of photos with it |
| FR-2 | System processes photos locally (studio side) and stores extracted embeddings + metadata to the central DB |
| FR-3 | System detects all faces in a photo and stores one embedding per detected face |
| FR-4 | Given a query photo of a customer, the system retrieves all photos from a specified event where that customer appears |
| FR-5 | Matching prioritizes **recall** — false negatives are not acceptable; false positives are tolerated |
| FR-6 | Operator can review matched photos and select a subset for printing |
| FR-7 | Operator finalizes an order, creating an `orders` record and one `order_items` row per selected photo |
| FR-8 | Application is accessible from multiple workstations within a studio |
| FR-11 | Customer portal: customers can request event access, browse matched photos, select photos, and place orders |
| FR-12 | Admin portal: full platform control — studios, subscriptions, events, orders, and users |
| FR-13 | Studio operator portal: create events, upload/process photos, create orders for walk-in customers, process and print orders |
| FR-14 | Customer embeddings are stored once at registration and never updated |
| FR-15 | Event access is granted via order ID — customers enter their order ID to access the linked event |
| FR-16 | Customer portal opens device camera to capture a photo; embedding is extracted client-side or server-side to perform face search |
| FR-17 | Studio operator creates an empty order (linked to an event) at point of payment and provides the order ID to the customer |
| FR-18 | Studio operator manually triggers photo processing for an event; processing does not start automatically on upload |
| FR-19 | Processing makes a single pass over all unprocessed photos; each photo is marked `processed` on completion |
| FR-20 | If processing is rerun, already-processed photos are skipped; only unprocessed photos are retried |
| FR-21 | During processing, if a detected face matches an existing person record, it is linked rather than creating a duplicate |
| FR-9 | Central server holds DB records only; raw photo files stay on studio-local storage |
| FR-10 | System tracks annual subscription payments per studio to gate access |

---

## Non-Functional Requirements

| # | Requirement |
|---|-------------|
| NFR-1 | Vector similarity search must support high-recall face matching at event scale (20,000+ photos, multiple faces per photo) |
| NFR-2 | Photo processing must not require uploading raw images to the central server |
| NFR-3 | Multi-workstation access requires a shared, consistent database state |
| NFR-4 | Subscription/payment records must be reliably enforced before granting access |
| NFR-5 | System should support multiple studios as independent tenants |

---

## Technology Considerations

- **Software policy**: all dependencies and tools must be open source
- **Backend**: Spring Boot (Java) — REST API, business logic, auth, and DB access
- **Frontend (web portals)**: React — admin portal and customer portal
- **Studio desktop app**: Electron — studio operator portal runs as a desktop application on studio workstations
- **ML / feature extraction**: Python — face detection, embedding extraction, and similarity search logic (runs locally at the studio via the Electron app or a bundled Python process)
- **Database**: PostgreSQL + `pgvector` extension (supports vector embeddings natively within Postgres, avoiding a separate vector DB service)
- **Embedding model**: FaceNet (via DeepFace) — already in use in the current PoC
- **Face detection**: OpenCV Haar Cascades (current PoC); consider upgrading to RetinaFace for production
- **Authentication**: SSO via Google (OAuth 2.0); no username/password auth
- **Order format**: persisted to DB as `orders` + `order_items` records; no file export required

---

## Portal Roles & Permissions

### Admin Portal (code owners)
- Full control over all platform data and configuration
- Manage studio accounts and subscriptions
- View and manage all events, orders, and users across all studios
- Platform-level settings and monitoring

### Studio Operator Portal
- Create and manage events/albums
- Upload photos and trigger local processing pipeline
- Take a photo of a walk-in customer to perform face search and create an order on their behalf
- Review and process orders (mark as printed/fulfilled)
- Print photos from finalized orders

### Customer Portal
- Enter an order ID (provided by studio at point of payment) to access the linked event
- Open device camera, take a self-photo, and trigger face search against the event
- Browse matched photos and filter out false positives
- Select desired photos and confirm the order

---

## Resolved Decisions

| Question | Decision |
|----------|----------|
| Similarity threshold | Tuned to minimize false negatives to near zero; exact threshold determined empirically during development |
| Customer record persistence | Customer records are persisted in a `users` table (keyed by national ID); retrieval searches against event embeddings using the stored embedding |
| Event scale | Up to 20,000+ photos per event, each potentially containing multiple faces; efficiency and throughput are critical design constraints |
| User-facing access | Both a studio operator interface and a customer-facing portal |
| Payment/subscription | Lahza (Palestinian payment provider) for digital payments; cash payments also supported |
