-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Studios
CREATE TABLE studios (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    google_sub VARCHAR(255) UNIQUE,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Users (customers), keyed by national ID
CREATE TABLE users (
    id BIGSERIAL PRIMARY KEY,
    id_number VARCHAR(50) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    google_sub VARCHAR(255) UNIQUE,
    face_embedding_id BIGINT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Events / Albums
CREATE TABLE events (
    id BIGSERIAL PRIMARY KEY,
    studio_id BIGINT NOT NULL REFERENCES studios(id),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    event_date DATE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_events_studio_id ON events(studio_id);

-- Photos
CREATE TABLE photos (
    id BIGSERIAL PRIMARY KEY,
    event_id BIGINT NOT NULL REFERENCES events(id),
    filename VARCHAR(500) NOT NULL,
    local_path VARCHAR(1000) NOT NULL,
    processed BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_photos_event_id ON photos(event_id);
CREATE INDEX idx_photos_processed ON photos(event_id, processed);

-- Face embeddings (vector store)
CREATE TABLE face_embeddings (
    id BIGSERIAL PRIMARY KEY,
    embedding vector(128) NOT NULL,
    photo_id BIGINT REFERENCES photos(id),
    user_id BIGINT REFERENCES users(id),
    event_id BIGINT NOT NULL REFERENCES events(id),
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_face_embeddings_event_id ON face_embeddings(event_id);
CREATE INDEX idx_face_embeddings_user_id ON face_embeddings(user_id);

-- IVFFlat index for ANN search within events
CREATE INDEX idx_face_embeddings_vector ON face_embeddings
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Add FK from users to face_embeddings now that both tables exist
ALTER TABLE users ADD CONSTRAINT fk_users_face_embedding
    FOREIGN KEY (face_embedding_id) REFERENCES face_embeddings(id);

-- Person-photo links (many-to-many between face_embeddings and photos)
CREATE TABLE person_photo_links (
    id BIGSERIAL PRIMARY KEY,
    face_embedding_id BIGINT NOT NULL REFERENCES face_embeddings(id),
    photo_id BIGINT NOT NULL REFERENCES photos(id),
    similarity_score DOUBLE PRECISION,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE (face_embedding_id, photo_id)
);

CREATE INDEX idx_person_photo_links_face ON person_photo_links(face_embedding_id);
CREATE INDEX idx_person_photo_links_photo ON person_photo_links(photo_id);

-- Subscriptions
CREATE TABLE subscriptions (
    id BIGSERIAL PRIMARY KEY,
    studio_id BIGINT NOT NULL REFERENCES studios(id),
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    payment_method VARCHAR(50) NOT NULL,
    amount_paid DECIMAL(10,2) NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_subscriptions_studio_id ON subscriptions(studio_id);

-- Orders
CREATE TABLE orders (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(id),
    event_id BIGINT NOT NULL REFERENCES events(id),
    studio_id BIGINT NOT NULL REFERENCES studios(id),
    status VARCHAR(50) NOT NULL DEFAULT 'CREATED',
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_orders_event_id ON orders(event_id);
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_studio_id ON orders(studio_id);

-- Order items
CREATE TABLE order_items (
    id BIGSERIAL PRIMARY KEY,
    order_id BIGINT NOT NULL REFERENCES orders(id),
    notes TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_order_items_order_id ON order_items(order_id);

-- Order item photos (junction)
CREATE TABLE order_item_photos (
    id BIGSERIAL PRIMARY KEY,
    order_item_id BIGINT NOT NULL REFERENCES order_items(id),
    photo_id BIGINT NOT NULL REFERENCES photos(id),
    UNIQUE (order_item_id, photo_id)
);

CREATE INDEX idx_order_item_photos_order_item ON order_item_photos(order_item_id);
CREATE INDEX idx_order_item_photos_photo ON order_item_photos(photo_id);
