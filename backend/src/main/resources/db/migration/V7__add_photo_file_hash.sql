ALTER TABLE photos ADD COLUMN file_hash VARCHAR(64);

CREATE UNIQUE INDEX photos_event_file_hash_idx ON photos (event_id, file_hash)
    WHERE file_hash IS NOT NULL;
