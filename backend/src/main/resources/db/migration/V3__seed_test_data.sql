-- Test studio for local development
INSERT INTO studios (name, email, active, created_at, updated_at)
VALUES ('Test Studio', 'admin@teststudio.com', TRUE, NOW(), NOW());

-- Active subscription for the test studio
INSERT INTO subscriptions (studio_id, start_date, end_date, payment_method, amount_paid, active, created_at)
VALUES (1, CURRENT_DATE, CURRENT_DATE + INTERVAL '1 year', 'cash', 0.00, TRUE, NOW());

-- Sample event
INSERT INTO events (studio_id, name, description, event_date, created_at, updated_at)
VALUES (1, 'Sample Event', 'A test event for development', CURRENT_DATE, NOW(), NOW());
