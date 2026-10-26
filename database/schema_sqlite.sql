
-- ============================================================
-- IndiaRail Bot Detection System
-- SQLite Database Schema for Python Flask
-- ============================================================

PRAGMA foreign_keys = ON;

-- ============================================================
-- 1. SESSIONS TABLE
-- Stores user sessions and bot detection results
-- ============================================================

CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL UNIQUE,
    ip_address      TEXT NOT NULL DEFAULT 'unknown',
    user_agent      TEXT,
    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at    DATETIME,
    is_bot          INTEGER NOT NULL DEFAULT 0
                    CHECK (is_bot IN (0, 1)),
    bot_probability REAL NOT NULL DEFAULT 0.0
                    CHECK (bot_probability BETWEEN 0.0 AND 1.0),
    verdict         TEXT NOT NULL DEFAULT 'pending',
    captcha_shown   INTEGER NOT NULL DEFAULT 0
                    CHECK (captcha_shown IN (0, 1)),
    captcha_passed  INTEGER NOT NULL DEFAULT 0
                    CHECK (captcha_passed IN (0, 1)),
    blocked         INTEGER NOT NULL DEFAULT 0
                    CHECK (blocked IN (0, 1)),
    risk_level      TEXT NOT NULL DEFAULT 'low'
                    CHECK (risk_level IN ('low', 'medium', 'high', 'critical'))
);

-- ============================================================
-- 2. BEHAVIORAL FEATURES TABLE
-- Stores user interaction features for ML detection
-- ============================================================

CREATE TABLE IF NOT EXISTS behavioral_features (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT NOT NULL,
    captured_at         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    elapsed_seconds     REAL NOT NULL DEFAULT 0,
    form_fill_time      REAL NOT NULL DEFAULT 0,

    mouse_moves         INTEGER NOT NULL DEFAULT 0,
    avg_mouse_speed     REAL NOT NULL DEFAULT 0,
    scroll_events       INTEGER NOT NULL DEFAULT 0,
    hover_events        INTEGER NOT NULL DEFAULT 0,

    avg_keystroke_ms    REAL NOT NULL DEFAULT 0,
    keystroke_stddev    REAL NOT NULL DEFAULT 0,

    avg_click_prec      REAL NOT NULL DEFAULT 0,
    total_clicks        INTEGER NOT NULL DEFAULT 0,
    copy_pastes         INTEGER NOT NULL DEFAULT 0,
    tab_presses         INTEGER NOT NULL DEFAULT 0,
    field_fills         INTEGER NOT NULL DEFAULT 0,
    corrections         INTEGER NOT NULL DEFAULT 0,
    window_blurs        INTEGER NOT NULL DEFAULT 0,

    fp_score            REAL NOT NULL DEFAULT 0,
    speed_score         REAL NOT NULL DEFAULT 0,
    mouse_score         REAL NOT NULL DEFAULT 0,
    ks_score            REAL NOT NULL DEFAULT 0,
    pattern_score       REAL NOT NULL DEFAULT 0,
    fp_ml_score         REAL NOT NULL DEFAULT 0,
    network_score       REAL NOT NULL DEFAULT 0,
    combined_score      REAL NOT NULL DEFAULT 0,

    FOREIGN KEY (session_id)
        REFERENCES sessions(session_id)
        ON DELETE CASCADE
);

-- ============================================================
-- 3. BOOKING ATTEMPTS TABLE
-- Stores railway booking attempt information
-- ============================================================

CREATE TABLE IF NOT EXISTS booking_attempts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    attempted_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    train_number    TEXT,
    train_name      TEXT,
    from_station    TEXT,
    to_station      TEXT,
    journey_date    TEXT,
    seat_class      TEXT,

    passenger_count INTEGER NOT NULL DEFAULT 1
                    CHECK (passenger_count >= 1),
    total_fare      REAL NOT NULL DEFAULT 0
                    CHECK (total_fare >= 0),

    payment_method  TEXT,
    pnr_number      TEXT,
    status          TEXT NOT NULL DEFAULT 'attempted',

    FOREIGN KEY (session_id)
        REFERENCES sessions(session_id)
        ON DELETE CASCADE
);

-- ============================================================
-- 4. IP INTELLIGENCE TABLE
-- Tracks IP activity and bot-related risks
-- ============================================================

CREATE TABLE IF NOT EXISTS ip_intelligence (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ip_address      TEXT NOT NULL UNIQUE,
    first_seen      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    total_attempts  INTEGER NOT NULL DEFAULT 0,
    bot_attempts    INTEGER NOT NULL DEFAULT 0,
    human_attempts  INTEGER NOT NULL DEFAULT 0,

    is_blocked      INTEGER NOT NULL DEFAULT 0
                    CHECK (is_blocked IN (0, 1)),
    block_reason    TEXT,
    risk_score      REAL NOT NULL DEFAULT 0
);

-- ============================================================
-- 5. FLAGGED SIGNALS TABLE
-- Stores suspicious behavior and detection signals
-- ============================================================

CREATE TABLE IF NOT EXISTS flagged_signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    flagged_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    severity        TEXT NOT NULL,
    signal_type     TEXT NOT NULL,
    description     TEXT,

    FOREIGN KEY (session_id)
        REFERENCES sessions(session_id)
        ON DELETE CASCADE
);

-- ============================================================
-- 6. SYSTEM EVENTS TABLE
-- Stores application events and diagnostic information
-- ============================================================

CREATE TABLE IF NOT EXISTS system_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_at        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    event_type      TEXT NOT NULL,
    session_id      TEXT,
    details         TEXT,

    FOREIGN KEY (session_id)
        REFERENCES sessions(session_id)
        ON DELETE SET NULL
);

-- ============================================================
-- 7. INDEXES
-- Improve performance for common queries
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_sessions_created
    ON sessions(created_at);

CREATE INDEX IF NOT EXISTS idx_sessions_bot
    ON sessions(is_bot);

CREATE INDEX IF NOT EXISTS idx_sessions_ip
    ON sessions(ip_address);

CREATE INDEX IF NOT EXISTS idx_bf_session
    ON behavioral_features(session_id);

CREATE INDEX IF NOT EXISTS idx_ba_session
    ON booking_attempts(session_id);

CREATE INDEX IF NOT EXISTS idx_ip_address
    ON ip_intelligence(ip_address);

CREATE INDEX IF NOT EXISTS idx_fs_session
    ON flagged_signals(session_id);

CREATE INDEX IF NOT EXISTS idx_events_session
    ON system_events(session_id);

CREATE INDEX IF NOT EXISTS idx_events_type
    ON system_events(event_type);

-- ============================================================
-- END OF INDIA RAIL BOT DETECTION DATABASE SCHEMA
-- ============================================================
