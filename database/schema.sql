 -- ═══════════════════════════════════════════════════
--  IndiaRail Bot Detection System — Database Schema
--  SQLite3 Compatible
-- ═══════════════════════════════════════════════════

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    NOT NULL UNIQUE,
    ip_address      TEXT    NOT NULL DEFAULT 'unknown',
    user_agent      TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at    DATETIME,
    is_bot          INTEGER NOT NULL DEFAULT 0,
    bot_probability REAL    NOT NULL DEFAULT 0.0,
    verdict         TEXT    NOT NULL DEFAULT 'pending',
    captcha_shown   INTEGER DEFAULT 0,
    captcha_passed  INTEGER DEFAULT 0,
    blocked         INTEGER DEFAULT 0,
    risk_level      TEXT    DEFAULT 'low'
);

CREATE TABLE IF NOT EXISTS behavioral_features (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT    NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    captured_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    elapsed_seconds  REAL    DEFAULT 0,
    form_fill_time   REAL    DEFAULT 0,
    mouse_moves      INTEGER DEFAULT 0,
    avg_mouse_speed  REAL    DEFAULT 0,
    scroll_events    INTEGER DEFAULT 0,
    hover_events     INTEGER DEFAULT 0,
    avg_keystroke_ms REAL    DEFAULT 0,
    keystroke_stddev REAL    DEFAULT 0,
    avg_click_prec   REAL    DEFAULT 0,
    total_clicks     INTEGER DEFAULT 0,
    copy_pastes      INTEGER DEFAULT 0,
    tab_presses      INTEGER DEFAULT 0,
    field_fills      INTEGER DEFAULT 0,
    corrections      INTEGER DEFAULT 0,
    window_blurs     INTEGER DEFAULT 0,
    fp_score         REAL    DEFAULT 0,
    speed_score      REAL    DEFAULT 0,
    mouse_score      REAL    DEFAULT 0,
    ks_score         REAL    DEFAULT 0,
    pattern_score    REAL    DEFAULT 0,
    fp_ml_score      REAL    DEFAULT 0,
    network_score    REAL    DEFAULT 0,
    combined_score   REAL    DEFAULT 0
);

CREATE TABLE IF NOT EXISTS booking_attempts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    attempted_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    train_number    TEXT,
    train_name      TEXT,
    from_station    TEXT,
    to_station      TEXT,
    journey_date    TEXT,
    seat_class      TEXT,
    passenger_count INTEGER DEFAULT 1,
    total_fare      REAL    DEFAULT 0,
    payment_method  TEXT,
    pnr_number      TEXT,
    status          TEXT    DEFAULT 'attempted'
);

CREATE TABLE IF NOT EXISTS ip_intelligence (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ip_address      TEXT    NOT NULL UNIQUE,
    first_seen      DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_seen       DATETIME DEFAULT CURRENT_TIMESTAMP,
    total_attempts  INTEGER DEFAULT 0,
    bot_attempts    INTEGER DEFAULT 0,
    human_attempts  INTEGER DEFAULT 0,
    is_blocked      INTEGER DEFAULT 0,
    block_reason    TEXT,
    risk_score      REAL    DEFAULT 0
);

CREATE TABLE IF NOT EXISTS flagged_signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    flagged_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    severity    TEXT    NOT NULL,
    signal_type TEXT    NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS system_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    event_type  TEXT    NOT NULL,
    session_id  TEXT,
    details     TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_created   ON sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_sessions_bot       ON sessions(is_bot);
CREATE INDEX IF NOT EXISTS idx_sessions_ip        ON sessions(ip_address);
CREATE INDEX IF NOT EXISTS idx_bf_session         ON behavioral_features(session_id);
CREATE INDEX IF NOT EXISTS idx_ba_session         ON booking_attempts(session_id);
CREATE INDEX IF NOT EXISTS idx_ip_address         ON ip_intelligence(ip_address);

CREATE VIEW IF NOT EXISTS v_session_summary AS
SELECT s.session_id, s.ip_address, s.created_at, s.verdict, s.risk_level,
    s.bot_probability, s.blocked, s.captcha_shown, s.captcha_passed,
    b.combined_score, b.elapsed_seconds, b.mouse_moves, b.avg_keystroke_ms,
    ba.train_name, ba.from_station, ba.to_station, ba.status AS booking_status
FROM sessions s
LEFT JOIN behavioral_features b ON s.session_id = b.session_id
LEFT JOIN booking_attempts ba   ON s.session_id = ba.session_id;

CREATE VIEW IF NOT EXISTS v_hourly_stats AS
SELECT strftime('%Y-%m-%d %H:00', created_at) AS hour,
    COUNT(*) AS total, SUM(is_bot) AS bots,
    COUNT(*) - SUM(is_bot) AS humans,
    ROUND(AVG(bot_probability)*100, 1) AS avg_bot_pct
FROM sessions
GROUP BY strftime('%Y-%m-%d %H:00', created_at)
ORDER BY hour DESC;
