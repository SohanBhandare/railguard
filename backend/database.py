"""
Database layer — SQLite3 with WAL mode for concurrent reads.
All queries centralized here, no raw SQL in app.py.
"""
import sqlite3, os, json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'database', 'railguard.db')
SCHEMA = os.path.join(os.path.dirname(__file__), '..', 'database', 'schema_sqlite.sql')

class Database:
    def __init__(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self._init_db()

    def _conn(self):
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        if os.path.exists(SCHEMA):
            with self._conn() as c:
                c.executescript(open(SCHEMA).read())
        else:
            self._create_tables()

    def _create_tables(self):
        ddl = """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL UNIQUE,
            ip_address TEXT NOT NULL DEFAULT 'unknown',
            user_agent TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            completed_at DATETIME,
            is_bot INTEGER NOT NULL DEFAULT 0,
            bot_probability REAL NOT NULL DEFAULT 0.0,
            verdict TEXT NOT NULL DEFAULT 'pending',
            captcha_shown INTEGER DEFAULT 0,
            captcha_passed INTEGER DEFAULT 0,
            blocked INTEGER DEFAULT 0,
            risk_level TEXT DEFAULT 'low'
        );
        CREATE TABLE IF NOT EXISTS behavioral_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            captured_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            elapsed_seconds REAL DEFAULT 0, form_fill_time REAL DEFAULT 0,
            mouse_moves INTEGER DEFAULT 0, avg_mouse_speed REAL DEFAULT 0,
            scroll_events INTEGER DEFAULT 0, hover_events INTEGER DEFAULT 0,
            avg_keystroke_ms REAL DEFAULT 0, keystroke_stddev REAL DEFAULT 0,
            avg_click_prec REAL DEFAULT 0, total_clicks INTEGER DEFAULT 0,
            copy_pastes INTEGER DEFAULT 0, tab_presses INTEGER DEFAULT 0,
            field_fills INTEGER DEFAULT 0, corrections INTEGER DEFAULT 0,
            window_blurs INTEGER DEFAULT 0, fp_score REAL DEFAULT 0,
            speed_score REAL DEFAULT 0, mouse_score REAL DEFAULT 0,
            ks_score REAL DEFAULT 0, pattern_score REAL DEFAULT 0,
            fp_ml_score REAL DEFAULT 0, network_score REAL DEFAULT 0,
            combined_score REAL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS booking_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            attempted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            train_number TEXT, train_name TEXT,
            from_station TEXT, to_station TEXT,
            journey_date TEXT, seat_class TEXT,
            passenger_count INTEGER DEFAULT 1,
            total_fare REAL DEFAULT 0, payment_method TEXT,
            pnr_number TEXT, status TEXT DEFAULT 'attempted'
        );
        CREATE TABLE IF NOT EXISTS ip_intelligence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ip_address TEXT NOT NULL UNIQUE,
            first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
            total_attempts INTEGER DEFAULT 0,
            bot_attempts INTEGER DEFAULT 0,
            human_attempts INTEGER DEFAULT 0,
            is_blocked INTEGER DEFAULT 0,
            block_reason TEXT, risk_score REAL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS flagged_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            flagged_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            severity TEXT NOT NULL, signal_type TEXT NOT NULL, description TEXT
        );
        CREATE TABLE IF NOT EXISTS system_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            event_type TEXT NOT NULL, session_id TEXT, details TEXT
        );
        """
        with self._conn() as c:
            c.executescript(ddl)

    def ping(self):
        try:
            with self._conn() as c:
                c.execute("SELECT 1")
            return 'ok'
        except Exception as e:
            return str(e)

    # ── Sessions ──────────────────────────────────────
    def create_session(self, sid, ip, ua):
        with self._conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO sessions (session_id,ip_address,user_agent) VALUES (?,?,?)",
                (sid, ip, ua))

    def update_session_verdict(self, sid, verdict, prob, is_bot, risk):
        with self._conn() as c:
            c.execute("""UPDATE sessions SET verdict=?,bot_probability=?,is_bot=?,
                         risk_level=?,blocked=? WHERE session_id=?""",
                      (verdict, prob, int(is_bot), risk, int(is_bot), sid))

    def update_session_completed(self, sid):
        with self._conn() as c:
            c.execute("UPDATE sessions SET completed_at=CURRENT_TIMESTAMP WHERE session_id=?", (sid,))

    def update_captcha(self, sid, passed):
        with self._conn() as c:
            c.execute("UPDATE sessions SET captcha_shown=1,captcha_passed=? WHERE session_id=?",
                      (int(passed), sid))

    def get_sessions(self, limit=50, offset=0):
        with self._conn() as c:
            rows = c.execute("""
                SELECT s.session_id,s.ip_address,s.created_at,s.verdict,s.risk_level,
                       s.bot_probability,s.blocked,s.captcha_shown,s.captcha_passed,
                       b.combined_score,b.elapsed_seconds,b.mouse_moves,
                       ba.train_name,ba.from_station,ba.to_station,ba.status AS booking_status
                FROM sessions s
                LEFT JOIN behavioral_features b ON s.session_id=b.session_id
                LEFT JOIN booking_attempts ba   ON s.session_id=ba.session_id
                ORDER BY s.created_at DESC LIMIT ? OFFSET ?""", (limit, offset)).fetchall()
        return [dict(r) for r in rows]

    def get_session_detail(self, sid):
        with self._conn() as c:
            s = c.execute("SELECT * FROM sessions WHERE session_id=?", (sid,)).fetchone()
            if not s: return None
            bf = c.execute("SELECT * FROM behavioral_features WHERE session_id=?", (sid,)).fetchone()
            ba = c.execute("SELECT * FROM booking_attempts WHERE session_id=?", (sid,)).fetchone()
            fs = c.execute("SELECT * FROM flagged_signals WHERE session_id=? ORDER BY flagged_at", (sid,)).fetchall()
        return {'session': dict(s), 'features': dict(bf) if bf else {},
                'booking': dict(ba) if ba else {}, 'flags': [dict(f) for f in fs]}

    # ── Features ──────────────────────────────────────
    def save_features(self, sid, features, scores):
        with self._conn() as c:
            c.execute("""INSERT OR REPLACE INTO behavioral_features
                (session_id,elapsed_seconds,form_fill_time,mouse_moves,avg_mouse_speed,
                 scroll_events,hover_events,avg_keystroke_ms,keystroke_stddev,
                 avg_click_prec,total_clicks,copy_pastes,tab_presses,field_fills,
                 corrections,window_blurs,fp_score,
                 speed_score,mouse_score,ks_score,pattern_score,fp_ml_score,
                 network_score,combined_score)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (sid,
                 features.get('elapsed_seconds',0), features.get('form_fill_time',0),
                 features.get('mouse_moves',0),      features.get('avg_mouse_speed',0),
                 features.get('scroll_events',0),    features.get('hover_events',0),
                 features.get('avg_keystroke_ms',0), features.get('keystroke_stddev',0),
                 features.get('avg_click_prec',0),   features.get('total_clicks',0),
                 features.get('copy_pastes',0),      features.get('tab_presses',0),
                 features.get('field_fills',0),       features.get('corrections',0),
                 features.get('window_blurs',0),      features.get('fp_score',0),
                 scores.get('speed',0), scores.get('mouse',0), scores.get('ks',0),
                 scores.get('pattern',0), scores.get('fp',0), scores.get('net',0),
                 scores.get('combined',0)))

    # ── Booking ───────────────────────────────────────
    def save_booking(self, sid, b):
        with self._conn() as c:
            c.execute("""INSERT INTO booking_attempts
                (session_id,train_number,train_name,from_station,to_station,
                 journey_date,seat_class,passenger_count,total_fare,payment_method,
                 pnr_number,status)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (sid, b['train_number'],b['train_name'],b['from_station'],b['to_station'],
                 b['journey_date'],b['seat_class'],b['passenger_count'],b['total_fare'],
                 b['payment_method'],b['pnr_number'],b['status']))

    # ── IP Intelligence ───────────────────────────────
    def upsert_ip(self, ip):
        with self._conn() as c:
            c.execute("""INSERT INTO ip_intelligence(ip_address) VALUES(?)
                         ON CONFLICT(ip_address) DO UPDATE SET
                         last_seen=CURRENT_TIMESTAMP,
                         total_attempts=total_attempts+1""", (ip,))

    def get_ip_info(self, ip):
        with self._conn() as c:
            row = c.execute("SELECT * FROM ip_intelligence WHERE ip_address=?", (ip,)).fetchone()
        return dict(row) if row else None

    def update_ip_stats(self, ip, is_bot):
        col = 'bot_attempts' if is_bot else 'human_attempts'
        with self._conn() as c:
            c.execute(f"""UPDATE ip_intelligence SET {col}={col}+1,
                         risk_score=ROUND(CAST(bot_attempts AS REAL)/
                         NULLIF(total_attempts,0),3),
                         last_seen=CURRENT_TIMESTAMP WHERE ip_address=?""", (ip,))

    def get_ip_intel(self):
        with self._conn() as c:
            rows = c.execute("""SELECT *,
                ROUND(CASE WHEN total_attempts>0 THEN bot_attempts*100.0/total_attempts ELSE 0 END,1) AS bot_rate_pct
                FROM ip_intelligence ORDER BY risk_score DESC LIMIT 100""").fetchall()
        return [dict(r) for r in rows]

    def block_ip(self, ip, reason):
        with self._conn() as c:
            c.execute("""INSERT INTO ip_intelligence(ip_address,is_blocked,block_reason)
                         VALUES(?,1,?) ON CONFLICT(ip_address) DO UPDATE SET
                         is_blocked=1,block_reason=?""", (ip, reason, reason))

    # ── Flags & Events ────────────────────────────────
    def save_flag(self, sid, severity, signal_type, description):
        with self._conn() as c:
            c.execute("INSERT INTO flagged_signals(session_id,severity,signal_type,description) VALUES(?,?,?,?)",
                      (sid, severity, signal_type, description))

    def log_event(self, event_type, sid, details=''):
        with self._conn() as c:
            c.execute("INSERT INTO system_events(event_type,session_id,details) VALUES(?,?,?)",
                      (event_type, sid, details))

    # ── Dashboard ─────────────────────────────────────
    def get_dashboard_data(self):
        with self._conn() as c:
            stats = c.execute("""
                SELECT COUNT(*) as total,
                       SUM(is_bot) as bots,
                       COUNT(*)-SUM(is_bot) as humans,
                       ROUND(AVG(bot_probability)*100,1) as avg_bot_pct,
                       SUM(blocked) as blocked,
                       SUM(captcha_shown) as captchas
                FROM sessions""").fetchone()
            recent = c.execute("""
                SELECT session_id,ip_address,strftime('%H:%M:%S',created_at) as time,
                       verdict,risk_level,ROUND(bot_probability*100,1) as pct,blocked
                FROM sessions ORDER BY created_at DESC LIMIT 10""").fetchall()
            risk_dist = c.execute("""
                SELECT risk_level, COUNT(*) as cnt
                FROM sessions GROUP BY risk_level""").fetchall()
            hourly = c.execute("""
                SELECT strftime('%H:00',created_at) as hour,
                       COUNT(*) as total, SUM(is_bot) as bots
                FROM sessions
                WHERE created_at >= datetime('now','-24 hours')
                GROUP BY strftime('%H',created_at)
                ORDER BY hour""").fetchall()
            top_flags = c.execute("""
                SELECT signal_type, COUNT(*) as cnt, severity
                FROM flagged_signals GROUP BY signal_type
                ORDER BY cnt DESC LIMIT 8""").fetchall()
        return {
            'stats': dict(stats) if stats else {},
            'recent_sessions': [dict(r) for r in recent],
            'risk_distribution': [dict(r) for r in risk_dist],
            'hourly_trend': [dict(r) for r in hourly],
            'top_flags': [dict(r) for r in top_flags],
        }

    def get_hourly_stats(self):
        with self._conn() as c:
            rows = c.execute("""
                SELECT strftime('%Y-%m-%d %H:00', created_at) AS hour,
                       COUNT(*) AS total, SUM(is_bot) AS bots
                FROM sessions WHERE created_at >= datetime('now','-48 hours')
                GROUP BY hour ORDER BY hour""").fetchall()
        return [dict(r) for r in rows]

    def get_risk_breakdown(self):
        with self._conn() as c:
            rows = c.execute("SELECT risk_level,COUNT(*) as cnt FROM sessions GROUP BY risk_level").fetchall()
        return [dict(r) for r in rows]

    def get_training_data(self):
        with self._conn() as c:
            rows = c.execute("""
                SELECT b.*,s.is_bot FROM behavioral_features b
                JOIN sessions s ON b.session_id=s.session_id
                WHERE s.verdict != 'pending'""").fetchall()
        return [dict(r) for r in rows]
