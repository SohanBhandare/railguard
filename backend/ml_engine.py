"""
ML Engine — Random Forest + Isolation Forest ensemble.
Trains on synthetic data at startup; can retrain on live DB data.
"""
import numpy as np, os, pickle, random, math
from datetime import datetime

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'database', 'model.pkl')

try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    SKLEARN = True
except ImportError:
    SKLEARN = False

FEATURE_NAMES = [
    'elapsed_seconds','mouse_moves','avg_mouse_speed','scroll_events',
    'avg_keystroke_ms','keystroke_stddev','avg_click_prec','total_clicks',
    'copy_pastes','tab_presses','field_fills','corrections','window_blurs',
    'fp_score','hover_events',
]

class BotDetector:
    def __init__(self):
        self.rf   = None
        self.iso  = None
        self.scaler = None
        self.trained = False
        self.trained_at = None
        self.train_samples = 0
        self._load_or_train()

    # ── Load / Train ──────────────────────────────────
    def _load_or_train(self):
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, 'rb') as f:
                    d = pickle.load(f)
                self.rf = d['rf']; self.iso = d['iso']; self.scaler = d['scaler']
                self.trained = True; self.trained_at = d.get('trained_at')
                self.train_samples = d.get('train_samples', 0)
                print(f"[ML] Model loaded from {MODEL_PATH}")
                return
            except Exception:
                pass
        self._train_synthetic()

    def _train_synthetic(self):
        if not SKLEARN:
            print("[ML] scikit-learn not available — using rule-based fallback")
            return
        print("[ML] Training on synthetic data...")
        X, y = [], []

        # Human profiles
        for _ in range(1200):
            X.append([
                random.uniform(30, 180),       # elapsed
                random.randint(40, 300),        # mouse moves
                random.uniform(100, 600),       # mouse speed
                random.randint(2, 20),          # scrolls
                random.uniform(120, 400),       # ks avg
                random.uniform(30, 120),        # ks stddev (HIGH = human)
                random.uniform(0.50, 0.90),     # click prec
                random.randint(3, 25),          # clicks
                1 if random.random() < 0.10 else 0,  # rarely pastes
                random.randint(0, 5),           # tabs
                random.randint(4, 15),          # field fills
                random.randint(0, 8),           # corrections (0 is possible)
                random.randint(0, 3),           # window blurs
                random.uniform(0, 0.1),         # fp score
                random.randint(5, 60),          # hovers
            ])
            y.append(0)

        # Bot — Speed
        for _ in range(250):
            X.append([
                random.uniform(0.5, 4),
                random.randint(0, 3),
                random.uniform(2000, 8000),
                0,
                random.uniform(5, 20),
                random.uniform(0, 3),           # LOW stddev = bot
                random.uniform(0.995, 1.0),
                random.randint(0, 3),
                1, random.randint(10, 20),
                random.randint(5, 12), 0,
                0, random.uniform(0.5, 1.0),
                random.randint(0, 2),
            ])
            y.append(1)

        # Bot — Smart
        for _ in range(250):
            X.append([
                random.uniform(5, 15),
                random.randint(3, 10),
                random.uniform(600, 2000),
                0,
                random.uniform(30, 60),
                random.uniform(1, 8),
                random.uniform(0.97, 1.0),
                random.randint(2, 8),
                1, random.randint(8, 15),
                random.randint(5, 10), 0,
                0, random.uniform(0.4, 0.9),
                random.randint(0, 5),
            ])
            y.append(1)

        # Bot — Sneaky (harder to catch)
        for _ in range(200):
            X.append([
                random.uniform(15, 40),
                random.randint(10, 30),
                random.uniform(200, 600),
                random.randint(0, 2),
                random.uniform(80, 150),
                random.uniform(5, 18),
                random.uniform(0.93, 0.99),
                random.randint(3, 12),
                1, random.randint(5, 12),
                random.randint(5, 10), 0,
                random.randint(0, 1),
                random.uniform(0.2, 0.6),
                random.randint(1, 10),
            ])
            y.append(1)

        # Bot — Human-like: copies human speed / mouse values, but the TELL-TALE signs are
        # perfectly even typing (std = ~3-7% of speed), zero corrections and pasted data.
        # Context counters (time on page, hovers, clicks, blurs) are varied widely because
        # a real browser adds human-looking values to them.
        for _ in range(400):
            ks = random.uniform(120, 300)
            X.append([
                random.uniform(20, 300),
                random.randint(10, 80),
                random.uniform(120, 400),
                random.randint(0, 6),
                ks,
                ks * random.uniform(0.02, 0.07),
                random.uniform(0.70, 0.95),
                random.randint(1, 20),
                1, random.randint(0, 4),
                random.randint(5, 14), 0,
                random.randint(0, 4),
                random.uniform(0.0, 0.3),
                random.randint(0, 80),
            ])
            y.append(1)

        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y)

        self.scaler = StandardScaler()
        X_sc = self.scaler.fit_transform(X_arr)

        Xtr, Xte, ytr, yte = train_test_split(X_sc, y_arr, test_size=0.2, random_state=42)

        self.rf = RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_leaf=3,
            class_weight='balanced', random_state=42, n_jobs=-1)
        self.rf.fit(Xtr, ytr)

        human_idx = y_arr == 0
        self.iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        self.iso.fit(X_sc[human_idx])

        acc = accuracy_score(yte, self.rf.predict(Xte))
        print(f"[ML] RF accuracy: {acc:.3f} | samples: {len(X)}")

        self.trained = True
        self.trained_at = datetime.utcnow().isoformat()
        self.train_samples = len(X)
        self._save_model()

    def _save_model(self):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump({'rf':self.rf,'iso':self.iso,'scaler':self.scaler,
                         'trained_at':self.trained_at,'train_samples':self.train_samples}, f)

    # ── Predict ───────────────────────────────────────
    def predict(self, features: dict, ip_bot_rate: float = 0.0) -> dict:
        vec = [features.get(n, 0) for n in FEATURE_NAMES]

        if SKLEARN and self.trained and self.rf:
            arr = np.array([vec], dtype=np.float32)
            arr_sc = self.scaler.transform(arr)
            rf_prob  = float(self.rf.predict_proba(arr_sc)[0][1])
            iso_score = float(self.iso.decision_function(arr_sc)[0])
            # Isolation Forest: negative = anomaly. Normalize to [0,1].
            iso_prob = max(0, min(1, 0.5 - iso_score * 2))
            # Ensemble
            combined = 0.65 * rf_prob + 0.25 * iso_prob + 0.10 * ip_bot_rate
        else:
            combined = self._rule_predict(features)
            rf_prob = combined; iso_prob = combined

        # Individual heuristic scores for UI display
        sc = self._heuristic_scores(features)
        combined = float(np.clip(0.65*rf_prob + 0.25*iso_prob + 0.10*(sc['combined']*0.3 + ip_bot_rate*0.7), 0, 1))

        return {
            'rf':      round(rf_prob, 4),
            'iso':     round(iso_prob, 4),
            'speed':   round(sc['speed'], 4),
            'mouse':   round(sc['mouse'], 4),
            'ks':      round(sc['ks'], 4),
            'pattern': round(sc['pattern'], 4),
            'fp':      round(sc['fp'], 4),
            'net':     round(min(ip_bot_rate * 2, 1.0), 4),
            'combined':round(combined, 4),
        }

    def _heuristic_scores(self, f):
        elapsed  = f.get('elapsed_seconds', 60)
        moves    = f.get('mouse_moves', 50)
        ksAvg    = f.get('avg_keystroke_ms', 200)
        ksDev    = f.get('keystroke_stddev', 60)
        prec     = f.get('avg_click_prec', 0.7)
        scrolls  = f.get('scroll_events', 5)
        paste    = f.get('copy_pastes', 0)
        tab      = f.get('tab_presses', 2)
        fills    = f.get('field_fills', 5)
        corr     = f.get('corrections', 2)
        wblur    = f.get('window_blurs', 1)
        fp       = f.get('fp_score', 0)
        mspd     = f.get('avg_mouse_speed', 200)

        # Speed
        speed = (0.98 if elapsed<2 else 0.90 if elapsed<5 else 0.65 if elapsed<10
                 else 0.38 if elapsed<20 else 0.15 if elapsed<45 else 0.05)
        if ksAvg > 0:
            speed = max(speed, 0.96 if ksAvg<20 else 0.78 if ksAvg<50 else 0.42 if ksAvg<100 else 0)

        # Mouse
        mouse = (0.97 if moves<2 else 0.82 if moves<8 else 0.50 if moves<20
                 else 0.22 if moves<50 else 0.08 if moves<100 else 0.03)
        if mspd > 1500: mouse = max(mouse, 0.72)
        if scrolls == 0 and elapsed > 6: mouse = max(mouse, 0.40)

        # Keystroke biometrics
        ks = 0
        if ksAvg > 0:
            ks = (0.96 if ksDev<5 and ksAvg<50 else
                  0.78 if ksDev<15 and ksAvg<80 else
                  0.40 if ksDev<30 and ksAvg<150 else 0.08)
        if prec > 0.995: ks = max(ks, 0.93)
        elif prec > 0.97: ks = max(ks, 0.62)

        # Pattern
        pattern = 0
        if paste > 0: pattern += 0.38
        if corr == 0 and fills > 3: pattern += 0.22
        if tab > 10: pattern += 0.18
        if wblur > 4: pattern += 0.12
        pattern = min(pattern, 1.0)

        # Fingerprint
        combined = 0.22*speed + 0.20*mouse + 0.18*ks + 0.15*pattern + 0.15*fp + 0.10*0
        return {'speed':speed,'mouse':mouse,'ks':ks,'pattern':pattern,'fp':fp,'combined':combined}

    def _rule_predict(self, f):
        sc = self._heuristic_scores(f)
        return min(sc['combined'], 1.0)

    # ── Flags ─────────────────────────────────────────
    def get_flags(self, features, scores):
        flags = []
        def add(sev, type_, desc): flags.append({'severity':sev,'type':type_,'desc':desc})
        e  = features.get('elapsed_seconds', 99)
        mm = features.get('mouse_moves', 99)
        ks = features.get('avg_keystroke_ms', 200)
        kd = features.get('keystroke_stddev', 60)
        pr = features.get('avg_click_prec', 0.5)
        sc = features.get('scroll_events', 5)
        pa = features.get('copy_pastes', 0)
        tb = features.get('tab_presses', 2)
        cr = features.get('corrections', 2)
        fi = features.get('field_fills', 4)

        if e < 5 and e > 0:
            add('critical','superhuman_speed', f"Form completed in {e:.1f}s (threshold 15s)")
        if ks > 0 and ks < 50:
            add('critical','machine_keystrokes', f"Avg keystroke {ks:.0f}ms, deviation ±{kd:.0f}ms (bot: low variance)")
        if mm < 5 and e > 3:
            add('high','no_mouse', f"Only {mm} mouse events recorded during session")
        if pr > 0.98:
            add('high','pixel_perfect', f"Click precision {pr*100:.2f}% (humans avg 60-85%)")
        if pa > 0:
            add('medium','copy_paste', f"Copy-paste detected in {pa} field(s)")
        if sc == 0 and e > 8:
            add('medium','no_scroll', "Page never scrolled during session")
        if tb > 12:
            add('medium','tab_nav', f"Excessive tab presses: {tb}")
        if cr == 0 and fi > 4:
            add('low','no_corrections', f"Zero corrections across {fi} fields (suspicious accuracy)")
        return flags

    # ── Retrain ───────────────────────────────────────
    def retrain(self, rows):
        if not SKLEARN: return {'error': 'scikit-learn not available'}
        if len(rows) < 20:
            return {'error': f'Need at least 20 sessions, got {len(rows)}'}
        X, y = [], []
        for r in rows:
            X.append([r.get(n, 0) for n in FEATURE_NAMES])
            y.append(int(r.get('is_bot', 0)))
        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y)
        X_sc = self.scaler.fit_transform(X_arr)
        Xtr, Xte, ytr, yte = train_test_split(X_sc, y_arr, test_size=0.2, random_state=42)
        self.rf.fit(Xtr, ytr)
        acc = accuracy_score(yte, self.rf.predict(Xte))
        self.trained_at = datetime.utcnow().isoformat()
        self.train_samples = len(rows)
        self._save_model()
        return {'ok': True, 'accuracy': round(acc, 4), 'samples': len(rows), 'trained_at': self.trained_at}

    # ── Feature Importance ────────────────────────────
    def feature_importance(self):
        if not (SKLEARN and self.trained and self.rf): return []
        imp = self.rf.feature_importances_
        return sorted([{'feature':FEATURE_NAMES[i],'importance':round(float(v),4)}
                        for i,v in enumerate(imp)], key=lambda x:-x['importance'])

    def status(self):
        return {'sklearn':SKLEARN,'trained':self.trained,
                'trained_at':self.trained_at,'samples':self.train_samples}