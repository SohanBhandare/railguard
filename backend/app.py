"""
IndiaRail Bot Detection System — Flask Backend v2.0
REST API for bot analysis, session management, and admin dashboard.
Database: SQLite3 | ML: scikit-learn Random Forest + Isolation Forest
"""
import os, json, random, string
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from database import Database
from ml_engine import BotDetector

BASE     = os.path.dirname(os.path.abspath(__file__))
FRONTEND = os.path.join(BASE, '..', 'frontend')

app = Flask(__name__, static_folder=os.path.join(FRONTEND, 'static'))
CORS(app, resources={r"/api/*": {"origins": "*"}})

db = Database()
ml = BotDetector()

# ── helpers ───────────────────────────────────────────
def gen_session_id():
    return 'RG-' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

def gen_pnr():
    return str(random.randint(1000000000, 9999999999))

def get_risk_level(prob):
    if prob >= 0.75: return 'critical'
    if prob >= 0.55: return 'high'
    if prob >= 0.35: return 'medium'
    return 'low'

def get_ip():
    return (request.headers.get('X-Forwarded-For','').split(',')[0].strip()
            or request.remote_addr or '127.0.0.1')

# ── static / pages ────────────────────────────────────
@app.route('/')
def index(): return send_from_directory(FRONTEND, 'index.html')

@app.route('/admin')
def admin(): return send_from_directory(FRONTEND, 'admin.html')


# ── API ───────────────────────────────────────────────
@app.route('/api/session/start', methods=['POST'])
def session_start():
    sid = gen_session_id()
    ip  = get_ip()
    ua  = request.headers.get('User-Agent','')
    db.create_session(sid, ip, ua)
    db.upsert_ip(ip)
    return jsonify({'session_id': sid, 'status': 'ok'})

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json(force=True) or {}
    sid  = data.get('session_id','')
    ip   = get_ip()

    features = {k: float(data.get(k,0)) if k not in ('mouse_moves','scroll_events','hover_events',
                'total_clicks','copy_pastes','tab_presses','field_fills','corrections','window_blurs')
                else int(data.get(k,0))
                for k in ('elapsed_seconds','form_fill_time','mouse_moves','avg_mouse_speed',
                          'scroll_events','hover_events','avg_keystroke_ms','keystroke_stddev',
                          'avg_click_prec','total_clicks','copy_pastes','tab_presses',
                          'field_fills','corrections','window_blurs','fp_score')}

    ip_info    = db.get_ip_info(ip)
    ip_bot_rate = 0.0
    if ip_info and ip_info['total_attempts'] > 0:
        ip_bot_rate = ip_info['bot_attempts'] / ip_info['total_attempts']

    scores = ml.predict(features, ip_bot_rate=ip_bot_rate)
    prob   = scores['combined']
    is_bot = prob >= 0.60
    risk   = get_risk_level(prob)

    db.save_features(sid, features, scores)
    verdict = 'bot' if is_bot else ('medium_risk' if prob >= 0.35 else 'human')
    db.update_session_verdict(sid, verdict, prob, is_bot, risk)
    db.update_ip_stats(ip, is_bot)

    flags = ml.get_flags(features, scores)
    for f in flags:
        db.save_flag(sid, f['severity'], f['type'], f['desc'])
    db.log_event('bot_blocked' if is_bot else 'human_passed', sid, f"prob={prob:.3f}")

    return jsonify({'session_id':sid,'is_bot':is_bot,'probability':round(prob,4),
                    'risk_level':risk,'verdict':verdict,'scores':scores,'flags':flags,
                    'ip_bot_rate':round(ip_bot_rate,3)})

@app.route('/api/booking/confirm', methods=['POST'])
def booking_confirm():
    data   = request.get_json(force=True) or {}
    sid    = data.get('session_id','')
    is_bot = data.get('is_bot', False)
    pnr    = gen_pnr() if not is_bot else None
    booking = {
        'train_number':    data.get('train_number',''),
        'train_name':      data.get('train_name',''),
        'from_station':    data.get('from_station',''),
        'to_station':      data.get('to_station',''),
        'journey_date':    data.get('journey_date',''),
        'seat_class':      data.get('seat_class',''),
        'passenger_count': int(data.get('passenger_count',1)),
        'total_fare':      float(data.get('total_fare',0)),
        'payment_method':  data.get('payment_method',''),
        'pnr_number':      pnr,
        'status':          'blocked' if is_bot else 'confirmed',
    }
    db.save_booking(sid, booking)
    db.update_session_completed(sid)
    return jsonify({'pnr': pnr, 'status': booking['status']})

@app.route('/api/captcha/result', methods=['POST'])
def captcha_result():
    data   = request.get_json(force=True) or {}
    sid    = data.get('session_id','')
    passed = bool(data.get('passed', False))
    db.update_captcha(sid, passed)
    db.log_event('captcha_result', sid, f"passed={passed}")
    return jsonify({'ok': True, 'passed': passed})

@app.route('/api/dashboard', methods=['GET'])
def dashboard():
    return jsonify(db.get_dashboard_data())

@app.route('/api/sessions', methods=['GET'])
def sessions():
    limit  = min(int(request.args.get('limit', 50)), 200)
    offset = int(request.args.get('offset', 0))
    return jsonify({'sessions': db.get_sessions(limit, offset), 'limit': limit, 'offset': offset})

@app.route('/api/sessions/<sid>', methods=['GET'])
def session_detail(sid):
    detail = db.get_session_detail(sid)
    if not detail: return jsonify({'error': 'Not found'}), 404
    return jsonify(detail)

@app.route('/api/ip-intel', methods=['GET'])
def ip_intel():
    return jsonify({'records': db.get_ip_intel()})

@app.route('/api/ip-intel/block', methods=['POST'])
def block_ip():
    data = request.get_json(force=True) or {}
    ip   = data.get('ip_address','')
    if not ip: return jsonify({'error': 'ip_address required'}), 400
    db.block_ip(ip, data.get('reason','Manual block'))
    return jsonify({'ok': True, 'ip': ip, 'blocked': True})

@app.route('/api/charts/hourly', methods=['GET'])
def chart_hourly():
    return jsonify({'data': db.get_hourly_stats()})

@app.route('/api/feature-importance', methods=['GET'])
def feature_importance():
    return jsonify({'importance': ml.feature_importance()})

@app.route('/api/retrain', methods=['POST'])
def retrain():
    rows   = db.get_training_data()
    result = ml.retrain(rows)
    db.log_event('model_retrained', None, f"sessions={len(rows)} acc={result.get('accuracy','?')}")
    return jsonify(result)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status':'ok','version':'2.0','db':db.ping(),
                    'ml':ml.status(),'time':datetime.utcnow().isoformat()+'Z'})

if __name__ == '__main__':
    print("="*52)
    print("  IndiaRail Bot Detection API  v2.0")
    print("  Booking : http://127.0.0.1:5000/")
    print("  Admin   : http://127.0.0.1:5000/admin")
    print("  Health  : http://127.0.0.1:5000/api/health")
    print("="*52)
    app.run(debug=True, host='0.0.0.0', port=5000)
