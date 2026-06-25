import os, re, json, time, secrets, smtplib, ssl, datetime
from email.message import EmailMessage
from dotenv import load_dotenv
load_dotenv()
from functools import wraps
import joblib, numpy as np
from flask import (Flask, request, jsonify, session,
                   redirect, url_for, render_template)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(BASE, "artifacts")
DATA_DIR  = os.path.join(BASE, "data")
USERS_FILE = os.path.join(DATA_DIR, "users.json")
os.makedirs(DATA_DIR, exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates")
app.secret_key = os.getenv("SECRET_KEY") or secrets.token_hex(32)
# GOOGLE_CLIENT_ID removed – Google sign‑in disabled

# ── User store (JSON file, no DB needed) ──────────────────────────────────────
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE) as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

from werkzeug.security import generate_password_hash, check_password_hash

def hash_pw(password):
    # Use a strong password hash (PBKDF2) via Werkzeug (built on top of hashlib)
    return generate_password_hash(password, method='pbkdf2:sha256')

# ── Stopwords ─────────────────────────────────────────────────────────────────
_SW = set("""a an the and or but in on at to of for is are was were be been being
have has had do does did will would shall should may might must can could
it its this that these those i me my we our you your he his she her they their them
with by from up about into over after what which who when where how all each every no not""".split())

_TAGLINES = [
    "the truth changes everything",
    "one wrong move means death",
    "the consequences are irreversible",
    "loyalties are tested",
    "nothing is what it seems",
    "the real enemy is closer than anyone thinks",
    "time is running out",
    "secrets have a way of surfacing",
    "every choice has a cost",
    "there is no going back",
]

def _strip_taglines(text):
    text = text.lower()
    for tag in _TAGLINES:
        text = text.replace(tag, "")
    return text.strip()

def preprocess(text):
    text = _strip_taglines(text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(t for t in text.split() if t not in _SW and len(t) > 2)

# ── Genre meta ────────────────────────────────────────────────────────────────
EMOJI = {"Action":"💥","Adventure":"🗺️","Comedy":"😂","Crime":"🕵️",
         "Drama":"🎭","Fantasy":"🧙","Horror":"👻","Romance":"❤️",
         "Sci-Fi":"🚀","Thriller":"😱"}

EXAMPLES = {
    "Action":    ["Die Hard (1988)", "Mad Max: Fury Road (2015)", "John Wick (2014)"],
    "Adventure": ["Indiana Jones (1981)", "The Mummy (1999)", "Jumanji (1995)"],
    "Comedy":    ["Superbad (2007)", "The Grand Budapest Hotel (2014)", "Bridesmaids (2011)"],
    "Crime":     ["Pulp Fiction (1994)", "The Godfather (1972)", "No Country for Old Men (2007)"],
    "Drama":     ["The Shawshank Redemption (1994)", "Forrest Gump (1994)", "Schindler's List (1993)"],
    "Fantasy":   ["The Lord of the Rings (2001)", "Harry Potter (2001)", "Pan's Labyrinth (2006)"],
    "Horror":    ["The Exorcist (1973)", "Get Out (2017)", "Hereditary (2018)"],
    "Romance":   ["Titanic (1997)", "Casablanca (1942)", "Notting Hill (1999)"],
    "Sci-Fi":    ["Interstellar (2014)", "The Matrix (1999)", "Blade Runner 2049 (2017)"],
    "Thriller":  ["Silence of the Lambs (1991)", "Se7en (1995)", "Parasite (2019)"],
}

KEYWORDS = {
    "Fantasy":   ["wizard","magic","spell","dragon","sorcerer","enchanted","kingdom","mage","curse","witch"],
    "Sci-Fi":    ["space","alien","robot","android","galaxy","wormhole","spacecraft","planet","astronaut","quantum","future"],
    "Romance":   ["love","romance","relationship","couple","marriage","wedding","heartbreak","soulmate","feelings"],
    "Horror":    ["ghost","haunted","demon","possessed","monster","supernatural","evil","nightmare","creature","spirit","ritual"],
    "Action":    ["battle","fight","assassin","soldier","weapon","explosion","chase","combat","mercenary","hostage","raid"],
    "Crime":     ["detective","murder","cartel","mafia","heist","robbery","criminal","corruption","evidence","smuggling"],
    "Thriller":  ["conspiracy","suspense","spy","secret","danger","surveillance","mole","blackmail","stalker","paranoia"],
    "Comedy":    ["funny","hilarious","mishap","prank","chaos","awkward","absurd","humor","misunderstanding","ridiculous"],
    "Drama":     ["struggle","grief","tragedy","redemption","conflict","emotional","loss","sacrifice","hardship","healing"],
    "Adventure": ["journey","quest","explore","expedition","treasure","survive","wilderness","discover","escape","ruins"],
}

# ── Load ML model ─────────────────────────────────────────────────────────────
model  = joblib.load(os.path.join(ARTIFACTS, "model.pkl"))
bundle = joblib.load(os.path.join(ARTIFACTS, "label_encoder.pkl"))
GENRES = bundle["genres"]

# ── Auth decorator ────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated

# ── Routes: Auth pages ────────────────────────────────────────────────────────
@app.route("/")
def home():
    if "username" in session:
        return redirect(url_for("predict_page"))
    return render_template("index.html")

@app.route("/login", methods=["GET"])
def login_page():
    # Render login page with Google client ID

    if "username" in session:
        return redirect(url_for("predict_page"))
    return render_template("login.html")

@app.route("/register", methods=["GET"])
def register_page():
    if "username" in session:
        return redirect(url_for("predict_page"))
    return render_template("register.html")

@app.route("/forgot", methods=["GET"])
def forgot_page():
    return render_template("forgot.html")

@app.route("/predict")
@login_required
def predict_page():
    return render_template("predict.html", username=session["username"])

# ── API: Register ─────────────────────────────────────────────────────────────
@app.route("/api/register", methods=["POST"])
def api_register():
    data     = request.get_json(silent=True) or {}
    name     = (data.get("name") or "").strip()
    username = (data.get("username") or "").strip().lower()
    email    = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "").strip()

    # Validation
    if not name or not username or not email or not password:
        return jsonify({"error": "All fields are required"}), 400
    if len(username) < 3:
        return jsonify({"error": "Username must be at least 3 characters"}), 400
    if not re.match(r"^[a-z0-9_]+$", username):
        return jsonify({"error": "Username can only contain letters, numbers, underscores"}), 400
    if not re.match(r"^[^@]+@[^@]+\.[^@]+$", email):
        return jsonify({"error": "Please enter a valid email address"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    users = load_users()
    if username in users:
        return jsonify({"error": "Username already taken"}), 409
    if any(u["email"] == email for u in users.values()):
        return jsonify({"error": "Email already registered"}), 409

    users[username] = {
        "name":       name,
        "email":      email,
        "password":   hash_pw(password),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    save_users(users)
    session["username"] = username
    session["name"]     = name
    return jsonify({"ok": True, "redirect": "/predict"}), 201

# ── API: Login ────────────────────────────────────────────────────────────────
@app.route("/api/login", methods=["POST"])
def api_login():
    data     = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip().lower()
    password = (data.get("password") or "").strip()

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    users = load_users()
    user  = users.get(username)
    if not user:
        return jsonify({"error": "User not found. Please register first.", "needs_register": True}), 401
    if not check_password_hash(user["password"], password):
        # Backward compat: old accounts used hashlib.sha256 (unsalted)
        import hashlib
        if user["password"] != hashlib.sha256(password.encode()).hexdigest():
            return jsonify({"error": "Invalid password. Please try again."}), 401
        # Upgrade old hash to werkzeug format
        user["password"] = hash_pw(password)
        save_users(users)

    session["username"] = username
    session["name"]     = user["name"]
    return jsonify({"ok": True, "redirect": "/predict"})

# ── API: Google Login (disabled) ─────────────────────────────────────────────────
@app.route("/api/login/google", methods=["POST"])
def api_login_google():
    # Google sign‑in has been disabled in this deployment.
    return jsonify({"error": "Google login is disabled"}), 400

# ── API: Logout ───────────────────────────────────────────────────────────────

# ── OTP handling for forgot password ───────────────────────────────────────────
OTP_FILE = os.path.join(DATA_DIR, "otps.json")

def load_otps():
    if not os.path.exists(OTP_FILE):
        return {}
    with open(OTP_FILE) as f:
        return json.load(f)

def save_otps(otps):
    with open(OTP_FILE, "w") as f:
        json.dump(otps, f, indent=2)

def generate_otp(length=6):
    return "".join(str(secrets.randbelow(10)) for _ in range(length))

def send_email(to_addr, subject, body):
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "465"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    if not all([smtp_host, smtp_user, smtp_pass]):
        return False, "SMTP configuration missing"
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_addr
    msg.set_content(body)
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context) as server:
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return True, "Email sent"
    except Exception as e:
        return False, str(e)

# ── API: Forgot password – request OTP ─────────────────────────────────────────────────
@app.route("/api/forgot/request", methods=["POST"])
def api_forgot_request():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    if not email:
        return jsonify({"error": "Email is required"}), 400
    users = load_users()
    user_key = None
    for u, d in users.items():
        if d.get("email") == email:
            user_key = u
            break
    if user_key:
        otp = generate_otp()
        expires = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=10)).isoformat().replace("+00:00", "Z")
        otps = load_otps()
        otps[email] = {"otp": otp, "expires_at": expires}
        save_otps(otps)
        subject = "MovieMind password reset OTP"
        body = f"Your OTP for resetting your MovieMind password is: {otp}\nIt expires in 10 minutes."
        ok, msg = send_email(email, subject, body)
        if not ok:
            return jsonify({"error": "Failed to send email. Check SMTP configuration."}), 500
    return jsonify({"ok": True, "message": "If the email exists, an OTP has been sent"}), 200

# ── API: Forgot password – verify OTP & set new password ───────────────────────────────────
@app.route("/api/forgot/verify", methods=["POST"])
def api_forgot_verify():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    otp = (data.get("otp") or "").strip()
    new_password = (data.get("new_password") or "").strip()
    if not email or not otp or not new_password:
        return jsonify({"error": "Email, OTP and new password are required"}), 400
    otps = load_otps()
    entry = otps.get(email)
    if not entry:
        return jsonify({"error": "Invalid or expired OTP"}), 400
    try:
        exp = datetime.datetime.fromisoformat(entry["expires_at"].replace("Z", "+00:00"))
    except Exception:
        exp = datetime.datetime.now(datetime.timezone.utc)
    if datetime.datetime.now(datetime.timezone.utc) > exp:
        otps.pop(email, None)
        save_otps(otps)
        return jsonify({"error": "OTP has expired"}), 400
    if entry["otp"] != otp:
        return jsonify({"error": "Invalid OTP"}), 400
    users = load_users()
    user_key = None
    for u, d in users.items():
        if d.get("email") == email:
            user_key = u
            break
    if not user_key:
        return jsonify({"error": "User not found"}), 404
    users[user_key]["password"] = hash_pw(new_password)
    save_users(users)
    otps.pop(email, None)
    save_otps(otps)
    return jsonify({"ok": True, "message": "Password has been reset"})

@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"ok": True, "redirect": "/login"})

# ── API: Predict ──────────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    data = request.get_json(silent=True) or {}
    plot = (data.get("plot") or "").strip()

    if not plot:
        return jsonify({"error": "Plot cannot be empty"}), 400
    if len(plot) < 20:
        return jsonify({"error": "Please enter a longer plot (at least 20 characters)"}), 400

    t0    = time.perf_counter()
    x     = preprocess(plot)
    probs = np.array(model.predict_proba([x])[0], dtype=float)
    elapsed = time.perf_counter() - t0

    top_idx = np.argsort(-probs)[:3]
    genre   = GENRES[top_idx[0]]
    confidence = round(min(float(probs[top_idx[0]]) * 100.0, 97.5), 1)
    top3 = [{"genre": GENRES[i], "score": round(float(probs[i]) * 100.0, 1)} for i in top_idx]

    # Keywords
    lower = plot.lower()
    kw = [k for k in KEYWORDS.get(genre, []) if k in lower]
    if len(kw) < 3:
        for g, keys in KEYWORDS.items():
            if g != genre:
                kw += [k for k in keys if k in lower and k not in kw]
            if len(kw) >= 8: break
    if not kw:
        kw = [w for w in lower.split() if w not in _SW and len(w) > 4][:8]

    return jsonify({
        "genre":           genre,
        "emoji":           EMOJI.get(genre, "🎬"),
        "confidence":      confidence,
        "top3":            top3,
        "keywords":        kw[:8],
        "movies":          EXAMPLES.get(genre, []),
        "prediction_time": f"{elapsed:.3f}s",
    })

# ── API: Session check ────────────────────────────────────────────────────────
@app.route("/api/me")
def api_me():
    if "username" in session:
        return jsonify({"logged_in": True, "username": session["username"], "name": session["name"]})
    return jsonify({"logged_in": False})

if __name__ == "__main__":
    print("[MovieMind] Running on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
