"""Abuse guards shared by the demo server (ece) and the HF Space:
 - one shloka per request (block bulk paste / bulk download)
 - a per-IP daily render quota
Pure-stdlib, in-memory (resets on process restart — fine for a public demo).

Both caps are env-overridable. Set the env var to 0 to disable the corresponding check —
useful when running demo/server.py locally on your own machine for personal use, where the
per-IP daily quota just gates *you* against yourself."""
import os, threading, datetime

DAILY_LIMIT  = int(os.environ.get("VAGDHENU_DAILY_LIMIT", "10"))    # renders per IP per day (0 = unlimited)
MAX_AKSHARAS = int(os.environ.get("VAGDHENU_MAX_AKSHARAS", "100"))  # one verse — sragdharā is 84 syllables (0 = unlimited)

_lock = threading.Lock()
_counts = {}  # ip -> [date_iso, count]


def _n_aksharas(s):
    """Count Devanagari/Kannada akṣaras (independent vowels + non-virāma-killed consonants)."""
    n = 0; L = len(s)
    for i, c in enumerate(s):
        o = ord(c)
        indep = (0x0905 <= o <= 0x0914) or (0x0C85 <= o <= 0x0C94)
        cons  = (0x0915 <= o <= 0x0939) or (0x0C95 <= o <= 0x0CB9)
        if indep:
            n += 1
        elif cons:
            nxt = s[i+1] if i+1 < L else ""
            if nxt not in ("्", "್"):
                n += 1
    return n


def client_ip(request):
    """Real visitor IP — prefer Cloudflare's header (tunnel), then X-Forwarded-For, then peer."""
    try:
        h = getattr(request, "headers", {}) or {}
        get = h.get if hasattr(h, "get") else (lambda k, d=None: d)
        ip = get("cf-connecting-ip") or (get("x-forwarded-for", "") or "").split(",")[0].strip()
        if not ip and getattr(request, "client", None):
            ip = request.client.host
        return ip or "unknown"
    except Exception:
        return "unknown"


def validate_one_shloka(text):
    """Return an error message if the input is not a single shloka, else None."""
    t = (text or "").replace("।।", "॥")
    if t.count("॥") >= 2:
        return "Please enter just one shloka at a time 🙏"
    if MAX_AKSHARAS and _n_aksharas(text) > MAX_AKSHARAS:
        return "That looks longer than one shloka — please paste a single verse 🙏"
    return None


def check_and_count(ip):
    """True if under today's limit (and records the use); False if the daily quota is exhausted.
    DAILY_LIMIT=0 disables the cap and always returns True (still records the count for logging)."""
    today = datetime.date.today().isoformat()
    with _lock:
        rec = _counts.get(ip)
        if not rec or rec[0] != today:
            rec = [today, 0]; _counts[ip] = rec
        if DAILY_LIMIT and rec[1] >= DAILY_LIMIT:
            return False
        rec[1] += 1
        return True
