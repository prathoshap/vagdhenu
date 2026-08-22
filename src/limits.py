"""Abuse guards shared by the demo server (ece) and the HF Space:
 - one shloka per request (block bulk paste / bulk download)
 - a per-IP daily render quota
Pure-stdlib, in-memory (resets on process restart — fine for a public demo).

Dry-run / noop paths call validate_one_shloka but must NOT call check_and_count
(planning is free and must not burn the daily GPU quota).
"""
import threading, datetime

DAILY_LIMIT  = 10    # renders per IP per day
MAX_AKSHARAS = 100   # one verse: longest common vṛtta (sragdharā) is 84 syllables; >100 = multiple shlokas

_lock = threading.Lock()
_counts = {}  # ip -> [date_iso, count]


def _is_vedic_mark(c):
    if not c:
        return False
    o = ord(c[0])
    return (0x0951 <= o <= 0x0954) or (0x1CD0 <= o <= 0x1CFF)


def _n_aksharas(s):
    """Count Devanagari/Kannada akṣaras (independent vowels + non-virāma-killed consonants).
    Skips Vedic pitch marks (U+0951–U+0954, U+1CD0–U+1CFF) so accented saṃhitā is not
    over-counted; counts ॐ (U+0950) as one akṣara."""
    n = 0; L = len(s)
    for i, c in enumerate(s):
        if _is_vedic_mark(c):
            continue
        o = ord(c)
        if o == 0x0950:                       # ॐ
            n += 1; continue
        indep = (0x0905 <= o <= 0x0914) or (0x0C85 <= o <= 0x0C94)
        cons  = (0x0915 <= o <= 0x0939) or (0x0C95 <= o <= 0x0CB9)
        if indep:
            n += 1
        elif cons:
            nxt = s[i+1] if i+1 < L else ""
            if nxt not in ("्", "್") and not _is_vedic_mark(nxt):
                n += 1
    return n


def has_vedic_marks(text):
    return any(_is_vedic_mark(c) for c in (text or ""))


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
    """Return an error message if the input is not a single shloka, else None.
    Vedic pitch marks are ignored for length checks (they are metadata, not extra akṣaras)."""
    t = (text or "").replace("।।", "॥")
    if t.count("॥") >= 2:
        return "Please enter just one shloka at a time 🙏"
    if _n_aksharas(text) > MAX_AKSHARAS:
        return "That looks longer than one shloka — please paste a single verse 🙏"
    return None


def vedic_noop_warning(text):
    """Advisory for dry-run/demo when input carries Vedic svara marks (not an error)."""
    if has_vedic_marks(text):
        return ("Vedic accents detected — plan/noop keeps them as syllable metadata; "
                "audio render would chant classical prosody (pitch marks are not synthesized).")
    return None


def check_and_count(ip):
    """True if under today's limit (and records the use); False if the daily quota is exhausted.
    Call ONLY on real GPU renders — never on dry-run/noop planning."""
    today = datetime.date.today().isoformat()
    with _lock:
        rec = _counts.get(ip)
        if not rec or rec[0] != today:
            rec = [today, 0]; _counts[ip] = rec
        if rec[1] >= DAILY_LIMIT:
            return False
        rec[1] += 1
        return True


def peek_count(ip):
    """Today's render count for ip without incrementing (for dry-run status lines)."""
    today = datetime.date.today().isoformat()
    with _lock:
        rec = _counts.get(ip)
        if not rec or rec[0] != today:
            return 0
        return rec[1]
