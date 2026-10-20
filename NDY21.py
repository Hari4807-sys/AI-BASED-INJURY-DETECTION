"""
Form Check - real-time movement & injury-risk tracking
FastAPI + MediaPipe Pose Landmarker.

Run:  python main.py      then open http://127.0.0.1:8000
Needs pose_landmarker_lite.task next to this file (or set POSE_MODEL_PATH).
Optional env: CAMERA_INDEX (default 0)
"""
import os
import re
import sys
import time
import uuid
import html
import shutil
import tempfile
import threading
from collections import deque, Counter

import cv2
import numpy as np
import requests
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse

app = FastAPI()

# ---------------- Config ----------------
MODEL_PATH = os.environ.get("POSE_MODEL_PATH", "pose_landmarker_lite.task")
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))

TELEGRAM_BOT_TOKEN = "8922151814:AAF9QKFUGbjeJjIwDtKPrR7YkoWiPYm5hF8"
TELEGRAM_CHAT_ID =  "8905406897"
ALERT_COOLDOWN_SECONDS = 30

DETECT_EVERY_N_FRAMES = 3     # run the pose model on 1 of every N frames
DETECTION_MAX_WIDTH = 480     # detection runs on a downscaled copy
DISPLAY_MAX_WIDTH = 960       # what the browser receives
JPEG_QUALITY = 80
IDLE_TIMEOUT_SECONDS = 15     # stop a session if the browser stops polling
MIN_VISIBILITY = 0.5

EXERCISE = "squat"
EXERCISES = {
    "squat":   {"angle_label": "KNEE ANGLE", "zone": (70, 110),  "metric": "REPS"},
    "plank":   {"angle_label": "BODY LINE",  "zone": (160, 180), "metric": "HOLD"},
    "running": {"angle_label": "KNEE ANGLE", "zone": None,       "metric": "GAIT"},
}

SQUAT_DOWN_TRIGGER = 140      # knee angle below this = a rep has started
SQUAT_UP_TRIGGER = 155        # back above this = rep finished
ASYMMETRY_THRESHOLD_DEGREES = 15
MISS_FRAME_THRESHOLD = 15     # consecutive bad detections before a confirmed alert
RUN_WINDOW_SECONDS = 2.0
RUN_MIN_ROM = 20              # knee range of motion (deg) below which a leg looks stiff
RUN_ROM_DIFF = 20             # left/right range-of-motion difference that looks uneven

# MediaPipe pose indices
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
LEG = {"L": (L_HIP, L_KNEE, L_ANKLE), "R": (R_HIP, R_KNEE, R_ANKLE)}
BODY = {"L": (L_SHOULDER, L_HIP, L_ANKLE), "R": (R_SHOULDER, R_HIP, R_ANKLE)}
BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
         (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (29, 31),
         (27, 31), (28, 30), (30, 32), (28, 32)]

LEVEL_BGR = {"good": (176, 227, 127), "bad": (140, 105, 226),
             "warn": (127, 232, 215), "idle": (163, 148, 138)}


# ---------------- Telegram ----------------
_last_alert_time = 0.0
_alert_lock = threading.Lock()


def send_telegram_alert(message: str, force: bool = False):
    """Non-blocking: never stalls the video stream."""
    global _last_alert_time
    if "YOUR_" in TELEGRAM_BOT_TOKEN or "YOUR_" in TELEGRAM_CHAT_ID:
        return
    with _alert_lock:
        now = time.time()
        if not force and now - _last_alert_time < ALERT_COOLDOWN_SECONDS:
            return
        _last_alert_time = now

    def _send():
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message,
                                     "parse_mode": "HTML", "disable_web_page_preview": True}, timeout=6)
        except Exception as e:
            print(f"Telegram alert failed: {e}")

    threading.Thread(target=_send, daemon=True).start()


# ---------------- Telegram message content ----------------
EX_META = {"squat": ("🏋️", "SQUAT"), "plank": ("🧘", "PLANK"), "running": ("🏃", "RUNNING")}

ADVICE = {
    "asym": {
        "title": "Uneven loading",
        "risk": "One side is doing more work, which can overload a single knee or hip over time.",
        "tips": [
            "Set your feet an equal distance from the centre line and turn toes out the same amount.",
            "Drop the load and do slow 3-second lowering reps to re-groove an even push.",
            "Add single-leg work (split squats, step-ups) to even out strength on the weaker side.",
            "Check ankle and hip mobility on the side that lags behind.",
        ],
    },
    "shallow": {
        "title": "Shallow reps",
        "risk": "Partial reps cut strength and muscle gains and can shift stress onto the knees.",
        "tips": [
            "Squat to a box or bench set at your target depth until the range feels automatic.",
            "Keep your chest up and brace your core before every rep.",
            "Work on ankle mobility, or raise the heels slightly, if depth is limited.",
            "Lower the load until you can reach full depth with control.",
        ],
    },
    "deep": {
        "title": "Very deep reps",
        "risk": "Dropping past your controlled range can strain the knees and round the lower back.",
        "tips": [
            "Stop the descent where your lower back starts to tuck under.",
            "Control the bottom position instead of bouncing out of it.",
            "Improve hip and ankle mobility before chasing more depth.",
        ],
    },
    "sag": {
        "title": "Hips sagging",
        "risk": "A sagging plank loads the lower back and takes the work away from your core.",
        "tips": [
            "Squeeze your glutes and tuck your pelvis slightly, with ribs pulled down.",
            "Keep elbows directly under shoulders and push the floor away.",
            "Shorten the hold and rest, or drop to a knee plank until form stays clean.",
        ],
    },
    "pike": {
        "title": "Hips piking",
        "risk": "Hips too high reduce core engagement and turn the plank into a shoulder hold.",
        "tips": [
            "Lower your hips until shoulders, hips and ankles form one straight line.",
            "Brace your abs as if bracing for a light punch and breathe steadily.",
            "Look at a spot just ahead of your hands to keep your neck neutral.",
        ],
    },
    "stiff": {
        "title": "Stiff leg / limited knee motion",
        "risk": "Reduced knee bend usually means overstriding or a tight, guarded leg, which raises impact load.",
        "tips": [
            "Warm up with leg swings and easy strides before running.",
            "Shorten your stride and land closer under your body.",
            "Check for tightness or a niggle on the stiff side, and rest it if it hurts.",
        ],
    },
    "uneven": {
        "title": "Uneven stride",
        "risk": "One leg moving less than the other is a common sign of compensation and overuse injuries.",
        "tips": [
            "Film from behind and side to see which leg is doing less.",
            "Add single-leg strength work (step-ups, calf raises, single-leg RDLs).",
            "Slow down and focus on an even, relaxed cadence.",
            "See a physio or gait specialist if it comes with pain.",
        ],
    },
}


def _esc(s):
    return html.escape(str(s), quote=False)


def format_alert(exercise, msg, key, context=None):
    icon, name = EX_META.get(exercise, ("🏃", exercise.upper()))
    adv = ADVICE.get(key)
    lines = [f"⚠️ <b>FORM ALERT</b>  ·  {icon} {name}",
             "━━━━━━━━━━━━━━━━",
             f"🔴 <b>Issue:</b> {_esc(msg)}"]
    if adv:
        lines.append(f"🩺 <b>Why it matters:</b> {_esc(adv['risk'])}")
    if context:
        lines.append(f"📊 <b>Progress:</b> {_esc(context)}")
    if adv:
        lines += ["", "💡 <b>Recommendations</b>"]
        lines += [f"  {i}. {_esc(t)}" for i, t in enumerate(adv["tips"][:3], 1)]
    lines += ["", f"🕐 {time.strftime('%H:%M:%S')}  ·  <i>Stop if you feel sharp pain.</i>"]
    return "\n".join(lines)


def _fmt_duration(sec):
    sec = int(sec)
    return f"{sec // 60}m {sec % 60:02d}s" if sec >= 60 else f"{sec}s"


def format_summary(exercise, source_label, duration, score, alerts, issues,
                   reps=0, bad_reps=0, hold=0.0):
    icon, name = EX_META.get(exercise, ("🏃", exercise.upper()))
    filled = max(0, min(10, round(score / 10)))
    bar = "▓" * filled + "░" * (10 - filled)
    if score >= 90:
        verdict = "🏆 Excellent form - keep it up!"
    elif score >= 75:
        verdict = "👍 Solid session with a few things to tidy up."
    elif score >= 50:
        verdict = "🛠 Needs work - focus on the fixes below."
    else:
        verdict = "🚧 Form broke down a lot - lighten up and review the cues."

    lines = [f"✅ <b>SESSION SUMMARY</b>  ·  {icon} {name}",
             "━━━━━━━━━━━━━━━━",
             f"📍 <b>Source:</b> {_esc(source_label)}",
             f"⏱ <b>Duration:</b> {_fmt_duration(duration)}",
             f"🎯 <b>Form score:</b> {bar} {score:.0f}%"]
    if exercise == "squat":
        lines.append(f"🔁 <b>Reps:</b> {reps}  ({reps - bad_reps} clean · {bad_reps} flagged)")
    elif exercise == "plank":
        lines.append(f"⏳ <b>Hold time:</b> {int(hold) // 60}:{int(hold) % 60:02d}")
    lines.append(f"🚨 <b>Alerts:</b> {alerts}")

    top = [(k, n) for k, n in issues.most_common(3) if k in ADVICE]
    if top:
        lines += ["", "📌 <b>What came up</b>"]
        lines += [f"  • {_esc(ADVICE[k]['title'])} ×{n}" for k, n in top]
        lines += ["", "💡 <b>Recommendations</b>"]
        for k, _ in top[:2]:
            lines.append(f"<b>{_esc(ADVICE[k]['title'])}</b>")
            lines += [f"  {i}. {_esc(t)}" for i, t in enumerate(ADVICE[k]["tips"][:3], 1)]
    else:
        lines += ["", "🎉 <b>No issues flagged</b> - nothing to fix this session."]
    lines += ["", verdict, "<i>Automated feedback, not medical advice.</i>"]
    return "\n".join(lines)


# ---------------- Sessions ----------------
class Session:
    """Per-stream state, shared between the frame loop and the /api/state poller."""

    def __init__(self, exercise, kind, sid=None):
        self.id = sid if sid and re.fullmatch(r"[A-Za-z0-9_-]{8,64}", sid) else uuid.uuid4().hex
        self.exercise = exercise
        self.kind = kind                  # "live" | "upload"
        self.path = None                  # temp file for uploads
        self.started = False
        self.created = time.time()
        self.last_poll = time.time()
        self.stop = threading.Event()
        self.finished = threading.Event()
        self.lock = threading.Lock()
        cfg = EXERCISES[exercise]
        self.angle_label = cfg["angle_label"]
        self.zone = cfg["zone"]
        self.metric_label = cfg["metric"]
        self.metric_value = "0" if exercise == "squat" else ("0s" if exercise == "plank" else "-")
        self.status = "STARTING"
        self.level = "idle"
        self.angle = None
        self.alerts = 0
        self.fps = 0.0
        self.tracking = False
        self.progress = None
        self.error = None
        self.done = False
        self.events = []
        _SESSIONS[self.id] = self

    def log(self, text, kind="good"):
        with self.lock:
            self.events.append({"id": len(self.events) + 1, "t": time.time(), "text": text, "kind": kind})

    def fail(self, msg):
        self.error = msg
        self.status = "ERROR"
        self.level = "bad"
        self.log(msg, "bad")

    def snapshot(self, since):
        with self.lock:
            self.last_poll = time.time()
            events = [e for e in self.events if e["id"] > since]
        return {
            "status": self.status, "level": self.level, "angle": self.angle,
            "angle_label": self.angle_label, "zone": self.zone,
            "metric_label": self.metric_label, "metric_value": self.metric_value,
            "alerts": self.alerts, "fps": round(self.fps), "tracking": self.tracking,
            "progress": self.progress, "error": self.error, "done": self.done,
            "events": events,
        }


_SESSIONS = {}
_ACTIVE_LIVE = None
_live_lock = threading.Lock()


def _purge_sessions():
    now = time.time()
    for sid, s in list(_SESSIONS.items()):
        if s.done and now - s.last_poll > 600:
            _SESSIONS.pop(sid, None)
        elif not s.started and s.path and now - s.created > 600:
            try:
                os.remove(s.path)
            except OSError:
                pass
            _SESSIONS.pop(sid, None)


# ---------------- Geometry ----------------
class EMA:
    def __init__(self, alpha=0.45):
        self.a, self.v = alpha, None

    def update(self, x):
        self.v = x if self.v is None else self.a * x + (1 - self.a) * self.v
        return self.v

    def reset(self):
        self.v = None


def angle3d(a, b, c):
    """Angle at b (degrees) from three landmarks. Uses world-space x,y,z (metres),
    so it doesn't depend on the video's aspect ratio or on normalisation."""
    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _vis(p):
    v = getattr(p, "visibility", None)
    return 1.0 if v is None else v          # (old code: `v or 1.0` turned 0.0 into 1.0)


def _visible(*pts):
    return all(_vis(p) >= MIN_VISIBILITY for p in pts)


# ---------------- Form analysis ----------------
class FormAnalyzer:
    """Turns landmarks into (level, message, angle) and keeps rep / hold / gait state."""

    def __init__(self, exercise, sess):
        self.ex, self.sess = exercise, sess
        self.knee_ema = {"L": EMA(), "R": EMA()}
        self.body_ema = EMA()
        self.stage, self.rep_min = "up", 180.0
        self.reps = self.bad_reps = 0
        self.flash = None                       # (until, level, message)
        self.streak_msg, self.streak, self.confirmed = None, 0, False
        self.hist = deque()
        self.hold, self.last_t = 0.0, None
        self.bad_samples = self.samples = 0
        self.issues = Counter()

    # -- helpers
    def lost(self):
        self._track(None)
        for e in self.knee_ema.values():
            e.reset()
        self.body_ema.reset()
        self.last_t = None

    def _context(self):
        if self.ex == "squat":
            return f"{self.reps} rep(s) completed so far"
        if self.ex == "plank":
            return f"{int(self.hold)}s held so far"
        return None

    def _raise(self, msg, key, kind="bad"):
        self.sess.alerts += 1
        self.issues[key] += 1
        self.sess.log(msg, kind)
        send_telegram_alert(format_alert(self.ex, msg, key, self._context()))

    def _track(self, msg, key=None):
        """Sustained-issue detector: alert once per episode, not once per frame."""
        if msg is None:
            self.streak_msg, self.streak, self.confirmed = None, 0, False
            return
        if msg == self.streak_msg or (self.streak_msg and msg.split(" - ")[0] == self.streak_msg.split(" - ")[0]):
            self.streak += 1
        else:
            self.streak_msg, self.streak, self.confirmed = msg, 1, False
        if self.streak >= MISS_FRAME_THRESHOLD and not self.confirmed:
            self.confirmed = True
            self._raise(msg, key)

    def _knees(self, norm, world):
        out = {}
        for s, (a, b, c) in LEG.items():
            if _visible(norm[a], norm[b], norm[c]):
                raw = angle3d(world[a], world[b], world[c])
                out[s] = raw if self.ex == "running" else self.knee_ema[s].update(raw)
            else:
                self.knee_ema[s].reset()
        return out

    # -- main entry
    def update(self, norm, world, t):
        dt = 0.0 if self.last_t is None else max(0.0, min(t - self.last_t, 0.5))
        self.last_t = t
        if self.ex == "plank":
            res = self._plank(norm, world, dt)
        else:
            knees = self._knees(norm, world)
            if not knees:
                self._track(None)
                res = ("warn", "LEGS NOT IN FRAME", None)
            else:
                primary = float(np.mean(list(knees.values())))
                res = self._squat(knees, primary) if self.ex == "squat" else self._running(knees, primary, t)
        self.samples += 1
        if res[0] == "bad":
            self.bad_samples += 1
        return res

    # -- squat: rep state machine + depth check at the bottom of each rep
    def _squat(self, knees, primary):
        lo, hi = EXERCISES["squat"]["zone"]
        if self.stage == "up":
            if primary < SQUAT_DOWN_TRIGGER:
                self.stage, self.rep_min = "down", primary
        else:
            self.rep_min = min(self.rep_min, primary)
            if primary >= SQUAT_UP_TRIGGER:
                self.stage = "up"
                self._finish_rep(lo, hi)

        asym = None
        if len(knees) == 2 and primary < SQUAT_DOWN_TRIGGER \
                and abs(knees["L"] - knees["R"]) > ASYMMETRY_THRESHOLD_DEGREES:
            asym = f"Asymmetric loading - L {knees['L']:.0f}° vs R {knees['R']:.0f}°"
        self._track(asym, "asym")
        if asym:
            return "bad", asym, primary
        if self.flash and time.time() < self.flash[0]:
            return self.flash[1], self.flash[2], primary
        if self.stage == "up":
            return "good", "STANDING - READY", primary
        if primary > hi:
            return "good", "IN MOTION", primary
        if primary >= lo:
            return "good", "GOOD DEPTH", primary
        return "bad", "TOO DEEP - CONTROL THE BOTTOM", primary

    def _finish_rep(self, lo, hi):
        self.reps += 1
        self.sess.metric_value = str(self.reps)
        m = self.rep_min
        if m > hi:
            self.bad_reps += 1
            msg = f"Rep {self.reps}: too shallow (bottomed at {m:.0f}°)"
            self.flash = (time.time() + 1.5, "bad", "SHALLOW REP - GO DEEPER")
            self._raise(msg, "shallow")
        elif m < lo:
            self.bad_reps += 1
            msg = f"Rep {self.reps}: very deep (bottomed at {m:.0f}°)"
            self.flash = (time.time() + 1.5, "warn", "VERY DEEP REP")
            self.sess.log(msg, "warn")
            self.issues["deep"] += 1
        else:
            self.flash = (time.time() + 1.2, "good", f"REP {self.reps} - CLEAN")
            self.sess.log(f"Rep {self.reps}: good depth ({m:.0f}°)", "good")

    # -- plank: shoulder-hip-ankle line, sag vs pike, hold timer
    def _plank(self, norm, world, dt):
        lo, hi = EXERCISES["plank"]["zone"]
        sides = {}
        for s, (a, b, c) in BODY.items():
            if _visible(norm[a], norm[b], norm[c]):
                vis = _vis(norm[a]) + _vis(norm[b]) + _vis(norm[c])
                sides[s] = (angle3d(world[a], world[b], world[c]), vis)
        if not sides:
            self._track(None)
            return "warn", "BODY NOT IN FRAME", None
        side = max(sides, key=lambda k: sides[k][1])
        ang = self.body_ema.update(sides[side][0])
        sh, hp, an = (norm[i] for i in BODY[side])
        if not abs(an.x - sh.x) > 1.2 * abs(an.y - sh.y):
            self._track(None)
            return "warn", "GET INTO PLANK - SIDE VIEW", ang
        if ang < lo:
            t = (hp.x - sh.x) / (an.x - sh.x + 1e-6)
            line_y = sh.y + t * (an.y - sh.y)
            sag = hp.y > line_y
            msg = ("Hips sagging - squeeze glutes, lift hips" if sag
                   else "Hips piking - lower hips into a straight line")
            self._track(msg, "sag" if sag else "pike")
            return "bad", msg, ang
        self._track(None)
        self.hold += dt
        self.sess.metric_value = f"{int(self.hold) // 60}:{int(self.hold) % 60:02d}"
        return "good", "SOLID PLANK - HOLD IT", ang

    # -- running: gait symmetry via knee range-of-motion over a rolling window
    def _running(self, knees, primary, t):
        if len(knees) < 2:
            self._track(None)
            return "warn", "SHOW BOTH LEGS - SIDE VIEW", primary
        self.hist.append((t, knees["L"], knees["R"]))
        while self.hist and t - self.hist[0][0] > RUN_WINDOW_SECONDS:
            self.hist.popleft()
        if len(self.hist) < 8 or t - self.hist[0][0] < 1.2:
            return "good", "READING GAIT...", primary
        ls, rs = [h[1] for h in self.hist], [h[2] for h in self.hist]
        rom_l, rom_r = max(ls) - min(ls), max(rs) - min(rs)
        self.sess.metric_value = f"{rom_l:.0f}/{rom_r:.0f}"
        if min(rom_l, rom_r) < RUN_MIN_ROM:
            msg, key = "Stiff leg - limited knee motion", "stiff"
        elif abs(rom_l - rom_r) > RUN_ROM_DIFF:
            msg, key = "Uneven stride - one leg moving less", "uneven"
        else:
            self._track(None)
            return "good", "GAIT SYMMETRIC", primary
        self._track(msg, key)
        return "bad", msg, primary


# ---------------- Drawing ----------------
def draw_skeleton(frame, lms, level, w, h):
    if lms is None:
        return
    color = LEVEL_BGR.get(level, LEVEL_BGR["idle"])
    for a, b in BONES:
        if _visible(lms[a], lms[b]):
            pa, pb = (int(lms[a].x * w), int(lms[a].y * h)), (int(lms[b].x * w), int(lms[b].y * h))
            cv2.line(frame, pa, pb, color, 3, cv2.LINE_AA)
    for i, lm in enumerate(lms):
        if i < 11 or not _visible(lm):
            continue
        key = i in (23, 24, 25, 26, 27, 28)
        c = (int(lm.x * w), int(lm.y * h))
        cv2.circle(frame, c, 7 if key else 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, c, 5 if key else 3, color, -1, cv2.LINE_AA)


def draw_pill(frame, text, level):
    text = text.encode("ascii", "ignore").decode().strip()
    if not text:
        return
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    h, w = frame.shape[:2]
    x1, y1 = 16, 16
    x2, y2 = min(w - 4, x1 + tw + 46), y1 + th + 22
    roi = frame[y1:y2, x1:x2]
    dark = np.full_like(roi, (16, 18, 13))
    frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.3, dark, 0.7, 0)
    cv2.circle(frame, (x1 + 16, (y1 + y2) // 2), 6, LEVEL_BGR.get(level, LEVEL_BGR["idle"]), -1, cv2.LINE_AA)
    cv2.putText(frame, text, (x1 + 32, y1 + th + 11), font, scale, (239, 242, 237), thick, cv2.LINE_AA)


# ---------------- Capture + detection ----------------
def make_detector():
    opts = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,      # temporal tracking, smoother than IMAGE mode
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(opts)


def open_capture(source):
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    return cv2.VideoCapture(source)


def generate_frames(sess: Session, source):
    live = isinstance(source, int)
    cap = open_capture(source)
    detector = None
    duration = 0.0
    analyzer = FormAnalyzer(sess.exercise, sess)
    try:
        if not cap.isOpened():
            sess.fail("Could not open camera - is another app using it?" if live else "Could not read that video file")
            return
        try:
            detector = make_detector()
        except Exception as e:
            sess.fail(f"Pose model failed to load: {e}")
            return

        fps_src = 0 if live else (cap.get(cv2.CAP_PROP_FPS) or 30)
        fps_src = fps_src if fps_src and fps_src > 1 else 30
        total = 0 if live else int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        sess.status, sess.level = ("TRACKING" if live else "ANALYZING"), "idle"
        sess.log("Live session started" if live else "Analysis started", "good")

        t0 = time.time()
        prev = t0
        fps = 0.0
        frame_idx, last_ts, read_failures = 0, -1, 0
        cached, level = None, "idle"

        while not sess.stop.is_set():
            if time.time() - sess.last_poll > IDLE_TIMEOUT_SECONDS:
                break                                   # browser went away
            ok, frame = cap.read()
            if not ok:
                if live:
                    read_failures += 1
                    if read_failures > 30:
                        sess.fail("Camera stopped delivering frames")
                        break
                    time.sleep(0.03)
                    continue
                break
            read_failures = 0

            if live:
                frame = cv2.flip(frame, 1)              # mirror view feels natural
            h, w = frame.shape[:2]
            if w > DISPLAY_MAX_WIDTH:
                frame = cv2.resize(frame, (DISPLAY_MAX_WIDTH, int(h * DISPLAY_MAX_WIDTH / w)))
                h, w = frame.shape[:2]

            frame_idx += 1
            media_t = (time.time() - t0) if live else frame_idx / fps_src
            duration = media_t

            if (frame_idx - 1) % DETECT_EVERY_N_FRAMES == 0:
                small = frame
                if w > DETECTION_MAX_WIDTH:
                    small = cv2.resize(frame, (DETECTION_MAX_WIDTH, int(h * DETECTION_MAX_WIDTH / w)))
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                ts = int(media_t * 1000)
                ts = ts if ts > last_ts else last_ts + 1
                last_ts = ts
                res = detector.detect_for_video(mp_img, ts)

                if res.pose_landmarks and res.pose_world_landmarks:
                    cached = res.pose_landmarks[0]
                    level, msg, angle = analyzer.update(cached, res.pose_world_landmarks[0], media_t)
                    sess.status, sess.level, sess.angle, sess.tracking = msg, level, angle, True
                else:
                    cached, level = None, "idle"
                    analyzer.lost()
                    sess.status, sess.level, sess.angle, sess.tracking = "NO PERSON DETECTED", "idle", None, False

            draw_skeleton(frame, cached, level, w, h)
            draw_pill(frame, sess.status, sess.level)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-3))
            prev, sess.fps = now, fps
            if total:
                sess.progress = min(1.0, frame_idx / total)

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
    finally:
        cap.release()
        if detector is not None:
            detector.close()
        if sess.path and os.path.exists(sess.path):
            try:
                os.remove(sess.path)
            except OSError:
                pass
        if not sess.error:
            sess.status = "ANALYSIS COMPLETE" if not live else "SESSION ENDED"
            sess.level = "idle"
        sess.angle, sess.tracking, sess.progress = None, False, (1.0 if not live and not sess.error else sess.progress)

        if analyzer.samples > 0:
            pct = 100 * analyzer.bad_samples / analyzer.samples
            extra = f"{analyzer.reps} rep(s), {analyzer.bad_reps} flagged, " if sess.exercise == "squat" else ""
            label = "Live session" if live else "Uploaded video"
            sess.log(f"Analysis complete ({sess.exercise}, {label.lower()}): {extra}"
                     f"{sess.alerts} alert(s), {pct:.0f}% of checked frames flagged", "good")
            send_telegram_alert(
                format_summary(sess.exercise, label, duration, 100 - pct, sess.alerts, analyzer.issues,
                               reps=analyzer.reps, bad_reps=analyzer.bad_reps, hold=analyzer.hold),
                force=True,
            )
        sess.done = True
        sess.finished.set()


# ---------------- Routes ----------------
MJPEG = "multipart/x-mixed-replace; boundary=frame"


def _clean_exercise(ex):
    return ex if ex in EXERCISES else EXERCISE


@app.get("/video_feed")
def video_feed(exercise: str = EXERCISE, sid: str = None):
    """Live webcam capture with real-time form tracking."""
    global _ACTIVE_LIVE
    _purge_sessions()
    sess = Session(_clean_exercise(exercise), "live", sid)
    with _live_lock:
        old, _ACTIVE_LIVE = _ACTIVE_LIVE, sess
    if old is not None and not old.finished.is_set():
        old.stop.set()
        old.finished.wait(3)                            # free the camera before opening it again
    return StreamingResponse(generate_frames(sess, CAMERA_INDEX), media_type=MJPEG)


@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...), exercise: str = Form(EXERCISE)):
    _purge_sessions()
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
    sess = Session(_clean_exercise(exercise), "upload")
    sess.path = tmp.name
    return {"sid": sess.id, "stream_url": f"/stream_uploaded/{sess.id}"}


@app.get("/stream_uploaded/{sid}")
def stream_uploaded(sid: str):
    sess = _SESSIONS.get(sid)
    if not sess or sess.kind != "upload" or sess.started or not sess.path or not os.path.exists(sess.path):
        raise HTTPException(status_code=404, detail="Upload not found or already streamed")
    sess.started = True
    return StreamingResponse(generate_frames(sess, sess.path), media_type=MJPEG)


@app.get("/api/state/{sid}")
def api_state(sid: str, since: int = 0):
    sess = _SESSIONS.get(sid)
    if not sess:
        raise HTTPException(status_code=404, detail="Unknown session")
    return sess.snapshot(since)


@app.post("/api/stop/{sid}")
def api_stop(sid: str):
    sess = _SESSIONS.get(sid)
    if sess:
        sess.stop.set()
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


# ---------------- Frontend ----------------
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Form Check</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0d1014;
    --surface: #151a20;
    --surface-2: #1b2129;
    --line: #232a33;
    --text: #e3e8ee;
    --dim: #8a94a3;
    --good: #7fe3b0;
    --bad: #e2698c;
    --warn: #d7e87f;
    --upload: #6aa9e0;
    --idle: #8a94a3;
    --accent: var(--idle);
    --radius: 10px;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; }
  body {
    background:
      radial-gradient(900px 420px at 15% -10%, rgba(127,227,176,.09), transparent 60%),
      radial-gradient(700px 380px at 100% 0%, rgba(106,169,224,.07), transparent 60%),
      var(--bg);
    color: var(--text);
    font-family: 'Archivo', system-ui, -apple-system, sans-serif;
    min-height: 100vh;
    -webkit-font-smoothing: antialiased;
  }
  .app { max-width: 1120px; margin: 0 auto; padding: 32px 24px 72px; }

  /* header */
  .top { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 26px; }
  .brand { display: flex; align-items: center; gap: 14px; }
  .mark { width: 40px; height: 40px; padding: 8px; border-radius: 10px; color: var(--good);
          background: radial-gradient(circle, rgba(255,255,255,.14), rgba(255,255,255,.02) 70%);
          border: 1px solid rgba(255,255,255,.18); box-shadow: 0 0 20px rgba(255,255,255,.20); }
  h1 { font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 30px; letter-spacing: -0.02em; margin: 0; line-height: 1; }
  .brand p { margin: 4px 0 0; color: var(--dim); font-size: 13px; }
  .conn { display: inline-flex; align-items: center; gap: 8px; font: 500 12px 'JetBrains Mono', ui-monospace, monospace; letter-spacing: .6px;
          text-transform: uppercase; color: var(--dim); background: var(--surface); border: 1px solid var(--line);
          padding: 7px 12px; border-radius: 999px; }
  .conn .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--idle); }
  .conn[data-s="live"] { color: var(--text); }
  .conn[data-s="live"] .dot { background: var(--good); box-shadow: 0 0 0 0 rgba(127,227,176,.6); animation: pulse 1.6s infinite; }
  .conn[data-s="warn"] .dot { background: var(--warn); }
  .conn[data-s="bad"] .dot { background: var(--bad); }
  @keyframes pulse { 70% { box-shadow: 0 0 0 8px rgba(127,227,176,0); } 100% { box-shadow: 0 0 0 0 rgba(127,227,176,0); } }

  /* controls */
  .controls { display: flex; align-items: flex-end; justify-content: space-between; gap: 16px; flex-wrap: wrap;
              border-bottom: 1px solid var(--line); margin-bottom: 22px; }
  nav.tabs { display: flex; gap: 26px; }
  nav.tabs button { background: none; border: none; cursor: pointer; color: var(--dim);
                    font: 500 15px 'Archivo', sans-serif; padding: 0 2px 14px; border-bottom: 2px solid transparent;
                    transition: color .15s, border-color .15s; }
  nav.tabs button:hover { color: var(--text); }
  nav.tabs button.active { color: var(--text); border-bottom-color: var(--good); }
  nav.tabs button[data-mode="upload"].active { border-bottom-color: var(--upload); }
  .seg { display: inline-flex; background: var(--surface); border: 1px solid var(--line); border-radius: 999px; padding: 3px; margin-bottom: 10px; }
  .seg button { background: none; border: none; color: var(--dim); cursor: pointer; padding: 7px 16px; border-radius: 999px;
                font: 500 13px 'Archivo', sans-serif; text-transform: capitalize; transition: background .15s, color .15s; }
  .seg button:hover:not(:disabled) { color: var(--text); }
  .seg button.active { background: var(--surface-2); color: var(--text); box-shadow: inset 0 0 0 1px var(--line); }
  .seg button:disabled { opacity: .5; cursor: not-allowed; }

  /* layout */
  .grid { display: grid; grid-template-columns: minmax(0, 1.65fr) minmax(290px, 1fr); gap: 20px; align-items: start; }
  @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
  .card { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: 18px 20px; }
  .label { font: 500 11px 'JetBrains Mono', ui-monospace, monospace; letter-spacing: 1.2px; text-transform: uppercase; color: var(--dim); }

  /* stage */
  .stage { position: relative; background: #070B09; border: 1px solid var(--line); border-radius: var(--radius);
           aspect-ratio: 16 / 9; overflow: hidden; display: flex; align-items: center; justify-content: center; }
  .stage img { width: 100%; height: 100%; object-fit: contain; display: block; }
  .stage::before, .stage::after { content: ""; position: absolute; width: 22px; height: 22px; z-index: 2; pointer-events: none;
                                   border: 2px solid var(--accent); opacity: .7; transition: border-color .2s; }
  .stage::before { top: 10px; right: 10px; border-left: 0; border-bottom: 0; border-top-right-radius: 4px; }
  .stage::after { bottom: 10px; left: 10px; border-right: 0; border-top: 0; border-bottom-left-radius: 4px; }
  .placeholder { text-align: center; color: var(--dim); font-size: 14px; padding: 0 24px; max-width: 340px; }
  .placeholder svg { width: 40px; height: 40px; margin-bottom: 12px; opacity: .85;
                       filter: drop-shadow(0 0 14px rgba(255,255,255,.45)) drop-shadow(0 0 3px rgba(255,255,255,.6)); }
  .placeholder.err { color: var(--bad); }
  .chips { position: absolute; left: 12px; bottom: 12px; display: flex; gap: 8px; z-index: 3; }
  .chip { font: 500 11px 'JetBrains Mono', ui-monospace, monospace; letter-spacing: .5px; padding: 5px 9px; border-radius: 6px;
          background: rgba(10,15,13,.72); border: 1px solid var(--line); color: var(--text); backdrop-filter: blur(6px); }
  .chip[hidden] { display: none; }
  .progress { height: 4px; background: var(--line); border-radius: 4px; margin-top: 12px; overflow: hidden; }
  .progress span { display: block; height: 100%; width: 0; background: var(--upload); transition: width .2s linear; }

  .actions { margin-top: 16px; }
  .btn { border: none; border-radius: 8px; padding: 12px 22px; cursor: pointer; font: 600 14px 'Archivo', sans-serif;
         transition: transform .1s, filter .15s; }
  .btn:hover { filter: brightness(1.08); }
  .btn:active { transform: translateY(1px); }
  .btn.start { background: var(--good); color: #0d1014; }
  .btn.stop { background: transparent; color: var(--bad); box-shadow: inset 0 0 0 1.5px var(--bad); }
  .dropzone { display: block; border: 1.5px dashed var(--line); border-radius: var(--radius); padding: 30px 20px; text-align: center;
              color: var(--dim); cursor: pointer; transition: border-color .15s, background .15s, color .15s; }
  .dropzone:hover, .dropzone.drag { border-color: var(--upload); background: rgba(106,169,224,.08); color: var(--text); }
  .dropzone input { display: none; }
  .dropzone strong { display: block; color: var(--text); font-weight: 500; margin-bottom: 4px; }
  .dropzone span { font-size: 13px; }
  .hidden { display: none !important; }

  /* selected-file / analyze control */
  .selected-file { display: flex; align-items: center; justify-content: space-between; gap: 14px; flex-wrap: wrap;
                    border: 1px solid var(--line); border-radius: var(--radius); padding: 14px 16px; background: var(--surface); }
  .selected-file-info { display: flex; align-items: center; gap: 10px; min-width: 0; }
  .selected-file-info svg { flex: none; width: 20px; height: 20px; color: var(--upload); }
  .selected-file-text { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
  .selected-file-name { color: var(--text); font-size: 14px; font-weight: 500; overflow: hidden; text-overflow: ellipsis;
                         white-space: nowrap; max-width: 320px; }
  .selected-file-meta { color: var(--dim); font: 400 12px 'JetBrains Mono', ui-monospace, monospace; }
  .selected-file-actions { display: flex; gap: 10px; flex-wrap: wrap; }
  .btn-ghost { background: transparent; color: var(--text); border: 1px solid var(--line); border-radius: 8px;
               padding: 12px 18px; min-height: 40px; cursor: pointer; font: 600 14px 'Archivo', sans-serif;
               transition: border-color .15s ease-out, background .15s ease-out; }
  .btn-ghost:hover { border-color: var(--upload); background: rgba(106,169,224,.08); }

  /* side panel */
  .panel { display: flex; flex-direction: column; gap: 16px; }
  .hud[data-level="good"] { --accent: var(--good); }
  .hud[data-level="bad"] { --accent: var(--bad); }
  .hud[data-level="warn"] { --accent: var(--warn); }
  .hud[data-level="idle"] { --accent: var(--idle); }
  .status-card { border-left: 3px solid var(--accent); transition: border-color .2s; }
  .status-text { font: 700 24px 'Archivo', sans-serif; letter-spacing: -0.01em; margin-top: 6px; line-height: 1.15;
                 color: var(--accent); transition: color .2s; min-height: 2.3em; display: flex; align-items: center; }
  .gauge { display: flex; flex-direction: column; align-items: center; }
  .gauge svg { width: 100%; max-width: 260px; overflow: visible; }
  .gauge .track { stroke: var(--line); }
  .gauge .zone { stroke: rgba(127,227,176,.55); }
  .gauge .needle { stroke: var(--accent); transition: stroke .2s; }
  .gauge .hub { fill: var(--accent); transition: fill .2s; }
  .gauge-value { font: 500 36px 'JetBrains Mono', ui-monospace, monospace; line-height: 1; margin-top: -30px; color: var(--accent); transition: color .2s; }
  .gauge-value small { font-size: 16px; color: var(--dim); }
  .gauge-meta { display: flex; justify-content: space-between; width: 100%; margin-bottom: 4px; }
  .zone-text { font: 400 12px 'JetBrains Mono', ui-monospace, monospace; color: var(--dim); }
  .metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .tile .value { font: 700 34px 'Archivo', sans-serif; margin-top: 6px; letter-spacing: -0.01em; }
  .tile.alerts .value.has { color: var(--bad); }

  /* log */
  .log { margin-top: 20px; }
  .log h2 { margin: 0 0 8px; font: 600 18px 'Archivo', sans-serif; letter-spacing: -0.01em; color: var(--dim); }
  ul.entries { list-style: none; margin: 0; padding: 0; max-height: 300px; overflow-y: auto; }
  ul.entries li { display: flex; align-items: baseline; gap: 12px; padding: 10px 0; border-bottom: 1px solid var(--line); font-size: 14px; animation: in .25s ease; }
  ul.entries li:last-child { border-bottom: 0; }
  ul.entries li i { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; align-self: center; background: var(--idle); }
  ul.entries li.good i { background: var(--good); }
  ul.entries li.bad i { background: var(--bad); }
  ul.entries li.warn i { background: var(--warn); }
  ul.entries time { font: 400 12px 'JetBrains Mono', ui-monospace, monospace; color: var(--dim); flex-shrink: 0; }
  ul.entries .empty { color: var(--dim); }
  @keyframes in { from { opacity: 0; transform: translateY(-4px); } to { opacity: 1; transform: none; } }
  @media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
</head>
<body>
<div class="app">

  <header class="top">
    <div class="brand">
      <svg class="mark" viewBox="0 0 32 32" aria-hidden="true"><path d="M6 27 L16 8 L26 27" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/><circle cx="16" cy="8" r="3.2" fill="currentColor"/></svg>
      <div>
        <h1>FORM CHECK</h1>
        <p>Real-time movement &amp; injury-risk tracking</p>
      </div>
    </div>
    <div class="conn" id="conn" data-s="idle"><span class="dot"></span><span id="connText">Idle</span></div>
  </header>

  <div class="controls">
    <nav class="tabs">
      <button data-mode="live" class="active">Live session</button>
      <button data-mode="upload">Upload recording</button>
    </nav>
    <div class="seg" id="exSeg" role="radiogroup" aria-label="Exercise">
      <button data-ex="squat" class="active" role="radio">squat</button>
      <button data-ex="plank" role="radio">plank</button>
      <button data-ex="running" role="radio">running</button>
    </div>
  </div>

  <main class="grid">
    <section>
      <div class="stage" id="stage">
        <img id="feed" alt="Pose tracking feed" hidden>
        <div class="placeholder" id="placeholder"></div>
        <div class="chips">
          <span class="chip" id="chipFps" hidden>0 FPS</span>
          <span class="chip" id="chipTrack" hidden>TRACKING</span>
        </div>
      </div>
      <div class="progress" id="progress" hidden><span id="progressBar"></span></div>

      <div class="actions" id="liveActions">
        <button class="btn start" id="liveBtn">Start live session</button>
      </div>
      <div class="actions hidden" id="uploadActions">
        <label class="dropzone" id="dropZone">
          <input type="file" id="fileInput" accept="video/*">
          <strong>Drop a video here, or click to choose one</strong>
          <span>MP4 or MOV · side or front view · full body in frame</span>
        </label>
        <div class="selected-file hidden" id="selectedFile">
          <div class="selected-file-info">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/><path d="M12 4v10"/><path d="m7.5 10.5 4.5 4 4.5-4"/></svg>
            <div class="selected-file-text">
              <span class="selected-file-name" id="selectedFileName"></span>
              <span class="selected-file-meta" id="selectedFileMeta"></span>
            </div>
          </div>
          <div class="selected-file-actions">
            <button class="btn-ghost" id="changeFileBtn">Choose different</button>
            <button class="btn start" id="analyzeBtn">Analyze video</button>
          </div>
        </div>
      </div>
    </section>

    <aside class="panel hud" id="hud" data-level="idle">
      <div class="card status-card">
        <div class="label">Status</div>
        <div class="status-text" id="statusText" aria-live="polite">Waiting to start</div>
      </div>

      <div class="card gauge">
        <div class="gauge-meta">
          <span class="label" id="angleLabel">KNEE ANGLE</span>
          <span class="zone-text" id="zoneText"></span>
        </div>
        <svg viewBox="0 0 200 112" aria-hidden="true">
          <path class="track" id="gTrack" fill="none" stroke-width="10" stroke-linecap="round"/>
          <path class="zone" id="gZone" fill="none" stroke-width="10" stroke-linecap="butt"/>
          <g id="needle" transform="rotate(0 100 100)" style="opacity:.25">
            <line class="needle" x1="100" y1="100" x2="34" y2="100" stroke-width="3" stroke-linecap="round"/>
          </g>
          <circle class="hub" cx="100" cy="100" r="6"/>
        </svg>
        <div class="gauge-value"><span id="angleText">&mdash;</span><small id="angleUnit"></small></div>
      </div>

      <div class="metrics">
        <div class="card tile"><div class="label" id="metricLabel">REPS</div><div class="value" id="metricValue">0</div></div>
        <div class="card tile alerts"><div class="label">ALERTS</div><div class="value" id="alertsValue">0</div></div>
      </div>
    </aside>
  </main>

  <section class="card log">
    <h2>Session log</h2>
    <ul class="entries" id="logList"><li class="empty">No events yet.</li></ul>
  </section>
</div>

<script>
const $ = id => document.getElementById(id);
const feed = $('feed'), placeholder = $('placeholder'), hud = $('hud');

const LOCAL = {
  squat:   { label: 'KNEE ANGLE', zone: [70, 110],  metric: 'REPS', value: '0' },
  plank:   { label: 'BODY LINE',  zone: [160, 180], metric: 'HOLD', value: '0s' },
  running: { label: 'KNEE ANGLE', zone: null,       metric: 'GAIT', value: '-' },
};
const ICON_CAM = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 8a2 2 0 0 1 2-2h2l1.5-2h7L17 6h2a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><circle cx="12" cy="13" r="3.5"/></svg>';
const ICON_UP = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 16V4m0 0L7 9m5-5l5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/></svg>';

let mode = 'live', exercise = 'squat';
let sid = null, running = false, pollTimer = null, lastEvent = 0;
let sessionStartedAt = 0, gotFirstState = false;

/* ---------- gauge ---------- */
const CX = 100, CY = 100, R = 80;
const pt = a => { const p = Math.PI * (1 - a / 180); return [CX + R * Math.cos(p), CY - R * Math.sin(p)]; };
const arc = (a1, a2) => {
  const [x1, y1] = pt(a1), [x2, y2] = pt(Math.min(a2, 179.9));
  return 'M' + x1.toFixed(2) + ' ' + y1.toFixed(2) + ' A' + R + ' ' + R + ' 0 0 1 ' + x2.toFixed(2) + ' ' + y2.toFixed(2);
};
$('gTrack').setAttribute('d', arc(0, 179.9));
let needleCur = 0, needleTarget = 0;
(function tick() {
  needleCur += (needleTarget - needleCur) * 0.25;
  $('needle').setAttribute('transform', 'rotate(' + needleCur.toFixed(2) + ' 100 100)');
  requestAnimationFrame(tick);
})();
function setZone(zone) {
  const z = $('gZone');
  if (zone) { z.setAttribute('d', arc(zone[0], zone[1])); z.style.display = ''; $('zoneText').textContent = 'target ' + zone[0] + '\u2013' + zone[1] + '\u00B0'; }
  else { z.style.display = 'none'; $('zoneText').textContent = 'range of motion'; }
}

/* ---------- ui helpers ---------- */
function setConn(s, text) { $('conn').dataset.s = s; $('connText').textContent = text; }
function showPlaceholder(html, err) { placeholder.innerHTML = html; placeholder.classList.toggle('err', !!err); placeholder.hidden = false; feed.hidden = true; }
function idlePlaceholder() {
  showPlaceholder(mode === 'live' ? ICON_CAM + '<div>Press start to begin live tracking</div>' : ICON_UP + '<div>Choose a video to analyze</div>');
}
function resetHud() {
  const c = LOCAL[exercise];
  hud.dataset.level = 'idle';
  $('statusText').textContent = 'Waiting to start';
  $('angleText').textContent = '\u2014'; $('angleUnit').textContent = '';
  $('angleLabel').textContent = c.label; setZone(c.zone);
  $('metricLabel').textContent = c.metric; $('metricValue').textContent = c.value;
  $('alertsValue').textContent = '0'; $('alertsValue').classList.remove('has');
  needleTarget = 0; $('needle').style.opacity = .25;
  $('chipFps').hidden = true; $('chipTrack').hidden = true;
  $('progress').hidden = true; $('progressBar').style.width = '0';
}
function logEvent(text, kind, t) {
  const empty = $('logList').querySelector('.empty'); if (empty) empty.remove();
  const li = document.createElement('li'); li.className = kind || 'good';
  const dot = document.createElement('i');
  const time = document.createElement('time'); time.textContent = new Date(t ? t * 1000 : Date.now()).toLocaleTimeString();
  const span = document.createElement('span'); span.textContent = text;
  li.append(dot, time, span);
  const list = $('logList'); list.prepend(li);
  while (list.children.length > 200) list.lastChild.remove();
}
function setButtons() {
  const b = $('liveBtn');
  b.textContent = running && mode === 'live' ? 'Stop session' : 'Start live session';
  b.className = 'btn ' + (running && mode === 'live' ? 'stop' : 'start');
  document.querySelectorAll('#exSeg button').forEach(x => x.disabled = running);
}

/* ---------- mode / exercise ---------- */
async function setMode(next) {
  if (running) await stopSession();
  mode = next;
  document.querySelectorAll('nav.tabs button').forEach(t => t.classList.toggle('active', t.dataset.mode === next));
  $('liveActions').classList.toggle('hidden', next !== 'live');
  $('uploadActions').classList.toggle('hidden', next !== 'upload');
  clearPickedFile();
  resetHud(); idlePlaceholder(); setConn('idle', 'Idle');
}
document.querySelectorAll('nav.tabs button').forEach(t => t.addEventListener('click', () => setMode(t.dataset.mode)));
document.querySelectorAll('#exSeg button').forEach(b => b.addEventListener('click', () => {
  if (running) return;
  exercise = b.dataset.ex;
  document.querySelectorAll('#exSeg button').forEach(x => { x.classList.toggle('active', x === b); x.setAttribute('aria-checked', x === b); });
  resetHud();
}));

/* ---------- session lifecycle ---------- */
const newSid = () => (window.crypto && crypto.randomUUID ? crypto.randomUUID().replace(/-/g, '') : (Date.now().toString(36) + Math.random().toString(36).slice(2, 12)));

function beginSession(id, url, label) {
  sid = id; running = true; lastEvent = 0;
  sessionStartedAt = Date.now(); gotFirstState = false;
  resetHud(); setButtons();
  showPlaceholder(ICON_CAM + '<div>' + label + '</div>');
  setConn('live', mode === 'live' ? 'Live' : 'Analyzing');
  feed.onload = () => { placeholder.hidden = true; feed.hidden = false; };
  feed.onerror = () => { if (running) { /* state poll reports the real reason */ } };
  feed.src = url;
  clearTimeout(pollTimer); poll();
}

function startLive() {
  const id = newSid();
  beginSession(id, '/video_feed?exercise=' + exercise + '&sid=' + id, 'Starting camera\u2026');
}

async function stopSession() {
  if (!sid) return;
  const id = sid;
  try { await fetch('/api/stop/' + id, { method: 'POST' }); } catch (e) {}
  feed.onload = feed.onerror = null; feed.removeAttribute('src'); feed.hidden = true;
  // let the poller pick up done=true; but make sure it eventually clears
  for (let i = 0; i < 30 && running; i++) await new Promise(r => setTimeout(r, 100));
  if (running) finish({ status: 'SESSION ENDED' });
}

function finish(s) {
  running = false; clearTimeout(pollTimer);
  feed.removeAttribute('src');
  setButtons();
  hud.dataset.level = 'idle';
  if (s && s.status) $('statusText').textContent = s.status;
  $('needle').style.opacity = .25; $('angleText').textContent = '\u2014'; $('angleUnit').textContent = '';
  $('chipFps').hidden = true; $('chipTrack').hidden = true;
  if (!s || !s.error) {
    setConn('idle', 'Complete');
    const last = feed.dataset.hadFrames; // keep the last frame visible only if we had one
    showPlaceholder(ICON_CAM + '<div>Session finished. Start another whenever you\u2019re ready.</div>');
  }
  if (mode === 'upload') clearPickedFile();
}

async function poll() {
  if (!sid || !running) return;
  try {
    const r = await fetch('/api/state/' + sid + '?since=' + lastEvent, { cache: 'no-store' });
    if (r.status === 404) {
      // the server can take a beat to register a brand-new session (esp. opening the camera);
      // don't treat an early 404 as "ended" until we've given it a short grace period.
      if (!gotFirstState && Date.now() - sessionStartedAt < 4000) {
        pollTimer = setTimeout(poll, 150);
        return;
      }
      finish({ status: 'SESSION ENDED' }); return;
    }
    const s = await r.json();
    gotFirstState = true;
    applyState(s);
    if (s.done) { finish(s); return; }
  } catch (e) { setConn('warn', 'Reconnecting'); }
  pollTimer = setTimeout(poll, 150);
}

function applyState(s) {
  if (s.error) {
    showPlaceholder('<div>' + s.error + '</div>', true);
    setConn('bad', 'Error');
  }
  hud.dataset.level = s.level;
  $('statusText').textContent = s.status;
  $('angleLabel').textContent = s.angle_label; setZone(s.zone);
  if (s.angle != null) {
    $('angleText').textContent = Math.round(s.angle); $('angleUnit').textContent = '\u00B0';
    needleTarget = Math.max(0, Math.min(180, s.angle)); $('needle').style.opacity = 1;
  } else {
    $('angleText').textContent = '\u2014'; $('angleUnit').textContent = ''; $('needle').style.opacity = .25;
  }
  $('metricLabel').textContent = s.metric_label; $('metricValue').textContent = s.metric_value;
  $('alertsValue').textContent = s.alerts; $('alertsValue').classList.toggle('has', s.alerts > 0);
  $('chipFps').hidden = false; $('chipFps').textContent = s.fps + ' FPS';
  $('chipTrack').hidden = false; $('chipTrack').textContent = s.tracking ? 'TRACKING' : 'SEARCHING';
  if (s.progress != null && mode === 'upload') { $('progress').hidden = false; $('progressBar').style.width = (s.progress * 100).toFixed(1) + '%'; }
  (s.events || []).forEach(e => { logEvent(e.text, e.kind, e.t); lastEvent = Math.max(lastEvent, e.id); });
}

/* ---------- live + upload wiring ---------- */
$('liveBtn').addEventListener('click', () => running ? stopSession() : startLive());

const dropZone = $('dropZone'), fileInput = $('fileInput'), selectedFile = $('selectedFile');
let pendingFile = null;

function pickFile(file) {
  pendingFile = file;
  dropZone.classList.add('hidden');
  selectedFile.classList.remove('hidden');
  $('selectedFileName').textContent = file.name;
  $('selectedFileMeta').textContent = (file.size / (1024 * 1024)).toFixed(1) + ' MB · ready to analyze';
}
function clearPickedFile() {
  pendingFile = null;
  selectedFile.classList.add('hidden');
  dropZone.classList.remove('hidden');
}

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag'));
dropZone.addEventListener('drop', e => { e.preventDefault(); dropZone.classList.remove('drag'); if (e.dataTransfer.files.length) pickFile(e.dataTransfer.files[0]); });
fileInput.addEventListener('change', () => { if (fileInput.files.length) pickFile(fileInput.files[0]); fileInput.value = ''; });
$('changeFileBtn').addEventListener('click', clearPickedFile);
$('analyzeBtn').addEventListener('click', () => { if (pendingFile) handleUpload(pendingFile); });

async function handleUpload(file) {
  if (running) await stopSession();
  selectedFile.classList.add('hidden');
  logEvent('Uploading ' + file.name + '\u2026', 'good');
  setConn('warn', 'Uploading');
  showPlaceholder(ICON_UP + '<div>Uploading ' + file.name + '\u2026</div>');
  const form = new FormData();
  form.append('file', file);
  form.append('exercise', exercise);
  try {
    const res = await fetch('/upload_video', { method: 'POST', body: form });
    if (!res.ok) throw new Error('server returned ' + res.status);
    const data = await res.json();
    beginSession(data.sid, data.stream_url, 'Analyzing ' + file.name + '\u2026');
  } catch (err) {
    logEvent('Upload failed: ' + err.message, 'bad');
    setConn('bad', 'Error'); idlePlaceholder();
    pickFile(file);
  }
}

window.addEventListener('beforeunload', () => { if (running && sid) navigator.sendBeacon('/api/stop/' + sid); });

resetHud(); idlePlaceholder(); setButtons();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Model file '{MODEL_PATH}' not found. Download pose_landmarker_lite.task "
              f"from the MediaPipe Pose Landmarker docs and put it next to this file.")
    uvicorn.run(app, host="127.0.0.1", port=8000)