import cv2
import mediapipe as mp
import math
import numpy as np
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO,   # IMPORTANT
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

canvas = None
webs = []
prev_draw_pt = None

last_web_time = 0
web_cooldown = 0.4  # seconds


def tip_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y

def thumb_up(lm):
    return lm[4].x < lm[3].x

def all_fingers_up(lm):
    return (tip_up(lm, 8, 6) and tip_up(lm, 12, 10) and
            tip_up(lm, 16, 14) and tip_up(lm, 20, 18))



def detect_spiderman(lm):
    return (tip_up(lm, 8, 6) and
            tip_up(lm, 20, 18) and
            thumb_up(lm) and
            not tip_up(lm, 12, 10) and
            not tip_up(lm, 16, 14))

def detect_two_fingers(lm):
    return (tip_up(lm, 8, 6) and
            tip_up(lm, 12, 10) and
            not tip_up(lm, 16, 14) and
            not tip_up(lm, 20, 18))


def build_zigzag(ox, oy, dx, dy, total_len, seg=25, amp=7):
    px, py = -dy, dx
    pts = [(int(ox), int(oy))]
    covered, side = 0, 1

    while covered < total_len:
        step = min(seg, total_len - covered)
        mid = covered + step * 0.5
        mx = ox + dx * mid + px * amp * side
        my = oy + dy * mid + py * amp * side
        ex = ox + dx * (covered + step)
        ey = oy + dy * (covered + step)
        pts.append((int(mx), int(my)))
        pts.append((int(ex), int(ey)))
        covered += step
        side *= -1

    return pts

def ray_to_boundary(x, y, dx, dy, w, h):
    ts = []
    if dx > 0: ts.append((w - 1 - x) / dx)
    elif dx < 0: ts.append(-x / dx)
    if dy > 0: ts.append((h - 1 - y) / dy)
    elif dy < 0: ts.append(-y / dy)
    t = min((v for v in ts if v >= 0), default=1)
    ex = max(0, min(w - 1, int(x + dx * t)))
    ey = max(0, min(h - 1, int(y + dy * t)))
    return math.hypot(ex - x, ey - y)

def spawn_web(lm, w, h):
    tip = lm[8]
    knuckle = lm[5]
    x1, y1 = int(tip.x * w), int(tip.y * h)
    dx = tip.x - knuckle.x
    dy = tip.y - knuckle.y
    mag = math.hypot(dx, dy)
    if mag < 1e-5:
        return
    dx /= mag
    dy /= mag

    max_len = ray_to_boundary(x1, y1, dx, dy, w, h)

    webs.append({
        "ox": x1,
        "oy": y1,
        "dx": dx,
        "dy": dy,
        "progress": 0,
        "max_len": max_len
    })

def draw_web(frame, pts):
    overlay = frame.copy()

    for i in range(len(pts) - 1):
        cv2.line(overlay, pts[i], pts[i + 1], (0, 90, 255), 8)

    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    for i in range(len(pts) - 1):
        cv2.line(frame, pts[i], pts[i + 1], (0, 0, 255), 4)

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp = int(time.time() * 1000)
    results = detector.detect_for_video(mp_image, timestamp)

    hands = results.hand_landmarks if results.hand_landmarks else []

    mode = "IDLE"

    for lm in hands:

        # ── OPEN PALM CLEARS SCREEN
        if all_fingers_up(lm):
            canvas = np.zeros_like(frame)
            webs.clear()
            prev_draw_pt = None
            mode = "CLEAR"
            break

        # ── SPIDER WEB
        if detect_spiderman(lm):
            mode = "WEB"
            current_time = time.time()
            if current_time - last_web_time > web_cooldown:
                spawn_web(lm, w, h)
                last_web_time = current_time

        # AIR DRAW
        elif detect_two_fingers(lm):
            mode = "DRAW"
            tip = lm[8]
            cx, cy = int(tip.x * w), int(tip.y * h)

            if prev_draw_pt is not None:
                cv2.line(canvas, prev_draw_pt, (cx, cy), (255, 140, 0), 8)

            prev_draw_pt = (cx, cy)
            break

    if mode != "DRAW":
        prev_draw_pt = None

    # Animate Webs
    alive = []
    for web in webs:
        web["progress"] = min(web["progress"] + 40, web["max_len"])

        pts = build_zigzag(
            web["ox"],
            web["oy"],
            web["dx"],
            web["dy"],
            web["progress"]
        )

        draw_web(frame, pts)

        if web["progress"] < web["max_len"]:
            alive.append(web)

    webs = alive

    #  Blend canvas
    mask = canvas.any(axis=2)
    frame[mask] = cv2.addWeighted(frame, 0.3, canvas, 0.7, 0)[mask]

    cv2.putText(frame, f"MODE: {mode}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    cv2.imshow("AR Hand System - Smooth", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()