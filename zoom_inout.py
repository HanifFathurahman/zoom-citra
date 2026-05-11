import os
import sys
import time
import urllib.request

try:
    import cv2
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ModuleNotFoundError as exc:
    missing_package = exc.name
    print(f"Package belum terinstall: {missing_package}")
    print("Jalankan perintah berikut dari folder proyek:")
    print("  python -m pip install -r requirements.txt")
    print("")
    print("Jika masih gagal, pastikan memakai Python 3.9 sampai 3.12 versi 64-bit.")
    sys.exit(1)


MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
OUTPUT_DIR = "outputs"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

ZOOM_MIN = 1.0
ZOOM_MAX = 4.0
ZOOM_SPEED = 0.12

FILTERS = [
    ("normal", "Normal"),
    ("grayscale", "Grayscale"),
    ("negative", "Negative"),
]


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Mengunduh model MediaPipe (~25MB), harap tunggu...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model berhasil diunduh.")


def palm_center(landmarks):
    palm_ids = [0, 1, 5, 9, 13, 17]
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in palm_ids])
    return pts.mean(axis=0)


def hand_size(landmarks):
    wrist = np.array([landmarks[0].x, landmarks[0].y])
    mid_mcp = np.array([landmarks[9].x, landmarks[9].y])
    return np.linalg.norm(wrist - mid_mcp)


def draw_hand(frame, landmarks, w, h, color=(200, 200, 200)):
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], color, 2, cv2.LINE_AA)
    for i, pt in enumerate(points):
        r = 6 if i == 0 else 4
        cv2.circle(frame, pt, r, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, r, (0, 0, 0), 1, cv2.LINE_AA)


def apply_zoom(frame, zoom_level):
    h, w = frame.shape[:2]
    if zoom_level <= 1.0:
        return frame

    new_w = int(w / zoom_level)
    new_h = int(h / zoom_level)
    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    cropped = frame[y1:y1 + new_h, x1:x1 + new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_filter(frame, filter_name):
    if filter_name == "normal":
        return frame.copy()

    if filter_name == "grayscale":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if filter_name == "negative":
        return cv2.bitwise_not(frame)

    return frame.copy()


def draw_zoom_bar(frame, zoom_level, w, h):
    bar_x = w - 38
    bar_top = 105
    bar_bot = h - 185
    bar_h = max(1, bar_bot - bar_top)

    cv2.rectangle(frame, (bar_x, bar_top), (bar_x + 16, bar_bot), (40, 40, 40), -1)
    cv2.rectangle(frame, (bar_x, bar_top), (bar_x + 16, bar_bot), (100, 100, 100), 1)

    fill_ratio = (zoom_level - ZOOM_MIN) / (ZOOM_MAX - ZOOM_MIN)
    fill_h = int(bar_h * fill_ratio)
    if fill_h > 0:
        color = (0, int(200 * fill_ratio + 55), int(255 * (1 - fill_ratio)))
        cv2.rectangle(frame, (bar_x, bar_bot - fill_h), (bar_x + 16, bar_bot), color, -1)

    cv2.putText(frame, f"{zoom_level:.1f}x", (bar_x - 10, bar_top - 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)


def draw_status_ui(frame, zoom_level, state, filter_label, saved_path):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 96), (12, 12, 12), -1)
    cv2.rectangle(overlay, (0, h - 72), (w, h), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    cv2.putText(frame, "Zoom Citra", (18, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.72, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(frame, "Kelompok 3 - R8E", (18, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Filter: {filter_label}", (18, 82),
                cv2.FONT_HERSHEY_DUPLEX, 0.74, (0, 220, 255), 1, cv2.LINE_AA)

    state_color = {
        "zoom_in": (0, 230, 100),
        "zoom_out": (0, 120, 255),
        "hold": (210, 210, 210),
        "no_hand": (120, 120, 120),
    }
    state_label = {
        "zoom_in": "GESTURE: ZOOM IN",
        "zoom_out": "GESTURE: ZOOM OUT",
        "hold": "GESTURE: TAHAN",
        "no_hand": "GESTURE: BUTUH 2 TANGAN",
    }
    col = state_color.get(state, (170, 170, 170))
    lbl = state_label.get(state, state)
    cv2.putText(frame, lbl, (w - 340, 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.58, col, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Zoom {zoom_level:.1f}x", (w - 340, 70),
                cv2.FONT_HERSHEY_DUPLEX, 0.58, (230, 230, 230), 1, cv2.LINE_AA)

    help_text = "1 Normal | 2 Grayscale | 3 Negatif | S Simpan | R Reset | Q Keluar"
    cv2.putText(frame, help_text, (18, h - 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(frame, "Gesture: dekatkan dua telapak tangan untuk zoom in, jauhkan untuk zoom out",
                (18, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 130, 130), 1, cv2.LINE_AA)

    if saved_path:
        cv2.putText(frame, f"Tersimpan: {saved_path}", (w - 390, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 120), 1, cv2.LINE_AA)

    draw_zoom_bar(frame, zoom_level, w, h)


def save_frame(frame):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = time.strftime("hasil_%Y%m%d_%H%M%S.png")
    path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(path, frame)
    return path


def create_detector():
    download_model()
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.6,
    )
    return vision.HandLandmarker.create_from_options(options)


def update_zoom_from_hands(frame, detector, zoom_level, prev_dist):
    h, w = frame.shape[:2]
    state = "no_hand"

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.hand_landmarks and len(result.hand_landmarks) == 2:
        lm0 = result.hand_landmarks[0]
        lm1 = result.hand_landmarks[1]
        c0 = palm_center(lm0)
        c1 = palm_center(lm1)
        avg_size = (hand_size(lm0) + hand_size(lm1)) / 2 + 1e-6
        dist = np.linalg.norm(c0 - c1)
        norm_dist = dist / (avg_size * 6)

        if prev_dist is not None:
            delta = norm_dist - prev_dist
            if delta < -0.005:
                zoom_level = min(ZOOM_MAX, zoom_level + ZOOM_SPEED * abs(delta) * 80)
                state = "zoom_in"
            elif delta > 0.005:
                zoom_level = max(ZOOM_MIN, zoom_level - ZOOM_SPEED * abs(delta) * 80)
                state = "zoom_out"
            else:
                state = "hold"

        prev_dist = norm_dist
        draw_hand(frame, lm0, w, h, color=(0, 210, 255))
        draw_hand(frame, lm1, w, h, color=(0, 210, 255))

        px0, py0 = int(c0[0] * w), int(c0[1] * h)
        px1, py1 = int(c1[0] * w), int(c1[1] * h)
        cv2.line(frame, (px0, py0), (px1, py1), (255, 200, 0), 2, cv2.LINE_AA)
        cv2.circle(frame, (px0, py0), 10, (255, 200, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, (px1, py1), 10, (255, 200, 0), -1, cv2.LINE_AA)
    elif result.hand_landmarks and len(result.hand_landmarks) == 1:
        draw_hand(frame, result.hand_landmarks[0], w, h, color=(120, 120, 120))
        prev_dist = None
    else:
        prev_dist = None

    return zoom_level, prev_dist, state


def main():
    detector = create_detector()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Kamera tidak dapat dibuka. Pastikan webcam tersedia dan tidak dipakai aplikasi lain.")
        detector.close()
        return

    zoom_level = 1.0
    prev_dist = None
    filter_index = 0
    saved_path = ""
    last_saved_time = 0

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break

        raw_frame = cv2.flip(raw_frame, 1)
        annotated = raw_frame.copy()
        zoom_level, prev_dist, state = update_zoom_from_hands(
            annotated, detector, zoom_level, prev_dist
        )

        zoomed_annotated = apply_zoom(annotated, zoom_level)
        display = apply_filter(zoomed_annotated, FILTERS[filter_index][0])

        if time.time() - last_saved_time > 2.5:
            saved_path = ""

        draw_status_ui(
            display,
            zoom_level,
            state,
            FILTERS[filter_index][1],
            saved_path,
        )

        cv2.imshow("Mini Studio Pengolahan Citra", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            zoom_level = 1.0
            prev_dist = None
        elif key == ord("s"):
            saved_path = save_frame(display)
            last_saved_time = time.time()
        elif ord("1") <= key <= ord("3"):
            filter_index = key - ord("1")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
