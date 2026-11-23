import os
import sys
import csv
import time
import math
import cv2 as cv
import numpy as np
import mediapipe as mp
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

# Ensure UTF-8 console output
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Label mapping
label_path = 'model/word_classifier/word_classifier_label.csv'
csv_path = 'model/word_classifier/word_keypoint.csv'
os.makedirs('model/word_classifier', exist_ok=True)

DEFAULT_WORDS = ['أشعر بألم', 'انا تمام', 'قف', 'ألم خفيف', 'ألم قوي', 'اريد أن أحك وجهي', 'غسل الفم', 'اريد أن اتحدث']
if not os.path.exists(label_path):
    with open(label_path, 'w', encoding='utf-8-sig', newline='') as f:
        for w in DEFAULT_WORDS:
            f.write(w + '\n')

with open(label_path, 'r', encoding='utf-8-sig') as f:
    words = [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]

def get_sample_counts():
    counts = {i: 0 for i in range(len(words))}
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            for row in csv.reader(f):
                if row:
                    try:
                        cid = int(row[0])
                        if cid in counts:
                            counts[cid] += 1
                    except ValueError:
                        pass
    return counts

def pre_process_landmarks_normalized(landmarks):
    coords = [[lm.x, lm.y] for lm in landmarks.landmark]
    arr = np.array(coords, dtype=np.float32)
    arr -= arr[0]  # wrist translation to (0, 0)
    flat = arr.flatten()
    max_value = np.max(np.abs(flat))
    if max_value > 0:
        flat /= max_value
    return flat.tolist()

# Arabic shaper
from app import format_arabic_hud_text

def draw_arabic_text(image, text, pos, font_size=24, color=(0, 255, 0), bg_color=(25, 25, 25)):
    try:
        shaped = format_arabic_hud_text(text)
        img_pil = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
        pil_pos = (pos[0], max(0, pos[1] - font_size))
        bbox = draw.textbbox(pil_pos, shaped, font=font)
        if bg_color:
            draw.rectangle((bbox[0]-5, bbox[1]-2, bbox[2]+5, bbox[3]+2), fill=bg_color)
        draw.text(pil_pos, shaped, font=font, fill=(color[2], color[1], color[0]))
        return cv.cvtColor(np.array(img_pil), cv.COLOR_RGB2BGR)
    except Exception:
        cv.putText(image, text, pos, cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return image

def main():
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    selected_idx = 0
    state = "IDLE"  # "IDLE", "COUNTDOWN", "RECORDING"
    countdown_start = 0
    samples_to_record = 60
    recorded_in_batch = 0

    print("==================================================")
    print("  مسجل إشارات طبيب الأسنان (Dental Signs Recorder)")
    print("==================================================")
    print(f"اختر رقم الإشارة لتسجيلها (0 - {len(words) - 1}):")
    for i, w in enumerate(words):
        print(f"  [{i}] : {w}")
    print("\nاضغط [SPACE] لبدء التسجيل (مع عد تنازلي 3 ثواني)")
    print("اضغط [T] لتدريب النموذج مباشرة")
    print("اضغط [ESC] أو [Q] للخروج\n")

    while True:
        key = cv.waitKey(1)
        if key in [27, ord('q'), ord('Q')]:
            break

        # Number selection 0-9
        if ord('0') <= key <= ord('9'):
            val = key - ord('0')
            if val < len(words):
                selected_idx = val
                state = "IDLE"

        # SPACE to trigger recording
        if key == 32 and state == "IDLE":
            state = "COUNTDOWN"
            countdown_start = time.time()
            recorded_in_batch = 0

        # T to trigger training
        if key in [ord('t'), ord('T')] and state == "IDLE":
            print("\nبدء تدريب نموذج الكلمات الطبية...")
            os.system(f"{sys.executable} train_word_model.py")
            print("اكتمل التدريب!\n")

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        h, w, _ = frame.shape
        debug_image = frame.copy()

        img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = hands.process(img_rgb)
        img_rgb.flags.writeable = True

        pre_processed_landmark_list = []
        hand_detected = False

        if results.multi_hand_landmarks:
            hand_detected = True
            no_of_hands = len(results.multi_hand_landmarks)
            if no_of_hands == 2:
                for hl in results.multi_hand_landmarks[:2]:
                    feats = pre_process_landmarks_normalized(hl)
                    pre_processed_landmark_list.extend(feats)
                    mp_draw.draw_landmarks(debug_image, hl, mp_hands.HAND_CONNECTIONS)
            elif no_of_hands == 1:
                hl = results.multi_hand_landmarks[0]
                feats = pre_process_landmarks_normalized(hl)
                pre_processed_landmark_list = [0.0] * 42 + feats
                mp_draw.draw_landmarks(debug_image, hl, mp_hands.HAND_CONNECTIONS)

            if len(pre_processed_landmark_list) != 84:
                pre_processed_landmark_list = (pre_processed_landmark_list + [0.0] * 84)[:84]

        now = time.time()
        counts = get_sample_counts()

        # Handle states
        if state == "COUNTDOWN":
            elapsed = now - countdown_start
            rem = max(0, 3 - int(elapsed))
            if rem > 0:
                cv.putText(debug_image, f"GET READY: {rem}", (int(w/2) - 150, int(h/2)),
                           cv.FONT_HERSHEY_DUPLEX, 1.8, (0, 215, 255), 3)
            else:
                state = "RECORDING"
                recorded_in_batch = 0

        elif state == "RECORDING":
            if hand_detected and len(pre_processed_landmark_list) == 84:
                with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([selected_idx, *pre_processed_landmark_list])
                recorded_in_batch += 1

            # Draw recording progress bar
            pct = min(1.0, recorded_in_batch / samples_to_record)
            bar_w = 400
            bar_x = int((w - bar_w) / 2)
            bar_y = h - 60
            cv.rectangle(debug_image, (bar_x, bar_y), (bar_x + bar_w, bar_y + 24), (30, 30, 30), -1)
            cv.rectangle(debug_image, (bar_x, bar_y), (bar_x + int(bar_w * pct), bar_y + 24), (0, 0, 255), -1)
            cv.putText(debug_image, f"RECORDING: {recorded_in_batch} / {samples_to_record}",
                       (bar_x + 80, bar_y + 18), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if recorded_in_batch >= samples_to_record:
                state = "IDLE"
                print(f"تم تسجيل {samples_to_record} عينة لكلمة: {words[selected_idx]}")

        # Top Overlay: Current selected word and counts
        selected_word_text = f"الكلمة المختارة [{selected_idx}]: {words[selected_idx]} (العينات: {counts.get(selected_idx, 0)})"
        debug_image = draw_arabic_text(debug_image, selected_word_text, (20, 40), font_size=26, color=(0, 255, 0))

        # Instructions Banner
        instructions = "اضغط [0-6] لاختيار الكلمة | [SPACE] للتسجيل | [T] للتدريب | [ESC] خروج"
        debug_image = draw_arabic_text(debug_image, instructions, (20, 80), font_size=18, color=(200, 200, 200))

        # Sidebar with all 7 words and counts
        side_y = 120
        for idx, wname in enumerate(words):
            color = (0, 255, 255) if idx == selected_idx else (180, 180, 180)
            txt = f"[{idx}] {wname}: {counts.get(idx, 0)}"
            debug_image = draw_arabic_text(debug_image, txt, (20, side_y), font_size=20, color=color, bg_color=(20, 20, 20))
            side_y += 34

        cv.imshow('Medical Signs Recorder (SignAI)', debug_image)

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
