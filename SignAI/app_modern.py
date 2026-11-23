import os
import sys
import csv
import time
import math
import argparse
import threading
from collections import deque, Counter

# Configure console encoding for UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import cv2 as cv
import numpy as np
import mediapipe as mp
from tkinter import (
    Tk, Frame, Label, Entry, Button, Text, Canvas, Scrollbar,
    Scale, HORIZONTAL, END, StringVar, BooleanVar, DoubleVar
)
from PIL import Image, ImageTk, ImageDraw, ImageFont

# Local system imports
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from model.word_classifier.word_classifier import WordClassifier
from functions.feature_engineering import extract_hand_features, extract_dual_hand_features
from functions.dialogue_manager import DialogueManager
from functions.tts_engine import TTSEngine

try:
    import mediapipe as mp
    if hasattr(mp, 'solutions'):
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
    else:
        import mediapipe.python.solutions.drawing_utils as mp_drawing
        import mediapipe.python.solutions.hands as mp_hands
except Exception:
    mp_drawing = None
    mp_hands = None


# ==============================================================================
# Arabic HUD Text Reshaper (Zero-Dependency Connected Arabic for OpenCV / PIL)
# ==============================================================================

ARABIC_FORMS = {
    '\u0621': ('\uFE80', '\uFE80', '\uFE80', '\uFE80'),
    '\u0622': ('\uFE81', '\uFE82', '\uFE81', '\uFE82'),
    '\u0623': ('\uFE83', '\uFE84', '\uFE83', '\uFE84'),
    '\u0624': ('\uFE85', '\uFE86', '\uFE85', '\uFE86'),
    '\u0625': ('\uFE87', '\uFE88', '\uFE87', '\uFE88'),
    '\u0626': ('\uFE89', '\uFE8A', '\uFE8B', '\uFE8C'),
    '\u0627': ('\uFE8D', '\uFE8E', '\uFE8D', '\uFE8E'),
    '\u0628': ('\uFE8F', '\uFE90', '\uFE91', '\uFE92'),
    '\u0629': ('\uFE93', '\uFE94', '\uFE93', '\uFE94'),
    '\u062A': ('\uFE95', '\uFE96', '\uFE97', '\uFE98'),
    '\u062B': ('\uFE99', '\uFE9A', '\uFE9B', '\uFE9C'),
    '\u062C': ('\uFE9D', '\uFE9E', '\uFE9F', '\uFEA0'),
    '\u062D': ('\uFEA1', '\uFEA2', '\uFEA3', '\uFEA4'),
    '\u062E': ('\uFEA5', '\uFEA6', '\uFEA7', '\uFEA8'),
    '\u062F': ('\uFEA9', '\uFEAA', '\uFEA9', '\uFEAA'),
    '\u0630': ('\uFEAB', '\uFEAC', '\uFEAB', '\uFEAC'),
    '\u0631': ('\uFEAD', '\uFEAE', '\uFEAD', '\uFEAE'),
    '\u0632': ('\uFEAF', '\uFEB0', '\uFEAF', '\uFEB0'),
    '\u0633': ('\uFEB1', '\uFEB2', '\uFEB3', '\uFEB4'),
    '\u0634': ('\uFEB5', '\uFEB6', '\uFEB7', '\uFEB8'),
    '\u0635': ('\uFEB9', '\uFEBA', '\uFEBB', '\uFEBC'),
    '\u0636': ('\uFEBD', '\uFEBE', '\uFEBF', '\uFEC0'),
    '\u0637': ('\uFEC1', '\uFEC2', '\uFEC3', '\uFEC4'),
    '\u0638': ('\uFEC5', '\uFEC6', '\uFEC7', '\uFEC8'),
    '\u0639': ('\uFEC9', '\uFECA', '\uFECB', '\uFECC'),
    '\u063A': ('\uFECD', '\uFECE', '\uFECF', '\uFED0'),
    '\u0641': ('\uFED1', '\uFED2', '\uFED3', '\uFED4'),
    '\u0642': ('\uFED5', '\uFED6', '\uFED7', '\uFED8'),
    '\u0643': ('\uFED9', '\uFEDA', '\uFEDB', '\uFEDC'),
    '\u0644': ('\uFEDD', '\uFEDE', '\uFEDF', '\uFEE0'),
    '\u0645': ('\uFEE1', '\uFEE2', '\uFEE3', '\uFEE4'),
    '\u0646': ('\uFEE5', '\uFEE6', '\uFEE7', '\uFEE8'),
    '\u0647': ('\uFEE9', '\uFEEA', '\uFEEB', '\uFEEC'),
    '\u0648': ('\uFEED', '\uFEEE', '\uFEED', '\uFEEE'),
    '\u0649': ('\uFEEF', '\uFEF0', '\uFEEF', '\uFEF0'),
    '\u064A': ('\uFEF1', '\uFEF2', '\uFEF3', '\uFEF4'),
}

NON_CONNECTING_AFTER = set('\u0621\u0622\u0623\u0624\u0625\u0627\u062F\u0630\u0631\u0632\u0648\u0629')


def reshape_word(word):
    n = len(word)
    reshaped = []
    i = 0
    while i < n:
        c = word[i]
        if c == '\u0644' and i + 1 < n and word[i+1] in '\u0627\u0623\u0625\u0622':
            alef = word[i+1]
            prev_connects = (i > 0 and word[i-1] in ARABIC_FORMS and word[i-1] not in NON_CONNECTING_AFTER)
            if alef == '\u0627':
                reshaped.append('\uFEFC' if prev_connects else '\uFEFB')
            elif alef == '\u0623':
                reshaped.append('\uFEF8' if prev_connects else '\uFEF7')
            elif alef == '\u0625':
                reshaped.append('\uFEFA' if prev_connects else '\uFEF9')
            elif alef == '\u0622':
                reshaped.append('\uFEF6' if prev_connects else '\uFEF5')
            i += 2
            continue

        if c in ARABIC_FORMS:
            prev_connects = (i > 0 and word[i-1] in ARABIC_FORMS and word[i-1] not in NON_CONNECTING_AFTER)
            next_connects = (i + 1 < n and word[i+1] in ARABIC_FORMS and c not in NON_CONNECTING_AFTER)
            if prev_connects and next_connects:
                reshaped.append(ARABIC_FORMS[c][3])
            elif prev_connects:
                reshaped.append(ARABIC_FORMS[c][1])
            elif next_connects:
                reshaped.append(ARABIC_FORMS[c][2])
            else:
                reshaped.append(ARABIC_FORMS[c][0])
        else:
            reshaped.append(c)
        i += 1
    return ''.join(reshaped)


def format_arabic_hud_text(arabic_label, percent_str=""):
    words = arabic_label.strip().split()
    shaped_words = []
    for w in words:
        has_open_paren = w.startswith('(') or w.startswith('[')
        has_close_paren = w.endswith(')') or w.endswith(']')
        core = w.strip('()[]')
        if any('\u0600' <= ch <= '\u06FF' for ch in core):
            shaped_core = reshape_word(core)[::-1]
        else:
            shaped_core = core

        if has_open_paren and has_close_paren:
            shaped_w = '(' + shaped_core + ')'
        elif has_open_paren:
            shaped_w = shaped_core + '('
        elif has_close_paren:
            shaped_w = ')' + shaped_core
        else:
            shaped_w = shaped_core
        shaped_words.append(shaped_w)

    shaped_label = ' '.join(reversed(shaped_words))
    if percent_str:
        return f"{percent_str}  {shaped_label}"
    return shaped_label


def draw_hud_badge(image, arabic_text, status_str, pos=(20, 40), font_size=22, border_color=(16, 185, 129)):
    """
    Renders a crisp, dark high-contrast badge with 1px hairline border on OpenCV image.
    border_color in RGB: default emerald (16, 185, 129).
    """
    try:
        shaped = format_arabic_hud_text(arabic_text, status_str)
        img_pil = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Windows system font fallback
        font_path = "C:/Windows/Fonts/segoeui.ttf"
        if not os.path.exists(font_path):
            font_path = "C:/Windows/Fonts/arial.ttf"
        font = ImageFont.truetype(font_path, font_size)

        pil_pos = (pos[0], max(0, pos[1] - font_size))
        bbox = draw.textbbox(pil_pos, shaped, font=font)
        pad = 8
        box = (bbox[0] - pad, bbox[1] - pad // 2, bbox[2] + pad, bbox[3] + pad // 2)

        # Draw dark solid background & 1px hairline border
        draw.rectangle(box, fill=(18, 18, 21), outline=border_color, width=1)
        draw.text(pil_pos, shaped, font=font, fill=(245, 245, 245))
        return cv.cvtColor(np.array(img_pil), cv.COLOR_RGB2BGR)
    except Exception:
        cv.putText(image, f"{arabic_text} {status_str}", pos, cv.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2)
        return image


# ==============================================================================
# Biomechanical Heuristic Detection (Fast-Path Kinematics)
# ==============================================================================

def is_thumbs_up(landmarks):
    lm = landmarks.landmark
    hand_scale = math.hypot(lm[9].x - lm[0].x, lm[9].y - lm[0].y)
    if hand_scale < 0.02:
        return False
    thumb_ext = (lm[2].y - lm[4].y) / hand_scale
    index_ext = (lm[5].y - lm[4].y) / hand_scale
    thumb_extended_up = (thumb_ext > 0.30) and (index_ext > 0.25)
    thumb_joints_up = (lm[4].y < lm[3].y < lm[2].y) and (lm[4].y < lm[0].y)
    other_curled = sum([
        (lm[8].y > lm[6].y),
        (lm[12].y > lm[10].y),
        (lm[16].y > lm[14].y),
        (lm[20].y > lm[18].y)
    ])
    return thumb_extended_up and thumb_joints_up and (other_curled >= 3)


def is_thumbs_down(landmarks):
    lm = landmarks.landmark
    hand_scale = math.hypot(lm[9].x - lm[0].x, lm[9].y - lm[0].y)
    if hand_scale < 0.02:
        return False
    thumb_ext_down = ((lm[4].y - lm[2].y) / hand_scale > 0.20) and (lm[4].y > lm[0].y)
    thumb_progression = (lm[4].y > lm[3].y > lm[2].y)
    other_curled = sum([
        (lm[8].y > lm[6].y),
        (lm[12].y > lm[10].y),
        (lm[16].y > lm[14].y),
        (lm[20].y > lm[18].y)
    ])
    return thumb_ext_down and thumb_progression and (other_curled >= 3)


def is_open_palm(landmarks):
    lm = landmarks.landmark
    return (
        lm[8].y < lm[6].y and
        lm[12].y < lm[10].y and
        lm[16].y < lm[14].y and
        lm[20].y < lm[18].y and
        (lm[4].y < lm[2].y or abs(lm[4].x - lm[0].x) > 0.08)
    )


def check_dental_c_pain(landmarks):
    lm = landmarks.landmark
    hand_scale = math.hypot(lm[9].x - lm[0].x, lm[9].y - lm[0].y)
    if hand_scale < 0.02:
        return None
    aperture = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y) / hand_scale
    mid_curled = (lm[12].y > lm[10].y) or (math.hypot(lm[12].x - lm[0].x, lm[12].y - lm[0].y) < hand_scale * 1.30)
    ring_curled = (lm[16].y > lm[14].y) or (math.hypot(lm[16].x - lm[0].x, lm[16].y - lm[0].y) < hand_scale * 1.20)
    pinky_curled = (lm[20].y > lm[18].y) or (math.hypot(lm[20].x - lm[0].x, lm[20].y - lm[0].y) < hand_scale * 1.10)
    curled_count = sum([mid_curled, ring_curled, pinky_curled])

    index_dist_wrist = math.hypot(lm[8].x - lm[0].x, lm[8].y - lm[0].y) / hand_scale
    thumb_dist_wrist = math.hypot(lm[4].x - lm[0].x, lm[4].y - lm[0].y) / hand_scale

    if curled_count >= 2 and index_dist_wrist > 0.80 and thumb_dist_wrist > 0.50:
        if aperture >= 0.48:
            return ("ألم قوي", 0.98)
        elif aperture < 0.44:
            return ("ألم خفيف", 0.98)
    return None


# ==============================================================================
# Modern Anti-AI High-Craft Application Main Class
# ==============================================================================

class SignAIModernApp:
    """
    World-class desktop application adhering to Moatassem.ai Anti-AI High-Craft Design Standard.
    - Zero Unicode emojis.
    - Zinc monochrome palette: Base #09090b, Card #141416, Border #27272a, Accent Emerald #10b981.
    - Tabular typography for counters and indicators.
    - Embedded OpenCV MediaPipe feed, dynamic hold progress slider, and audio TTS.
    """
    def __init__(self, root, start_loop=True):
        self.root = root
        self.root.title("SignAI - Clinical Translation Workstation")
        self.root.geometry("1340x740")
        self.root.minsize(1240, 680)
        self.root.configure(bg="#09090b")

        # Core State
        self.current_mode = "WORDS"  # "WORDS" (Dental Clinic) or "LETTERS" (Fingerspelling)
        self.expression = ""
        self.camera_active = True
        self.mirror_view = True

        # Interactive Tuning Parameters (Observable variables)
        self.hold_threshold_var = DoubleVar(value=1.5)  # Snappy 1.5s default hold
        self.confidence_threshold_var = DoubleVar(value=0.80)
        self.tts_enabled_var = BooleanVar(value=True)

        # Dialogue & TTS Engines
        self.dialogue_mgr = DialogueManager()
        self.tts_engine = TTSEngine(enabled=True)

        # Recognition Models
        self.keypoint_classifier = KeyPointClassifier()
        self.word_classifier = WordClassifier()

        # Load Labels
        self.alphabet_labels = self._load_csv_labels('model/keypoint_classifier/keypoint_classifier_label.csv')
        self.word_labels = self._load_csv_labels('model/word_classifier/word_classifier_label.csv')

        # Load Reference Chart Posters
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dental_img_path = os.path.join(base_dir, "images", "dental_signs.png")
        alpha_img_path = os.path.join(base_dir, "images", "arabic_alphabet.jpg")

        self.photo_dental = None
        self.photo_alphabet = None
        if os.path.exists(dental_img_path):
            try:
                chart_d = Image.open(dental_img_path).resize((340, 540), Image.Resampling.LANCZOS)
                self.photo_dental = ImageTk.PhotoImage(chart_d, master=self.root)
            except Exception:
                pass

        if os.path.exists(alpha_img_path):
            try:
                chart_a = Image.open(alpha_img_path).resize((340, 540), Image.Resampling.LANCZOS)
                self.photo_alphabet = ImageTk.PhotoImage(chart_a, master=self.root)
            except Exception:
                pass

        # Video capture & tracking
        if start_loop:
            self.cap = cv.VideoCapture(0)
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            self.cap = None

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) if mp_hands is not None else None

        # Gesture temporal smoothing state
        self.pred_history = deque(maxlen=7)
        self.candidate_sign = None
        self.candidate_start_time = None
        self.sign_already_committed = False
        self.last_valid_time = 0.0
        self.grace_period = 0.25

        # Performance monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.last_inference_ms = 0.0

        # Build UI Architecture
        self._build_top_navbar()
        self._build_main_workspace()
        self._build_bottom_statusbar()

        # Keyboard shortcuts
        self.root.bind_all("<Tab>", self._toggle_mode)
        self.root.bind_all("<Key-m>", self._toggle_mode)
        self.root.bind_all("<Key-M>", self._toggle_mode)
        self.root.bind_all("<F1>", self._toggle_mode)

        # Start continuous video loop
        self.is_running = start_loop
        if start_loop and self.cap is not None:
            self._run_video_loop()

    def _load_csv_labels(self, path):
        labels = []
        if not os.path.isabs(path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            alt = os.path.join(base_dir, path)
            if os.path.exists(alt):
                path = alt
        if os.path.exists(path):
            with open(path, encoding='utf-8-sig') as f:
                labels = [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]
        return labels

    # --------------------------------------------------------------------------
    # UI Components Construction
    # --------------------------------------------------------------------------

    def _build_top_navbar(self):
        nav_frame = Frame(self.root, bg="#141416", height=54, highlightbackground="#27272a", highlightthickness=1)
        nav_frame.pack(side="top", fill="x", padx=12, pady=(10, 6))

        # Brand Title (Crisp Anti-AI Minimalist Typography)
        title_label = Label(
            nav_frame,
            text="SIGNAI | CLINICAL TRANSLATION WORKSTATION",
            font=("Segoe UI", 12, "bold"),
            fg="#F4F4F5",
            bg="#141416",
            padx=16
        )
        title_label.pack(side="left", pady=10)

        # Subtitle badge
        sub_label = Label(
            nav_frame,
            text="EGYPTIAN ARABIC ARSL",
            font=("Segoe UI", 9, "bold"),
            fg="#10B981",
            bg="#18231E",
            padx=8,
            pady=3
        )
        sub_label.pack(side="left", padx=4)

        # Right-side CTA controls
        self.btn_mode_toggle = Button(
            nav_frame,
            text="الوضع: إشارات طبيب الأسنان [Tab / M]",
            font=("Segoe UI", 10, "bold"),
            bg="#10B981",
            fg="#09090B",
            activebackground="#059669",
            activeforeground="#FFFFFF",
            relief="flat",
            padx=14,
            pady=4,
            command=self._toggle_mode
        )
        self.btn_mode_toggle.pack(side="right", padx=12)

        self.btn_tts_toggle = Button(
            nav_frame,
            text="الصوت: مفعل",
            font=("Segoe UI", 9, "bold"),
            bg="#27272A",
            fg="#E4E4E7",
            activebackground="#3F3F46",
            relief="flat",
            padx=10,
            pady=4,
            command=self._toggle_tts
        )
        self.btn_tts_toggle.pack(side="right", padx=6)

    def _build_main_workspace(self):
        main_container = Frame(self.root, bg="#09090b")
        main_container.pack(fill="both", expand=True, padx=12, pady=4)

        # Column 1: Video HUD & Parameter Sliders (Left, 540px)
        col_video = Frame(main_container, bg="#141416", highlightbackground="#27272a", highlightthickness=1)
        col_video.pack(side="left", fill="both", expand=False, padx=(0, 6), pady=4)

        self.lbl_camera = Label(col_video, bg="#09090b")
        self.lbl_camera.pack(padx=8, pady=8)

        # Sliders Container
        ctrl_frame = Frame(col_video, bg="#141416")
        ctrl_frame.pack(fill="x", padx=12, pady=4)

        Label(ctrl_frame, text="مدة تثبيت الإشارة (ثواني):", font=("Segoe UI", 9), fg="#A1A1AA", bg="#141416").grid(row=0, column=0, sticky="w")
        self.lbl_hold_val = Label(ctrl_frame, text="1.5s", font=("Segoe UI", 9, "bold"), fg="#10B981", bg="#141416")
        self.lbl_hold_val.grid(row=0, column=1, sticky="e")

        self.slider_hold = Scale(
            ctrl_frame,
            from_=0.5, to=3.0,
            resolution=0.1,
            orient=HORIZONTAL,
            showvalue=False,
            variable=self.hold_threshold_var,
            command=lambda v: self.lbl_hold_val.config(text=f"{float(v):.1f}s"),
            bg="#141416",
            troughcolor="#27272A",
            activebackground="#10B981",
            highlightthickness=0,
            length=300
        )
        self.slider_hold.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 6))

        Label(ctrl_frame, text="حد دقة التعرف (Confidence):", font=("Segoe UI", 9), fg="#A1A1AA", bg="#141416").grid(row=2, column=0, sticky="w")
        self.lbl_conf_val = Label(ctrl_frame, text="80%", font=("Segoe UI", 9, "bold"), fg="#10B981", bg="#141416")
        self.lbl_conf_val.grid(row=2, column=1, sticky="e")

        self.slider_conf = Scale(
            ctrl_frame,
            from_=0.60, to=0.95,
            resolution=0.05,
            orient=HORIZONTAL,
            showvalue=False,
            variable=self.confidence_threshold_var,
            command=lambda v: self.lbl_conf_val.config(text=f"{int(float(v)*100)}%"),
            bg="#141416",
            troughcolor="#27272A",
            activebackground="#10B981",
            highlightthickness=0,
            length=300
        )
        self.slider_conf.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(2, 6))

        # Column 2: Composer & Clinical Dialogue (Center, ~440px)
        col_center = Frame(main_container, bg="#141416", highlightbackground="#27272a", highlightthickness=1)
        col_center.pack(side="left", fill="both", expand=True, padx=6, pady=4)

        lbl_comp_title = Label(col_center, text="صياغة رسالة المريض (Composer)", font=("Segoe UI", 11, "bold"), fg="#F4F4F5", bg="#141416")
        lbl_comp_title.pack(anchor="ne", padx=14, pady=(8, 2))

        # Large RTL Arabic entry
        self.entry_composer = Entry(
            col_center,
            font=("Segoe UI", 18, "bold"),
            fg="#FFFFFF",
            bg="#1E1E22",
            insertbackground="#10B981",
            justify="right",
            relief="flat",
            highlightbackground="#27272A",
            highlightcolor="#10B981",
            highlightthickness=1
        )
        self.entry_composer.pack(fill="x", padx=14, pady=4, ipady=4)
        self.entry_composer.bind("<Return>", lambda e: self._submit_sentence())

        # Action Buttons
        btn_action_frame = Frame(col_center, bg="#141416")
        btn_action_frame.pack(fill="x", padx=14, pady=4)

        btn_submit = Button(
            btn_action_frame,
            text="إرسال للجلسة [Enter]",
            font=("Segoe UI", 9, "bold"),
            bg="#10B981",
            fg="#09090B",
            activebackground="#059669",
            relief="flat",
            padx=8,
            pady=3,
            command=self._submit_sentence
        )
        btn_submit.pack(side="right", padx=2)

        btn_space = Button(
            btn_action_frame,
            text="مسافة",
            font=("Segoe UI", 9),
            bg="#27272A",
            fg="#E4E4E7",
            activebackground="#3F3F46",
            relief="flat",
            padx=8,
            pady=3,
            command=lambda: self._enter_text(" ")
        )
        btn_space.pack(side="right", padx=2)

        btn_backspace = Button(
            btn_action_frame,
            text="مسح حرف",
            font=("Segoe UI", 9),
            bg="#27272A",
            fg="#E4E4E7",
            activebackground="#3F3F46",
            relief="flat",
            padx=6,
            pady=3,
            command=self._delete_char
        )
        btn_backspace.pack(side="right", padx=2)

        btn_clear = Button(
            btn_action_frame,
            text="مسح الكل",
            font=("Segoe UI", 9),
            bg="#27272A",
            fg="#E4E4E7",
            activebackground="#3F3F46",
            relief="flat",
            padx=6,
            pady=3,
            command=self._clear_composer
        )
        btn_clear.pack(side="right", padx=2)

        # Quick Palette Section (Dynamically toggles between Words and Letters!)
        self.f_quick_container = Frame(col_center, bg="#141416")
        self.f_quick_container.pack(fill="x", padx=14, pady=4)

        # Dental phrases frame
        self.f_quick_words = Frame(self.f_quick_container, bg="#141416")
        lbl_quick_w = Label(self.f_quick_words, text="إشارات عيادة الأسنان السريعة:", font=("Segoe UI", 9, "bold"), fg="#10B981", bg="#141416")
        lbl_quick_w.pack(anchor="ne", pady=(0, 2))
        grid_w = Frame(self.f_quick_words, bg="#141416")
        grid_w.pack(fill="x")
        dental_presets = [
            ("أشعر بألم", "أشعر بألم "),
            ("ألم خفيف", "ألم خفيف "),
            ("ألم قوي", "ألم قوي "),
            ("قف", "قف "),
            ("أنا تمام", "أنا تمام "),
            ("غسل الفم", "اريد غسل الفم "),
            ("اريد التحدث", "اريد أن اتحدث "),
            ("أحك وجهي", "اريد أن أحك وجهي ")
        ]
        for i, (btn_text, phrase) in enumerate(dental_presets):
            b = Button(
                grid_w, text=btn_text, font=("Segoe UI", 8), bg="#1E1E22", fg="#F4F4F5",
                activebackground="#27272A", relief="flat", highlightbackground="#27272A",
                highlightthickness=1, padx=4, pady=4, command=lambda p=phrase: self._enter_text(p)
            )
            b.grid(row=i // 4, column=i % 4, sticky="ew", padx=2, pady=2)
        for c in range(4):
            grid_w.columnconfigure(c, weight=1)
        self.f_quick_words.pack(fill="x")

        # Alphabet letters frame (initially hidden, shows in LETTERS mode)
        self.f_quick_alpha = Frame(self.f_quick_container, bg="#141416")
        lbl_quick_a = Label(self.f_quick_alpha, text="حروف أبجدية شائعة:", font=("Segoe UI", 9, "bold"), fg="#0284C7", bg="#141416")
        lbl_quick_a.pack(anchor="ne", pady=(0, 2))
        grid_a = Frame(self.f_quick_alpha, bg="#141416")
        grid_a.pack(fill="x")
        alpha_presets = ["ا", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ر", "س", "ش", "ص", "ع", "ف", "ق", "ك", "ل", "م", "ن", "هـ", "و", "ي"]
        for i, letter in enumerate(alpha_presets):
            b = Button(
                grid_a, text=letter, font=("Segoe UI", 8, "bold"), bg="#1E1E22", fg="#38BDF8",
                activebackground="#27272A", relief="flat", highlightbackground="#27272A",
                highlightthickness=1, padx=2, pady=2, command=lambda l=letter: self._enter_text(l)
            )
            b.grid(row=i // 11, column=i % 11, sticky="ew", padx=1, pady=1)
        for c in range(11):
            grid_a.columnconfigure(c, weight=1)

        # Dialogue Transcript
        lbl_trans = Label(col_center, text="سجل المحادثة السريرية:", font=("Segoe UI", 10, "bold"), fg="#A1A1AA", bg="#141416")
        lbl_trans.pack(anchor="ne", padx=14, pady=(6, 2))

        self.txt_transcript = Text(
            col_center, wrap="word", bg="#18181B", fg="#F4F4F5", font=("Segoe UI", 10),
            relief="flat", padx=8, pady=6, height=7, highlightbackground="#27272A", highlightthickness=1
        )
        self.txt_transcript.tag_configure("rtl", justify="right")
        self.txt_transcript.tag_configure("patient_tag", foreground="#10B981", font=("Segoe UI", 10, "bold"))
        self.txt_transcript.tag_configure("doctor_tag", foreground="#38BDF8", font=("Segoe UI", 10, "bold"))
        self.txt_transcript.tag_configure("time_tag", foreground="#71717A", font=("Segoe UI", 8))
        self.txt_transcript.pack(fill="both", expand=True, padx=14, pady=2)

        # Doctor quick replies & export row
        doc_row = Frame(col_center, bg="#141416")
        doc_row.pack(fill="x", padx=14, pady=(2, 6))
        for preset in DialogueManager.CLINICAL_DOCTOR_PRESETS[:2]:
            Button(
                doc_row, text=preset, font=("Segoe UI", 8), bg="#18181B", fg="#E4E4E7",
                relief="flat", highlightbackground="#27272A", highlightthickness=1,
                padx=4, pady=2, command=lambda p=preset: self._add_doctor_message(p)
            ).pack(side="right", padx=2)

        Button(
            doc_row, text="تصدير السجل", font=("Segoe UI", 8, "bold"), bg="#27272A", fg="#F4F4F5",
            relief="flat", padx=6, pady=2, command=self._export_transcript
        ).pack(side="left", padx=2)

        Button(
            doc_row, text="مسح السجل", font=("Segoe UI", 8), bg="#27272A", fg="#A1A1AA",
            relief="flat", padx=6, pady=2, command=self._clear_transcript
        ).pack(side="left", padx=2)

        # Column 3: Live Sign Reference Poster (Right, ~350px)
        col_poster = Frame(main_container, bg="#141416", highlightbackground="#27272a", highlightthickness=1)
        col_poster.pack(side="right", fill="both", expand=False, padx=(6, 0), pady=4)

        self.lbl_chart_title = Label(
            col_poster,
            text="دليل إشارات عيادة الأسنان المعتمدة",
            font=("Segoe UI", 10, "bold"),
            fg="#10B981",
            bg="#141416"
        )
        self.lbl_chart_title.pack(anchor="center", padx=8, pady=(8, 4))

        self.lbl_reference_chart = Label(col_poster, bg="#141416")
        if self.photo_dental is not None:
            self.lbl_reference_chart.configure(image=self.photo_dental)
            self.lbl_reference_chart.image = self.photo_dental
        self.lbl_reference_chart.pack(padx=8, pady=4)

    def _build_bottom_statusbar(self):
        status_bar = Frame(self.root, bg="#141416", height=28, highlightbackground="#27272a", highlightthickness=1)
        status_bar.pack(side="bottom", fill="x", padx=12, pady=(0, 6))

        self.lbl_status_left = Label(
            status_bar,
            text="SYSTEM READY | CAMERA CONNECTED",
            font=("Segoe UI", 8, "bold"),
            fg="#71717A",
            bg="#141416",
            padx=10
        )
        self.lbl_status_left.pack(side="left", pady=4)

        self.lbl_status_perf = Label(
            status_bar,
            text="FPS: -- | INFERENCE: -- ms",
            font=("Segoe UI", 8, "bold"),
            fg="#10B981",
            bg="#141416",
            padx=10
        )
        self.lbl_status_perf.pack(side="right", pady=4)

    # --------------------------------------------------------------------------
    # Interaction Logic & Actions
    # --------------------------------------------------------------------------

    def _toggle_mode(self, event=None):
        if self.current_mode == "WORDS":
            self.current_mode = "LETTERS"
            self.btn_mode_toggle.config(
                text="الوضع: الحروف الأبجدية [Tab / M]",
                bg="#0284C7",
                fg="#FFFFFF"
            )
            if hasattr(self, 'lbl_reference_chart') and self.photo_alphabet is not None:
                self.lbl_reference_chart.configure(image=self.photo_alphabet)
                self.lbl_reference_chart.image = self.photo_alphabet
            if hasattr(self, 'lbl_chart_title'):
                self.lbl_chart_title.config(text="لوحة الحروف الأبجدية (31 إشارة)", fg="#0284C7")
            if hasattr(self, 'f_quick_words'):
                self.f_quick_words.pack_forget()
            if hasattr(self, 'f_quick_alpha'):
                self.f_quick_alpha.pack(fill="x")
        else:
            self.current_mode = "WORDS"
            self.btn_mode_toggle.config(
                text="الوضع: إشارات طبيب الأسنان [Tab / M]",
                bg="#10B981",
                fg="#09090B"
            )
            if hasattr(self, 'lbl_reference_chart') and self.photo_dental is not None:
                self.lbl_reference_chart.configure(image=self.photo_dental)
                self.lbl_reference_chart.image = self.photo_dental
            if hasattr(self, 'lbl_chart_title'):
                self.lbl_chart_title.config(text="لوحة إشارات عيادة الأسنان (8 إشارات)", fg="#10B981")
            if hasattr(self, 'f_quick_alpha'):
                self.f_quick_alpha.pack_forget()
            if hasattr(self, 'f_quick_words'):
                self.f_quick_words.pack(fill="x")

        self.pred_history.clear()
        self.candidate_sign = None
        self.candidate_start_time = None
        self.sign_already_committed = False
        if event is not None:
            return "break"

    def _toggle_tts(self):
        enabled = self.tts_engine.toggle()
        self.tts_enabled_var.set(enabled)
        self.btn_tts_toggle.config(
            text="الصوت: مفعل" if enabled else "الصوت: معطل",
            fg="#10B981" if enabled else "#71717A"
        )

    def _enter_text(self, text):
        self.expression += text
        self.entry_composer.delete(0, END)
        self.entry_composer.insert(0, self.expression)

    def _delete_char(self):
        if len(self.expression) > 0:
            if self.current_mode == "WORDS":
                trimmed = self.expression.rstrip()
                last_space = trimmed.rfind(" ")
                self.expression = trimmed[:last_space + 1] if last_space != -1 else ""
            else:
                self.expression = self.expression[:-1]
            self.entry_composer.delete(0, END)
            self.entry_composer.insert(0, self.expression)

    def _clear_composer(self):
        self.expression = ""
        self.entry_composer.delete(0, END)

    def _submit_sentence(self):
        text = self.expression.strip()
        if text:
            entry = self.dialogue_mgr.add_message("patient", text)
            if entry:
                self._render_transcript()
                if self.tts_enabled_var.get():
                    self.tts_engine.speak(text)
            self._clear_composer()

    def _add_doctor_message(self, text):
        entry = self.dialogue_mgr.add_message("doctor", text)
        if entry:
            self._render_transcript()

    def _render_transcript(self):
        self.txt_transcript.delete("1.0", END)
        for msg in self.dialogue_mgr.get_messages():
            role_text = "المريض: " if msg["role"] == "patient" else "الطبيب: "
            tag = "patient_tag" if msg["role"] == "patient" else "doctor_tag"

            self.txt_transcript.insert(END, f"[{msg['timestamp']}] ", "time_tag")
            self.txt_transcript.insert(END, role_text, tag)
            self.txt_transcript.insert(END, f"{msg['text']}\n", "rtl")
        self.txt_transcript.see(END)

    def _export_transcript(self):
        filepath = self.dialogue_mgr.export_transcript()
        self.lbl_status_left.config(text=f"EXPORTED: {os.path.basename(filepath)}", fg="#10B981")

    def _clear_transcript(self):
        self.dialogue_mgr.clear()
        self._render_transcript()

    # --------------------------------------------------------------------------
    # Real-Time Perception & Gesture Classification Loop
    # --------------------------------------------------------------------------

    def _run_video_loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            if self.mirror_view:
                frame = cv.flip(frame, 1)

            t_start = time.time()
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = self.hands.process(rgb_frame) if self.hands is not None else None
            rgb_frame.flags.writeable = True

            current_time = time.time()
            raw_sign = ""
            raw_confidence = 0.0
            display_text = ""
            hold_progress = 0.0
            hold_threshold = self.hold_threshold_var.get()
            min_confidence = self.confidence_threshold_var.get()

            # Persistent Mode Badge on Camera Feed
            mode_badge_text = "إشارات عيادة الأسنان" if self.current_mode == "WORDS" else "الحروف الأبجدية"
            mode_badge_col = (16, 185, 129) if self.current_mode == "WORDS" else (2, 132, 199)
            frame = draw_hud_badge(frame, mode_badge_text, "[Tab/M]", pos=(20, 32), font_size=15, border_color=mode_badge_col)

            if results and results.multi_hand_landmarks:
                no_of_hands = len(results.multi_hand_landmarks)
                hand_features_list = []

                # Draw landmark connections with crisp high-visibility styling
                for hand_lms in results.multi_hand_landmarks:
                    if mp_drawing and hasattr(mp_hands, 'HAND_CONNECTIONS'):
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_lms,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(129, 185, 16), thickness=2)
                        )
                    # Extract 74 kinematic features
                    coords = [[lm.x, lm.y] for lm in hand_lms.landmark]
                    feats = extract_hand_features(coords)
                    hand_features_list.append(feats)

                # 1. Kinematic Fast-Path Rules
                is_two_thumbs = (
                    no_of_hands == 2 and
                    is_thumbs_up(results.multi_hand_landmarks[0]) and
                    is_thumbs_up(results.multi_hand_landmarks[1])
                )
                is_two_palms = (
                    no_of_hands == 2 and
                    is_open_palm(results.multi_hand_landmarks[0]) and
                    is_open_palm(results.multi_hand_landmarks[1])
                )
                has_thumbs_down = (
                    is_thumbs_down(results.multi_hand_landmarks[0]) or
                    (no_of_hands == 2 and is_thumbs_down(results.multi_hand_landmarks[1]))
                )

                if is_two_thumbs:
                    raw_sign = "space"
                    raw_confidence = 1.0
                    display_text = "مسافة"
                elif is_two_palms:
                    raw_sign = "finish"
                    raw_confidence = 1.0
                    display_text = "إرسال الجملة"
                elif has_thumbs_down:
                    raw_sign = "delete"
                    raw_confidence = 1.0
                    display_text = "مسح"
                else:
                    if self.current_mode == "WORDS":
                        # Check C-pain aperture check
                        c_pain = check_dental_c_pain(results.multi_hand_landmarks[0])
                        if c_pain is not None:
                            raw_sign, raw_confidence = c_pain
                            display_text = raw_sign
                        elif self.word_classifier.is_loaded:
                            idx, raw_confidence = self.word_classifier.classify_hands(hand_features_list)
                            if 0 <= idx < len(self.word_labels):
                                raw_sign = self.word_labels[idx]
                                display_text = raw_sign
                    else:
                        idx, raw_confidence = self.keypoint_classifier.classify_hands(hand_features_list)
                        if 0 <= idx < len(self.alphabet_labels):
                            raw_sign = self.alphabet_labels[idx]
                            display_text = raw_sign

                self.last_inference_ms = (time.time() - t_start) * 1000.0

                # 2. Majority-Vote Temporal Smoothing
                if raw_sign != "":
                    self.pred_history.append((raw_sign, raw_confidence))
                    counts = Counter([s for s, c in self.pred_history])
                    best_sign, num_votes = counts.most_common(1)[0]
                    confs = [c for s, c in self.pred_history if s == best_sign]
                    if num_votes >= 4:
                        sign = best_sign
                        confidence = sum(confs) / len(confs)
                    else:
                        sign = ""
                        confidence = 0.0
                else:
                    sign = ""
                    confidence = 0.0

                # 3. Hold Progress & Trigger Engine
                if confidence >= min_confidence:
                    if self.candidate_sign != sign:
                        self.candidate_sign = sign
                        self.candidate_start_time = current_time
                        self.last_valid_time = current_time
                        self.sign_already_committed = False
                    else:
                        self.last_valid_time = current_time

                    elapsed = current_time - self.candidate_start_time
                    hold_progress = min(1.0, elapsed / hold_threshold)

                    if elapsed >= hold_threshold and not self.sign_already_committed:
                        # Commit action
                        if sign in ["finish", "نقطة"]:
                            self._submit_sentence()
                        elif sign in ["space", "مسافة"]:
                            self._enter_text(" ")
                        elif sign in ["delete", "مسح"]:
                            self._delete_char()
                        else:
                            if self.current_mode == "WORDS":
                                self._enter_text(sign + " ")
                            else:
                                self._enter_text(sign)

                        self.sign_already_committed = True
                else:
                    if self.candidate_start_time and (current_time - self.last_valid_time > self.grace_period):
                        self.candidate_sign = None
                        self.candidate_start_time = None
                        self.sign_already_committed = False
                        hold_progress = 0.0
                        self.pred_history.clear()
                    elif self.candidate_start_time:
                        elapsed = current_time - self.candidate_start_time
                        hold_progress = min(1.0, elapsed / hold_threshold)

                # Draw Sleek High-Craft HUD
                if display_text and (confidence >= 0.70 or self.sign_already_committed):
                    status_text = "[تم]" if self.sign_already_committed else f"{int(confidence*100)}%"
                    border_color = (16, 185, 129) if (self.sign_already_committed or hold_progress >= 1.0) else (38, 38, 42)
                    frame = draw_hud_badge(frame, display_text, status_text, pos=(20, 72), font_size=20, border_color=border_color)

                    # Dynamic Progress Bar underneath badge
                    if hold_progress > 0:
                        bx, by, bw, bh = 20, 88, 180, 6
                        cv.rectangle(frame, (bx, by), (bx + bw, by + bh), (24, 24, 27), -1)
                        fill_w = int(bw * hold_progress)
                        bar_col = (16, 185, 129) if self.sign_already_committed else (245, 158, 11)
                        if fill_w > 0:
                            cv.rectangle(frame, (bx, by), (bx + fill_w, by + bh), bar_col, -1)
            else:
                frame = draw_hud_badge(frame, "في انتظار اليدين", "ضع يدك أمام الكاميرا", pos=(20, 72), font_size=13, border_color=(75, 85, 99))
                if self.candidate_start_time and (current_time - self.last_valid_time > self.grace_period):
                    self.candidate_sign = None
                    self.candidate_start_time = None
                    self.sign_already_committed = False
                    self.pred_history.clear()

            # FPS computation
            self.fps_counter += 1
            if current_time - self.fps_start_time >= 1.0:
                self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
                self.fps_counter = 0
                self.fps_start_time = current_time
                self.lbl_status_perf.config(
                    text=f"FPS: {self.current_fps:.1f} | INFERENCE: {self.last_inference_ms:.1f} ms"
                )

            # Update Tkinter Camera Display
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.lbl_camera.imgtk = img_tk
            self.lbl_camera.configure(image=img_tk)

        self.root.after(16, self._run_video_loop)

    def close(self):
        self.is_running = False
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.tts_engine.stop()
        self.root.destroy()


def main():
    root = Tk()
    app = SignAIModernApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()


if __name__ == '__main__':
    main()
