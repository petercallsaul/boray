import os
import csv
import argparse
import sys
import time
import math
import cv2 as cv
import numpy as np
import mediapipe as mp
import argostranslate.package
import argostranslate.translate
from collections import deque, Counter
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
mp_drawing = mp.solutions.drawing_utils
from tkinter import *
from PIL import Image, ImageTk
from gingerit.gingerit import GingerIt



expression = ' '
stored_inputs = []

from_code = "en"
to_code = "ar"
# Download and install Argos Translate package

installed_packages = argostranslate.package.get_installed_packages()
is_installed = any(
    x.from_code == from_code and x.to_code == to_code for x in installed_packages
)
if not is_installed:
    print('checking network ...')
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        filter(
            lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
        )
    )
    argostranslate.package.install_from_path(package_to_install.download())


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.4)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=float,
                        default=0.5)
    parser.add_argument("--hold_time",
                        help='seconds to hold sign steadily to confirm (default: 4.0)',
                        type=float,
                        default=4.0)
    args = parser.parse_args()
    return args

def main():
    global expression, stored_inputs
    try:

        def cls():
            global expression
            expression = ''
            e.delete(0, "end")

        def delete_char():
            global expression
            if len(expression) > 0:
                if current_app_mode == "WORDS":
                    trimmed = expression.rstrip()
                    last_space = trimmed.rfind(" ")
                    if last_space != -1:
                        expression = trimmed[:last_space + 1]
                    else:
                        expression = ""
                else:
                    expression = expression[:-1]
                e.delete(0, "end")
                e.insert(0, expression)

        def submit_sentence():
            global expression
            var = expression.strip()
            if var:
                add_input(f'المريض: {var}')
                cls()

        def enter_text(x):
            global expression
            e.delete(0, "end")
            expression = expression + str(x)
            e.insert(0, expression)

        def add_input(input_text):
            if input_text:
                stored_inputs.append(input_text)
                update_text()

        def update_text():
            text.delete(1.0, END)
            for input_text in stored_inputs:
                text.insert(END, input_text + '\n', "rtl")

        def cls_stored():
            global stored_inputs
            stored_inputs = []
            add_input("سجل المحادثة :")

        args = get_args()
        cap_device = args.device
        cap_width = args.width
        cap_height = args.height
        use_static_image_mode = args.use_static_image_mode
        min_detection_confidence = args.min_detection_confidence
        min_tracking_confidence = args.min_tracking_confidence
        hold_time_threshold = args.hold_time

        
        root = Tk()
        root.title("Sign.AI")
        root.geometry("1200x600")
        photo = PhotoImage(file = "images\logo.png")
        root.iconphoto(False,photo)
        root.config(bg="#1C1C1C")

        Label(root, text="نظام ترجمة لغة الإشارة المصرية", font=("Segoe UI", 26, "bold"), bg="#1C1C1C", fg="#78F093").grid(row=0, column=0, columnspan=4, pady=4)

        current_app_mode = "WORDS"  # "WORDS" (Dental Clinic Signs) or "LETTERS" (Fingerspelling)

        # Charts: Alphabet & Dental Clinic Signs
        chart_alpha = Image.open("images/arabic_alphabet.jpg").resize((340, 540), Image.Resampling.LANCZOS)
        photo_alphabet = ImageTk.PhotoImage(chart_alpha)

        chart_dental = Image.open("images/dental_signs.png").resize((340, 540), Image.Resampling.LANCZOS)
        photo_dental = ImageTk.PhotoImage(chart_dental)

        def toggle_app_mode(event=None):
            nonlocal current_app_mode
            if current_app_mode == "WORDS":
                current_app_mode = "LETTERS"
                btn_mode_toggle.config(text="الوضع: حروف أبجدية [M / Tab]", bg="#2980b9")
                label_chart.configure(image=photo_alphabet)
                label_chart.image = photo_alphabet
            else:
                current_app_mode = "WORDS"
                btn_mode_toggle.config(text="الوضع: إشارات طبيب الأسنان [M / Tab]", bg="#27ae60")
                label_chart.configure(image=photo_dental)
                label_chart.image = photo_dental
            if event is not None:
                return "break"

        btn_mode_toggle = Button(root, text="الوضع: إشارات طبيب الأسنان [M / Tab]", font=("Segoe UI", 11, "bold"), bg="#27ae60", fg="white", bd=2, padx=10, command=toggle_app_mode)
        btn_mode_toggle.grid(row=0, column=4, padx=10, pady=5)

        # Global Key bindings for Tab and M/m to always toggle mode
        root.bind_all("<Tab>", toggle_app_mode)
        root.bind_all("<Key-m>", toggle_app_mode)
        root.bind_all("<Key-M>", toggle_app_mode)
        root.bind_all("<F1>", toggle_app_mode)
        root.bind_all("<Control-m>", toggle_app_mode)
        root.bind_all("<Control-M>", toggle_app_mode)
        f1 = LabelFrame(root, bg="#DEDEDE")
        f1.grid(row=1, column=1, columnspan=3)
        l1 = Label(f1, bg="#1C1C1C")
        l1.grid(row=1, column=1, columnspan=3)

        # Arabic Entry box aligned Right-to-Left
        e = Entry(root, width=38, fg="#FFFFFF", font=("Segoe UI", 22, "bold"), bg="#252526", justify="right")
        e.grid(row=2, column=1, columnspan=3, pady=6)
        e.bind("<Tab>", toggle_app_mode)
        e.bind("<Key-m>", toggle_app_mode)
        e.bind("<Key-M>", toggle_app_mode)

        # Action Buttons Row
        f_buttons = Frame(root, bg="#1C1C1C")
        f_buttons.grid(row=3, column=1, columnspan=3, pady=4)

        btn_clear = Button(f_buttons, text='مسح الكل', font=("Segoe UI", 10, "bold"), bg="#3A3A3A", fg="white", bd=1, padx=8, command=cls)
        btn_clear.grid(row=0, column=0, padx=4)

        btn_backspace = Button(f_buttons, text='مسح', font=("Segoe UI", 10, "bold"), bg="#D9534F", fg="white", bd=1, padx=8, command=delete_char)
        btn_backspace.grid(row=0, column=1, padx=4)

        btn_space = Button(f_buttons, text='مسافة', font=("Segoe UI", 10, "bold"), bg="#0275D8", fg="white", bd=1, padx=8, command=lambda: enter_text(" "))
        btn_space.grid(row=0, column=2, padx=4)

        btn_taa = Button(f_buttons, text='ة', font=("Segoe UI", 11, "bold"), bg="#337AB7", fg="white", bd=1, padx=10, command=lambda: enter_text("ة"))
        btn_taa.grid(row=0, column=3, padx=4)

        btn_laa = Button(f_buttons, text='لا', font=("Segoe UI", 11, "bold"), bg="#337AB7", fg="white", bd=1, padx=10, command=lambda: enter_text("لا"))
        btn_laa.grid(row=0, column=4, padx=4)

        btn_submit = Button(f_buttons, text='إرسال الجملة', font=("Segoe UI", 10, "bold"), bg="#5CB85C", fg="white", bd=1, padx=10, command=submit_sentence)
        btn_submit.grid(row=0, column=5, padx=4)

        # Dental Signs Quick Buttons Row (from dental sign chart)
        f_med_words = Frame(root, bg="#1C1C1C")
        f_med_words.grid(row=4, column=1, columnspan=3, pady=2)
        Label(f_med_words, text="إشارات العيادة:", font=("Segoe UI", 10, "bold"), bg="#1C1C1C", fg="#78F093").grid(row=0, column=0, padx=4)
        med_quick_list = [
            ('قف', '#c0392b', 'قف '),
            ('انا تمام', '#27ae60', 'انا تمام '),
            ('أشعر بألم', '#e74c3c', 'أشعر بألم '),
            ('ألم خفيف', '#f39c12', 'ألم خفيف '),
            ('ألم قوي', '#d35400', 'ألم قوي '),
            ('أحك وجهي', '#8e44ad', 'اريد أن أحك وجهي '),
            ('غسل الفم', '#2980b9', 'غسل الفم '),
            ('اتحدث', '#16a085', 'اريد أن اتحدث ')
        ]
        for col_i, (b_title, b_bg, b_phrase) in enumerate(med_quick_list, start=1):
            Button(f_med_words, text=b_title, font=("Segoe UI", 9, "bold"), bg=b_bg, fg="white", bd=1, padx=3,
                   command=lambda p=b_phrase: enter_text(p)).grid(row=0, column=col_i, padx=2)

        frame = Frame(root, bg="#282c34")
        frame.grid(row=1, column=0, padx=10, pady=10)

        text = Text(frame, wrap="word", bg="#282c34", fg="white", font=('Segoe UI', 13))
        text.tag_configure("rtl", justify="right")
        text.config(spacing1=8)
        text.config(spacing2=8)
        text.config(spacing3=8)
        text.grid(row=0, column=0, sticky="nsew")

        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=6)

        root.grid_rowconfigure(1, weight=4)
        root.grid_columnconfigure(0, weight=6)

        # Start with dental poster since default mode is WORDS
        label_chart = Label(root, image=photo_dental, bg="#1C1C1C")
        label_chart.image = photo_dental
        label_chart.grid(row=1, column=4, rowspan=4)

        add_input("سجل المحادثة : ")
        button_clear_predictions = Button(root, text='مسح السجل', font=("Segoe UI", 10), height=1, width=15, bd=2, command=cls_stored)
        button_clear_predictions.grid(row=2, column=0)

        cap = cv.VideoCapture(cap_device)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=use_static_image_mode,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        keypoint_classifier = KeyPointClassifier()

        with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            keypoint_classifier_labels = [
                row[0].strip() for row in reader if len(row) > 0 and row[0].strip()
            ]

        from model.word_classifier.word_classifier import WordClassifier
        word_classifier = WordClassifier()
        word_classifier_labels = []
        if os.path.exists('model/word_classifier/word_classifier_label.csv'):
            with open('model/word_classifier/word_classifier_label.csv', encoding='utf-8-sig') as f:
                word_classifier_labels = [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]

        mode = 0
        number = len(keypoint_classifier_labels) - 1
        times_captured = 0
        candidate_sign = None
        candidate_start_time = None
        sign_already_committed = False
        last_valid_time = 0.0
        grace_period = 0.25
        pred_history = deque(maxlen=9)

        while True:
            key = cv.waitKey(1)
            if key == 27:  # ESC
                break
            if key in [9, ord('m'), ord('M')]:  # Tab or M
                toggle_app_mode()

            number, mode = select_mode(key, mode, number, keypoint_classifier_labels)

            ret, image = cap.read()
            if not ret:
                break

            image = cv.flip(image, 1)
            debug_image = image.copy()
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            current_time = time.time()
            hold_progress = 0.0

            if results.multi_hand_landmarks is not None:
                no_of_hands = len(results.multi_hand_landmarks)
                pre_processed_landmark_list = []
                hand_features_list = []

                if no_of_hands == 2:
                    for hand_landmarks in results.multi_hand_landmarks[:2]:
                        feats = pre_process_landmarks_normalized(hand_landmarks)
                        hand_features_list.append(feats)
                        pre_processed_landmark_list.extend(feats)
                        mp_drawing.draw_landmarks(image=debug_image, landmark_list=hand_landmarks,
                                                connections=mp_hands.HAND_CONNECTIONS)
                elif no_of_hands == 1:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    feats = pre_process_landmarks_normalized(hand_landmarks)
                    hand_features_list.append(feats)
                    pre_processed_landmark_list = [0.0] * 42 + feats
                    mp_drawing.draw_landmarks(image=debug_image, landmark_list=hand_landmarks,
                                            connections=mp_hands.HAND_CONNECTIONS)
                else:
                    pre_processed_landmark_list = [0.0] * 84

                if len(pre_processed_landmark_list) != 84:
                    pre_processed_landmark_list = (pre_processed_landmark_list + [0.0] * 84)[:84]

                # 1. Two Thumbs Up -> Space
                is_two_thumbs_up = (
                    no_of_hands == 2 and
                    is_thumbs_up(results.multi_hand_landmarks[0]) and
                    is_thumbs_up(results.multi_hand_landmarks[1])
                )
                # 2. Two Open Palms -> Submit / Finish
                is_two_palms_open = (
                    no_of_hands == 2 and
                    is_open_palm(results.multi_hand_landmarks[0]) and
                    is_open_palm(results.multi_hand_landmarks[1])
                )
                # 3. Thumbs Down (1 or 2 hands) -> Delete / Backspace
                has_thumbs_down = (
                    is_thumbs_down(results.multi_hand_landmarks[0]) or
                    (no_of_hands == 2 and is_thumbs_down(results.multi_hand_landmarks[1]))
                )
                # 4. Special alphabet letters: لا (Laa) and ة (Taa Marbuta)
                has_laa = is_laa_sign(results.multi_hand_landmarks[0]) or (no_of_hands == 2 and is_laa_sign(results.multi_hand_landmarks[1]))
                has_taa_marbuta = is_taa_marbuta_sign(results.multi_hand_landmarks[0]) or (no_of_hands == 2 and is_taa_marbuta_sign(results.multi_hand_landmarks[1]))

                if is_two_thumbs_up:
                    raw_sign = "space"
                    raw_confidence = 1.0
                    display_text = "مسافة (إبهامين لأعلى)"
                elif is_two_palms_open:
                    raw_sign = "نقطة"
                    raw_confidence = 1.0
                    display_text = "إنهاء الجملة (كفين مفتوحين)"
                elif has_thumbs_down:
                    raw_sign = "مسح"
                    raw_confidence = 1.0
                    display_text = "مسح (إبهام لأسفل)"
                elif current_app_mode == "LETTERS" and has_laa:
                    raw_sign = "لا"
                    raw_confidence = 0.98
                    display_text = "لا (لام ألف)"
                elif current_app_mode == "LETTERS" and has_taa_marbuta:
                    raw_sign = "ة"
                    raw_confidence = 0.98
                    display_text = "ة (تاء مربوطة)"
                else:
                    if current_app_mode == "WORDS":
                        c_pain = None
                        if results.multi_hand_landmarks:
                            c_pain = check_dental_c_pain(results.multi_hand_landmarks[0])
                        if c_pain is not None:
                            raw_sign, raw_confidence = c_pain
                            display_text = raw_sign
                        elif word_classifier is not None and word_classifier.is_loaded:
                            hand_sign_id, raw_confidence = word_classifier.classify_hands(hand_features_list)
                            if 0 <= hand_sign_id < len(word_classifier_labels):
                                raw_sign = word_classifier_labels[hand_sign_id]
                            else:
                                raw_sign = ""
                            display_text = raw_sign
                        else:
                            raw_sign = ""
                            display_text = ""
                    else:
                        times_captured = logging_csv(number, mode, pre_processed_landmark_list, times_captured)
                        hand_sign_id, raw_confidence = keypoint_classifier.classify_hands(hand_features_list)
                        if 0 <= hand_sign_id < len(keypoint_classifier_labels):
                            raw_sign = keypoint_classifier_labels[hand_sign_id]
                        else:
                            raw_sign = ""
                        display_text = raw_sign

                # Majority-vote temporal smoothing over recent predictions (9-frame window)
                if raw_sign != "":
                    pred_history.append((raw_sign, raw_confidence))
                    counts = Counter([s for s, c in pred_history])
                    best_sign, num_votes = counts.most_common(1)[0]
                    confs = [c for s, c in pred_history if s == best_sign]
                    if num_votes >= 5:
                        sign = best_sign
                        confidence = sum(confs) / len(confs)
                    else:
                        sign = ""
                        confidence = 0.0
                else:
                    sign = ""
                    confidence = 0.0

                if confidence >= 0.80:
                    if candidate_sign != sign:
                        candidate_sign = sign
                        candidate_start_time = current_time
                        last_valid_time = current_time
                        sign_already_committed = False
                    else:
                        last_valid_time = current_time

                    elapsed = current_time - candidate_start_time
                    hold_progress = min(1.0, elapsed / hold_time_threshold)

                    if elapsed >= hold_time_threshold and not sign_already_committed:
                        if sign in ["fullstop", "نقطة"]:
                            submit_sentence()
                        elif sign in ["space", "مسافة"]:
                            if expression and expression[-1] != " ":
                                enter_text(" ")
                        elif sign in ["delete", "مسح"]:
                            delete_char()
                        else:
                            if current_app_mode == "WORDS":
                                enter_text(sign + " ")
                            else:
                                enter_text(sign)

                        sign_already_committed = True
                else:
                    if candidate_start_time and (current_time - last_valid_time > grace_period):
                        candidate_sign = None
                        candidate_start_time = None
                        sign_already_committed = False
                        hold_progress = 0.0
                        pred_history.clear()
                    elif candidate_start_time:
                        elapsed = current_time - candidate_start_time
                        hold_progress = min(1.0, elapsed / hold_time_threshold)

                debug_image = draw_info_text(
                    debug_image,
                    cap_width,
                    confidence,
                    display_text,
                    hold_progress=hold_progress,
                    hold_time_threshold=hold_time_threshold,
                    committed=sign_already_committed,
                )
            else:
                if candidate_start_time and (current_time - last_valid_time > grace_period):
                    candidate_sign = None
                    candidate_start_time = None
                    sign_already_committed = False
                    hold_progress = 0.0
                    pred_history.clear()

            debug_image = draw_info(debug_image, mode, number, times_captured, app_mode=current_app_mode)
            debug_image = cv.cvtColor(debug_image, cv.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(debug_image))
            l1['image'] = img
            root.update()

            # cv.imshow('Hand Gesture Recognition', debug_image)

        cap.release()
        cv.destroyAllWindows()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        try:
            cap.release()
            cv.destroyAllWindows()
        except Exception:
            pass




def select_mode(key, mode, number, labels=None):
    if key == 110:  # n
        mode = 0
    elif key == 107:  # k
        mode = 1
        sign_name = input("enter sign name:").strip()
        if sign_name:
            with open('model/keypoint_classifier/keypoint_classifier_label.csv', mode="a", newline="", encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([sign_name])
            if labels is not None:
                labels.append(sign_name)
                number = len(labels) - 1
            else:
                number += 1
    return number, mode


def is_thumbs_up(landmarks):
    lm = landmarks.landmark
    # Hand scale based on palm size (distance from wrist 0 to middle MCP 9)
    hand_scale = math.hypot(lm[9].x - lm[0].x, lm[9].y - lm[0].y)
    if hand_scale < 0.02:
        return False

    # 1. Thumb is vertically extended upwards relative to palm scale
    thumb_ext = (lm[2].y - lm[4].y) / hand_scale
    index_ext = (lm[5].y - lm[4].y) / hand_scale
    thumb_extended_up = (thumb_ext > 0.30) and (index_ext > 0.25)

    # 2. Sequential upward joint progression
    thumb_joints_up = (lm[4].y < lm[3].y < lm[2].y) and (lm[4].y < lm[0].y)

    # 3. Other 4 fingers curled (fingertip is lower than PIP or MCP)
    index_curled = (lm[8].y > lm[6].y) or (lm[8].y > lm[5].y)
    middle_curled = (lm[12].y > lm[10].y) or (lm[12].y > lm[9].y)
    ring_curled = (lm[16].y > lm[14].y) or (lm[16].y > lm[13].y)
    pinky_curled = (lm[20].y > lm[18].y) or (lm[20].y > lm[17].y)

    other_tips_y = [lm[8].y, lm[12].y, lm[16].y, lm[20].y]
    thumb_highest = lm[4].y < min(other_tips_y)

    curled_count = sum([index_curled, middle_curled, ring_curled, pinky_curled])
    return thumb_extended_up and thumb_joints_up and thumb_highest and (curled_count >= 3)


def is_thumbs_down(landmarks):
    lm = landmarks.landmark
    hand_scale = math.hypot(lm[9].x - lm[0].x, lm[9].y - lm[0].y)
    if hand_scale < 0.02:
        return False
    thumb_ext_down = ((lm[4].y - lm[2].y) / hand_scale > 0.20) and (lm[4].y > lm[0].y)
    thumb_progression = (lm[4].y > lm[3].y > lm[2].y)
    index_curled = (lm[8].y > lm[6].y) or (lm[8].y > lm[5].y)
    middle_curled = (lm[12].y > lm[10].y) or (lm[12].y > lm[9].y)
    ring_curled = (lm[16].y > lm[14].y) or (lm[16].y > lm[13].y)
    pinky_curled = (lm[20].y > lm[18].y) or (lm[20].y > lm[17].y)
    curled_count = sum([index_curled, middle_curled, ring_curled, pinky_curled])
    return thumb_ext_down and thumb_progression and (curled_count >= 3)


def is_open_palm(landmarks):
    lm = landmarks.landmark
    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up = lm[16].y < lm[14].y
    pinky_up = lm[20].y < lm[18].y
    thumb_up = lm[4].y < lm[2].y or abs(lm[4].x - lm[0].x) > 0.08
    return index_up and middle_up and ring_up and pinky_up and thumb_up


def is_laa_sign(landmarks):
    """لا (Laam-Alef) gesture: Index and Pinky extended UP (Horns 🤘), Middle and Ring curled."""
    lm = landmarks.landmark
    hand_scale = math.hypot(lm[9].x - lm[0].x, lm[9].y - lm[0].y)
    if hand_scale < 0.02:
        return False
    index_up = (lm[6].y - lm[8].y) / hand_scale > 0.25
    pinky_up = (lm[18].y - lm[20].y) / hand_scale > 0.20
    middle_curled = (lm[12].y > lm[10].y) or (lm[12].y > lm[9].y)
    ring_curled = (lm[16].y > lm[14].y) or (lm[16].y > lm[13].y)
    return index_up and pinky_up and middle_curled and ring_curled


def is_taa_marbuta_sign(landmarks):
    """ة (Taa Marbuta) gesture: Index, Middle, and Ring fingers bent forward at PIP in an arch/claw."""
    lm = landmarks.landmark
    hand_scale = math.hypot(lm[9].x - lm[0].x, lm[9].y - lm[0].y)
    if hand_scale < 0.02:
        return False
    index_bent = (lm[8].y >= lm[6].y - 0.02) and (lm[6].y < lm[5].y)
    middle_bent = (lm[12].y >= lm[10].y - 0.02) and (lm[10].y < lm[9].y)
    ring_bent = (lm[16].y >= lm[14].y - 0.02) and (lm[14].y < lm[13].y)
    pinky_curled = (lm[20].y > lm[17].y)
    return index_bent and middle_bent and ring_bent and pinky_curled


def check_dental_c_pain(landmarks):
    """
    Distinguishes between 'ألم خفيف' (Mild pain, small pinch gap) and 'ألم قوي' (Severe pain, wide C gap).
    Both signs have middle, ring, pinky curled into palm while index and thumb form a C aperture.
    """
    lm = landmarks.landmark
    hand_scale = math.hypot(lm[9].x - lm[0].x, lm[9].y - lm[0].y)
    if hand_scale < 0.02:
        return None

    # Distance between thumb tip (4) and index tip (8) normalized by palm scale
    aperture = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y) / hand_scale

    # Check if middle, ring, pinky are curled into palm
    mid_curled = (lm[12].y > lm[10].y) or (math.hypot(lm[12].x - lm[0].x, lm[12].y - lm[0].y) < hand_scale * 1.30)
    ring_curled = (lm[16].y > lm[14].y) or (math.hypot(lm[16].x - lm[0].x, lm[16].y - lm[0].y) < hand_scale * 1.20)
    pinky_curled = (lm[20].y > lm[18].y) or (math.hypot(lm[20].x - lm[0].x, lm[20].y - lm[0].y) < hand_scale * 1.10)
    curled_count = sum([mid_curled, ring_curled, pinky_curled])

    # Index finger extended / curved (not curled into palm)
    index_dist_wrist = math.hypot(lm[8].x - lm[0].x, lm[8].y - lm[0].y) / hand_scale
    thumb_dist_wrist = math.hypot(lm[4].x - lm[0].x, lm[4].y - lm[0].y) / hand_scale

    if curled_count >= 2 and index_dist_wrist > 0.80 and thumb_dist_wrist > 0.50:
        # If aperture is large (open C shape) -> ألم قوي
        if aperture >= 0.48:
            return ("ألم قوي", 0.98)
        # If aperture is small (tight pinch) -> ألم خفيف
        elif aperture < 0.44:
            return ("ألم خفيف", 0.98)

    return None


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    return [
        [min(int(lm.x * image_width), image_width - 1),
         min(int(lm.y * image_height), image_height - 1)]
        for lm in landmarks.landmark
    ]


def pre_process_landmarks_normalized(landmarks):
    """Direct normalized [x, y] coordinates in [0, 1] matching ArSL dataset without aspect distortion."""
    coords = [[lm.x, lm.y] for lm in landmarks.landmark]
    arr = np.array(coords, dtype=np.float32)
    arr -= arr[0]  # wrist translation to (0, 0)
    flat = arr.flatten()
    max_value = np.max(np.abs(flat))
    if max_value > 0:
        flat /= max_value
    return flat.tolist()


def pre_process_landmark(landmark_list):
    arr = np.array(landmark_list, dtype=np.float32)
    arr -= arr[0]
    flat = arr.flatten()
    max_value = np.max(np.abs(flat))
    if max_value > 0:
        flat /= max_value
    return flat.tolist()


def logging_csv(number, mode, landmark_list, times_captured):
    if mode == 0:
        times_captured = 0
    if mode == 1:
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
        times_captured = times_captured+1
    return times_captured


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


def draw_text(image, arabic_label, status_str, pos, font_size=26, color=(0, 255, 0)):
    try:
        from PIL import ImageDraw, ImageFont
        shaped_text = format_arabic_hud_text(arabic_label, status_str)
        img_pil = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
        pil_pos = (pos[0], max(0, pos[1] - font_size))
        
        # Dark badge background for crisp contrast
        bbox = draw.textbbox(pil_pos, shaped_text, font=font)
        draw.rectangle((bbox[0] - 6, bbox[1] - 3, bbox[2] + 6, bbox[3] + 3), fill=(25, 25, 25))
        
        draw.text(pil_pos, shaped_text, font=font, fill=(color[2], color[1], color[0]))
        return cv.cvtColor(np.array(img_pil), cv.COLOR_RGB2BGR)
    except Exception:
        cv.putText(image, f"{arabic_label} {status_str}", pos, cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv.LINE_AA)
        return image


def draw_info_text(image, img_width, confidence, hand_sign_text, hold_progress=0.0, hold_time_threshold=4.0, committed=False):
    if hand_sign_text != "" and (confidence >= 0.70 or committed):
        text_x = int(img_width / 2 + 10)
        color = (0, 255, 0) if (committed or confidence >= 0.80) else (0, 255, 255)
        status_str = "[تم الحفظ]" if committed else f"({int(confidence * 100)}%)"

        image = draw_text(image, hand_sign_text, status_str, (text_x, 38), font_size=26, color=color)

        if (confidence >= 0.80 or committed) and hold_progress > 0:
            bar_x = text_x
            bar_y = 52
            bar_w = 220
            bar_h = 14
            # Dark background and border
            cv.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
            cv.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)

            # Fill bar: Yellow/cyan while holding, bright green when complete
            fill_w = int(bar_w * min(1.0, hold_progress))
            bar_color = (0, 255, 0) if (committed or hold_progress >= 1.0) else (0, 210, 255)
            if fill_w > 0:
                cv.rectangle(image, (bar_x + 1, bar_y + 1), (bar_x + fill_w - 1, bar_y + bar_h - 1), bar_color, -1)

            # Time text
            if committed:
                time_text = "CONFIRMED"
            else:
                time_text = f"{hold_progress * hold_time_threshold:.1f}s / {hold_time_threshold:.1f}s"
            cv.putText(image, time_text, (bar_x + bar_w + 10, bar_y + 11), cv.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv.LINE_AA)

    return image


def draw_info(image, mode, number, times_captured, app_mode="WORDS"):
    mode_text = "الوضع: إشارات عيادة الأسنان" if app_mode == "WORDS" else "الوضع: حروف أبجدية"
    image = draw_text(image, mode_text, "[M/Tab]", (10, 30), font_size=18, color=(255, 215, 0))
    if mode == 1:
        cv.putText(image, "Recording Data...", (10, 58), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv.LINE_AA)
        cv.putText(image, "Label:" + str(number), (10, 78), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv.LINE_AA)
        cv.putText(image, "Captured:" + str(times_captured), (10, 98), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
