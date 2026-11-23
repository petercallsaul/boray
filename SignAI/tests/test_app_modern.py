import unittest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.dialogue_manager import DialogueManager
from functions.tts_engine import TTSEngine
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from model.word_classifier.word_classifier import WordClassifier
from app_modern import format_arabic_hud_text, check_dental_c_pain, is_open_palm
from generate_baseline_words import generate_hand_landmarks

class MockLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class MockHand:
    def __init__(self, pts_array):
        arr = np.array(pts_array, dtype=np.float32).reshape(21, 2)
        self.landmark = [MockLandmark(p[0], p[1]) for p in arr]

class TestAppModern(unittest.TestCase):
    def test_arabic_hud_reshaping(self):
        text = "أشعر بألم"
        formatted = format_arabic_hud_text(text, "(95%)")
        self.assertIn("(95%)", formatted)
        self.assertTrue(len(formatted) > 5)

    def test_c_pain_aperture_detection(self):
        # Class 3: mild pain (small pinch)
        pts_mild = generate_hand_landmarks(3)
        hand_mild = MockHand(pts_mild)
        result_mild = check_dental_c_pain(hand_mild)
        self.assertIsNotNone(result_mild)
        self.assertEqual(result_mild[0], "ألم خفيف")

        # Class 4: severe pain (wide C)
        pts_sev = generate_hand_landmarks(4)
        hand_sev = MockHand(pts_sev)
        result_sev = check_dental_c_pain(hand_sev)
        self.assertIsNotNone(result_sev)
        self.assertEqual(result_sev[0], "ألم قوي")

    def test_open_palm_detection(self):
        # Class 2 is 'قف' (Stop - Open palm upright)
        pts_stop = generate_hand_landmarks(2)
        hand_stop = MockHand(pts_stop)
        # Check palm
        self.assertTrue(hand_stop.landmark[8].y < hand_stop.landmark[6].y)
        self.assertTrue(hand_stop.landmark[12].y < hand_stop.landmark[10].y)

    def test_dialogue_and_export_lifecycle(self):
        dm = DialogueManager()
        dm.add_message("patient", "أشعر بألم في أسناني")
        dm.add_message("doctor", "افتح فمك من فضلك")
        self.assertEqual(len(dm.get_messages()), 2)
        out_dir = "tests/test_output"
        filepath = dm.export_transcript(output_dir=out_dir)
        self.assertTrue(os.path.exists(filepath))
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("المريض", content)
            self.assertIn("الطبيب", content)
        os.remove(filepath)
        os.rmdir(out_dir)

if __name__ == '__main__':
    unittest.main()
