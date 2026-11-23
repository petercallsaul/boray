import unittest
import os
import sys
import numpy as np

# Add parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from model.word_classifier.word_classifier import WordClassifier
from functions.feature_engineering import extract_hand_features
from generate_baseline_words import generate_hand_landmarks

class TestModelInference(unittest.TestCase):
    def setUp(self):
        self.keypoint_clf = KeyPointClassifier()
        self.word_clf = WordClassifier()

    def test_keypoint_classifier_loaded(self):
        self.assertTrue(self.keypoint_clf.has_npz or self.keypoint_clf.interpreter is not None)
        self.assertEqual(self.keypoint_clf.input_dim, 148)
        self.assertEqual(self.keypoint_clf.output_dim, 31)

    def test_word_classifier_loaded(self):
        self.assertTrue(self.word_clf.is_loaded)
        self.assertEqual(self.word_clf.input_dim, 148)
        self.assertEqual(self.word_clf.output_dim, 8)

    def test_word_classification_accuracy_on_canonical_poses(self):
        """Verify that canonical baseline poses classify to their exact class with high confidence (>90%)."""
        for expected_class in range(8):
            pts = generate_hand_landmarks(expected_class)
            coords = pts.flatten().tolist()
            feats = extract_hand_features(coords)
            pred_id, conf = self.word_clf.classify_hands([feats])
            self.assertEqual(pred_id, expected_class, f"Failed on word class {expected_class}")
            self.assertGreater(conf, 0.90, f"Confidence too low for class {expected_class}: {conf}")

    def test_keypoint_single_hand_inference(self):
        pts = generate_hand_landmarks(0)
        feats = extract_hand_features(pts.flatten().tolist())
        pred_id, conf = self.keypoint_clf.classify_hands([feats])
        self.assertIsInstance(pred_id, int)
        self.assertGreaterEqual(pred_id, 0)
        self.assertLess(pred_id, 31)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_keypoint_two_hands_inference(self):
        pts1 = generate_hand_landmarks(0)
        pts2 = generate_hand_landmarks(1)
        f1 = extract_hand_features(pts1.flatten().tolist())
        f2 = extract_hand_features(pts2.flatten().tolist())
        pred_id, conf = self.keypoint_clf.classify_hands([f1, f2])
        self.assertIsInstance(pred_id, int)
        self.assertGreaterEqual(pred_id, 0)
        self.assertLess(pred_id, 31)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

if __name__ == '__main__':
    unittest.main()
