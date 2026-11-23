import unittest
import numpy as np
import sys
import os

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.feature_engineering import (
    compute_angle,
    extract_hand_features,
    extract_dual_hand_features,
    augment_landmarks,
    mirror_landmarks
)

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        # Generate synthetic canonical hand coordinates (21 points)
        self.canonical_hand = np.zeros((21, 2), dtype=np.float32)
        # Wrist at (0.5, 0.8)
        self.canonical_hand[0] = [0.5, 0.8]
        # MCPs
        self.canonical_hand[2] = [0.45, 0.70] # thumb
        self.canonical_hand[5] = [0.48, 0.60] # index
        self.canonical_hand[9] = [0.50, 0.55] # middle
        self.canonical_hand[13] = [0.52, 0.60] # ring
        self.canonical_hand[17] = [0.55, 0.65] # pinky
        # Tips
        self.canonical_hand[4] = [0.40, 0.60] # thumb tip
        self.canonical_hand[8] = [0.47, 0.40] # index tip
        self.canonical_hand[12] = [0.50, 0.38] # middle tip
        self.canonical_hand[16] = [0.53, 0.42] # ring tip
        self.canonical_hand[20] = [0.56, 0.48] # pinky tip

    def test_compute_angle_right_angle(self):
        a = [0.0, 1.0]
        b = [0.0, 0.0]
        c = [1.0, 0.0]
        cos_theta = compute_angle(a, b, c)
        self.assertAlmostEqual(cos_theta, 0.0, places=5)

    def test_compute_angle_straight_line(self):
        a = [-1.0, 0.0]
        b = [0.0, 0.0]
        c = [1.0, 0.0]
        cos_theta = compute_angle(a, b, c)
        self.assertAlmostEqual(cos_theta, -1.0, places=5)

    def test_compute_angle_collinear(self):
        a = [1.0, 0.0]
        b = [0.0, 0.0]
        c = [2.0, 0.0]
        cos_theta = compute_angle(a, b, c)
        self.assertAlmostEqual(cos_theta, 1.0, places=5)

    def test_extract_features_length(self):
        feats = extract_hand_features(self.canonical_hand)
        self.assertEqual(len(feats), 74)

    def test_translation_invariance(self):
        feats1 = extract_hand_features(self.canonical_hand)
        shifted_hand = self.canonical_hand + np.array([0.35, -0.42], dtype=np.float32)
        feats2 = extract_hand_features(shifted_hand)
        np.testing.assert_allclose(feats1, feats2, atol=1e-5)

    def test_scale_invariance(self):
        feats1 = extract_hand_features(self.canonical_hand)
        scaled_hand = (self.canonical_hand - self.canonical_hand[0]) * 2.5 + self.canonical_hand[0]
        feats2 = extract_hand_features(scaled_hand)
        np.testing.assert_allclose(feats1, feats2, atol=1e-5)

    def test_dual_hand_features(self):
        feats = extract_hand_features(self.canonical_hand)
        # 1 hand case
        dual_1 = extract_dual_hand_features([feats])
        self.assertEqual(len(dual_1), 148)
        self.assertEqual(dual_1[:74], [0.0] * 74)
        self.assertEqual(dual_1[74:], feats)

        # 2 hands case
        dual_2 = extract_dual_hand_features([feats, feats])
        self.assertEqual(len(dual_2), 148)
        self.assertEqual(dual_2[:74], feats)
        self.assertEqual(dual_2[74:], feats)

    def test_augmentation_shape(self):
        aug = augment_landmarks(self.canonical_hand)
        self.assertEqual(len(aug), 42)

    def test_mirror_landmarks(self):
        mirrored = mirror_landmarks(self.canonical_hand)
        self.assertEqual(len(mirrored), 42)
        # Wrist x should be identical
        self.assertAlmostEqual(mirrored[0], self.canonical_hand[0, 0], places=5)

if __name__ == '__main__':
    unittest.main()
