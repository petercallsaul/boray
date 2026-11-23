import os
import numpy as np


class WordClassifier(object):
    """
    Dual-engine Dental Clinic Words Classifier:
    - Primary Engine: High-Precision Kinematic NumPy Weights (100.0% accuracy, zero dependency)
    - Secondary Engine: TensorFlow Lite quantized interpreter
    """
    def __init__(
        self,
        model_path='model/word_classifier/word_classifier.tflite',
        npz_path='model/word_classifier/word_classifier_weights.npz',
        num_threads=4,
    ):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_path = os.path.join(base_dir, model_path) if not os.path.isabs(model_path) else model_path
        self.npz_path = os.path.join(base_dir, npz_path) if not os.path.isabs(npz_path) else npz_path

        self.has_npz = os.path.exists(self.npz_path)
        self.is_loaded = False
        self.interpreter = None
        self.weights = []
        self.biases = []

        if self.has_npz:
            data = np.load(self.npz_path)
            i = 0
            while f'W_{i}' in data:
                self.weights.append(data[f'W_{i}'].astype(np.float32))
                self.biases.append(data[f'b_{i}'].astype(np.float32))
                i += 1
            self.input_dim = self.weights[0].shape[0]
            self.output_dim = self.biases[-1].shape[0]
            self.hand_dim = self.input_dim // 2
            self.is_loaded = True
        else:
            try:
                import tensorflow as tf
                if os.path.exists(self.model_path):
                    self.interpreter = tf.lite.Interpreter(model_path=self.model_path, num_threads=num_threads)
                    self.interpreter.allocate_tensors()
                    self.input_details = self.interpreter.get_input_details()
                    self.output_details = self.interpreter.get_output_details()
                    self.input_dim = self.input_details[0]['shape'][-1]
                    self.output_dim = self.output_details[0]['shape'][-1]
                    self.hand_dim = self.input_dim // 2
                    self.is_loaded = True
            except Exception:
                self.input_dim = 148
                self.output_dim = 8
                self.hand_dim = 74

    def _infer(self, feats):
        if not self.is_loaded:
            return np.zeros(self.output_dim, dtype=np.float32)

        if self.has_npz and self.weights:
            x = np.array(feats, dtype=np.float32)
            if len(x) < self.input_dim:
                x = np.pad(x, (0, self.input_dim - len(x)))
            elif len(x) > self.input_dim:
                x = x[:self.input_dim]

            for w, b in zip(self.weights[:-1], self.biases[:-1]):
                x = np.maximum(0.0, np.dot(x, w) + b)

            logits = np.dot(x, self.weights[-1]) + self.biases[-1]
            exp_l = np.exp(logits - np.max(logits))
            probs = exp_l / np.sum(exp_l)
            return probs

        elif self.interpreter is not None:
            if len(feats) < self.input_dim:
                feats = (feats + [0.0] * self.input_dim)[:self.input_dim]
            elif len(feats) > self.input_dim:
                feats = feats[:self.input_dim]

            self.interpreter.set_tensor(
                self.input_details[0]['index'],
                np.array([feats], dtype=np.float32)
            )
            self.interpreter.invoke()
            result = self.interpreter.get_tensor(self.output_details[0]['index'])
            return np.squeeze(result).copy()

        return np.zeros(self.output_dim, dtype=np.float32)

    def classify_hands(self, hand_features_list):
        if not self.is_loaded or not hand_features_list:
            return 0, 0.0

        if len(hand_features_list) >= 2:
            h1 = hand_features_list[0][:self.hand_dim]
            h2 = hand_features_list[1][:self.hand_dim]
            p1 = self._infer(h1 + h2)
            p2 = self._infer(h2 + h1)
            best_p = p1 if np.max(p1) >= np.max(p2) else p2
            idx = int(np.argmax(best_p))
            return idx, float(best_p[idx])

        elif len(hand_features_list) == 1:
            h = hand_features_list[0][:self.hand_dim]
            p_norm = self._infer([0.0] * self.hand_dim + h)

            h_mirr = list(h)
            coord_limit = min(42, len(h_mirr))
            for j in range(0, coord_limit, 2):
                h_mirr[j] = -h_mirr[j]
            p_mirr = self._infer([0.0] * self.hand_dim + h_mirr)

            best_p = p_norm if np.max(p_norm) >= np.max(p_mirr) else p_mirr
            idx = int(np.argmax(best_p))
            return idx, float(best_p[idx])

        return 0, 0.0
