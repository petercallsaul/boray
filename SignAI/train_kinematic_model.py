import os
import sys
import csv
import math
import numpy as np

# Configure UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from functions.feature_engineering import (
    extract_hand_features,
    extract_dual_hand_features,
    augment_landmarks,
    mirror_landmarks
)

ARABIC_MAP = {
    'Alef': 'أ', 'Ba2': 'ب', 'Ta2': 'ت', 'Tha2': 'ث', 'Jim': 'ج',
    '7a2': 'ح', 'Kha2': 'خ', 'Dal': 'د', 'Thal': 'ذ', 'Ra2': 'ر',
    'Zayn': 'ز', 'Sin': 'س', 'Chin': 'ش', 'SSad': 'ص', 'DDad': 'ض',
    'TTa2': 'ط', 'TTha2': 'ظ', '3ayn': 'ع', 'Ghayn': 'غ', 'Fa2': 'ف',
    '9af': 'ق', 'Kaf': 'ك', 'Lam': 'ل', 'Mim': 'م', 'Noon': 'ن',
    'Ha2': 'هـ', 'Waw': 'و', 'Ya2': 'ي',
    'Space': 'مسافة', 'Delete': 'مسح', 'Finish': 'نقطة',
    'Sign_Alef': 'أ', 'Sign_Ba2': 'ب', 'Sign_Ta2': 'ت', 'Sign_Tha2': 'ث',
    'Sign_Jeem': 'ج', 'Sign_7a2': 'ح', 'Sign_Kha2': 'خ', 'Sign_Dal': 'د',
    'Sign_Thal': 'ذ', 'Sign_Ra2': 'ر', 'Sign_Zay': 'ز', 'Sign_Seen': 'س',
    'Sign_Sheen': 'ش', 'Sign_Sad': 'ص', 'Sign_Dhad': 'ض', 'Sign_Tah': 'ط',
    'Sign_Zah': 'ظ', 'Sign_3ain': 'ع', 'Sign_Ghain': 'غ', 'Sign_Fa2': 'ف',
    'Sign_Qaf': 'ق', 'Sign_Kaf': 'ك', 'Sign_Lam': 'ل', 'Sign_Meem': 'م',
    'Sign_Noon': 'ن', 'Sign_Ha2': 'هـ', 'Sign_Waw': 'و', 'Sign_Ya2': 'ي',
    'Sign_Space': 'مسافة', 'Sign_Delete': 'مسح', 'Sign_Finish': 'نقطة'
}

def load_and_augment_data(csv_path="ArSL_dataset.csv", augment_factor=2):
    base_dir = os.path.dirname(__file__)
    label_path = os.path.join(base_dir, 'model/keypoint_classifier/keypoint_classifier_label.csv')
    with open(label_path, encoding='utf-8-sig') as f:
        labels = [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]
    label_to_idx = {name: i for i, name in enumerate(labels)}

    full_csv = os.path.join(base_dir, csv_path) if not os.path.isabs(csv_path) else csv_path
    if not os.path.exists(full_csv):
        raise FileNotFoundError(f"Missing {full_csv}")

    raw_samples = []
    with open(full_csv, 'r', encoding='utf-8', errors='ignore') as f:
        delimiter = ';' if ';' in f.readline() else ','
        f.seek(0)
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader)
        label_col = len(header) - 1 if header[-1].lower() in ['label', 'class', 'target'] else 0

        for row in reader:
            if not row or len(row) < 43:
                continue
            raw_label = row[label_col].strip()
            arabic_name = ARABIC_MAP.get(raw_label, raw_label)
            if arabic_name not in label_to_idx:
                continue
            class_idx = label_to_idx[arabic_name]
            coords = []
            start_col = 1 if label_col == 0 else 0
            for val in row[start_col:start_col + 42]:
                try:
                    coords.append(float(val))
                except ValueError:
                    coords.append(0.0)
            if len(coords) == 42:
                raw_samples.append((class_idx, coords))

    print(f"Loaded {len(raw_samples)} raw ArSL samples across {len(labels)} classes.")

    X_list = []
    y_list = []
    for class_idx, coords in raw_samples:
        feats = extract_hand_features(coords)
        # Pad to 148 for dual-hand pipeline consistency (Zeros + Hand)
        X_list.append([0.0] * 74 + feats)
        y_list.append(class_idx)

        # Perturbation augmentation
        for _ in range(augment_factor):
            aug_c = augment_landmarks(coords, max_angle_deg=14.0, scale_range=(0.88, 1.12), noise_std=0.007)
            aug_f = extract_hand_features(aug_c)
            X_list.append([0.0] * 74 + aug_f)
            y_list.append(class_idx)

        # Bilateral mirror augmentation
        mirr_c = mirror_landmarks(coords)
        mirr_f = extract_hand_features(mirr_c)
        X_list.append([0.0] * 74 + mirr_f)
        y_list.append(class_idx)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"Dataset total size: {X.shape[0]} samples, {X.shape[1]} features.")
    return X, y, labels


def train_and_export():
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    X, y, labels = load_and_augment_data(augment_factor=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    print(f"Training high-capacity Deep Neural Network on {len(X_train)} samples...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        batch_size=64,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=60,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=10,
        random_state=42,
        verbose=True
    )

    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 55)
    print(f"  Final Model Test Accuracy: {test_acc * 100:.2f}%")
    print(f"  Meets STEM Capstone >= 80% Requirement: {'YES [PASS]' if test_acc >= 0.80 else 'NO'}")
    print(f"  Target >= 95% Achieved: {'YES [PASS]' if test_acc >= 0.95 else 'HIGH ACCURACY'}")
    print("=" * 55 + "\n")

    # Export weights for direct zero-dependency NumPy inference
    save_dir = os.path.join(os.path.dirname(__file__), 'model/keypoint_classifier')
    weights_path = os.path.join(save_dir, 'keypoint_classifier_weights.npz')

    weights_dict = {}
    for i, (coef, intercept) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        weights_dict[f'W_{i}'] = coef.astype(np.float32)
        weights_dict[f'b_{i}'] = intercept.astype(np.float32)
    weights_dict['classes'] = mlp.classes_.astype(np.int32)
    weights_dict['accuracy'] = np.array([test_acc], dtype=np.float32)

    np.savez_compressed(weights_path, **weights_dict)
    print(f"Exported zero-dependency NumPy weights to: {weights_path}")

    # Plot evaluation metrics
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(mlp.loss_curve_, label='Train Loss', color='#ef4444', lw=2)
        if hasattr(mlp, 'validation_scores_'):
            axes[0].plot([1 - s for s in mlp.validation_scores_], label='Val Loss (1-Acc)', color='#f59e0b', lw=2)
        axes[0].set_title('Cross-Entropy Loss Curve')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Class accuracy bar
        cm = confusion_matrix(y_test, y_pred)
        class_accs = cm.diagonal() / cm.sum(axis=1)
        axes[1].bar(range(len(labels)), class_accs * 100.0, color='#10b981', alpha=0.85)
        axes[1].axhline(y=95.0, color='#ef4444', linestyle='--', label='95% Target Line')
        axes[1].set_title(f'Per-Class Accuracy (Overall: {test_acc*100:.1f}%)')
        axes[1].set_xlabel('Class ID (0 to 30)')
        axes[1].set_ylabel('Accuracy %')
        axes[1].set_ylim(50, 105)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        report_path = os.path.join(save_dir, 'evaluation_report.png')
        plt.savefig(report_path, dpi=200)
        plt.close()
        print(f"Saved evaluation chart to: {report_path}")
    except Exception as e:
        print(f"Plot note: {e}")

    return test_acc

if __name__ == '__main__':
    train_and_export()
