import os
import sys
import csv
import math
import numpy as np

# Configure console for UTF-8
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
from generate_baseline_words import generate_hand_landmarks

DEFAULT_WORDS = [
    'أشعر بألم',
    'انا تمام',
    'قف',
    'ألم خفيف',
    'ألم قوي',
    'اريد أن أحك وجهي',
    'غسل الفم',
    'اريد أن اتحدث'
]

def generate_augmented_word_dataset(samples_per_class=350):
    X_list = []
    y_list = []

    for class_id in range(len(DEFAULT_WORDS)):
        base_pts = generate_hand_landmarks(class_id)
        base_coords = base_pts.flatten().tolist()
        base_feats = extract_hand_features(base_coords)
        X_list.append([0.0] * 74 + base_feats)
        y_list.append(class_id)

        for _ in range(samples_per_class):
            jittered_pts = base_pts.copy()
            jitter = np.random.normal(0, 0.014, size=jittered_pts.shape).astype(np.float32)
            jitter[0] = 0.0
            if class_id == 3: # mild pain (tight pinch)
                jitter[4] *= 0.4
                jitter[8] *= 0.4
            elif class_id == 4: # severe pain (wide C)
                jitter[4] *= 0.5
                jitter[8] *= 0.5

            varied_pts = jittered_pts + jitter
            varied_coords = varied_pts.flatten().tolist()

            aug_coords = augment_landmarks(
                varied_coords,
                max_angle_deg=12.0,
                scale_range=(0.88, 1.12),
                noise_std=0.005
            )
            aug_feats = extract_hand_features(aug_coords)
            X_list.append([0.0] * 74 + aug_feats)
            y_list.append(class_id)

            # Bilateral mirroring
            mirr_coords = mirror_landmarks(aug_coords)
            mirr_feats = extract_hand_features(mirr_coords)
            X_list.append([0.0] * 74 + mirr_feats)
            y_list.append(class_id)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"Generated Dental Words Dataset: {X.shape[0]} samples, {len(DEFAULT_WORDS)} classes.")
    return X, y, DEFAULT_WORDS


def train_and_export():
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X, y, labels = generate_augmented_word_dataset(samples_per_class=350)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        batch_size=32,
        learning_rate_init=0.001,
        max_iter=50,
        early_stopping=True,
        validation_fraction=0.10,
        random_state=42,
        verbose=True
    )

    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 55)
    print(f"  Dental Words Model Accuracy: {test_acc * 100:.2f}%")
    print(f"  Target >= 95% Achieved: {'YES [PASS]' if test_acc >= 0.95 else 'NO'}")
    print("=" * 55 + "\n")

    save_dir = os.path.join(os.path.dirname(__file__), 'model/word_classifier')
    os.makedirs(save_dir, exist_ok=True)
    weights_path = os.path.join(save_dir, 'word_classifier_weights.npz')

    weights_dict = {}
    for i, (coef, intercept) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        weights_dict[f'W_{i}'] = coef.astype(np.float32)
        weights_dict[f'b_{i}'] = intercept.astype(np.float32)
    weights_dict['classes'] = mlp.classes_.astype(np.int32)
    weights_dict['accuracy'] = np.array([test_acc], dtype=np.float32)

    np.savez_compressed(weights_path, **weights_dict)
    print(f"Exported zero-dependency NumPy weights to: {weights_path}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(mlp.loss_curve_, label='Train Loss', color='#ef4444', lw=2)
        axes[0].set_title('Dental Signs Loss')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        class_accs = cm.diagonal() / cm.sum(axis=1)
        axes[1].bar(range(len(labels)), class_accs * 100.0, color='#10b981', alpha=0.85)
        axes[1].set_title(f'Dental Signs Accuracy ({test_acc*100:.1f}%)')
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Accuracy %')
        axes[1].set_ylim(80, 105)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        eval_path = os.path.join(save_dir, 'word_evaluation_report.png')
        plt.savefig(eval_path, dpi=200)
        plt.close()
        print(f"Saved evaluation chart to: {eval_path}")
    except Exception as e:
        print(f"Plot note: {e}")

    return test_acc

if __name__ == '__main__':
    train_and_export()
