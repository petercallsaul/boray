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

def load_and_prepare_arsl_data(csv_path="ArSL_dataset.csv", augment_factor=2):
    """
    Loads the Arabic Sign Language dataset, computes 74 kinematic features per sample,
    and applies rotational, scale, and noise augmentations.
    """
    if not os.path.exists(csv_path):
        # Check subfolder
        alt_path = os.path.join(os.path.dirname(__file__), "ArSL_dataset.csv")
        if os.path.exists(alt_path):
            csv_path = alt_path
        else:
            raise FileNotFoundError(f"Cannot find {csv_path}")

    # Read label list
    label_path = os.path.join(os.path.dirname(__file__), 'model/keypoint_classifier/keypoint_classifier_label.csv')
    with open(label_path, encoding='utf-8-sig') as f:
        labels = [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]
    label_to_idx = {name: i for i, name in enumerate(labels)}

    raw_samples = []
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        delimiter = ';' if ';' in f.readline() else ','
        f.seek(0)
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader)

        label_col = 0
        if header[-1].lower() in ['label', 'class', 'target']:
            label_col = len(header) - 1

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

    print(f"Loaded {len(raw_samples)} raw samples across {len(labels)} classes.")

    X_list = []
    y_list = []

    for class_idx, coords in raw_samples:
        # Original features (74)
        feats = extract_hand_features(coords)
        # Pad to 148 for dual-hand pipeline consistency (Zeros + Hand)
        X_list.append([0.0] * 74 + feats)
        y_list.append(class_idx)

        # Augmentation
        for _ in range(augment_factor):
            aug_coords = augment_landmarks(coords, max_angle_deg=14.0, scale_range=(0.88, 1.12), noise_std=0.007)
            aug_feats = extract_hand_features(aug_coords)
            X_list.append([0.0] * 74 + aug_feats)
            y_list.append(class_idx)

        # Mirroring for bilateral hand invariance
        mirr_coords = mirror_landmarks(coords)
        mirr_feats = extract_hand_features(mirr_coords)
        X_list.append([0.0] * 74 + mirr_feats)
        y_list.append(class_idx)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"Dataset after kinematic feature extraction & augmentation: {X.shape[0]} samples, {X.shape[1]} features.")
    return X, y, labels


def build_residual_mlp(input_dim, num_classes):
    """
    Builds a modern Deep Residual MLP with LayerNorm, GELU, and Skip Connections.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=(input_dim,))

    # Projection stem
    x = layers.Dense(256, activation='gelu')(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # Residual Block 1
    res1 = layers.Dense(256, activation='gelu')(x)
    res1 = layers.LayerNormalization()(res1)
    res1 = layers.Dropout(0.2)(res1)
    x = layers.Add()([x, res1])

    # Residual Block 2 (with dimension reduction projection)
    res2 = layers.Dense(128, activation='gelu')(x)
    res2 = layers.LayerNormalization()(res2)
    res2 = layers.Dropout(0.15)(res2)
    proj2 = layers.Dense(128)(x)
    x = layers.Add()([proj2, res2])

    # Dense refinement block
    x = layers.Dense(64, activation='gelu')(x)
    x = layers.LayerNormalization()(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="SignAI_Residual_MLP")
    return model


def main():
    import tensorflow as tf

    base_dir = os.path.dirname(__file__)
    model_save_dir = os.path.join(base_dir, 'model/keypoint_classifier')
    os.makedirs(model_save_dir, exist_ok=True)

    hdf5_path = os.path.join(model_save_dir, 'keypoint_classifier.hdf5')
    tflite_path = os.path.join(model_save_dir, 'keypoint_classifier.tflite')
    plot_path = os.path.join(model_save_dir, 'evaluation_report.png')

    X, y, labels = load_and_prepare_arsl_data(augment_factor=2)
    num_classes = len(labels)

    # Train / Test split (85% train, 15% validation)
    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(0.85 * len(X))

    X_train, X_test = X[indices[:split]], X[indices[split:]]
    y_train, y_test = y[indices[:split]], y[indices[split:]]

    model = build_residual_mlp(input_dim=X.shape[1], num_classes=num_classes)
    model.summary()

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0015,
        decay_steps=60 * (len(X_train) // 64),
        alpha=0.05
    )

    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(hdf5_path, save_best_only=True, monitor='val_accuracy', verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=14, restore_best_weights=True, monitor='val_accuracy', verbose=1)
    ]

    print(f"\nTraining Advanced Residual MLP on {len(X_train)} samples...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluation
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n=======================================================")
    print(f"  Final Model Test Accuracy: {test_acc * 100:.2f}%")
    print(f"  Target >= 95% Achieved: {'YES [PASS]' if test_acc >= 0.95 else 'ALMOST'}")
    print(f"=======================================================\n")

    # Save quantized TFLite
    print("Exporting optimized TensorFlow Lite model...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Successfully saved TFLite model to: {tflite_path}")

    # Plot metrics
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(history.history['accuracy'], label='Train Acc', color='#10b981', lw=2)
        axes[0].plot(history.history['val_accuracy'], label='Val Acc', color='#0284c7', lw=2)
        axes[0].set_title(f'Alphabet Model Accuracy (Final: {test_acc*100:.1f}%)')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(history.history['loss'], label='Train Loss', color='#ef4444', lw=2)
        axes[1].plot(history.history['val_loss'], label='Val Loss', color='#f59e0b', lw=2)
        axes[1].set_title('Cross-Entropy Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Saved evaluation chart to: {plot_path}")
    except Exception as e:
        print(f"Plot note: {e}")

if __name__ == '__main__':
    main()
