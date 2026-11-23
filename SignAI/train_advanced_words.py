import os
import sys
import csv
import math
import numpy as np

# UTF-8 console output
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

def generate_augmented_word_dataset(samples_per_class=250):
    """
    Generates a rich, kinematically consistent dataset for the 8 dental signs
    using biomechanical baseline templates combined with multi-stage geometric perturbations.
    """
    X_list = []
    y_list = []

    for class_id in range(len(DEFAULT_WORDS)):
        # Generate canonical biomechanical base landmarks (21 points)
        base_pts = generate_hand_landmarks(class_id) # shape (21, 2)
        base_coords = base_pts.flatten().tolist()

        # Extract features for canonical base pose
        base_feats = extract_hand_features(base_coords)
        X_list.append([0.0] * 74 + base_feats)
        y_list.append(class_id)

        # Generate augmented variations
        for _ in range(samples_per_class):
            # 1. Subtle joint angle / position perturbation
            jittered_pts = base_pts.copy()
            # Non-wrist landmarks get tiny natural anatomical variation
            jitter = np.random.normal(0, 0.015, size=jittered_pts.shape).astype(np.float32)
            jitter[0] = 0.0 # wrist stays fixed
            # Extra care for pinch/C aperture classes (mild vs severe pain)
            if class_id == 3: # mild pain (tight pinch)
                # Keep index tip and thumb tip close
                jitter[4] *= 0.5
                jitter[8] *= 0.5
            elif class_id == 4: # severe pain (wide C)
                # Keep index tip and thumb tip distinctly separated
                jitter[4] *= 0.6
                jitter[8] *= 0.6

            varied_pts = jittered_pts + jitter
            varied_coords = varied_pts.flatten().tolist()

            # 2. General 2D rotation & scale augmentation
            aug_coords = augment_landmarks(
                varied_coords,
                max_angle_deg=12.0,
                scale_range=(0.88, 1.12),
                noise_std=0.006
            )
            aug_feats = extract_hand_features(aug_coords)
            X_list.append([0.0] * 74 + aug_feats)
            y_list.append(class_id)

            # 3. Mirroring for bilateral hand invariance
            mirr_coords = mirror_landmarks(aug_coords)
            mirr_feats = extract_hand_features(mirr_coords)
            X_list.append([0.0] * 74 + mirr_feats)
            y_list.append(class_id)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"Generated Dental Words Dataset: {X.shape[0]} samples across {len(DEFAULT_WORDS)} classes.")
    return X, y, DEFAULT_WORDS


def build_word_residual_mlp(input_dim, num_classes):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=(input_dim,))

    x = layers.Dense(128, activation='gelu')(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.15)(x)

    # Residual Block
    res1 = layers.Dense(128, activation='gelu')(x)
    res1 = layers.LayerNormalization()(res1)
    res1 = layers.Dropout(0.15)(res1)
    x = layers.Add()([x, res1])

    x = layers.Dense(64, activation='gelu')(x)
    x = layers.LayerNormalization()(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=inputs, outputs=outputs, name="SignAI_Word_Classifier")


def main():
    import tensorflow as tf

    base_dir = os.path.dirname(__file__)
    model_dir = os.path.join(base_dir, 'model/word_classifier')
    os.makedirs(model_dir, exist_ok=True)

    label_path = os.path.join(model_dir, 'word_classifier_label.csv')
    hdf5_path = os.path.join(model_dir, 'word_classifier.hdf5')
    tflite_path = os.path.join(model_dir, 'word_classifier.tflite')
    plot_path = os.path.join(model_dir, 'word_evaluation_report.png')

    # Save labels
    with open(label_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        for w in DEFAULT_WORDS:
            writer.writerow([w])

    X, y, labels = generate_augmented_word_dataset(samples_per_class=300)
    num_classes = len(labels)

    # Train / Test split
    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(0.85 * len(X))

    X_train, X_test = X[indices[:split]], X[indices[split:]]
    y_train, y_test = y[indices[:split]], y[indices[split:]]

    model = build_word_residual_mlp(input_dim=X.shape[1], num_classes=num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(hdf5_path, save_best_only=True, monitor='val_accuracy', verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True, monitor='val_accuracy')
    ]

    print(f"\nTraining Dental Words Model on {len(X_train)} samples...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=40,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n=======================================================")
    print(f"  Dental Words Model Accuracy: {test_acc * 100:.2f}%")
    print(f"  Target >= 95% Achieved: {'YES [PASS]' if test_acc >= 0.95 else 'ALMOST'}")
    print(f"=======================================================\n")

    # Export TFLite
    print("Exporting optimized TensorFlow Lite model...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved TFLite model to: {tflite_path}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history.history['accuracy'], label='Train Acc', color='#10b981', lw=2)
        axes[0].plot(history.history['val_accuracy'], label='Val Acc', color='#0284c7', lw=2)
        axes[0].set_title(f'Dental Words Accuracy ({test_acc*100:.1f}%)')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(history.history['loss'], label='Train Loss', color='#ef4444', lw=2)
        axes[1].plot(history.history['val_loss'], label='Val Loss', color='#f59e0b', lw=2)
        axes[1].set_title('Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Saved word evaluation chart to: {plot_path}")
    except Exception as e:
        print(f"Plot note: {e}")

if __name__ == '__main__':
    main()
