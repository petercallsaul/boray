import os
import sys
import csv
import numpy as np
import tensorflow as tf

# Configure console for UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

dataset_path = 'model/word_classifier/word_keypoint.csv'
label_path = 'model/word_classifier/word_classifier_label.csv'
model_save_path = 'model/word_classifier/word_classifier.hdf5'
tflite_save_path = 'model/word_classifier/word_classifier.tflite'
eval_plot_path = 'model/word_classifier/word_evaluation_report.png'

if not os.path.exists(dataset_path):
    print(f"Error: {dataset_path} not found. Please record samples with collect_word_data.py first.")
    exit(1)

with open(label_path, encoding='utf-8-sig') as f:
    labels = [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]
num_classes = len(labels)
print(f"Loaded {num_classes} medical word classes: {', '.join(labels)}")

# Load dataset
X = np.loadtxt(dataset_path, delimiter=',', dtype='float32', usecols=list(range(1, 85)))
y = np.loadtxt(dataset_path, delimiter=',', dtype='int32', usecols=(0,))

if len(X.shape) == 1:
    X = np.array([X])
    y = np.array([y])

total_samples = len(X)
print(f"Total recorded word samples: {total_samples}")

if total_samples < 14:
    print("Warning: Dataset is very small. Record at least 20-30 samples per word for high accuracy.")

# Train / Validation split
np.random.seed(42)
indices = np.arange(total_samples)
np.random.shuffle(indices)

split_idx = max(int(0.80 * total_samples), total_samples - num_classes)
X_train, X_test = X[indices[:split_idx]], X[indices[split_idx:]]
y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]

# Build Neural Network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Input((84,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=20, verbose=1, restore_best_weights=True)
]

print("\nTraining Medical Words Neural Network...")
epochs = 80 if total_samples >= 100 else 40
history = model.fit(
    X_train, y_train,
    batch_size=min(32, max(4, len(X_train))),
    epochs=epochs,
    validation_data=(X_test, y_test) if len(X_test) > 0 else None,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
if len(X_test) > 0:
    val_loss, val_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n==========================================")
    print(f"Medical Word Model Test Accuracy: {val_acc * 100:.2f}%")
    print(f"Meets STEM Capstone >= 80% Requirement: {'YES (PASS)' if val_acc >= 0.80 else 'NO'}")
    print(f"==========================================\n")

# Save HDF5
model.save(model_save_path)

# Export quantized TFLite
print("Exporting optimized TensorFlow Lite model...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(tflite_save_path, 'wb') as f:
    f.write(tflite_model)
print(f"Saved optimized model to: {tflite_save_path}")

# Plot learning curves
try:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['accuracy'], label='Train Acc', color='#2ecc71', lw=2)
    if 'val_accuracy' in history.history:
        axes[0].plot(history.history['val_accuracy'], label='Val Acc', color='#3498db', lw=2)
    axes[0].set_title('Medical Words Model - Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(history.history['loss'], label='Train Loss', color='#e74c3c', lw=2)
    if 'val_loss' in history.history:
        axes[1].plot(history.history['val_loss'], label='Val Loss', color='#e67e22', lw=2)
    axes[1].set_title('Medical Words Model - Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(eval_plot_path, dpi=200)
    plt.close()
    print(f"Saved evaluation chart to: {eval_plot_path}")
except Exception as e:
    print(f"Note on plotting: {e}")
