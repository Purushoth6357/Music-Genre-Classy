"""
model.py — Improved CNN Architecture for Music Genre Classification

Architecture: 5x [Conv2D → BatchNorm → ReLU → Dropout → MaxPool] → GAP → Dense → Softmax
Pipeline: Audio → Chunk → Mel Spectrogram → CNN → Genre

Improvements over v1:
  - 5 conv blocks (32→64→128→256→256) for richer feature extraction
  - L2 regularization on conv layers to reduce overfitting
  - Dropout after every block (not just before softmax)
  - Dual dense layers (256→128) for better classification head
  - Still lightweight: ~1.5M parameters
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.layers import GlobalAveragePooling2D


def build_model(inputShape=(128, 128, 1), numClasses=10):
    """
    Build an improved CNN model for music genre classification.

    Args:
        inputShape: Shape of input spectrogram (height, width, channels)
        numClasses: Number of genre classes (default 10 for GTZAN)

    Returns:
        Compiled Keras model
    """
    weightDecay = 1e-4

    model = models.Sequential([
        # --- Block 1: 32 filters ---
        layers.Conv2D(32, (3, 3), padding='same', input_shape=inputShape,
                      kernel_regularizer=regularizers.l2(weightDecay)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # --- Block 2: 64 filters ---
        layers.Conv2D(64, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(weightDecay)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # --- Block 3: 128 filters ---
        layers.Conv2D(128, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(weightDecay)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # --- Block 4: 256 filters ---
        layers.Conv2D(256, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(weightDecay)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # --- Block 5: 256 filters ---
        layers.Conv2D(256, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(weightDecay)),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        # --- Global Average Pooling ---
        GlobalAveragePooling2D(),

        # --- Classification Head ---
        layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(weightDecay)),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(weightDecay)),
        layers.Dropout(0.4),
        layers.Dense(numClasses, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == '__main__':
    # Quick test: build model and print summary
    model = build_model()
    model.summary()
