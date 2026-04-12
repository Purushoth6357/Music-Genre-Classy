"""
train.py — Improved Training Pipeline for Music Genre Classification

Pipeline: Audio -> Chunk -> Augment -> Mel Spectrogram -> CNN -> Genre

Improvements over v1:
  - Data augmentation (time-stretch, pitch-shift, noise injection)
  - ReduceLROnPlateau for adaptive learning rate
  - Better mel spectrogram params (n_fft, hop_length)
  - 3s chunks with 1.5s overlap (more training samples)
  - Higher patience (8) with 50 max epochs
  - 80/20 train/test split for more training data
  - Properly handles split BEFORE chunking (no data leakage)
"""

import os
import random
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from model import build_model


# --- Configuration -----------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), 'Data', 'genres_original')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'music_genre_model.keras')
NORM_STATS_PATH = os.path.join(MODEL_DIR, 'norm_stats.npz')
HISTORY_PLOT_PATH = os.path.join(MODEL_DIR, 'training_history.png')

SAMPLE_RATE = 22050
CHUNK_DURATION = 4        # seconds per chunk
OVERLAP_DURATION = 2    # seconds overlap between chunks
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TARGET_SHAPE = (128, 128)
TEST_SIZE = 0.3
RANDOM_STATE = 42
EPOCHS = 30
BATCH_SIZE = 32

# Data augmentation settings
AUGMENT_FACTOR = 2        # Number of augmented copies per chunk


# --- Genre Labels ------------------------------------------------------------
GENRES = [
    'blues', 'classical', 'country', 'disco', 'hiphop',
    'jazz', 'metal', 'pop', 'reggae', 'rock'
]


# --- Data Augmentation -------------------------------------------------------
def augment_audio(audioData, sampleRate):
    """
    Apply random augmentations to audio data for training.

    Augmentations (randomly applied):
      - Time stretch (0.85x to 1.15x speed)
      - Pitch shift (-2 to +2 semitones)
      - Additive Gaussian noise
    """
    augmented = audioData.copy()

    # Time stretch (random rate between 0.85 and 1.15)
    if random.random() < 0.5:
        rate = random.uniform(0.85, 1.15)
        augmented = librosa.effects.time_stretch(augmented, rate=rate)
        # Ensure same length
        if len(augmented) > len(audioData):
            augmented = augmented[:len(audioData)]
        elif len(augmented) < len(audioData):
            augmented = np.pad(augmented, (0, len(audioData) - len(augmented)), mode='constant')

    # Pitch shift (-2 to +2 semitones)
    if random.random() < 0.5:
        nSteps = random.uniform(-2, 2)
        augmented = librosa.effects.pitch_shift(augmented, sr=sampleRate, n_steps=nSteps)

    # Add Gaussian noise
    if random.random() < 0.5:
        noise = np.random.normal(0, 0.005, len(augmented))
        augmented = augmented + noise

    return augmented


def extract_mel_spectrogram(audioData, sampleRate, n_mels=N_MELS, targetShape=TARGET_SHAPE):
    """
    Extract a mel spectrogram from audio data and resize to target shape.

    Uses improved FFT parameters for richer frequency resolution.
    """
    spec = librosa.feature.melspectrogram(
        y=audioData,
        sr=sampleRate,
        n_mels=n_mels,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    specDb = librosa.power_to_db(spec, ref=np.max)

    # Resize to target shape using interpolation
    from PIL import Image
    specImage = Image.fromarray(specDb)
    specImage = specImage.resize((targetShape[1], targetShape[0]), Image.BILINEAR)
    specResized = np.array(specImage)

    return specResized


def process_audio_file(filePath, sampleRate=SAMPLE_RATE, augment=False):
    """
    Load audio file and split into overlapping chunks, returning mel spectrograms.
    Optionally applies data augmentation for training.
    """
    try:
        audioData, sr = librosa.load(filePath, sr=sampleRate)
    except Exception as e:
        print(f"  [!] Error loading {filePath}: {e}")
        return []

    chunkSamples = int(CHUNK_DURATION * sampleRate)
    overlapSamples = int(OVERLAP_DURATION * sampleRate)
    stepSamples = chunkSamples - overlapSamples

    spectrograms = []
    numChunks = max(1, (len(audioData) - chunkSamples) // stepSamples + 1)

    for i in range(numChunks):
        startSample = i * stepSamples
        endSample = startSample + chunkSamples
        chunk = audioData[startSample:endSample]

        # Pad if chunk is shorter than expected
        if len(chunk) < chunkSamples:
            chunk = np.pad(chunk, (0, chunkSamples - len(chunk)), mode='constant')

        # Original spectrogram
        spec = extract_mel_spectrogram(chunk, sampleRate)
        spectrograms.append(spec)

        # Augmented copies (training only)
        if augment:
            for _ in range(AUGMENT_FACTOR):
                augChunk = augment_audio(chunk, sampleRate)
                augSpec = extract_mel_spectrogram(augChunk, sampleRate)
                spectrograms.append(augSpec)

    return spectrograms


def load_dataset():
    """
    Load all audio files and split into train/test sets BEFORE chunking
    to prevent data leakage.
    """
    print("=" * 60)
    print("  LOADING DATASET")
    print("=" * 60)

    allFiles = []
    allLabels = []

    for genreIdx, genre in enumerate(GENRES):
        genreDir = os.path.join(DATA_DIR, genre)
        if not os.path.isdir(genreDir):
            print(f"  [!] Genre folder not found: {genreDir}")
            continue

        files = [
            os.path.join(genreDir, f)
            for f in sorted(os.listdir(genreDir))
            if f.endswith('.wav')
        ]
        print(f"  [OK] {genre:<12} -> {len(files)} files")

        allFiles.extend(files)
        allLabels.extend([genreIdx] * len(files))

    print(f"\n  Total files: {len(allFiles)}")

    # -- Split BEFORE chunking (prevents data leakage) --
    trainFiles, testFiles, trainLabels, testLabels = train_test_split(
        allFiles, allLabels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=allLabels
    )

    print(f"  Train files: {len(trainFiles)}")
    print(f"  Test files:  {len(testFiles)}")

    # -- Process train files (WITH augmentation) --
    print("\n  Processing training data (with augmentation)...")
    X_train = []
    y_train = []
    for i, (filePath, label) in enumerate(zip(trainFiles, trainLabels)):
        specs = process_audio_file(filePath, augment=True)
        X_train.extend(specs)
        y_train.extend([label] * len(specs))
        if (i + 1) % 50 == 0:
            print(f"    Processed {i + 1}/{len(trainFiles)} train files...")

    # -- Process test files (NO augmentation) --
    print("  Processing test data...")
    X_test = []
    y_test = []
    for i, (filePath, label) in enumerate(zip(testFiles, testLabels)):
        specs = process_audio_file(filePath, augment=False)
        X_test.extend(specs)
        y_test.extend([label] * len(specs))
        if (i + 1) % 50 == 0:
            print(f"    Processed {i + 1}/{len(testFiles)} test files...")

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print(f"\n  X_train shape: {X_train.shape}")
    print(f"  X_test shape:  {X_test.shape}")

    return X_train, X_test, y_train, y_test


def plot_training_history(history, savePath):
    """Save accuracy and loss plots from training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(savePath, dpi=150)
    plt.close()
    print(f"\n  [OK] Training plots saved -> {savePath}")


def main():
    """Main training pipeline."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # -- 1. Load & process dataset --
    X_train, X_test, y_train, y_test = load_dataset()

    # -- 2. Normalize --
    trainMean = np.mean(X_train)
    trainStd = np.std(X_train)
    X_train = (X_train - trainMean) / trainStd
    X_test = (X_test - trainMean) / trainStd

    # Save normalization stats for inference
    np.savez(NORM_STATS_PATH, mean=trainMean, std=trainStd)
    print(f"\n  [OK] Normalization stats saved -> {NORM_STATS_PATH}")
    print(f"    Mean: {trainMean:.4f}  |  Std: {trainStd:.4f}")

    # -- 3. Reshape for CNN (add channel dimension) --
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    print(f"\n  Final shapes:")
    print(f"    X_train: {X_train.shape}")
    print(f"    X_test:  {X_test.shape}")

    # -- 4. Build model --
    print("\n" + "=" * 60)
    print("  BUILDING MODEL")
    print("=" * 60)
    inputShape = X_train.shape[1:]
    numClasses = len(GENRES)
    model = build_model(inputShape=inputShape, numClasses=numClasses)
    model.summary()

    # -- 5. Callbacks --
    earlyStop = EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    reduceLR = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    # -- 6. Train --
    print("\n" + "=" * 60)
    print("  TRAINING")
    print("=" * 60)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[earlyStop, checkpoint, reduceLR],
        verbose=1
    )

    # -- 7. Evaluate --
    print("\n" + "=" * 60)
    print("  EVALUATION")
    print("=" * 60)
    testLoss, testAccuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Loss:     {testLoss:.4f}")
    print(f"  Test Accuracy: {testAccuracy:.4f}  ({testAccuracy * 100:.1f}%)")

    # -- 8. Save plots --
    plot_training_history(history, HISTORY_PLOT_PATH)

    # -- 9. Save genre labels --
    genreLabelsPath = os.path.join(MODEL_DIR, 'genre_labels.npy')
    np.save(genreLabelsPath, np.array(GENRES))
    print(f"  [OK] Genre labels saved -> {genreLabelsPath}")

    print("\n" + "=" * 60)
    print("  [OK] TRAINING COMPLETE")
    print(f"  [OK] Model saved -> {MODEL_PATH}")
    print("=" * 60)


if __name__ == '__main__':
    main()
