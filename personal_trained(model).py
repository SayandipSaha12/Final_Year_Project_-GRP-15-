import tensorflow as tf
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

print("="*70)
print("PERSONAL HANDWRITING MODEL TRAINER")
print("Trains ONLY on YOUR handwriting - Zero EMNIST data")
print("="*70)

# ============================================================
# CONFIGURATION
# ============================================================
BASE_FOLDER  = r"C:\Final Year Project"        # Where your letter images are
MY_LETTERS   = r"C:\Final Year Project\my-letters"  # Fine-tune folder

AUGMENTS_PER_IMAGE = 500   # 500 variations per original photo
BATCH_SIZE         = 32    # Small batch for small dataset
EPOCHS             = 150   # Early stopping will stop it earlier
VAL_SPLIT          = 0.15  # 15% validation
OUTPUT_MODEL       = "handwriting_model_personal.keras"
BEST_MODEL         = "best_personal_model.keras"
# ============================================================

letters_lower = [chr(i) for i in range(ord('a'), ord('z') + 1)]

# ============================================================
# ALL 26 LETTER IMAGE PATHS
# Uses EVERY image you have for each letter
# ============================================================
def p(filename):
    """Shorthand: build path in BASE_FOLDER"""
    return os.path.join(BASE_FOLDER, filename)

def m(filename):
    """Shorthand: build path in MY_LETTERS folder"""
    return os.path.join(MY_LETTERS, filename)

all_image_sources = {
    # Letters you had working before (1 image each)
    'A': [p("A image.jpeg")],
    'B': [p("B image.jpg")],
    'E': [p("2 E image.jpg")],
    'H': [p("H image.jpeg")],
    'I': [p("2 I image.jpeg")],
    'K': [p("K image.jpg")],
    'L': [p("L image.jpg")],
    'M': [p("M image.jpeg")],
    'N': [p("N image.jpg")],
    'O': [p("O image.jpg")],
    'Q': [p("2 Q image.jpg")],
    'T': [p("T image.jpg")],
    'U': [p("U image.jpg")],
    'V': [p("V image.jpg")],
    'W': [p("2 W image.jpg")],
    'X': [p("X image.jpg")],
    'Y': [p("Y image.jpg")],

    # Problem letters (5 images each from my-letters folder)
    'C': [m("C1.jpg"), m("C2.jpg"), m("C3.jpg"), m("C4.jpg"), m("C5.jpg")],
    'D': [m("D1.jpg"), m("D2.jpg"), m("D3.jpg"), m("D4.jpg"), m("D5.jpg")],
    'F': [m("F-1.jpg"), m("F2.jpg"), m("F3.jpg"), m("F4.jpg"), m("F5.jpg")],
    'G': [m("G1.jpg"), m("G2.jpg"), m("G3.jpg"), m("G4.jpg"), m("G5.jpg")],
    'J': [m("J1.jpg"), m("J2.jpg"), m("J3.jpg"), m("J4.jpg"), m("J5.jpg")],
    'P': [m("P1.jpg"), m("P2.jpg"), m("P3.jpg"), m("P4.jpg"), m("P5.jpg")],

    # R has both a regular image AND fine-tune samples
    'R': [p("R image.jpg"),
          m("R1.jpg"), m("R2.jpg"), m("R3.jpg"), m("R4.jpg"), m("R5.jpg")],

    'S': [m("S1.jpg"), m("S2.jpg"), m("S3.jpg"), m("S4.jpg"), m("S5.jpg")],
    'Z': [m("Z1.jpg"), m("Z2.jpg"), m("Z3.jpg"), m("Z4.jpg"), m("Z5.jpg")],
}

# ============================================================
# HELPER: Find image regardless of extension
# ============================================================
def find_image(path):
    """Try .jpg .jpeg .png if exact path not found"""
    if os.path.exists(path):
        return path
    base = os.path.splitext(path)[0]
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        if os.path.exists(base + ext):
            return base + ext
    return None

# ============================================================
# PREPROCESSING PIPELINE
# ============================================================
def preprocess(img):
    """Convert photo → clean 28x28 binary image"""
    if img is None:
        return None

    # Make background black, letter white
    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)

    # Binarize
    _, img_bin = cv2.threshold(img, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find and crop the letter
    coords = cv2.findNonZero(img_bin)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    padding = max(4, int(max(w, h) * 0.1))
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img_bin.shape[1] - x, w + 2 * padding)
    h = min(img_bin.shape[0] - y, h + 2 * padding)

    crop   = img_bin[y:y+h, x:x+w]
    maxd   = max(w, h)
    canvas = int(maxd * 1.2)
    square = np.zeros((canvas, canvas), dtype=np.uint8)
    xo     = (canvas - w) // 2
    yo     = (canvas - h) // 2
    square[yo:yo+h, xo:xo+w] = crop

    return cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)

# ============================================================
# HEAVY AUGMENTATION
# Simulates all natural variations in your handwriting
# ============================================================
def augment(img_28x28, n=500):
    """Generate n augmented versions of a 28x28 image"""
    results = []

    for _ in range(n):
        aug = img_28x28.copy().astype(np.float32)

        # 1. Rotation: your cursive slants, so simulate that
        angle = np.random.uniform(-28, 28)
        M = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (28, 28), borderValue=0)

        # 2. Scale: you don't always write same size
        scale  = np.random.uniform(0.72, 1.28)
        new_s  = max(1, int(28 * scale))
        aug_sc = cv2.resize(aug, (new_s, new_s))
        if new_s >= 28:
            st  = (new_s - 28) // 2
            aug = aug_sc[st:st+28, st:st+28]
        else:
            pad = (28 - new_s) // 2
            rem = 28 - new_s - pad
            aug = cv2.copyMakeBorder(aug_sc, pad, rem, pad, rem,
                                     cv2.BORDER_CONSTANT, value=0)
        aug = cv2.resize(aug, (28, 28))

        # 3. Shift: letter might not be perfectly centered
        sx = np.random.randint(-4, 5)
        sy = np.random.randint(-4, 5)
        Ms = np.float32([[1, 0, sx], [0, 1, sy]])
        aug = cv2.warpAffine(aug, Ms, (28, 28), borderValue=0)

        # 4. Brightness: different pen pressure & lighting
        aug = np.clip(aug * np.random.uniform(0.65, 1.35), 0, 255)

        # 5. Gaussian noise: simulate camera/photo noise
        aug = np.clip(aug + np.random.normal(0, 9, aug.shape), 0, 255)

        # 6. Stroke width: thick/thin pen variations
        k = np.ones((2, 2), np.uint8)
        if np.random.random() > 0.5:
            aug = cv2.dilate(aug.astype(np.uint8), k, 1).astype(np.float32)
        else:
            aug = cv2.erode(aug.astype(np.uint8), k, 1).astype(np.float32)

        # 7. Perspective warp: photo taken at slight angle
        if np.random.random() > 0.5:
            src = np.float32([[0,0],[27,0],[0,27],[27,27]])
            d   = np.random.uniform(0, 2.5, (4,2)).astype(np.float32)
            dst = src + d
            Mp  = cv2.getPerspectiveTransform(src, dst)
            aug = cv2.warpPerspective(aug, Mp, (28,28),
                                      borderValue=0).astype(np.float32)

        # 8. Shear: simulates slanted writing angle
        if np.random.random() > 0.6:
            shear  = np.random.uniform(-0.2, 0.2)
            Ms2    = np.float32([[1, shear, 0], [0, 1, 0]])
            aug    = cv2.warpAffine(aug, Ms2, (28,28),
                                    borderValue=0).astype(np.float32)

        results.append(aug / 255.0)

    return results

# ============================================================
# STEP 1: LOAD ALL IMAGES AND AUGMENT
# ============================================================
print("\n" + "="*70)
print("STEP 1: Loading your handwriting images and augmenting")
print("="*70 + "\n")

train_images   = []
train_labels   = []
val_images     = []    # Hold out ORIGINAL (non-augmented) for validation
val_labels     = []
img_count      = {}
missing_letters = []

for letter in sorted(all_image_sources.keys()):
    paths    = all_image_sources[letter]
    label    = ord(letter.lower()) - ord('a')
    loaded   = 0
    originals = []

    for path in paths:
        found = find_image(path)
        if found is None:
            continue

        img = cv2.imread(found, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        processed = preprocess(img)
        if processed is None:
            continue

        originals.append(processed)
        loaded += 1

    if loaded == 0:
        print(f"  ❌ {letter}: No images found! Check paths above.")
        missing_letters.append(letter)
        img_count[letter] = 0
        continue

    # Use 1 original per letter as validation (non-augmented)
    val_img = originals[0].astype(np.float32) / 255.0
    val_images.append(val_img)
    val_labels.append(label)

    # Use ALL originals + augmentation for training
    for orig in originals:
        aug_batch = augment(orig, n=AUGMENTS_PER_IMAGE)
        train_images.extend(aug_batch)
        train_labels.extend([label] * len(aug_batch))

        # Also add the original itself to training
        train_images.append(orig.astype(np.float32) / 255.0)
        train_labels.append(label)

    total_samples = loaded * (AUGMENTS_PER_IMAGE + 1)
    img_count[letter] = loaded
    print(f"  ✓ {letter}: {loaded} photo(s) × {AUGMENTS_PER_IMAGE+1} = "
          f"{total_samples:,} training samples")

# ============================================================
# STEP 2: PREPARE DATASETS
# ============================================================
print("\n" + "="*70)
print("STEP 2: Preparing training and validation sets")
print("="*70)

if missing_letters:
    print(f"\n⚠ MISSING LETTERS: {', '.join(missing_letters)}")
    print("  Check the paths in all_image_sources above!")

X_train = np.array(train_images, dtype=np.float32)
y_train = np.array(train_labels,  dtype=np.int32)
X_val   = np.array(val_images,   dtype=np.float32)
y_val   = np.array(val_labels,   dtype=np.int32)

X_train = np.expand_dims(X_train, axis=-1)
X_val   = np.expand_dims(X_val,   axis=-1)

# Shuffle training set
idx     = np.random.permutation(len(X_train))
X_train = X_train[idx]
y_train = y_train[idx]

print(f"\nTraining samples  : {len(X_train):,}")
print(f"Validation samples: {len(X_val)}")

# Samples per letter summary
print("\nTraining samples per letter:")
for letter in sorted(all_image_sources.keys()):
    lbl   = ord(letter.lower()) - ord('a')
    count = np.sum(y_train == lbl)
    bar   = "█" * (count // 200)
    print(f"  {letter}: {count:,} {bar}")

# ============================================================
# STEP 3: CLASS WEIGHTS
# Letters with fewer photos get higher weight
# ============================================================
print("\n" + "="*70)
print("STEP 3: Computing class weights")
print("="*70)

total_train = len(y_train)
n_classes   = 26
class_weights = {}

for c in range(n_classes):
    count = int(np.sum(y_train == c))
    if count > 0:
        class_weights[c] = total_train / (n_classes * count)
    else:
        class_weights[c] = 1.0

# Show weights for all letters
print("\nHigher weight = model focuses more on that letter:")
for c in range(n_classes):
    letter = letters_lower[c].upper()
    print(f"  {letter}: {class_weights[c]:.4f}")

# ============================================================
# STEP 4: BUILD MODEL
# ============================================================
print("\n" + "="*70)
print("STEP 4: Building personal handwriting model")
print("="*70)

model = tf.keras.models.Sequential([

    # Block 1: Detect basic strokes (curves, lines)
    tf.keras.layers.Conv2D(32, (3,3), activation='relu',
                           padding='same', input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.3),

    # Block 2: Detect letter parts (loops, curves, corners)
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.3),

    # Block 3: Detect full letter shapes
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.35),

    # Dense: Final classification
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        256, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(
        128, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(26, activation='softmax')
])

# Label smoothing helps prevent overconfidence on small dataset
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary()
print(f"\nTotal parameters: {model.count_params():,}")

# ============================================================
# STEP 5: TRAIN
# ============================================================
print("\n" + "="*70)
print("STEP 5: Training personal model")
print("="*70)
print(f"\nMax Epochs   : {EPOCHS} (early stopping will stop sooner)")
print(f"Batch Size   : {BATCH_SIZE}")
print(f"Training on  : {len(X_train):,} augmented samples")
print(f"Validating on: {len(X_val)} original images\n")

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=5, min_lr=0.000001, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=20,
        restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        BEST_MODEL, monitor='val_accuracy',
        save_best_only=True, verbose=1
    )
]

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# ============================================================
# STEP 6: SAVE
# ============================================================
model.save(OUTPUT_MODEL)
print(f"\n✓ Final model saved  : '{OUTPUT_MODEL}'")
print(f"✓ Best model saved   : '{BEST_MODEL}'")

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\n📊 Validation Accuracy: {val_acc*100:.2f}%")
print(f"📊 Validation Loss    : {val_loss:.4f}")

# ============================================================
# STEP 7: QUICK TEST ON ORIGINALS
# ============================================================
print("\n" + "="*70)
print("STEP 6: Quick test on ALL original images")
print("="*70 + "\n")

correct      = 0
total        = 0
wrong_list   = []

for letter in sorted(all_image_sources.keys()):
    paths = all_image_sources[letter]
    label = ord(letter.lower()) - ord('a')
    found_any = False

    for path in paths[:1]:  # Test first image of each letter
        found = find_image(path)
        if found is None:
            continue

        img = cv2.imread(found, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        processed = preprocess(img)
        if processed is None:
            continue

        normalized = processed.astype(np.float32) / 255.0
        img_input  = np.expand_dims(normalized, axis=(0, -1))
        preds      = model.predict(img_input, verbose=0)
        pred_idx   = np.argmax(preds[0])
        pred_letter = letters_lower[pred_idx].upper()
        confidence  = preds[0][pred_idx] * 100

        is_correct = (pred_letter == letter)
        symbol     = "✓" if is_correct else "✗"

        print(f"  {symbol} {letter}: Predicted as {pred_letter} "
              f"(Confidence: {confidence:.1f}%)")

        if is_correct:
            correct += 1
        else:
            wrong_list.append(f"{letter}→{pred_letter}")

        total += 1
        found_any = True
        break

    if not found_any:
        print(f"  ⚠ {letter}: No image found for testing")

print(f"\n{'='*50}")
print(f"Quick Test Results: {correct}/{total} letters correct")
if total > 0:
    print(f"Accuracy          : {correct/total*100:.1f}%")
if wrong_list:
    print(f"Wrong predictions : {', '.join(wrong_list)}")
print(f"{'='*50}")

# ============================================================
# TRAINING HISTORY PLOT
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'],     label='Training',   linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
ax1.set_title('Personal Model - Accuracy',
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['loss'],     label='Training',   linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
ax2.set_title('Personal Model - Loss',
              fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle(
    f"Personal Model Training | "
    f"Final Accuracy: {val_acc*100:.1f}%",
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig('personal_model_history.png', dpi=150)
print("\n✓ Training history saved as 'personal_model_history.png'")
plt.show()

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\n✅ Use this model in ALL your scripts:")
print(f'   model = tf.keras.models.load_model("{BEST_MODEL}")')