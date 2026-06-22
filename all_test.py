#THIS CODE IS SAME AS NEW_TEST.PY BUT WITH EXTRA VISUALIZATION AND SUMMARY DETAILS
#THIS CODE IS SAME AS NEW_TEST.PY BUT WITH EXTRA VISUALIZATION AND SUMMARY DETAILS
#THIS CODE IS SAME AS NEW_TEST.PY BUT WITH EXTRA VISUALIZATION AND SUMMARY DETAILS
#THIS CODE IS SAME AS NEW_TEST.PY BUT WITH EXTRA VISUALIZATION AND SUMMARY DETAILS
#THIS CODE IS SAME AS NEW_TEST.PY BUT WITH EXTRA VISUALIZATION AND SUMMARY DETAILS
#THIS CODE IS SAME AS NEW_TEST.PY BUT WITH EXTRA VISUALIZATION AND SUMMARY DETAILS
#THIS CODE IS SAME AS NEW_TEST.PY BUT WITH EXTRA VISUALIZATION AND SUMMARY DETAILS


import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

print("="*70)
print("FULL 26-LETTER RECOGNITION TEST - NEW COMBINED MODEL")
print("="*70)

# ============================================================
# LOAD THE NEW MODEL
# ============================================================
MODEL_PATH = "best_personal_model.keras"  # Use best checkpoint

if not os.path.exists(MODEL_PATH):
    # Fallback to final model if best checkpoint not found
    MODEL_PATH = "handwriting_model_personalized.keras"
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found!")
        print("   Make sure you ran retrain_combined.py first")
        exit()

model = tf.keras.models.load_model(MODEL_PATH)
print(f"✓ Model loaded: {MODEL_PATH}\n")

# ============================================================
# DEFINE ALL 26 LETTER IMAGE PATHS HERE
# ============================================================
image_paths = {
    'A': r"C:\Final Year Project\letters_output_20260604_004830_YD42\ANRUA.jpg",
    'B': r"C:\Final Year Project\letters_output_20260604_004830_YD42\BKAQK.jpg",
    'C': r"C:\Final Year Project\letters_output_20260604_004830_YD42\CKBXR.jpg",
    'D': r"C:\Final Year Project\letters_output_20260604_004830_YD42\D6P78.jpg",
    'E': r"C:\Final Year Project\letters_output_20260604_004830_YD42\EI2FL.jpg",
    'F': r"C:\Final Year Project\letters_output_20260604_004830_YD42\FU5JA.jpg",
    'G': r"C:\Final Year Project\letters_output_20260604_004830_YD42\GXIUA.jpg",
    'H': r"C:\Final Year Project\letters_output_20260604_004830_YD42\HMLW7.jpg",
    'I': r"C:\Final Year Project\letters_output_20260604_004830_YD42\I801J.jpg",
    'J': r"C:\Final Year Project\letters_output_20260604_004830_YD42\JXC00.jpg",
    'K': r"C:\Final Year Project\letters_output_20260604_004830_YD42\KOD04.jpg",
    'L': r"C:\Final Year Project\letters_output_20260604_004830_YD42\LJ5EW.jpg",
    'M': r"C:\Final Year Project\letters_output_20260604_004830_YD42\MPU2U.jpg",
    'N': r"C:\Final Year Project\letters_output_20260604_004830_YD42\NTYJ4.jpg",
    'O': r"C:\Final Year Project\letters_output_20260604_004830_YD42\O2NQE.jpg",
    'P': r"C:\Final Year Project\letters_output_20260604_004830_YD42\PN8F5.jpg",
    'Q': r"C:\Final Year Project\letters_output_20260604_004830_YD42\QDMJF.jpg",
    'R': r"C:\Final Year Project\letters_output_20260604_004830_YD42\RBXYY.jpg",
    'S': r"C:\Final Year Project\letters_output_20260604_004830_YD42\SCE0Q.jpg",
    'T': r"C:\Final Year Project\letters_output_20260604_004830_YD42\T34OH.jpg",
    'U': r"C:\Final Year Project\letters_output_20260604_004830_YD42\U0TXN.jpg",
    'V': r"C:\Final Year Project\letters_output_20260604_004830_YD42\V9EIM.jpg",
    'W': r"C:\Final Year Project\letters_output_20260604_004830_YD42\W509S.jpg",
    'X': r"C:\Final Year Project\letters_output_20260604_004830_YD42\XT7YO.jpg",
    'Y': r"C:\Final Year Project\letters_output_20260604_004830_YD42\YJQEG.jpg",
    'Z': r"C:\Final Year Project\letters_output_20260604_004830_YD42\Z93NO.jpg",
}
# ============================================================

all_letters   = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
letters_lower = [chr(i) for i in range(ord('a'), ord('z') + 1)]

# Previously failing - highlight these in results
problem_letters = ['C', 'D', 'F', 'G', 'J', 'P', 'R', 'S', 'Z']

print(f"Testing all 26 letters: {', '.join(all_letters)}")
print(f"Previously failing (now testing): {', '.join(problem_letters)}")
print("="*70 + "\n")

# ============================================================
# PREPROCESSING FUNCTION
# ============================================================
def preprocess_image(img):
    """Standard preprocessing pipeline"""
    if img is None:
        return None, None

    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)

    _, img_bin = cv2.threshold(img, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    coords = cv2.findNonZero(img_bin)
    if coords is None:
        return None, None

    x, y, w, h = cv2.boundingRect(coords)
    padding = max(4, int(max(w, h) * 0.1))
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img_bin.shape[1] - x, w + 2 * padding)
    h = min(img_bin.shape[0] - y, h + 2 * padding)

    char_crop  = img_bin[y:y+h, x:x+w]
    max_dim    = max(w, h)
    canvas_sz  = int(max_dim * 1.2)
    square_img = np.zeros((canvas_sz, canvas_sz), dtype=np.uint8)
    xo = (canvas_sz - w) // 2
    yo = (canvas_sz - h) // 2
    square_img[yo:yo+h, xo:xo+w] = char_crop

    resized    = cv2.resize(square_img, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized, resized

# ============================================================
# PREDICT FUNCTION
# ============================================================
def predict_letter(image_path):
    """Load image, preprocess, predict"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, 0.0, None

    normalized, display_img = preprocess_image(img)
    if normalized is None:
        return None, 0.0, None

    img_input   = np.expand_dims(normalized, axis=(0, -1))
    predictions = model.predict(img_input, verbose=0)
    pred_class  = np.argmax(predictions[0])
    confidence  = predictions[0][pred_class]
    pred_letter = letters_lower[pred_class].upper()

    return pred_letter, confidence, display_img

# ============================================================
# TEST ALL 26 LETTERS
# ============================================================
print("Processing all 26 letters...\n")

results          = []
previously_fixed = []  # Track if problem letters are now working

for letter in all_letters:
    is_problem = letter in problem_letters

    # Check if path is defined
    if letter not in image_paths:
        print(f"⚠ {letter}: No path defined in image_paths")
        results.append({
            'true_label' : letter,
            'predicted'  : 'NO PATH',
            'confidence' : 0,
            'correct'    : False,
            'image'      : None,
            'is_problem' : is_problem
        })
        continue

    # Check if file exists
    if not os.path.exists(image_paths[letter]):
        print(f"⚠ {letter}: Image not found → {image_paths[letter]}")
        results.append({
            'true_label' : letter,
            'predicted'  : 'NOT FOUND',
            'confidence' : 0,
            'correct'    : False,
            'image'      : None,
            'is_problem' : is_problem
        })
        continue

    # Predict
    predicted, confidence, display_img = predict_letter(image_paths[letter])

    if predicted is None:
        print(f"⚠ {letter}: Failed to process image")
        results.append({
            'true_label' : letter,
            'predicted'  : 'FAILED',
            'confidence' : 0,
            'correct'    : False,
            'image'      : None,
            'is_problem' : is_problem
        })
        continue

    is_correct = (predicted == letter)
    symbol     = "✓" if is_correct else "✗"

    # Extra tag for previously failing letters
    tag = " ← [WAS FAILING]" if is_problem else ""

    print(f"{symbol} {letter}: Predicted as {predicted} "
          f"(Confidence: {confidence*100:.1f}%){tag}")

    results.append({
        'true_label' : letter,
        'predicted'  : predicted,
        'confidence' : confidence,
        'correct'    : is_correct,
        'image'      : display_img,
        'is_problem' : is_problem
    })

    if is_problem and is_correct:
        previously_fixed.append(letter)

# ============================================================
# RESULTS SUMMARY
# ============================================================
valid   = [r for r in results
           if r['predicted'] not in ['NO PATH', 'NOT FOUND', 'FAILED']]
correct = [r for r in valid if r['correct']]
wrong   = [r for r in valid if not r['correct']]

prev_valid   = [r for r in valid   if r['is_problem']]
prev_correct = [r for r in correct if r['is_problem']]

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print(f"\nTotal Letters Tested : {len(valid)}/26")
print(f"Correct Predictions  : {len(correct)}")
print(f"Incorrect Predictions: {len(wrong)}")
if valid:
    print(f"Overall Accuracy     : {len(correct)/len(valid)*100:.1f}%")

print(f"\n--- Previously Failing Letters (C,D,F,G,J,P,R,S,Z) ---")
if prev_valid:
    print(f"Now Correct : {len(prev_correct)}/{len(prev_valid)}")
    if prev_correct:
        print(f"Fixed       : {', '.join([r['true_label'] for r in prev_correct])} ✓")
    still_wrong = [r for r in prev_valid if not r['correct']]
    if still_wrong:
        print(f"Still wrong : {', '.join([r['true_label'] for r in still_wrong])} ✗")

if wrong:
    print(f"\n--- All Incorrect Predictions ---")
    for r in wrong:
        tag = " [was problem letter]" if r['is_problem'] else ""
        print(f"  ✗ {r['true_label']} → {r['predicted']} "
              f"({r['confidence']*100:.1f}%){tag}")

# ============================================================
# VISUALIZATION - ALL 26 LETTERS
# ============================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATION")
print("="*70)

valid_img_results = [r for r in results if r['image'] is not None]
n = len(valid_img_results)

if n > 0:
    cols = 7
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 2.5, rows * 3.2))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.ravel()

    for idx, r in enumerate(valid_img_results):
        ax = axes[idx]
        ax.imshow(r['image'], cmap='gray')

        # Color logic:
        # Green        = correct
        # Orange       = correct AND was a problem letter (extra highlight!)
        # Red          = wrong
        if r['correct'] and r['is_problem']:
            color = 'darkorange'   # Fixed! Highlight these
        elif r['correct']:
            color = 'green'
        else:
            color = 'red'

        title = f"True: {r['true_label']}\nPred: {r['predicted']}"
        if r['correct']:
            title += f"\n✓ {r['confidence']*100:.0f}%"
            if r['is_problem']:
                title += " 🔥"   # Mark newly fixed letters
        else:
            title += f"\n✗ {r['confidence']*100:.0f}%"

        ax.set_title(title, color=color,
                     fontweight='bold', fontsize=9)
        ax.axis('off')

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    accuracy_str = (f"{len(correct)}/{len(valid)} Correct "
                    f"({len(correct)/len(valid)*100:.1f}%)"
                    if valid else "")

    plt.suptitle(
        f"Full 26-Letter Recognition Test | {accuracy_str}\n"
        f"Orange 🔥 = Previously failing, now fixed!",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('full_26_letter_test.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved as 'full_26_letter_test.png'")
    plt.show()

# ============================================================
# FINAL VERDICT
# ============================================================
print("\n" + "="*70)
print("FINAL VERDICT")
print("="*70)

if valid:
    acc = len(correct) / len(valid) * 100
    if acc == 100:
        print("\n🏆 PERFECT! All 26 letters recognized correctly!")
    elif acc >= 90:
        print(f"\n🎉 EXCELLENT! {acc:.1f}% accuracy - almost perfect!")
    elif acc >= 75:
        print(f"\n✅ GOOD! {acc:.1f}% accuracy - significant improvement!")
    else:
        print(f"\n⚠ {acc:.1f}% accuracy - some letters still need work")

    if wrong:
        print(f"\nStill needs improvement: "
              f"{', '.join([r['true_label'] for r in wrong])}")
        print("\nFor remaining wrong letters try:")
        print("  1. Add more samples (10 instead of 5)")
        print("  2. Make sure photos are clear and well-lit")
        print("  3. Write the letter bigger with thicker pen")

print("\n" + "="*70)
print("TEST COMPLETE!")
print("="*70)