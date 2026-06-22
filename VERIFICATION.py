"""
Handwriting Match Checker
=========================
Compares two images of handwriting and estimates the likelihood that they
were written by the same person, expressed as a percentage match.

HOW IT WORKS
------------
This does NOT do pixel-matching (the two samples can contain completely
different words). Instead it extracts STYLE features that tend to be
consistent for a given writer, regardless of what word/sentence was written:

  1. HOG            - shapes/orientation of strokes
  2. LBP             - micro-texture of the ink strokes
  3. GLCM            - contrast/homogeneity texture statistics
  4. Stroke width    - mean & variability of pen stroke thickness
  5. Slant angle      - average tilt of vertical strokes
  6. Ink density     - how much of the writing area is covered by ink

Each feature group is compared between the two images and combined into a
single weighted similarity score (0-100%).

LIMITATIONS (please read)
--------------------------
This is a heuristic similarity tool built from classical image-processing
features, NOT a forensically validated handwriting-verification system and
NOT a trained machine-learning classifier (that would require a large
labeled dataset of known writers, e.g. IAM or CEDAR, to train a model on).
Treat the output as a supporting signal, not proof of authorship. Image
quality, scan angle, pen type, and writing surface all affect the score.

USAGE
-----
    python handwriting_match.py
        -> opens a window where you click "Browse..." to pick each photo
           from your folders (no typing paths needed).

    python handwriting_match.py image1.jpg image2.jpg
        -> classic command-line mode: prints a text report, no window.

REQUIREMENTS
------------
    pip install opencv-python scikit-image scipy numpy matplotlib pillow
    (tkinter ships with most Python installs; on Ubuntu/Debian, if missing,
     run: sudo apt install python3-tk)
"""

import argparse
import sys

import cv2
import numpy as np
from scipy.spatial.distance import cosine
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage.morphology import skeletonize

# Standard size every handwriting crop is normalized to before feature
# extraction, so that feature vectors are always comparable.
STD_SIZE = (600, 200)  # (width, height)


# --------------------------------------------------------------------------
# 1. PREPROCESSING
# --------------------------------------------------------------------------
def load_and_preprocess(path):
    """Load an image, binarize it, crop to the handwriting, and resize."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Otsu binarization. THRESH_BINARY_INV -> ink becomes white (255) on a
    # black background, which is the convention used by the rest of the code.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Crop to the bounding box of the actual writing to remove blank margins.
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        binary = binary[y : y + h, x : x + w]

    if binary.size == 0 or binary.shape[0] == 0 or binary.shape[1] == 0:
        raise ValueError(f"No handwriting detected in image: {path}")

    resized = cv2.resize(binary, STD_SIZE, interpolation=cv2.INTER_AREA)
    # Re-threshold after resize to keep it strictly binary (resize can blur edges).
    _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

    return resized  # uint8 array, ink = 255, background = 0


# --------------------------------------------------------------------------
# 2. FEATURE EXTRACTION
# --------------------------------------------------------------------------
def feature_hog(binary_img):
    features = hog(
        binary_img,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features


def feature_lbp(binary_img):
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(binary_img, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp, bins=n_points + 2, range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-7
    return hist


def feature_glcm(binary_img):
    glcm = graycomatrix(
        binary_img, distances=[1, 2], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256, symmetric=True, normed=True,
    )
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    vec = np.hstack([graycoprops(glcm, p).flatten() for p in props])
    return np.nan_to_num(vec)


def feature_stroke_width(binary_img):
    """Estimate mean & std of pen stroke thickness using a distance transform
    sampled along the skeleton (medial axis) of the strokes."""
    mask = (binary_img > 0).astype(np.uint8)
    if mask.sum() == 0:
        return 0.0, 0.0

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    skeleton = skeletonize(mask.astype(bool))
    widths = dist[skeleton] * 2  # distance transform gives radius; widths = 2x

    if widths.size == 0:
        return 0.0, 0.0
    return float(np.mean(widths)), float(np.std(widths))


def feature_slant_angle(binary_img):
    """Estimate the dominant slant of strokes using Hough line detection."""
    edges = cv2.Canny(binary_img, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=20, minLineLength=15, maxLineGap=4
    )
    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            angles.append(90.0)
            continue
        angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
        # Fold into a -90..90 range, then only consider near-vertical strokes
        # (handwriting slant), discarding mostly-horizontal segments.
        if 45 <= abs(angle) <= 135:
            angles.append(angle)

    if not angles:
        return 0.0
    return float(np.median(angles))


def feature_ink_density(binary_img):
    return float(np.count_nonzero(binary_img)) / binary_img.size


def extract_all_features(binary_img):
    mean_w, std_w = feature_stroke_width(binary_img)
    return {
        "hog": feature_hog(binary_img),
        "lbp": feature_lbp(binary_img),
        "glcm": feature_glcm(binary_img),
        "stroke_mean": mean_w,
        "stroke_std": std_w,
        "slant": feature_slant_angle(binary_img),
        "density": feature_ink_density(binary_img),
    }


# --------------------------------------------------------------------------
# 3. SIMILARITY SCORING
# --------------------------------------------------------------------------
def cosine_similarity(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    sim = 1 - cosine(a, b)
    return float(np.clip(sim, 0.0, 1.0))


def scalar_similarity(a, b, scale):
    """Convert an absolute difference between two scalars into a 0-1
    similarity score, using `scale` as the difference that counts as 0%."""
    diff = abs(a - b)
    sim = 1 - (diff / scale)
    return float(np.clip(sim, 0.0, 1.0))


def compare_features(f1, f2):
    scores = {
        "Stroke shape (HOG)": cosine_similarity(f1["hog"], f2["hog"]),
        "Ink texture (LBP)": cosine_similarity(f1["lbp"], f2["lbp"]),
        "Texture statistics (GLCM)": cosine_similarity(f1["glcm"], f2["glcm"]),
        "Stroke width": scalar_similarity(f1["stroke_mean"], f2["stroke_mean"], scale=8.0),
        "Stroke width variation": scalar_similarity(f1["stroke_std"], f2["stroke_std"], scale=5.0),
        "Writing slant": scalar_similarity(f1["slant"], f2["slant"], scale=45.0),
        "Ink density": scalar_similarity(f1["density"], f2["density"], scale=0.5),
    }

    weights = {
        "Stroke shape (HOG)": 0.32,
        "Ink texture (LBP)": 0.18,
        "Texture statistics (GLCM)": 0.15,
        "Stroke width": 0.13,
        "Stroke width variation": 0.07,
        "Writing slant": 0.10,
        "Ink density": 0.05,
    }

    overall = sum(scores[k] * weights[k] for k in scores)
    return scores, overall * 100.0


# --------------------------------------------------------------------------
# 4. MAIN COMPARISON PIPELINE
# --------------------------------------------------------------------------
def compare_handwriting(path1, path2, threshold=65.0):
    img1 = load_and_preprocess(path1)
    img2 = load_and_preprocess(path2)

    f1 = extract_all_features(img1)
    f2 = extract_all_features(img2)

    breakdown, overall_pct = compare_features(f1, f2)
    verdict = "Likely the SAME writer" if overall_pct >= threshold else "Likely DIFFERENT writers"

    return {
        "overall_percent": overall_pct,
        "verdict": verdict,
        "breakdown": breakdown,
        "processed_images": (img1, img2),
    }


def print_report(result, path1, path2):
    print("\n" + "=" * 55)
    print("HANDWRITING MATCH REPORT")
    print("=" * 55)
    print(f"Image 1: {path1}")
    print(f"Image 2: {path2}")
    print("-" * 55)
    print("Feature-level similarity:")
    for name, score in result["breakdown"].items():
        print(f"   {name:<28s}: {score * 100:5.1f}%")
    print("-" * 55)
    print(f"OVERALL MATCH SCORE : {result['overall_percent']:.1f}%")
    print(f"VERDICT              : {result['verdict']}")
    print("=" * 55)
    print(
        "Note: heuristic similarity score, not a forensic or legally "
        "validated authentication result.\n"
    )


def visualize(img1, img2, path1, path2, result):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img1, cmap="gray")
    axes[0].set_title(path1.split("/")[-1])
    axes[0].axis("off")
    axes[1].imshow(img2, cmap="gray")
    axes[1].set_title(path2.split("/")[-1])
    axes[1].axis("off")
    fig.suptitle(
        f"Match score: {result['overall_percent']:.1f}%  —  {result['verdict']}"
    )
    plt.tight_layout()
    plt.savefig("comparison_result.png", dpi=150)
    print("Visualization saved to comparison_result.png")


# --------------------------------------------------------------------------
# 5. GUI (file-picker) MODE
# --------------------------------------------------------------------------
def launch_gui(threshold=65.0):
    """Opens a desktop window with 'Browse...' buttons so the user can pick
    each handwriting photo from their own folders via the native OS file
    dialog, then shows the match report in the window."""
    import tkinter as tk
    from tkinter import filedialog, messagebox
    from PIL import Image, ImageTk

    class HandwritingApp:
        def __init__(self, root):
            self.root = root
            self.path1 = None
            self.path2 = None
            self._thumb_refs = []  # keep references so Tkinter doesn't GC images

            root.title("Handwriting Match Checker")
            root.geometry("780x620")
            root.configure(bg="#f3f4f6")

            tk.Label(
                root, text="Handwriting Match Checker",
                font=("Helvetica", 18, "bold"), bg="#f3f4f6",
            ).pack(pady=(18, 2))
            tk.Label(
                root, text="Browse for two handwriting photos to compare",
                font=("Helvetica", 11), bg="#f3f4f6", fg="#555",
            ).pack(pady=(0, 12))

            row = tk.Frame(root, bg="#f3f4f6")
            row.pack()
            self.panel1 = self._build_panel(row, "Image 1", 0, self.browse_image1)
            self.panel2 = self._build_panel(row, "Image 2", 1, self.browse_image2)

            self.compare_btn = tk.Button(
                root, text="Compare Handwriting", font=("Helvetica", 12, "bold"),
                bg="#3b6fd6", fg="white", activebackground="#2f5bb8",
                padx=18, pady=8, bd=0, state="disabled", command=self.run_comparison,
            )
            self.compare_btn.pack(pady=14)

            result_box = tk.Frame(root, bg="white", bd=1, relief="solid")
            result_box.pack(padx=20, pady=(0, 18), fill="both", expand=True)
            self.result_label = tk.Label(
                result_box, text="Results will appear here after you compare.",
                font=("Courier New", 11), bg="white", fg="#888",
                justify="left", anchor="nw", wraplength=720,
            )
            self.result_label.pack(padx=15, pady=15, fill="both", expand=True, anchor="nw")

        def _build_panel(self, parent, title, col, browse_cmd):
            panel = tk.Frame(parent, bg="white", bd=1, relief="solid", width=340, height=260)
            panel.grid(row=0, column=col, padx=12)
            tk.Label(panel, text=title, font=("Helvetica", 11, "bold"), bg="white").pack(pady=(10, 5))
            img_lbl = tk.Label(panel, text="No image selected", bg="#eee", width=40, height=10)
            img_lbl.pack(padx=10, pady=5)
            tk.Button(panel, text="Browse...", command=browse_cmd).pack(pady=10)
            panel.img_lbl = img_lbl
            return panel

        def _choose_file(self):
            return filedialog.askopenfilename(
                title="Select a handwriting photo",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                    ("All files", "*.*"),
                ],
            )

        def browse_image1(self):
            path = self._choose_file()
            if path:
                self.path1 = path
                self._show_thumbnail(self.panel1, path)
                self._check_ready()

        def browse_image2(self):
            path = self._choose_file()
            if path:
                self.path2 = path
                self._show_thumbnail(self.panel2, path)
                self._check_ready()

        def _show_thumbnail(self, panel, path):
            try:
                img = Image.open(path)
                img.thumbnail((300, 200))
                photo = ImageTk.PhotoImage(img)
            except Exception as e:
                messagebox.showerror("Could not open image", str(e))
                return
            panel.img_lbl.configure(image=photo, text="")
            panel.img_lbl.image = photo
            self._thumb_refs.append(photo)

        def _check_ready(self):
            if self.path1 and self.path2:
                self.compare_btn.configure(state="normal")

        def run_comparison(self):
            self.result_label.configure(
                text="Comparing... please wait.", fg="#555", justify="left"
            )
            self.root.update_idletasks()
            try:
                result = compare_handwriting(self.path1, self.path2, threshold=threshold)
            except Exception as e:
                messagebox.showerror("Error", str(e))
                self.result_label.configure(text=f"Error: {e}", fg="#c0392b")
                return

            lines = [
                f"OVERALL MATCH SCORE : {result['overall_percent']:.1f}%",
                f"VERDICT              : {result['verdict']}",
                "",
                "Feature-level similarity:",
            ]
            for name, score in result["breakdown"].items():
                lines.append(f"   {name:<28s}: {score * 100:5.1f}%")
            lines.append("")
            lines.append(
                "Note: heuristic similarity score, not a forensic or\n"
                "legally validated authentication result."
            )

            color = "#1a7f37" if "SAME" in result["verdict"] else "#c0392b"
            self.result_label.configure(text="\n".join(lines), fg=color)

    root = tk.Tk()
    HandwritingApp(root)
    root.mainloop()


# --------------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compare two handwriting samples.")
    parser.add_argument("image1", nargs="?", help="Path to first handwriting image (omit to use the file-picker window)")
    parser.add_argument("image2", nargs="?", help="Path to second handwriting image (omit to use the file-picker window)")
    parser.add_argument(
        "--threshold", type=float, default=65.0,
        help="Percentage above which the verdict is 'same writer' (default: 65)",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="(CLI mode only) Save a side-by-side image of the two processed samples with the result",
    )
    args = parser.parse_args()

    # No image paths given -> launch the file-picker GUI window.
    if not args.image1 and not args.image2:
        launch_gui(threshold=args.threshold)
        return

    # Paths given on the command line -> classic text-report mode.
    path1, path2 = args.image1, args.image2
    if not path1 or not path2:
        print("Please provide both image paths, or run with no arguments to use the GUI.", file=sys.stderr)
        sys.exit(1)

    try:
        result = compare_handwriting(path1, path2, threshold=args.threshold)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print_report(result, path1, path2)

    if args.visualize:
        img1, img2 = result["processed_images"]
        visualize(img1, img2, path1, path2, result)


if __name__ == "__main__":
    main()