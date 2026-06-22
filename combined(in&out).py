"""
Handwritten A-Z Letter Extractor + Text Generator
---------------------------------------------------
Step 1: Select an image containing handwritten capital letters A to Z (in order).
        The script detects each character, crops it, and saves as A<random>.jpg,
        B<random>.jpg ... Z<random>.jpg inside a uniquely-named output folder.

Step 2: Enter any text and the script will generate a handwritten image using
        the letters extracted in Step 1.

Requirements:
    pip install opencv-python numpy pillow matplotlib
    (tkinter is built into Python on Windows/macOS; on Linux: sudo apt install python3-tk)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import string
import random
import string as strlib
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — EXTRACTION CODE (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def random_suffix(length: int = 4) -> str:
    chars = strlib.ascii_uppercase + strlib.digits
    return "".join(random.choices(chars, k=length))


def unique_folder(base: str = "letters_output") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"{base}_{timestamp}_{random_suffix(4)}"
    os.makedirs(folder, exist_ok=True)
    return folder


def crop_dark_borders(img: np.ndarray, brightness_threshold: int = 80) -> np.ndarray:
    """
    Remove dark strips (camera shadow / phone edge) from all four sides.
    Any column or row whose mean brightness is below brightness_threshold
    at the edge of the image is considered a dark border and cropped away.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    col_mean = gray.mean(axis=0)
    row_mean = gray.mean(axis=1)

    left = 0
    while left < w and col_mean[left] < brightness_threshold:
        left += 1

    right = w - 1
    while right > left and col_mean[right] < brightness_threshold:
        right -= 1

    top = 0
    while top < h and row_mean[top] < brightness_threshold:
        top += 1

    bottom = h - 1
    while bottom > top and row_mean[bottom] < brightness_threshold:
        bottom -= 1

    return img[top:bottom + 1, left:right + 1]


def is_scribble(roi_gray: np.ndarray, ink_threshold: float = 0.13) -> bool:
    """
    A scribble has much denser ink coverage than a clean letter.
    Real letters: ink_ratio approx 0.05-0.10
    Scribbles / cross-outs: ink_ratio approx 0.15+
    """
    _, bw = cv2.threshold(roi_gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_ratio = np.count_nonzero(bw) / bw.size
    return ink_ratio > ink_threshold


def reading_order_sort(bboxes: list) -> list:
    """
    Sort bounding boxes in reading order (top row left to right,
    then next row, etc.). Uses y-centre clustering.
    """
    if not bboxes:
        return []

    heights = sorted(h for _, _, _, h in bboxes)
    median_h = heights[len(heights) // 2]
    row_tol = median_h * 0.7

    def y_center(b):
        return b[1] + b[3] // 2

    rows = {}
    for b in bboxes:
        yc = y_center(b)
        matched = False
        for rep_y in list(rows.keys()):
            if abs(yc - rep_y) < row_tol:
                rows[rep_y].append(b)
                matched = True
                break
        if not matched:
            rows[float(yc)] = [b]

    result = []
    for rep_y in sorted(rows.keys()):
        result.extend(sorted(rows[rep_y], key=lambda b: b[0]))
    return result


def extract_letters(image_path: str, output_folder: str) -> list:
    """
    Detect up to 26 letter blobs in the image (reading order),
    skip scribbles, save each as <LETTER><suffix>.jpg.
    Returns list of saved file paths.
    """
    img_raw = cv2.imread(image_path)
    if img_raw is None:
        raise FileNotFoundError(f"Cannot open: {image_path}")

    img = crop_dark_borders(img_raw, brightness_threshold=80)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"  Image size after border crop: {img.shape[1]}x{img.shape[0]}")

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    IH, IW = gray.shape
    min_area = int(IH * IW * 0.001)
    bboxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w * h < max(min_area, 500):
            continue

        if y <= 2 and h > IH * 0.25:
            print(f"  Skipping border artefact at x={x} y={y} w={w} h={h}")
            continue

        aspect = w / h
        if aspect < 0.15 or aspect > 6.0:
            print(f"  Skipping extreme aspect ({aspect:.2f}) at x={x} y={y}")
            continue

        bboxes.append((x, y, w, h))

    if not bboxes:
        raise RuntimeError(
            "No letter-like regions found. "
            "Try better lighting or a flatter sheet of paper."
            "MAKE SURE YOU ARE USING CAPITAL LETTERS A-Z IN ORDER (NO LOWERCASE, NO NUMBERS, NO SYMBOLS)"
            "ALSO MAKE SURE THERE IS A GAP AT THE END OF EACH ROW IF YOU ARE WRITING THE ALPHABET IN DIFFERENT ROWS PART BY"
        )

    bboxes = reading_order_sort(bboxes)
    print(f"  Detected {len(bboxes)} candidate region(s)")

    PADDING = 12
    alphabet = string.ascii_uppercase
    saved = []
    letter_idx = 0

    for (x, y, w, h) in bboxes:
        if letter_idx >= 26:
            break

        x1 = max(0, x - PADDING)
        y1 = max(0, y - PADDING)
        x2 = min(IW, x + w + PADDING)
        y2 = min(IH, y + h + PADDING)

        roi      = img[y1:y2, x1:x2]
        roi_gray = gray[y1:y2, x1:x2]

        if is_scribble(roi_gray, ink_threshold=0.13):
            print(f"  Scribble detected at position #{letter_idx + 1} "
                  f"(x={x} y={y}) - skipping")
            continue

        letter   = alphabet[letter_idx]
        filename = f"{letter}{random_suffix(4)}.jpg"
        out_path = os.path.join(output_folder, filename)

        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        bg = Image.new("RGB", roi_pil.size, (255, 255, 255))
        bg.paste(roi_pil)
        bg.save(out_path, "JPEG", quality=95)

        print(f"  [+]  [{letter}]  ->  {filename}")
        saved.append(out_path)
        letter_idx += 1

    return saved


# ── BRIDGE: run extraction GUI, return letter_image_paths dict ────────────────

def run_extraction() -> dict:
    """
    Shows the file picker, runs extract_letters(), and returns
    { 'A': 'path/A1234.jpg', 'B': 'path/B5678.jpg', ... }
    built from the first character of each saved filename.
    """
    root = tk.Tk()
    root.withdraw()

    messagebox.showinfo(
        "Handwritten A-Z Extractor",
        "Select an image containing handwritten capital letters A to Z (in order).\n\n"
        "Tips for best results:\n"
        "  - Good lighting, white paper\n"
        "  - Letters should have clear gaps between them  and must be in capital\n"
        "  - Make sure there are no Scribbles / cross-outs or dots skipped\n"
        "  - If you are writing the alphabet in different rows, make sure there is a big gap at the end of each row\n"
    )

    image_path = filedialog.askopenfilename(
        title="Choose handwriting image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
            ("All files", "*.*"),
        ]
    )

    if not image_path:
        messagebox.showwarning("Cancelled", "No file selected. Exiting.")
        return {}

    output_folder = unique_folder("letters_output")
    print(f"\n  Output folder : {os.path.abspath(output_folder)}")
    print(f"  Image         : {image_path}\n")

    try:
        saved = extract_letters(image_path, output_folder)

        if len(saved) < 26:
            missing = 26 - len(saved)
            note = (f"\n\n  {missing} letter(s) could not be found or were skipped as scribbles.\n"
                    "Check the terminal output for details.")
        else:
            note = ""

        msg = (f"Done!  Extracted {len(saved)} / 26 letter(s).{note}\n\n"
               f"Saved to:\n{os.path.abspath(output_folder)}")

        print(f"\n{msg}")
        messagebox.showinfo("Extraction Done", msg)

        # Build { 'A': 'path', 'B': 'path', ... }
        # saved = ['folder/A1234.jpg', 'folder/B5678.jpg', ...]
        # The letter is the first character of each filename
        letter_image_paths = {}
        for path in saved:
            letter = os.path.basename(path)[0].upper()
            letter_image_paths[letter] = path
        return letter_image_paths

    except Exception as e:
        msg = f"Error: {e}"
        print(f"\n{msg}")
        messagebox.showerror("Error", msg)
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 - GENERATION CODE (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

letter_cache = {}


def load_and_process_letter(letter):
    """Load and preprocess a letter image"""
    letter = letter.upper()

    if letter in letter_cache:
        return letter_cache[letter]

    if letter not in letter_image_paths:
        return None

    image_path = letter_image_paths[letter]

    if not os.path.exists(image_path):
        print(f"Warning: Image not found for '{letter}' at {image_path}")
        return None

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)

    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    coords = cv2.findNonZero(img_bin)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)

    padding = max(4, int(max(w, h) * 0.1))
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img_bin.shape[1] - x, w + 2 * padding)
    h = min(img_bin.shape[0] - y, h + 2 * padding)

    char_crop = img_bin[y:y+h, x:x+w]

    max_dim = max(w, h)
    canvas_size = int(max_dim * 1.2)
    square_img = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    x_offset = (canvas_size - w) // 2
    y_offset = (canvas_size - h) // 2
    square_img[y_offset:y_offset+h, x_offset:x_offset+w] = char_crop

    resized_img = cv2.resize(square_img, (100, 100), interpolation=cv2.INTER_AREA)
    resized_img = cv2.bitwise_not(resized_img)

    letter_cache[letter] = resized_img
    return resized_img


def generate_text_image(text, letter_size=100, letter_spacing=5, line_spacing=20, margin=50):
    """Generate a document with the input text using your handwriting"""
    text = text.upper()

    lines = text.split('\n')

    processed_lines = []
    max_line_width = 0

    for line in lines:
        words = line.split(' ')
        processed_words = []

        for word in words:
            letters_in_word = []
            for char in word:
                if char.isalpha():
                    letter_img = load_and_process_letter(char)
                    if letter_img is not None:
                        letters_in_word.append(letter_img)

            if letters_in_word:
                processed_words.append(letters_in_word)

        processed_lines.append(processed_words)

        line_width = sum(len(word) * (letter_size + letter_spacing) for word in processed_words)
        line_width += max(0, len(processed_words) - 1) * (letter_size * 2)
        max_line_width = max(max_line_width, line_width)

    if max_line_width == 0:
        return None

    canvas_width = max_line_width + 2 * margin
    canvas_height = len(processed_lines) * (letter_size + line_spacing) + 2 * margin

    canvas = np.ones((canvas_height, canvas_width), dtype=np.uint8) * 255

    y_pos = margin

    for line_words in processed_lines:
        if not line_words:
            y_pos += letter_size + line_spacing
            continue

        x_pos = margin

        for word_idx, word_letters in enumerate(line_words):
            for letter_img in word_letters:
                canvas[y_pos:y_pos+letter_size, x_pos:x_pos+letter_size] = letter_img
                x_pos += letter_size + letter_spacing

            x_pos += letter_size * 2

        y_pos += letter_size + line_spacing

    return canvas


def sanitize_filename(text):
    """Create a safe filename from text"""
    text = text.replace('\n', ' ')
    text = re.sub(r'[<>:"/\\|?*]', '', text)
    text = text.strip()

    if len(text) > 30:
        text = text[:30]

    text = text.replace(' ', '_')

    if not text:
        text = "handwritten_text"

    return text


def save_as_pdf(image, filename):
    """Save image as PDF (optional - requires PIL)"""
    try:
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(image)
        pdf_filename = filename.replace('.png', '.pdf')
        pil_img.save(pdf_filename, 'PDF', resolution=100.0)
        return pdf_filename
    except Exception as e:
        print(f"Could not save as PDF: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN - Step 1: extract, Step 2: generate
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── STEP 1: Run extraction ────────────────────────────────────────────────
    print("="*70)
    print("STEP 1: EXTRACTING LETTERS FROM YOUR HANDWRITING IMAGE")
    print("="*70)

    letter_image_paths = run_extraction()

    if not letter_image_paths:
        print("No letters extracted. Exiting.")
        exit()

    # ── STEP 2: Run generator ─────────────────────────────────────────────────
    print("\n" + "="*70)
    print("HANDWRITING TEXT GENERATOR")
    print("="*70)

    print("Ready to generate text\n")

    available_letters = list(letter_image_paths.keys())
    problem_letters   = [l for l in string.ascii_uppercase if l not in letter_image_paths]

    print("This tool generates documents using your handwriting!")
    print(f"Available letters: {', '.join(sorted(available_letters))}")
    print(f"Unavailable letters: {', '.join(problem_letters)}")
    print("\nYou can:")
    print("  - Use spaces between words")
    print("  - Type multiple sentences\n")

    while True:
        print("="*70)
        text_input = input("Enter text to generate (or 'quit' to exit):\n> ").strip()

        if text_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break

        if not text_input:
            print("Please enter some text!\n")
            continue

        text_upper = text_input.upper()
        unavailable = [char for char in text_upper if char.isalpha() and char not in letter_image_paths]

        if unavailable:
            unique_unavailable = sorted(set(unavailable))
            print(f"\nWARNING: Text contains unavailable letters: {', '.join(unique_unavailable)}")
            print("These letters will be skipped in the output.\n")

            response = input("Continue anyway? (y/n): ").strip().lower()
            if response != 'y':
                print()
                continue

        print(f"\nGenerating handwritten document...")
        result_image = generate_text_image(text_input)

        if result_image is None:
            print("Failed to generate document! (No valid letters found)\n")
            continue

        safe_filename = sanitize_filename(text_input)
        output_filename = f"handwritten_{safe_filename}.png"

        try:
            cv2.imwrite(output_filename, result_image)
            print(f"Saved as '{output_filename}'")
        except Exception as e:
            print(f"Failed to save: {e}")
            continue

        plt.figure(figsize=(16, 10))
        plt.imshow(result_image, cmap='gray', vmin=0, vmax=255)
        plt.title(f'Generated Handwriting: "{text_input}"', fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        pdf_name = save_as_pdf(result_image, output_filename)
        if pdf_name:
            print(f"Also saved as '{pdf_name}'")

        print()

    print("\n" + "="*70)
    print("TEXT GENERATOR CLOSED")
    print("="*70)