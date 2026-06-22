import fitz  # PyMuPDF
import os
from tkinter import Tk, filedialog

# Hide main Tkinter window
root = Tk()
root.withdraw()

# Open file picker
pdf_path = filedialog.askopenfilename(
    title="Select a PDF file",
    filetypes=[("PDF Files", "*.pdf")]
)

if not pdf_path:
    print("No file selected.")
    exit()

# Create output folder
output_folder = os.path.join(os.path.dirname(pdf_path), "pdf_images")
os.makedirs(output_folder, exist_ok=True)

# Open PDF
pdf = fitz.open(pdf_path)

# Convert pages to JPG
for page_num in range(len(pdf)):
    page = pdf[page_num]
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))

    output_path = os.path.join(output_folder, f"page_{page_num + 1}.jpg")
    pix.save(output_path)

    print(f"Saved: {output_path}")

pdf.close()

print("\nConversion completed!")
print(f"Images saved in: {output_folder}")