## Project Description

This repository contains the Final Year Project of Group 15: **Personalized Handwriting Generation and Verification Software**.

The system allows users to create a personalized handwriting profile by providing handwritten samples of the letters A–Z. Using these samples, the software can generate digital text that closely replicates the user's handwriting style.

In addition, the project includes a handwriting verification module that compares generated handwriting with another handwriting sample to determine the likelihood that both were written by the same person. This feature can help detect potential fraud, forgery, or handwriting imitation attempts.


## Prerequisites

### 1. Install Visual Studio Code (Recommended)

Download and install **Visual Studio Code (VS Code)**:

https://code.visualstudio.com/

Open the project folder in VS Code and use the integrated terminal to run all commands.

---

### 2. Install Python

Install **Python 3.10 or newer**:

https://www.python.org/downloads/

Verify the installation:

```bash
python --version
```

---

### 3. Install Required Python Libraries

Run the following commands in the VS Code terminal:

```bash
pip install tensorflow
pip install opencv-python
pip install numpy
pip install matplotlib
pip install pillow
pip install scipy
pip install scikit-image
pip install pymupdf
```

---

### 4. Tkinter

Tkinter is included with Python on Windows and does not require a separate installation.

---

After completing the above steps, the project is ready to run on your local machine.

Just run combined(in&out).py  to give user input and generate the handwritting.

Run pdf_to_img_converter.py to convert the generated handwritting pdf to img for verification.

Run Verification.py to verify two handwrittings.

The app for this project is under development and will be available as soon as it is ready.
