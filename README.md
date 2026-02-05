# 🤖 AutoGrade Agent

**AutoGrade Agent** is a computer vision pipeline designed to automate the grading of multiple-choice exams. It acts as an autonomous agent that ingests raw images of answer keys ("Gabaritos") and student response sheets, processes them using Optical Character Recognition (OCR) and Optical Mark Recognition (OMR), and outputs structured grading reports.

## 🧠 Capabilities

* **Dynamic Key Extraction:** Uses Tesseract OCR to read the correct answers directly from a teacher's "Gabarito" image (no hardcoding required).
* **Optical Mark Recognition (OMR):** Detects bubbles, identifies column layouts, and determines filled answers based on pixel density.
* **Batch Processing:** Iterates through multiple class directories (e.g., `6o`, `7oA`), identifying the specific key for that class and grading all students within it.
* **Result Aggregation:** Generates individual CSV reports for each class containing detailed item analysis and total scores.

## 📂 Directory Structure

The agent expects the following strict directory hierarchy to function correctly:

```text
project_root/
│
├── auto_grader_multi.py    # The main agent script
├── results_6o.csv          # (Generated Output)
├── results_7oA.csv         # (Generated Output)
│
└── STUDENT_EXAMS/          # Root input directory
    ├── 6o/                 # Class Subfolder
    │   ├── gabarito.jpg    # MUST contain "gabarito" in filename
    │   ├── Mariana.jpg     # Rename file to Student Name
    │   └── Joao.jpg
    │
    ├── 7oA/                # Class Subfolder
    │   ├── Gabarito_7A.jpg
    │   └── Pedro.jpg
    │
    └── ...

```

## 🛠️ Prerequisites & Installation

### 1. System Dependencies

You must install the **Tesseract OCR engine** separately from Python packages.

* **Windows:** [Download Installer](https://www.google.com/search?q=https://github.com/UB-Mannheim/tesseract/wiki) (Note the path, e.g., `C:\Program Files\Tesseract-OCR\tesseract.exe`)
* **Linux:** `sudo apt-get install tesseract-ocr`
* **MacOS:** `brew install tesseract`

### 2. Python Environment

Install the required libraries:

```bash
pip install opencv-python numpy pandas pytesseract

```

### 3. Agent Configuration

If you are on Windows, open `auto_grader_multi.py` and uncomment/update the Tesseract path:

```python
# auto_grader_multi.py line 10
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

```

## 🚀 Usage

1. **Preprocessing:** Ensure images are rotated correctly (text upright) and have reasonable luminosity. Rename student images to `{StudentName}.jpg`.
2. **Run the Agent:**

```bash
python auto_grader_multi.py

```

3. **Review Output:** Check the generated `results_{class}.csv` files in the root directory.

## ⚙️ Fine-Tuning The Agent

Computer vision relies on thresholds. If the agent is missing bubbles or reading noise, tweak these constants in `auto_grader_multi.py`:

| Parameter | Line (Approx) | Description |
| --- | --- | --- |
| **Binarization Threshold** | `cv2.threshold(..., 170, ...)` | Controls sensitivity to darkness. Lower (150) for dark scans, Higher (190) for bright photos. |
| **Bubble Size** | `if w >= 20...` | Min/Max size of a bubble in pixels. Increase if detecting dust/noise; decrease if missing small bubbles. |
| **Filled Threshold** | `if max_pixels > 100` | Minimum "ink" required to count a bubble as marked. |
| **Column Gap** | `if curr[0] - prev[0] > 50` | Horizontal distance (px) required to detect a new column of questions. |

## ⚠️ Known Limitations

1. **Lighting:** The OMR logic assumes relatively even lighting. Shadows across the paper can cause false positives.
2. **Alignment:** The agent is robust to minor skew, but extreme rotation (>5 degrees) requires external preprocessing.
3. **OCR Accuracy:** Handwritten "Gabaritos" are difficult to read. Typed answer keys are recommended for 100% accuracy.