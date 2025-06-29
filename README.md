# Realtime Vehicle Re‑Identification with LFFT

This repository implements a realtime vehicle re‑identification system using a custom Vision Transformer architecture (LFFT) with local feature enhancement and jigsaw select patches. It captures live camera frames, extracts deep features, and either saves new vehicle embeddings or matches against known vehicles.

---

## Table of Contents

1. [Features](#features)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Directory Structure](#directory-structure)
5. [Configuration](#configuration)
6. [Running the Application](#running-the-application)
7. [Usage](#usage)
8. [Utility Scripts](#utility-scripts)
9. [Troubleshooting](#troubleshooting)
10. [License](#license)

---

## Features

* **Custom Transformer (LFFT)**: Frequency layer + attention + JSPM for robust feature extraction.
* **Realtime Processing**: Capture frames from webcam, extract embeddings on GPU/CPU.
* **Interactive Controls**:

  * `a`: Save current frame embedding as a new vehicle reference.
  * `s`: Search and display the closest match among saved embeddings.
  * `q`: Quit application and release camera.
* **Persistence**: Embeddings saved in `saved_features/` as `.pt` files.

---

## Requirements

* Python 3.10+ or Python 3.11
* CUDA drivers (if using GPU acceleration)

Python packages:

```
torch>=2.0
transformers==4.53.0  # for HuggingFace ViT model
opencv-python>=4.5
Pillow
numpy
msvcrt (built-in on Windows)
```

Install via:

```
pip install torch transformers opencv-python pillow numpy
```

---

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/vehicle-reid-realtime.git
   cd vehicle-reid-realtime
   ```
2. **(Optional) Create & activate a virtual environment**:

   ```bash
   python -m venv venv
   venv\Scripts\activate  # on Windows
   source venv/bin/activate  # on Linux/macOS
   ```
3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

## Directory Structure

```
├── model/                # Model definitions
│   └── lfft.py           # LFFT architecture class
│
├── utils/                # Utility functions
│   └── feature_extractor.py  # extract_feature, cosine_sim
│
├── saved_features/       # Saved `.pt` vehicle embeddings
│   └── vehicle_001.pt
│   └── vehicle_002.pt
│
├── main.py               # Entrypoint script for realtime RE-ID
└── README.md             # This documentation
```

---

## Configuration

* **SAVE\_DIR**: Folder for storing embeddings (`saved_features/` by default).
* **Camera Index**: In `main.py` the line `cv2.VideoCapture(0)` can be changed if you have multiple cameras.

---

## Running the Application

Launch the realtime RE‑ID loop:

```bash
python main.py
```

On startup, you should see:

```
Model yüklendi.
✅ Kamera açıldı.
   'a' → Yeni araç kaydet
   's' → Mevcutlara karşı kontrol et
   'q' → Çıkış
```

Then, press keys in the console:

* **a**: Save the current frame as a new vehicle embedding.
* **s**: Search among saved embeddings and print the best match + similarity score.
* **q**: Quit and release the camera.

---

## Usage Examples

1. **Save a new vehicle**:

   * Hold the target vehicle in view.
   * Switch to console and press `a`.
   * You’ll see: `\[A\] Yeni araç kaydedildi: vehicle_003`.
2. **Search for a vehicle**:

   * Point camera at a previously saved vehicle.
   * Press `s`.
   * You’ll see: `\[S\] En iyi eşleşme → vehicle_001 (0.85)`.
3. **Exit**:

   * Press `q` → `🛑 Çıkılıyor...` → Camera closed.

---

## Utility Scripts

You can inspect saved embeddings:

```python
import torch
embed = torch.load("saved_features/vehicle_001.pt")
print(embed.shape)  # (1, feature_dim)
```

Compute cosine similarity manually:

```python
from utils.feature_extractor import cosine_sim
sim = cosine_sim(embed1, embed2)
```

---

## Troubleshooting

* **Camera not opening**: Ensure no other application uses the webcam. Try `cv2.VideoCapture(1)` if multiple cameras.
* **Feature extraction slow**: Verify GPU is used (`torch.cuda.is_available()`).
* **Module import errors**: Check your `PYTHONPATH` or run from repository root.

---

## License

MIT © Muharrem Şimşek
