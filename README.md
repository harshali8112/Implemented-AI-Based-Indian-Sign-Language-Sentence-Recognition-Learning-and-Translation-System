# 🤟 ISL Sentence Translator: Real-Time Indian Sign Language Recognition

**A real-time Indian Sign Language (ISL) translator that recognizes 46 conversational sentences from a live webcam and speaks them out loud in the browser.**

## Overview

Communication between sign language users and non-signers is hard. This project reads body pose and hand movements from a webcam, predicts which sentence is being signed, shows it on screen, and reads it aloud with browser text-to-speech.

Instead of classifying raw video frames, the model works on **skeleton keypoints** (body pose and both hands) extracted with MediaPipe Holistic. This makes it less sensitive to lighting, background, clothing and camera distance than a raw-pixel approach.

## Features

- 🎥 **Live webcam recognition** of 46 ISL sentences
- 🧍 **Pose and hand landmarks** from MediaPipe Holistic, normalized to be independent of camera distance
- 🧠 **1D-CNN + Bi-LSTM** sequence model built in PyTorch
- ⚡ **Real-time prediction** over a rolling 35-frame window, with motion detection, confidence thresholds, prediction smoothing and a cooldown to avoid repeated outputs
- 🔊 **Browser text-to-speech** that speaks each newly recognized sentence once
- 🌐 **Flask web app** with live MJPEG video streaming and a sentence history
- 🛠️ **Full pipeline included:** recording, augmentation, keypoint extraction, training, evaluation and diagram generation

## How It Works

```text
Webcam / video
   → MediaPipe Holistic (pose + both hands)
   → 235 normalized features per frame (nose-centred, scaled by shoulder width, plus finger-flexion angles)
   → 35-frame sequence + frame-to-frame velocity = 470 features per frame
   → 1D-CNN (local patterns) → 2-layer Bi-LSTM (motion context) → classifier (46 classes)
   → Flask app → on-screen sentence + spoken output
```

| Stage | Details |
|---|---|
| Landmarks | 33 pose points + 21 points per hand |
| Features per frame | 235 position features (hands 63 + 63, pose 99, finger angles 5 + 5), plus 235 velocity features |
| Sequence length | 35 frames, so the input shape is `(35, 470)` |
| Model | Conv1D (256 → 128 channels, BatchNorm, Dropout) → Bi-LSTM (128 hidden, 2 layers) → fully connected head |
| Training | AdamW, label smoothing 0.1, gradient clipping, early stopping, 85/15 stratified split, fixed seed |
| Live decoding | Motion-energy gate, softmax averaged over the last 6 predictions, confidence threshold 0.86 (0.91 for one similar-looking sentence), 1.5 s cooldown |

## Results and Limitations

| Evaluation | Result |
|---|---|
| Held-out 15% stratified split (138 sequences, 46 classes) | 100% |
| Full-dataset diagnostic (920 sequences) | 99.89% (919 / 920) |

**Please read these numbers with care.** The dataset was recorded by a single signer: each sentence has one real webcam recording plus 19 generated variants (slower and faster speed, horizontal shifts, zoom, and brightness/contrast changes). Because variants of the same recording appear in both the training and test split, and the full-dataset score includes training data, these figures show that the model is consistent on this vocabulary. They are **not** a measure of accuracy on new signers. Testing with several signers and measuring real-time latency on CPU are the next steps.

## Tech Stack

| Area | Technology |
|---|---|
| Landmark detection | MediaPipe Holistic |
| Model | PyTorch (1D-CNN + Bi-LSTM) |
| Backend | Python, Flask, OpenCV |
| Frontend | HTML, CSS, JavaScript (browser Text-to-Speech) |
| Evaluation | scikit-learn, Matplotlib |

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/harshali8112/Implemented-AI-Based-Indian-Sign-Language-Sentence-Recognition-Learning-and-Translation-System.git
cd Implemented-AI-Based-Indian-Sign-Language-Sentence-Recognition-Learning-and-Translation-System

# 2. Create and activate a virtual environment (Python 3.9 - 3.12, required by MediaPipe 0.10.21)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the web app
python app.py
```

Open `http://127.0.0.1:5000` and allow webcam access. The trained model (`isl_keypoint_model.pth`) is included, so no training is needed to try the app.

Pages: `/` (home), `/translator` (live translator), `/feedback`.

A webcam-only demo without the web app is also available:

```bash
python live_inference.py
```

## Rebuild the Pipeline From Scratch

The training videos and extracted keypoints are not stored in this repository because of their size and because they contain video of a person.

```bash
# 1. Record clips for a sentence (saved under custom_dataset/<sentence>/)
python recorder.py --sentence "thank you" --count 5

#    ...or record one clip and auto-generate 19 variants
python record_one_and_generate_variants.py

# 2. Convert videos to (35, 470) keypoint sequences
python videos_to_keypoints.py --videos custom_dataset --output keypoint_sequences

# 3. Train the model
python train_keypoint_model.py --data keypoint_sequences --epochs 100

# 4. Evaluate and plot confusion matrices
python evaluate_confusion.py
```

## Project Structure

```text
├── app.py                              # Flask server + real-time inference worker
├── live_inference.py                   # Standalone webcam inference
├── keypoint_features.py                # MediaPipe feature extraction and preprocessing
├── train_keypoint_model.py             # Model definition and training loop
├── evaluate_confusion.py               # Evaluation and confusion matrix plots
├── recorder.py                         # Record webcam clips per sentence
├── record_one_and_generate_variants.py # 1 recorded clip + 19 generated variants
├── augment_dataset_from_one_clip.py    # Extra clips from one reference video
├── videos_to_keypoints.py              # Videos → (35, 470) .npy sequences
├── generate_project_diagrams.py        # Architecture and pipeline figures
├── isl_keypoint_model.pth              # Trained model checkpoint
├── requirements.txt
├── templates/                          # Flask HTML pages
├── static/                             # CSS and JavaScript
└── docs/
    └── PROJECT_MASTER_DOCUMENTATION.md # Detailed technical documentation
```

<details>
<summary><b>Supported sentences (46)</b></summary>

are you free today · are you hiding something · bring water for me · can you repeat that please · congratulations · do me a favour · do you need something · go and sleep · had your food · he she is my friend · help me · how are you · i am afraid of that · i am feeling cold · i am having fever · i am hungry · i am tired · i am very happy · i appreciate it · i got hurt · i need water · i promise · my name is · shall we go outside · speak softly · take care of yourself · tell the truth · thank you · this is so kind of you · this place is beautiful · turn on light · we are all with you · what are you doing · what do you think · what do you want to become · what happened · what you want · where are you from · who are you · why are you crying · you are bad · you are good · you are welcome · you can do it · you comb your hair · you need medicine, take this

</details>

## Future Improvements

- Record more signers to measure accuracy on people the model has not seen
- Measure real-time latency (FPS) on a CPU
- Add more sentences and a learning mode for beginners

---

👩‍💻 Built by [Harshali](https://github.com/harshali8112) · 📫 info.harshali268@gmail.com
