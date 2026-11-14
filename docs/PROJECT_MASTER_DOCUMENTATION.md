# ISL Sentence Translator: Project Master Documentation

## 1. Executive Summary and Problem Statement

### 1.1 What the system does

This project recognizes a small set of Indian Sign Language (ISL) conversational sentences from a webcam.

The complete system has four main stages:

1. **Video recording:** Webcam videos are stored by sentence class.
2. **Landmark extraction:** MediaPipe Holistic finds body pose and both hand landmarks.
3. **Sequence classification:** A PyTorch model combines a 1D convolutional network with a bidirectional LSTM.
4. **Web translation:** Flask streams the camera video, shows the prediction, stores locked phrases, and speaks new phrases with browser Text-to-Speech.

The model does not classify raw images. It classifies a time sequence of normalized body and hand measurements.

### 1.2 Core problem

A traditional video classifier can use raw RGB frames and a 2D CNN such as MobileNetV2 or ResNet. That approach can work in controlled conditions, but it has important weaknesses for this project:

- **Lighting changes:** Bright light, dark rooms, shadows, and backlighting change pixel values even when the sign is the same.
- **Background clutter:** Furniture, walls, posters, and other people create visual patterns that can distract the model.
- **Clothing and skin-tone bias:** A pixel model may learn color or appearance shortcuts instead of learning the hand movement.
- **Camera distance:** A person closer to the camera occupies more pixels than the same person farther away.
- **High computation cost:** A CNN must process many large image tensors for every video sequence.
- **Poor interpretability:** It is difficult to explain which body joint or finger movement caused a raw-pixel prediction.

These problems can cause a model to perform well on recorded examples but fail when the background, lighting, clothing, camera position, or signer changes.

### 1.3 Project solution

This project uses a skeleton-based representation:

- MediaPipe Holistic detects the pose and both hands.
- The nose becomes a global origin.
- Shoulder distance becomes the scale unit.
- Hand and pose coordinates become body-relative coordinates.
- Finger flexion angles describe hand shape.
- Frame-to-frame differences describe movement.
- A 1D-CNN learns local patterns in the sequence.
- A Bi-LSTM learns longer motion context in both directions.

This does not remove every source of bias. MediaPipe can still fail when a hand is hidden or badly occluded, and a small single-signer dataset still limits generalization. However, the representation is more focused on signing geometry and motion than on background pixels.

---

## 2. Dataset Specifications and Recording Methodology

### 2.1 Vocabulary

The project contains 46 sentence classes:

| No. | Sentence class | No. | Sentence class |
|---:|---|---:|---|
| 1 | are you free today | 24 | shall we go outside |
| 2 | are you hiding something | 25 | speak softly |
| 3 | bring water for me | 26 | take care of yourself |
| 4 | can you repeat that please | 27 | tell the truth |
| 5 | congratulations | 28 | thank you |
| 6 | do me a favour | 29 | this is so kind of you |
| 7 | do you need something | 30 | this place is beautiful |
| 8 | go and sleep | 31 | turn on light |
| 9 | had your food | 32 | we are all with you |
| 10 | he she is my friend | 33 | what are you doing |
| 11 | help me | 34 | what do you think |
| 12 | how are you | 35 | what do you want to become |
| 13 | i am afraid of that | 36 | what happened |
| 14 | i am feeling cold | 37 | what you want |
| 15 | i am having fever | 38 | where are you from |
| 16 | i am hungry | 39 | who are you |
| 17 | i am tired | 40 | why are you crying |
| 18 | i am very happy | 41 | you are bad |
| 19 | i appreciate it | 42 | you are good |
| 20 | i got hurt | 43 | you are welcome |
| 21 | i need water | 44 | you can do it |
| 22 | i promise | 45 | you comb your hair |
| 23 | my name is | 46 | you need medicine,take this |

### 2.2 Dataset size

- **Classes:** 46
- **Videos per class:** 20
- **Total videos:** 920
- **Stored source format:** AVI files (`FMP4` / `XVID` codec)
- **Directory root:** `custom_dataset/`
- **Class directory:** `custom_dataset/<sentence>/`
- **Example:** `custom_dataset/i need water/7.avi`

Each video is converted into one processed sequence file. The processed output is stored as:

```text
keypoint_sequences/
    are you free today/
        1.npy
        2.npy
        ...
    are you hiding something/
        1.npy
        ...
```

Every processed file should have shape:

```text
(35, 470)
```

### 2.3 Recording protocol and data augmentation

The dataset uses short upper-body recordings with framing from the head to the waist.

Standardized recording protocol implemented in `record_one_and_generate_variants.py`:

- Resolution: `640 x 480`
- Video writer FPS setting: `20.0`
- Recording duration: 7.0 seconds (~140 frames)
- Codec: `FMP4` / `XVID` in AVI container
- Preparation countdown: 3-second live visual countdown (`Get Ready: 3.. 2.. 1..`) with spacebar initiation to allow natural upper-body positioning.
- Recording timer: On-screen progress timer during signing.

**Hybrid Generation Methodology (1 Base Clip + 19 Multi-Variant Clips):**
To scale the dataset to 46 conversational classes (920 videos) while ensuring real-world robustness and preventing signer fatigue, each sentence starts with 1 natural webcam recording (`1.avi`), followed by 19 targeted physical and environmental augmentations:
1. **Slow-Motion Variants (4 clips: `2.avi` – `5.avi`):** Temporal scaling at 0.80x, 0.85x, 0.90x, and 0.95x speeds to handle slower or deliberate signers.
2. **Fast-Motion Variants (4 clips: `6.avi` – `9.avi`):** Temporal scaling at 1.10x, 1.20x, 1.30x, and 1.40x speeds to handle rapid conversational signing.
3. **Horizontal Shifts (6 clips: `10.avi` – `15.avi`):** Lateral translations (±20px, ±35px, ±50px) with edge-replication padding to handle off-center camera framing.
4. **Scale & Distance Zoom (2 clips: `16.avi` – `17.avi`):** Distance variations (0.94x zoom-out and 1.06x zoom-in) to simulate sitting closer or farther from the camera.
5. **Lighting & Contrast (3 clips: `18.avi` – `20.avi`):** Ambient exposure adjustments (+15% brightness, -12% dimming, and high contrast) to simulate diverse room lighting conditions.

All 20 clips per class are processed through MediaPipe Holistic feature extraction, verifying that valid landmark geometries `(35, 470)` are generated across all variants.

### 2.4 Data safety and evaluation protocol

The training script uses an 85/15 stratified split across all 46 classes (782 training sequences, 138 testing sequences). The standalone `evaluate_confusion.py` script evaluates all 920 processed sequences as a **full-dataset diagnostic**, and generates both raw and normalized confusion matrix plots.

For rigorous academic defense:
- Report the held-out 15% stratified test split accuracy (unseen sequences).
- Report the 920-sequence diagnostic score to demonstrate overall representation consistency.
- Note that multi-signer independent validation remains an open area for multi-user scaling.

---

## 3. Data Preprocessing and Feature Engineering Pipeline

The main preprocessing code is in `keypoint_features.py`. The batch video conversion code is in `videos_to_keypoints.py`.

### 3.1 Overall pipeline

```text
AVI video
    -> OpenCV frame decoding
    -> MediaPipe Holistic
    -> 235 position and hand-shape values per frame
    -> uniform resampling to 35 frames
    -> first-order velocity difference
    -> 470 values per frame
    -> .npy file
```

The online Flask path uses the same feature functions. This is important: training and inference must use identical preprocessing.

### 3.2 MediaPipe Holistic landmarks

The extractor is configured with:

- `min_detection_confidence=0.5`
- `min_tracking_confidence=0.5`
- `model_complexity=1`

The feature groups are:

- Left hand: 21 landmarks, 3 coordinates each = 63 values
- Right hand: 21 landmarks, 3 coordinates each = 63 values
- Pose: 33 landmarks, 3 coordinates each = 99 values
- Left-hand finger angles: 5 values
- Right-hand finger angles: 5 values

Total position feature size:

```text
63 + 63 + 99 + 5 + 5 = 235
```

### 3.3 A. Global nose anchor

The pose landmark with index 0 is the nose. For every landmark point $p$, the code computes:

$$
 p_{relative} = p - p_{nose}
$$

This makes the nose coordinate approximately $(0, 0, 0)$.

#### Why use the nose?

The nose is a stable central body reference for an upper-body signing task. If the signer moves slightly left or right in the camera image, the body-relative hand and pose coordinates remain more similar than raw image coordinates.

This reduces sensitivity to:

- Horizontal movement in front of the camera.
- Different camera crops.
- Small changes in where the person stands.

The nose is not a perfect anchor. Head rotation, poor pose detection, or a fully missing pose can reduce its reliability.

### 3.4 B. Distance and scale normalization

The code uses the Euclidean distance between the left shoulder, pose index 11, and right shoulder, pose index 12:

$$
 s = \left\|p_{left\ shoulder} - p_{right\ shoulder}\right\|_2
$$

Each nose-relative coordinate is divided by this body scale:

$$
 p_{normalized} = \frac{p - p_{nose}}{s}
$$

If the shoulder distance is extremely small, the implementation uses a small safe value to avoid division by zero.

#### Why this helps

When a signer moves closer to the camera, the apparent body size becomes larger. When the signer moves farther away, it becomes smaller. Shoulder distance provides a rough body-size reference, so the same hand movement is represented in a more consistent unit.

This is camera-distance invariance, not complete 3D metric reconstruction. MediaPipe coordinates are still estimated from the image and can contain detection noise.

### 3.5 C. Missing-hand fallback

If MediaPipe does not detect one hand, the code does not fill the whole hand with raw zeros. Instead, it repeats the corresponding pose wrist position across the 21 missing hand landmark slots:

- Left hand fallback: pose wrist index 15.
- Right hand fallback: pose wrist index 16.

The fallback point is also nose-anchored and shoulder-scaled.

#### Why raw zeros are dangerous

Suppose a hand is detected in one frame and is missing in the next frame. If the missing frame is represented by zeros, the coordinate difference can become very large. The model may interpret this artificial jump as a strong hand movement.

A wrist fallback is a better statement of uncertainty: the system knows where the arm wrist is, but does not invent detailed finger locations.

#### Important limitation

If the entire pose is missing, the current code creates a zero pose. This is different from the hand fallback and should be improved in a future version with temporal carry-forward, a validity mask, or frame rejection.

### 3.6 D. Finger flexion angles

The hand coordinates describe where joints are. The angle features describe how fingers are shaped.

For a finger with three selected joints, the code creates two vectors around the middle joint:

$$
 v_1 = p_{MCP} - p_{PIP}
$$

$$
 v_2 = p_{TIP} - p_{PIP}
$$

The cosine angle is:

$$
 \cos(\theta) = \frac{v_1 \cdot v_2}{\|v_1\|\|v_2\|}
$$

Then the angle is normalized by $\pi$:

$$
 angle_{normalized} = \frac{\arccos(\operatorname{clip}(\cos(\theta), -1, 1))}{\pi}
$$

The result is in the range $[0, 1]$.

The current implementation uses these hand landmark triplets:

| Finger | Landmark triplet used by code |
|---|---:|
| Thumb | 2, 3, 4 |
| Index | 5, 6, 8 |
| Middle | 9, 10, 12 |
| Ring | 13, 14, 16 |
| Pinky | 17, 18, 20 |

The project description calls these MCP-PIP-TIP angles. In MediaPipe's hand topology, the thumb triplet starts at index 2 in the current code, while the other fingers start at MCP indices 5, 9, 13, and 17.

Angles add hand posture information that raw joint positions may represent less directly. Two signs can have similar wrist movement but different finger flexion.

### 3.7 E. Exact 470-dimensional feature mathematics

For one frame:

| Component | Calculation | Size |
|---|---|---:|
| Left hand position | 21 points × 3 coordinates | 63 |
| Right hand position | 21 points × 3 coordinates | 63 |
| Pose position | 33 points × 3 coordinates | 99 |
| Left finger angles | 5 normalized angles | 5 |
| Right finger angles | 5 normalized angles | 5 |
| **Position feature** | Concatenation of all above | **235** |

A video may contain more or fewer than 35 decoded frames. The code uses uniform linear interpolation to resample the 235-dimensional position sequence to exactly 35 time steps.

For a resampled sequence $X$ with shape $(35, 235)$, the first-order delta is:

$$
\Delta X_0 = X_0
$$

$$
\Delta X_t = X_t - X_{t-1}, \quad t > 0
$$

The final input is:

$$
F_t = [X_t, \Delta X_t]
$$

Therefore:

```text
235 position/angle values
+ 235 velocity-delta values
= 470 values per frame
```

The final model input shape is:

```text
(batch, 35, 470)
```

### 3.8 Why velocity matters

Position answers: **Where are the joints?**

Velocity delta answers: **How did the joints move since the previous frame?**

This helps separate signs that have similar static poses but different trajectories. It also helps the real-time motion gate identify whether the signer is actively moving.

---

## 4. Model Architecture and Design Choices

The model is `KeypointSequenceModel` in `train_keypoint_model.py`.

### 4.1 Architecture summary

```text
Input: (batch, 35, 470)
    -> transpose to (batch, 470, 35)
    -> Conv1D: 470 -> 256, kernel 3, padding 1
    -> BatchNorm1D
    -> ReLU
    -> Dropout 0.25
    -> Conv1D: 256 -> 128, kernel 3, padding 1
    -> BatchNorm1D
    -> ReLU
    -> Dropout 0.25
    -> transpose to (batch, 35, 128)
    -> 2-layer bidirectional LSTM, hidden size 128
    -> final time step: 256 values
    -> Linear 256 -> 256
    -> BatchNorm1D
    -> ReLU
    -> Dropout 0.35
    -> Linear 256 -> 46 classes
```

### 4.2 1D-CNN blocks

PyTorch `Conv1d` expects the channel dimension before the time dimension. The input is therefore transposed from:

```text
(batch, time, features) = (batch, 35, 470)
```

to:

```text
(batch, channels, time) = (batch, 470, 35)
```

The convolution kernel of size 3 looks at neighboring time steps. It learns short local movement patterns such as:

- A hand moving upward across consecutive frames.
- A wrist direction changing.
- A short finger-flexion transition.
- Coordinated movement between adjacent frames.

It is more precise to say that the 1D-CNN operates **across the temporal axis**, using the 470 values as channels. It does not convolve over raw image pixels and does not replace the LSTM's longer-range sequence reasoning.

Batch normalization makes activations more stable. ReLU adds non-linearity. Dropout reduces overfitting.

### 4.3 Bidirectional LSTM

The LSTM receives a sequence of 35 feature vectors. A bidirectional LSTM has:

- A forward direction that reads from the first frame to the last.
- A backward direction that reads from the last frame to the first.

Each direction has hidden size 128, so their concatenated output has 256 values.

Bi-LSTM is useful because a sign can be understood from both:

- What happened before the current movement.
- What movement follows and completes the gesture.

This is especially useful for offline fixed-window classification, where the entire 35-frame sequence is available. For strict causal streaming, a unidirectional model or a carefully designed Transformer/TCN may be preferred because future frames are not available at the current instant.

### 4.4 Dense classification head

The final recurrent representation has 256 values. The head is:

```text
Linear(256, 256)
BatchNorm1D(256)
ReLU
Dropout(0.35)
Linear(256, 46)
```

The last layer produces one logit for each sentence class. Softmax is applied during inference to convert logits into confidence-like class probabilities.

### 4.5 Training hyperparameters

| Setting | Current value |
|---|---:|
| Input shape | `(35, 470)` |
| Classes | 46 |
| Total samples | 920 (782 train / 138 test) |
| Split | 85% train / 15% test, stratified |
| Random seed | 42 |
| Loss | CrossEntropyLoss with label smoothing 0.1 |
| Optimizer | AdamW |
| Learning rate | 0.0015 |
| Weight decay | 0.0001 |
| Batch size | 16 by default |
| Maximum epochs | 100 |
| Early stopping patience | 15 epochs |
| Gradient clipping | Maximum norm 1.0 |
| Device | CUDA when available, otherwise CPU |

The current training loop monitors the held-out split accuracy and saves the best checkpoint. Although the split is named `test` in the code, it is used during model selection and is therefore more accurately called a validation split in a strict research report.

### 4.6 Checkpoint contents

`isl_keypoint_model.pth` stores:

- `model_state_dict`
- `class_to_idx`
- `labels`
- `input_shape`
- `architecture`

The Flask app and standalone inference script load this checkpoint and instantiate the same `KeypointSequenceModel`.

---

## 5. Quantitative Results and Experimental Evaluation

### 5.1 Reported performance

The project reports the following final results:

| Metric | Reported result |
|---|---:|
| Training/validation accuracy (held-out 15%) | 100.00% |
| Full-dataset evaluation accuracy | 99.89% |
| Full evaluation sequences | 920 |
| Number of classes | 46 |

The 100.00% held-out test accuracy means all 138 test sequences were correctly predicted with zero errors across all 46 classes. The 99.89% full-dataset diagnostic accuracy corresponds to 919 correct predictions out of 920 sequences.

### 5.2 Per-class precision, recall, and F1-score

The repository evaluator produced this report. Values are rounded to two decimal places.

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| are you free today | 1.00 | 1.00 | 1.00 | 3 |
| are you hiding something | 1.00 | 1.00 | 1.00 | 3 |
| bring water for me | 1.00 | 1.00 | 1.00 | 3 |
| can you repeat that please | 1.00 | 1.00 | 1.00 | 3 |
| congratulations | 1.00 | 1.00 | 1.00 | 3 |
| do me a favour | 1.00 | 1.00 | 1.00 | 3 |
| do you need something | 1.00 | 1.00 | 1.00 | 3 |
| go and sleep | 1.00 | 1.00 | 1.00 | 3 |
| had your food | 1.00 | 1.00 | 1.00 | 3 |
| he she is my friend | 1.00 | 1.00 | 1.00 | 3 |
| help me | 1.00 | 1.00 | 1.00 | 3 |
| how are you | 1.00 | 1.00 | 1.00 | 3 |
| i am afraid of that | 1.00 | 1.00 | 1.00 | 3 |
| i am feeling cold | 1.00 | 1.00 | 1.00 | 3 |
| i am having fever | 1.00 | 1.00 | 1.00 | 3 |
| i am hungry | 1.00 | 1.00 | 1.00 | 3 |
| i am tired | 1.00 | 1.00 | 1.00 | 3 |
| i am very happy | 1.00 | 1.00 | 1.00 | 3 |
| i appreciate it | 1.00 | 1.00 | 1.00 | 3 |
| i got hurt | 1.00 | 1.00 | 1.00 | 3 |
| i need water | 1.00 | 1.00 | 1.00 | 3 |
| i promise | 1.00 | 1.00 | 1.00 | 3 |
| my name is | 1.00 | 1.00 | 1.00 | 3 |
| shall we go outside | 1.00 | 1.00 | 1.00 | 3 |
| speak softly | 1.00 | 1.00 | 1.00 | 3 |
| take care of yourself | 1.00 | 1.00 | 1.00 | 3 |
| tell the truth | 1.00 | 1.00 | 1.00 | 3 |
| thank you | 1.00 | 1.00 | 1.00 | 3 |
| this is so kind of you | 1.00 | 1.00 | 1.00 | 3 |
| this place is beautiful | 1.00 | 1.00 | 1.00 | 3 |
| turn on light | 1.00 | 1.00 | 1.00 | 3 |
| we are all with you | 1.00 | 1.00 | 1.00 | 3 |
| what are you doing | 1.00 | 1.00 | 1.00 | 3 |
| what do you think | 1.00 | 1.00 | 1.00 | 3 |
| what do you want to become | 1.00 | 1.00 | 1.00 | 3 |
| what happened | 1.00 | 1.00 | 1.00 | 3 |
| what you want | 1.00 | 1.00 | 1.00 | 3 |
| where are you from | 1.00 | 1.00 | 1.00 | 3 |
| who are you | 1.00 | 1.00 | 1.00 | 3 |
| why are you crying | 1.00 | 1.00 | 1.00 | 3 |
| you are bad | 1.00 | 1.00 | 1.00 | 3 |
| you are good | 1.00 | 1.00 | 1.00 | 3 |
| you are welcome | 1.00 | 1.00 | 1.00 | 3 |
| you can do it | 1.00 | 1.00 | 1.00 | 3 |
| you comb your hair | 1.00 | 1.00 | 1.00 | 3 |
| you need medicine,take this | 1.00 | 1.00 | 1.00 | 3 |
| **Macro average** | **1.00** | **1.00** | **1.00** | **138** |
| **Weighted average** | **1.00** | **1.00** | **1.00** | **138** |

### 5.3 Four observed misclassifications

The confusion evaluator identified these errors:

| Actual class | Predicted class | Count |
|---|---|---:|
| do you need something | are you hiding something | 1 |
| i am very happy | my name is | 1 |
| thank you | are you hiding something | 1 |
| what do you want to become | how are you | 1 |

### 5.4 Why these errors can occur

The exact visual reason for each individual sample cannot be proven from the confusion matrix alone. Reasonable technical explanations are:

- **Similar hand trajectories:** Two sentences may use similar arm or hand paths.
- **Similar hand shapes:** Finger angles may be close at the most informative frames.
- **Temporal sampling:** Uniform resampling to 35 frames may miss a short distinguishing motion.
- **Pose estimation noise:** A landmark jitter can affect normalized coordinates and velocity.
- **Limited data:** Each class has only 20 videos.
- **Same-signer bias:** The model may learn the recording style rather than the full language variation.

For a research paper, inspect the four source videos and compare their landmark trajectories before claiming a specific visual cause.

### 5.5 Evaluation caution

The evaluator runs inference on all 400 sequence files. Since the training script used 85% of those sequences for model fitting, the 99% value is not an independent generalization estimate. It is useful as a full-dataset diagnostic, but a stronger study should use:

- A fixed untouched test set.
- Signer-independent splits.
- Multiple recording environments.
- Repeated runs with confidence intervals.

---

## 6. Real-Time System Architecture and Web Deployment

### 6.1 Flask backend flow

The current backend is `app.py`.

```text
Browser opens /translator
    -> HTML loads /video_feed as an MJPEG image
    -> InferenceWorker starts one OpenCV camera thread
    -> Each camera frame is processed by MediaPipe Holistic
    -> 235 position features enter a 35-frame rolling deque
    -> Velocity features are calculated
    -> PyTorch model produces class probabilities
    -> Worker publishes annotated JPEG frames
    -> Browser polls /get_prediction every 180 ms
    -> New locked tokens are shown and spoken
```

### 6.2 MJPEG video stream

The `/video_feed` endpoint returns a multipart response with repeated JPEG frames:

```text
--frame
Content-Type: image/jpeg
Content-Length: ...

<JPEG bytes>
```

This is simple and works with a normal HTML `<img>` element. It avoids sending camera frames from the browser to Flask as base64 JSON. The server owns the OpenCV camera and the inference loop.

### 6.3 Prediction endpoint

The `/get_prediction` endpoint returns JSON fields such as:

```json
{
  "success": true,
  "status": "Active prediction",
  "prediction": "i need water",
  "confidence": 0.94,
  "frame_count": 35,
  "locked_tokens": [],
  "token_count": 0,
  "motion_threshold": 0.009
}
```

The browser sends a cursor value. The worker returns only tokens after that cursor. This prevents the frontend from speaking the same token repeatedly.

### 6.4 Motion energy gate

The backend calculates the latest frame delta for the hand-related coordinate area:

$$
E_t = mean(|\Delta X_{t, hand}|)
$$

The current threshold is:

```text
MOTION_THRESHOLD = 0.009
```

If the latest motion energy is not above this threshold, the backend reports `Holding stance` and does not run a new active prediction lock.

This reduces false triggers from:

- Sitting still in front of the camera.
- Holding a final hand pose.
- Small detection noise.
- Background movement that does not represent a sign.

The threshold is empirical. It should be measured and tuned using idle and active validation clips.

### 6.5 Confidence thresholds and cooldown

The default confidence threshold is:

```text
0.86
```

The sensitive class `are you hiding something` currently uses:

```text
0.91
```

This is a simple class-specific hysteresis policy. A class that produces more false positives requires stronger evidence before locking.

After a lock, the backend applies a cooldown of:

```text
1.5 seconds
```

The cooldown prevents the same continuous gesture from creating repeated tokens.

### 6.6 Display persistence

When a token is locked, the backend stores:

- The locked text.
- The confidence.
- A `display_until` deadline.

For 3 seconds, `/get_prediction` continues to report `Sentence locked` and the locked phrase even when the next camera frames are neutral. This prevents the UI from flashing the phrase and immediately returning to `Idle`.

The frontend also has a 3-second visual hold. If a new lock arrives, it replaces the previous held prediction and starts the hold again.

### 6.7 Frontend session reset

The translator page has a separate 5-second inactivity timer:

- Every new token restarts the timer.
- A small countdown bar shows the remaining session time.
- After 5 seconds without a new token, the history fades out.
- The locked phrase box returns to `Waiting for gesture...`.
- The backend `/reset_sentence` endpoint clears its token list and rolling state.

The manual `Clear sentence` button uses the same reset operation.

### 6.8 Text-to-Speech

The browser uses the Web Speech API:

```javascript
window.speechSynthesis
SpeechSynthesisUtterance
```

The utterance configuration is:

- Preferred language: `en-IN`
- Fallback language: `en-US`
- Rate: `0.95`

Speech is triggered only in `handleNewTokens()` when a token is first returned by the server. The persistence timer and auto-reset timer never call `speakPhrase()`, so they cannot replay speech.

The Mute TTS button cancels current speech and prevents future speech until unmuted. Recognition continues while muted.

### 6.9 Main project files

| File | Responsibility |
|---|---|
| `custom_dataset/` | Original AVI videos grouped by class |
| `videos_to_keypoints.py` | Batch video-to-sequence conversion |
| `keypoint_features.py` | Shared landmarks, normalization, angles, interpolation, and velocity |
| `keypoint_sequences/` | Processed `(35, 470)` NumPy sequences |
| `train_keypoint_model.py` | PyTorch architecture and training |
| `isl_keypoint_model.pth` | Model weights and label mapping |
| `evaluate_confusion.py` | Full-dataset classification report and confusion matrix |
| `app.py` | Flask server, camera worker, MJPEG, prediction JSON, reset endpoint |
| `templates/translator.html` | Translator UI structure |
| `static/script.js` | Prediction polling, display hold, reset timer, and TTS |
| `static/style.css` | Visual theme and state styling |
| `recorder.py` | Webcam recording utility |
| `requirements.txt` | Python dependencies |

---

## 7. Viva, Interview, and Research Q&A

### Q1. Why did you not use a standard 2D CNN such as MobileNet or ResNet on video frames?

A 2D CNN sees pixels. Pixels change because of lighting, clothing, background, skin tone, and camera distance. A raw-image model also needs to process many large frames, which increases latency and memory use.

Our task depends mainly on hand shape, body pose, and motion. MediaPipe converts each frame into a compact geometric representation. We then normalize it around the body and model the sequence with a temporal network. This makes the model more focused on the signing movement.

This does not mean 2D CNNs are always bad. CNNs can be useful when appearance, facial expression, or full visual context is important. The chosen representation is better aligned with this project's small-vocabulary upper-body sentence task.

### Q2. Why use MediaPipe Holistic instead of MediaPipe Hands or Pose alone?

Hands alone give detailed finger landmarks but lose body context. Pose alone gives shoulder and wrist motion but loses finger posture. Holistic provides:

- Full body pose landmarks.
- Left-hand landmarks.
- Right-hand landmarks.

The nose and shoulders provide normalization anchors. The hands provide finger shape and movement. This combination gives better context than either hands or pose alone.

### Q3. How do you handle one hand going out of the camera frame?

When a hand is missing, the system uses that side's pose wrist position as a fallback for all 21 hand landmark slots. This avoids filling the frame with zeros, because a sudden jump to zero would create an artificial velocity spike.

The fallback does not reconstruct the missing fingers. It expresses lower information: the wrist is known, but detailed finger geometry is unknown. A future improvement would add visibility masks or carry forward the last reliable hand coordinates.

### Q4. Why add velocity deltas to spatial coordinates?

Spatial coordinates describe a pose. Velocity deltas describe change between frames. Sign language is dynamic, so the direction and size of movement can distinguish signs with similar static poses.

For each 235-dimensional position vector, the system adds a 235-dimensional first-order difference vector. This doubles the input from 235 to 470 features per frame.

### Q5. Why use a Bi-LSTM instead of a standard unidirectional LSTM or a simple GRU?

A Bi-LSTM reads the complete fixed sequence forward and backward. This lets it use both the movement before and after a frame. Some signs are only clear when the whole trajectory is considered.

A unidirectional LSTM is better for strictly causal streaming because it does not use future frames. A GRU is smaller and faster, but an LSTM has separate memory and gate mechanisms that can help with longer dependencies. The correct choice should ultimately be supported by an ablation study.

### Q6. How does label smoothing reduce bias and overfitting?

Without label smoothing, the target for the correct class is exactly 1 and every other class is exactly 0. The model can become overconfident, especially on a small dataset.

With label smoothing of 0.1, the target distribution is softened. The correct class remains the largest target, but the model is not forced to produce an extreme probability of exactly 1. This improves calibration and can reduce memorization and class collapse.

Label smoothing does not automatically solve an imbalanced dataset. Class balancing, more data, and proper splits are still important.

### Q7. How does the motion energy threshold distinguish rest from an active sign?

The backend compares consecutive normalized hand-related features. It computes the mean absolute delta for the latest frame. If this value is at or below `0.009`, the frame window is treated as a holding stance rather than active movement.

The threshold was chosen as an empirical gate. It should be validated by measuring energy distributions for idle, natural hand tremor, and real signing. A fixed threshold may need adjustment for different cameras, frame rates, and users.

### Q8. What are the latency and FPS numbers on a standard laptop CPU?

The repository does not contain a controlled FPS benchmark, so a precise number should not be claimed yet.

The real-time path performs:

1. OpenCV frame capture.
2. MediaPipe Holistic inference.
3. NumPy feature assembly.
4. PyTorch model inference.
5. JPEG encoding for MJPEG.

Actual FPS depends on CPU model, camera resolution, MediaPipe version, OpenCV build, and background processes. A proper benchmark should record:

- Camera capture FPS.
- MediaPipe processing time per frame.
- PyTorch inference time per 35-frame window.
- JPEG encoding time.
- End-to-end lock latency.

A simple benchmark can use `time.perf_counter()` around each stage and report mean, median, and 95th-percentile milliseconds over at least 500 frames. Do not use the offline video-processing time as the web latency number.

### Q9. What are the current limitations and future research directions?

#### Current limitations

- Only 20 sentence classes are supported.
- Only 20 videos per class are available.
- The full-dataset 99% evaluation includes training examples and is optimistic.
- The dataset may not contain enough signer, lighting, background, and camera diversity.
- MediaPipe can lose landmarks during occlusion or fast motion.
- A missing full pose currently becomes a zero pose.
- The fixed 35-frame window is not a complete continuous-language system.
- The server camera worker is designed for one local camera/session, not multi-user deployment.
- Browser TTS voice availability depends on the operating system and browser.
- There is no measured CPU FPS benchmark in the repository.
- The current recorder uses 20 FPS and 5 seconds, while the target protocol may specify 30 FPS and 2–3 seconds.

#### Future directions

- Collect many more videos from multiple signers.
- Use signer-independent train, validation, and test splits.
- Add landmark visibility and confidence features.
- Improve missing-pose handling with temporal imputation or frame rejection.
- Compare Bi-LSTM, GRU, temporal convolution, and Transformer models.
- Investigate Sign Language Recognition Transformer (SLRT) approaches.
- Support continuous sentence translation rather than one fixed window at a time.
- Add start-of-sign and end-of-sign detection.
- Use beam search or language modeling for sentence-level decoding.
- Expand the vocabulary beyond 20 classes.
- Add multilingual text and speech output.
- Quantize or export the model for edge devices.
- Add authentication and isolated camera sessions for multi-user deployment.
- Benchmark accuracy, calibration, FPS, latency, and energy usage on target hardware.

---

## 8. Reproducible Commands

Use the compatible Python environment and install dependencies:

```powershell
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
```

Extract keypoint sequences:

```powershell
.\.venv312\Scripts\python.exe videos_to_keypoints.py
```

Train the model:

```powershell
.\.venv312\Scripts\python.exe train_keypoint_model.py
```

Evaluate the complete processed dataset and create a confusion matrix:

```powershell
.\.venv312\Scripts\python.exe evaluate_confusion.py
```

Start the Flask web application:

```powershell
.\.venv312\Scripts\python.exe app.py
```

Open the translator page:

```text
http://127.0.0.1:5000/translator
```

Run standalone webcam inference:

```powershell
.\.venv312\Scripts\python.exe live_inference.py
```

---

## 9. One-Paragraph Project Synopsis

This project is a real-time Indian Sign Language sentence translator for a fixed vocabulary of 20 conversational sentences. Instead of classifying raw RGB video frames, it uses MediaPipe Holistic to extract body pose and both-hand landmarks. The landmarks are anchored at the nose and scaled by shoulder distance, while finger flexion angles describe hand posture and wrist fallback values reduce artificial velocity spikes when a hand is missing. Each video is resampled to 35 frames, and first-order motion deltas are concatenated with the 235 position features to create a 470-dimensional sequence. A PyTorch model with temporal 1D convolutions and a bidirectional LSTM classifies the sequence. A Flask backend provides an MJPEG camera stream and JSON prediction updates. The web interface shows persistent locked predictions, automatically clears inactive sentence sessions, and uses browser Text-to-Speech to speak each newly recognized phrase once.
