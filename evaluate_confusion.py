import os
import sys
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from train_keypoint_model import KeypointSequenceModel

# Ensure safe printing on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "isl_keypoint_model.pth"
DATA_DIR = "keypoint_sequences"

if not os.path.exists(CHECKPOINT_PATH):
    print(f"Error: Checkpoint file '{CHECKPOINT_PATH}' not found!")
    sys.exit(1)

# 1. Load Checkpoint
print(f"[1/5] Loading model checkpoint from '{CHECKPOINT_PATH}'...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
id2label = checkpoint.get("id2label", checkpoint.get("labels"))
if isinstance(id2label, dict):
    id2label = {int(i): w for i, w in id2label.items()}
else:
    id2label = {i: w for i, w in enumerate(id2label)}
label2id = {w: i for i, w in id2label.items()}
classes = [id2label[i] for i in range(len(id2label))]

# 2. Load Model Architecture
print(f"[2/5] Initializing architecture with {len(classes)} classes...")
model = KeypointSequenceModel(len(classes)).to(DEVICE)
state_key = "model_state_dict" if "model_state_dict" in checkpoint else "model_state"
model.load_state_dict(checkpoint[state_key])
model.eval()

# 3. Gather All Test/Validation Sequences
print(f"[3/5] Loading keypoint sequences from '{DATA_DIR}'...")
all_sequences, all_labels, file_paths = [], [], []

class_counts = {}
for class_name in classes:
    class_folder = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_folder):
        print(f"  [!] Warning: Folder missing for class '{class_name}'")
        continue
    npy_files = sorted(glob.glob(os.path.join(class_folder, "*.npy")))
    class_counts[class_name] = len(npy_files)
    for f in npy_files:
        seq = np.load(f)  # Expected shape (35, 470)
        all_sequences.append(seq)
        all_labels.append(label2id[class_name])
        file_paths.append(f)

X = torch.tensor(np.array(all_sequences), dtype=torch.float32).to(DEVICE)
y_true = np.array(all_labels)

print(f"      Total samples loaded: {len(X)} across {len(classes)} classes.")

# 4. Perform Inference
print("[4/5] Running model inference...")
with torch.no_grad():
    outputs = model(X)
    probs = torch.softmax(outputs, dim=1).cpu().numpy()
    y_pred = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

overall_acc = accuracy_score(y_true, y_pred)
print(f"\n============================================================")
print(f"  OVERALL DATASET ACCURACY: {overall_acc * 100:.2f}% ({np.sum(y_true == y_pred)} / {len(y_true)} correct)")
print(f"============================================================")

# 5. Diagnostic Report
print("\n" + "="*70)
print("  DETAILED CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_true, y_pred, target_names=classes, digits=2, zero_division=0))

# 6. Confusion Matrix Computation
cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

confused_pairs = []
for i in range(len(classes)):
    for j in range(len(classes)):
        if i != j and cm[i][j] > 0:
            confused_pairs.append((classes[i], classes[j], int(cm[i][j]), cm_norm[i][j]))

confused_pairs.sort(key=lambda x: x[2], reverse=True)

print("\n" + "="*70)
print("  TOP CONFUSED GESTURE PAIRS (Actual Sign -> Misclassified As)")
print("="*70)
if confused_pairs:
    for actual, predicted, count, ratio in confused_pairs:
        print(f"  * Actual: '{actual}' --> Misclassified as: '{predicted}' | Count: {count} ({ratio*100:.1f}%)")
else:
    print("  [OK] No confusion detected on keypoint dataset! (100% classification)")
print("="*70)

# Specific Check on user-mentioned classes
print("\n" + "="*70)
print("  SPECIFIC FOCUS ANALYSIS: 'i promise', 'what happened', 'who are you'")
print("="*70)

for target_name in ["i promise", "what happened", "who are you"]:
    if target_name in label2id:
        idx = label2id[target_name]
        mask = (y_true == idx)
        class_samples = np.sum(mask)
        class_correct = np.sum((y_true == idx) & (y_pred == idx))
        class_acc = class_correct / class_samples if class_samples > 0 else 0
        avg_conf = np.mean(confidences[mask]) if class_samples > 0 else 0
        min_conf = np.min(confidences[mask]) if class_samples > 0 else 0
        max_conf = np.max(confidences[mask]) if class_samples > 0 else 0

        # Misclassifications for this class
        wrong_preds = y_pred[mask & (y_pred != idx)]
        misclass_dict = {}
        for wp in wrong_preds:
            w_name = id2label[wp]
            misclass_dict[w_name] = misclass_dict.get(w_name, 0) + 1

        print(f"\n-> Class: '{target_name}' (Index: {idx})")
        print(f"   Total samples: {class_samples}")
        print(f"   Correct predictions: {class_correct}/{class_samples} ({class_acc*100:.1f}%)")
        print(f"   Confidence on actual '{target_name}' samples:")
        print(f"     Average Confidence: {avg_conf*100:.2f}% (Min: {min_conf*100:.2f}%, Max: {max_conf*100:.2f}%)")
        if misclass_dict:
            print(f"   Misclassified into:")
            for m_cls, m_cnt in misclass_dict.items():
                print(f"     - '{m_cls}': {m_cnt} times")
        else:
            print(f"   No misclassifications inside the offline dataset.")

        # Check how many OTHER classes got predicted as this class (False Positives)
        fp_mask = (y_true != idx) & (y_pred == idx)
        fp_count = np.sum(fp_mask)
        print(f"   False Positives (Other classes wrongly predicted as '{target_name}'): {fp_count}")
        if fp_count > 0:
            fp_actual_indices = y_true[fp_mask]
            fp_dict = {}
            for ai in fp_actual_indices:
                a_name = id2label[ai]
                fp_dict[a_name] = fp_dict.get(a_name, 0) + 1
            for a_cls, a_cnt in fp_dict.items():
                print(f"     - Actual '{a_cls}' was predicted as '{target_name}': {a_cnt} times")
    else:
        print(f"-> Class '{target_name}' not found in model vocabulary!")

# 7. Generate High-Quality Confusion Matrix Plots
print("\n[5/5] Generating Confusion Matrix visualizations...")

# Plot 1: Standard Count Confusion Matrix
fig, ax = plt.subplots(figsize=(18, 16))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar(im, fraction=0.046, pad=0.04)

ax.set_title("ISL 30-Sentence Keypoint Confusion Matrix (Raw Sample Counts)", fontsize=16, fontweight='bold', pad=20)
tick_marks = np.arange(len(classes))
ax.set_xticks(tick_marks)
ax.set_xticklabels(classes, rotation=90, ha='right', fontsize=9)
ax.set_yticks(tick_marks)
ax.set_yticklabels(classes, fontsize=9)

# Overlay numbers on cells
thresh = cm.max() / 2.0
for i in range(len(classes)):
    for j in range(len(classes)):
        val = cm[i, j]
        if val > 0:
            ax.text(j, i, format(val, 'd'),
                    ha="center", va="center",
                    fontsize=7,
                    color="white" if val > thresh else "black")

ax.set_ylabel('True Sentence Label', fontsize=13, fontweight='bold')
ax.set_xlabel('Predicted Sentence Label', fontsize=13, fontweight='bold')
plt.tight_layout()
out_png_count = "confusion_matrix.png"
plt.savefig(out_png_count, dpi=300, bbox_inches="tight")
plt.close()
print(f"  -> Saved count confusion matrix to: {out_png_count}")

# Plot 2: Normalized Confusion Matrix (Percentages)
fig, ax = plt.subplots(figsize=(18, 16))
im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1.0)
cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.set_label('Recall / Accuracy Rate (0.0 to 1.0)', fontsize=11)

ax.set_title("ISL 30-Sentence Keypoint Confusion Matrix (Normalized Recall %)", fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(tick_marks)
ax.set_xticklabels(classes, rotation=90, ha='right', fontsize=9)
ax.set_yticks(tick_marks)
ax.set_yticklabels(classes, fontsize=9)

for i in range(len(classes)):
    for j in range(len(classes)):
        val = cm_norm[i, j]
        if val >= 0.01:
            ax.text(j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=7,
                    color="white" if val > 0.5 else "black")

ax.set_ylabel('True Sentence Label', fontsize=13, fontweight='bold')
ax.set_xlabel('Predicted Sentence Label', fontsize=13, fontweight='bold')
plt.tight_layout()
out_png_norm = "confusion_matrix_normalized.png"
plt.savefig(out_png_norm, dpi=300, bbox_inches="tight")
plt.close()
print(f"  -> Saved normalized confusion matrix to: {out_png_norm}")

print("\nDone! Evaluation and Confusion Matrix Generation Complete.")
