"""
evaluate.py — how good is the classifier, really?

Run:   python evaluate.py

Produces the numbers and plots you'll put in your dissertation.
The test set has not been touched until this point, which is what
makes these numbers honest.
"""
import sys
import time
from collections import Counter

import numpy as np
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import config as C
import skeleton as S
from train import MotionNet


def load_everything():
    try:
        X = np.load(C.DATA_DIR / "X_test.npy")
        y = np.load(C.DATA_DIR / "y_test.npy")
    except FileNotFoundError:
        print("No test data. Run:  python train.py")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MotionNet().to(device)
    try:
        model.load_state_dict(
            torch.load(C.WEIGHTS / "motion_net.pth", map_location=device))
        svm, scaler = joblib.load(C.WEIGHTS / "classifier.pkl")
    except FileNotFoundError:
        print("No trained model. Run:  python train.py")
        sys.exit(1)

    model.eval()
    return X, y, model, svm, scaler, device


def predict_all(X, model, svm, scaler, device):
    """Run everything through and time it."""
    features, times = [], []
    with torch.no_grad():
        for i in range(0, len(X), C.BATCH_SIZE):
            batch = torch.tensor(S.flatten(X[i:i + C.BATCH_SIZE])).to(device)
            start = time.perf_counter()
            out = model.summarise(batch).cpu().numpy()
            times.append((time.perf_counter() - start) / len(batch) * 1000)
            features.append(out)

    features = np.concatenate(features)
    scaled = scaler.transform(features)
    return svm.predict(scaled), svm.predict_proba(scaled), np.mean(times)


def safety_numbers(y_true, y_pred):
    """
    The two numbers that decide whether this could actually be deployed.

    A false dispatch wastes an ambulance. A missed one could cost someone
    their life. They matter far more than overall accuracy.
    """
    normal_id = C.LABEL_ID["normal"]
    red_ids = {C.LABEL_ID[a] for a in C.RED_ACTIONS if a in C.LABEL_ID}

    was_normal = y_true == normal_id
    called_anyway = was_normal & np.isin(y_pred, list(red_ids))
    false_rate = called_anyway.sum() / max(was_normal.sum(), 1)

    was_serious = np.isin(y_true, list(red_ids))
    called_it_nothing = was_serious & (y_pred == normal_id)
    missed_rate = called_it_nothing.sum() / max(was_serious.sum(), 1)

    print("\n" + "-" * 46)
    print("  What matters for deployment")
    print("-" * 46)
    print(f"  False alarms   {false_rate:>7.1%}   "
          f"({called_anyway.sum()} of {was_normal.sum()} normal clips)")
    print(f"                 target: under 2%")
    print()
    print(f"  Missed events  {missed_rate:>7.1%}   "
          f"({called_it_nothing.sum()} of {was_serious.sum()} serious clips)")
    print(f"                 target: zero")

    return false_rate, missed_rate


def draw_confusion(y_true, y_pred, present):
    names = [C.LABELS[i] for i in present]
    cm = confusion_matrix(y_true, y_pred, labels=present)
    pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    img = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("what the model said")
    ax.set_ylabel("what it actually was")
    ax.set_title("Where the model gets confused (%)")

    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center",
                    fontsize=10,
                    fontweight="bold" if i == j else "normal",
                    color="white" if pct[i, j] > 50 else "#333333")

    fig.colorbar(img, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(C.LOGS / "confusion_matrix.png", dpi=160)
    plt.close()
    np.save(C.LOGS / "confusion_matrix.npy", cm)


def draw_per_class(y_true, y_pred, present):
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=present, zero_division=0)
    names = [C.LABELS[i] for i in present]

    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(x - w, p, w, label="precision", color="#0B7285")
    ax.bar(x,     r, w, label="recall",    color="#B07219")
    ax.bar(x + w, f, w, label="f1",        color="#5F3D7E")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Performance per movement")
    ax.legend(frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.22))
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    plt.subplots_adjust(bottom=0.30)
    plt.savefig(C.LOGS / "per_class.png", dpi=160)
    plt.close()

    np.save(C.LOGS / "precision.npy", p)
    np.save(C.LOGS / "recall.npy", r)
    np.save(C.LOGS / "f1.npy", f)


def main():
    print("=" * 46)
    print("  Evaluating on the held-out test set")
    print("=" * 46)

    X, y, model, svm, scaler, device = load_everything()
    print(f"\n{len(X)} test samples, never seen during training")

    y_pred, y_prob, ms_per_clip = predict_all(X, model, svm, scaler, device)

    present = sorted(set(y.tolist()) | set(y_pred.tolist()))
    names = [C.LABELS[i] for i in present]

    print("\n" + "-" * 46)
    print("  Breakdown by movement")
    print("-" * 46)
    print(classification_report(y, y_pred, labels=present,
                               target_names=names, zero_division=0))

    overall = f1_score(y, y_pred, average="macro", zero_division=0)
    print(f"  Overall F1: {overall:.3f}")

    false_rate, missed_rate = safety_numbers(y, y_pred)

    print("\n" + "-" * 46)
    print("  Speed")
    print("-" * 46)
    print(f"  {ms_per_clip:.1f} ms per clip on {device}")

    draw_confusion(y, y_pred, present)
    draw_per_class(y, y_pred, present)

    print("\n" + "=" * 46)
    print("  Plots saved to logs/")
    print("    confusion_matrix.png")
    print("    per_class.png")
    print("=" * 46)
    print("\nNext:  python run_live.py")


if __name__ == "__main__":
    main()
