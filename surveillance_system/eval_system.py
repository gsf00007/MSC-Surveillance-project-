"""
eval_system.py — what does the *whole* system do, not just the classifier?

WHY THIS IS A DIFFERENT NUMBER
------------------------------
`evaluate.py` measures the classifier on its own. Every clip gets a label,
and if a normal clip is labelled "assault" that counts as a false alarm.

But the classifier never gets to raise an alarm by itself. Three things sit
between it and a phone call:

    confidence must reach 90%
    the event must persist for 10 seconds
    a person must not cancel within 10 seconds

A single confused clip passes none of those. So the classifier's false alarm
rate and the system's false alarm rate are different numbers, and the second
one is what actually matters.

This measures both, so you can say what the filtering buys you.

Run:   python evaluate.py     (first, to build the test split)
       python eval_system.py
"""
import sys
from collections import defaultdict

import numpy as np

import config as C
import skeleton as S
from threat import ThreatEngine


# A clip is 30 frames. Played at 15fps that's 2 seconds of footage.
# To simulate someone genuinely collapsing and staying down, the same clip
# is fed repeatedly, which is what a real continuous event looks like.
FPS = 15.0
SIMULATE_SECONDS = 14.0


def load():
    try:
        X = np.load(C.DATA_DIR / "X_test.npy")
        y = np.load(C.DATA_DIR / "y_test.npy")
    except FileNotFoundError:
        print("  no test split. run:  python train.py")
        sys.exit(1)

    import torch, joblib
    from train import MotionNet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = MotionNet().to(device)
    try:
        net.load_state_dict(torch.load(C.WEIGHTS / "motion_net.pth",
                                       map_location=device))
        svm, scaler = joblib.load(C.WEIGHTS / "classifier.pkl")
    except FileNotFoundError:
        print("  no trained model. run:  python train.py")
        sys.exit(1)
    net.eval()
    return X, y, net, svm, scaler, device, torch


def classify_all(X, net, svm, scaler, device, torch):
    """Label and confidence for every test clip."""
    features = []
    with torch.no_grad():
        for i in range(0, len(X), C.BATCH_SIZE):
            batch = torch.tensor(S.flatten(X[i:i + C.BATCH_SIZE])).to(device)
            features.append(net.summarise(batch).cpu().numpy())
    features = np.concatenate(features)

    probabilities = svm.predict_proba(scaler.transform(features))
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    return predictions, confidences


def run_through_engine(predictions, confidences, truths):
    """
    Feed each clip through the threat engine as if it were a continuous
    event, and record the worst level it ever reached.
    """
    results = []

    for i, (predicted, confidence, actual) in enumerate(
            zip(predictions, confidences, truths)):

        engine = ThreatEngine()
        label = C.LABELS[predicted]
        worst = "GREEN"

        # Simulate the same classification repeating, which is what a
        # sustained event produces.
        steps = int(SIMULATE_SECONDS * FPS / C.PROCESS_EVERY)
        for step in range(steps):
            t = step * C.PROCESS_EVERY / FPS
            engine.set_time(t)
            alert = engine.update(person=1, action=label,
                                  confidence=float(confidence))
            if alert.level == "RED":
                worst = "RED"
                break
            if alert.level == "YELLOW" and worst == "GREEN":
                worst = "YELLOW"

        results.append({
            "actual": C.LABELS[actual],
            "predicted": label,
            "confidence": float(confidence),
            "level": worst,
        })

    return results


def report(results, predictions, truths):
    normal_id = C.LABEL_ID["normal"]
    red_ids = {C.LABEL_ID[a] for a in C.RED_ACTIONS if a in C.LABEL_ID}

    truths = np.asarray(truths)
    predictions = np.asarray(predictions)

    # ── classifier alone ──────────────────────────────────────────────
    was_normal = truths == normal_id
    labelled_serious = np.isin(predictions, list(red_ids))
    classifier_false = (was_normal & labelled_serious).sum()
    normal_total = was_normal.sum()

    was_serious = np.isin(truths, list(red_ids))
    labelled_normal = predictions == normal_id
    classifier_missed = (was_serious & labelled_normal).sum()
    serious_total = was_serious.sum()

    # ── whole system ──────────────────────────────────────────────────
    system_false = sum(1 for r, actual_normal in zip(results, was_normal)
                       if actual_normal and r["level"] == "RED")
    system_missed = sum(1 for r, actual_serious in zip(results, was_serious)
                        if actual_serious and r["level"] == "GREEN")

    print("\n" + "=" * 60)
    print("  Classifier alone versus the whole system")
    print("=" * 60)
    print(f"  {'':<26}{'classifier':>15}{'full system':>16}")
    print("  " + "-" * 56)

    c_false = classifier_false / max(normal_total, 1)
    s_false = system_false / max(normal_total, 1)
    print(f"  {'false alarms':<26}{c_false:>14.1%}{s_false:>16.1%}")
    print(f"  {'':<26}{f'({classifier_false} of {normal_total})':>15}"
          f"{f'({system_false} of {normal_total})':>16}")

    print()
    c_missed = classifier_missed / max(serious_total, 1)
    s_missed = system_missed / max(serious_total, 1)
    print(f"  {'missed serious events':<26}{c_missed:>14.1%}{s_missed:>16.1%}")
    print(f"  {'':<26}{f'({classifier_missed} of {serious_total})':>15}"
          f"{f'({system_missed} of {serious_total})':>16}")

    if classifier_false > 0:
        cut = 1 - (system_false / classifier_false)
        print(f"\n  The confidence and duration filters remove "
              f"{cut:.0%} of the classifier's false alarms.")
    if system_false == 0:
        print("  No normal clip would have triggered a call.")

    # ── where the alerts landed ───────────────────────────────────────
    print("\n" + "-" * 60)
    print("  What each kind of clip would actually trigger")
    print("-" * 60)

    grid = defaultdict(lambda: defaultdict(int))
    for r in results:
        grid[r["actual"]][r["level"]] += 1

    print(f"  {'actual movement':<14}{'GREEN':>8}{'YELLOW':>9}{'RED':>7}")
    print("  " + "-" * 38)
    for name in C.LABELS:
        if name not in grid:
            continue
        row = grid[name]
        print(f"  {name:<14}{row['GREEN']:>8}{row['YELLOW']:>9}"
              f"{row['RED']:>7}")

    # ── the important one ─────────────────────────────────────────────
    collapse_clips = [r for r in results if r["actual"] == "collapse"]
    if collapse_clips:
        caught = sum(1 for r in collapse_clips if r["level"] == "RED")
        noticed = sum(1 for r in collapse_clips if r["level"] != "GREEN")
        print(f"\n  Of {len(collapse_clips)} real collapses:")
        print(f"    {caught} would have called an ambulance")
        print(f"    {noticed} would have alerted someone at all")


def main():
    print("=" * 60)
    print("  Evaluating the whole system, not just the classifier")
    print("=" * 60)

    X, y, net, svm, scaler, device, torch = load()
    print(f"\n  {len(X)} test clips, never seen during training")
    print(f"  simulating each as a {SIMULATE_SECONDS:.0f}s continuous event")
    print(f"  RED needs {C.RED_CONFIDENCE:.0%} held for "
          f"{C.RED_HOLD_SECONDS:.0f}s")

    predictions, confidences = classify_all(X, net, svm, scaler, device, torch)
    results = run_through_engine(predictions, confidences, y)
    report(results, predictions, y)

    print("\n" + "=" * 60)
    print("  Use both numbers in your write-up. The classifier's rate")
    print("  shows how good the model is. The system's rate shows what")
    print("  would actually happen in a building.")
    print("=" * 60)


if __name__ == "__main__":
    main()
