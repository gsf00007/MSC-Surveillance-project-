"""
tune_thresholds.py — where should the thresholds actually sit?

THE PROBLEM THIS SOLVES
-----------------------
`eval_system.py` shows the trade-off but not what to do about it. With RED at
90% confidence held for 10 seconds, false alarms drop to 2% — but only a
quarter of real collapses actually call an ambulance. Most stop at YELLOW.

Whether that's right depends on something the numbers can't tell you: is
there someone watching the dashboard?

    Someone is watching        YELLOW is fine. It alerts them, they look.
                               Keep the bar high, avoid false calls.

    Nobody is watching         YELLOW does nothing. Only RED matters, so the
                               bar has to come down or real events go
                               unanswered.

This sweeps both settings and shows the whole trade-off, so you pick a point
deliberately rather than accepting a default.

Run:   python tune_thresholds.py
"""
import sys
from collections import defaultdict

import numpy as np

import config as C
import skeleton as S
from threat import ThreatEngine


FPS = 15.0
SIMULATE_SECONDS = 14.0


def load_and_classify():
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

    features = []
    with torch.no_grad():
        for i in range(0, len(X), C.BATCH_SIZE):
            batch = torch.tensor(S.flatten(X[i:i + C.BATCH_SIZE])).to(device)
            features.append(net.summarise(batch).cpu().numpy())
    features = np.concatenate(features)

    probabilities = svm.predict_proba(scaler.transform(features))
    return (probabilities.argmax(axis=1),
            probabilities.max(axis=1),
            y)


def simulate(predictions, confidences, truths,
             red_confidence, red_seconds):
    """
    Run every clip through the engine with these settings and count what
    would happen.
    """
    # Temporarily override the settings the engine reads
    old_conf = C.RED_CONFIDENCE
    old_secs = C.RED_HOLD_SECONDS
    C.RED_CONFIDENCE = red_confidence
    C.RED_HOLD_SECONDS = red_seconds

    normal_id = C.LABEL_ID["normal"]
    red_ids = {C.LABEL_ID[a] for a in C.RED_ACTIONS if a in C.LABEL_ID}

    counts = {"red_on_normal": 0, "red_on_serious": 0,
              "alerted_serious": 0, "silent_serious": 0,
              "normal_total": 0, "serious_total": 0}

    steps = int(SIMULATE_SECONDS * FPS / C.PROCESS_EVERY)

    try:
        for predicted, confidence, actual in zip(predictions, confidences,
                                                  truths):
            engine = ThreatEngine()
            label = C.LABELS[predicted]

            worst = "GREEN"
            for step in range(steps):
                engine.set_time(step * C.PROCESS_EVERY / FPS)
                alert = engine.update(person=1, action=label,
                                      confidence=float(confidence))
                if alert.level == "RED":
                    worst = "RED"
                    break
                if alert.level == "YELLOW" and worst == "GREEN":
                    worst = "YELLOW"

            if actual == normal_id:
                counts["normal_total"] += 1
                if worst == "RED":
                    counts["red_on_normal"] += 1
            elif actual in red_ids:
                counts["serious_total"] += 1
                if worst == "RED":
                    counts["red_on_serious"] += 1
                if worst != "GREEN":
                    counts["alerted_serious"] += 1
                else:
                    counts["silent_serious"] += 1
    finally:
        C.RED_CONFIDENCE = old_conf
        C.RED_HOLD_SECONDS = old_secs

    return {
        "false_alarm": counts["red_on_normal"] / max(counts["normal_total"], 1),
        "dispatch_rate": counts["red_on_serious"] / max(counts["serious_total"], 1),
        "alerted_rate": counts["alerted_serious"] / max(counts["serious_total"], 1),
        "silent_rate": counts["silent_serious"] / max(counts["serious_total"], 1),
        "false_count": counts["red_on_normal"],
        "normal_total": counts["normal_total"],
        "dispatch_count": counts["red_on_serious"],
        "serious_total": counts["serious_total"],
    }


def sweep_confidence(predictions, confidences, truths, seconds):
    print(f"\n  Holding duration at {seconds:.0f}s, varying confidence:\n")
    print(f"  {'confidence':>11}{'auto-dispatch':>15}{'someone alerted':>17}"
          f"{'false alarms':>14}")
    print("  " + "-" * 55)

    rows = []
    for threshold in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        result = simulate(predictions, confidences, truths,
                          threshold, seconds)
        rows.append((threshold, result))
        marker = "  <-- current" if abs(threshold - 0.90) < 0.01 else ""
        print(f"  {threshold:>10.0%}{result['dispatch_rate']:>14.0%}"
              f"{result['alerted_rate']:>16.0%}"
              f"{result['false_alarm']:>13.0%}{marker}")
    return rows


def sweep_duration(predictions, confidences, truths, confidence):
    print(f"\n  Holding confidence at {confidence:.0%}, varying duration:\n")
    print(f"  {'seconds':>9}{'auto-dispatch':>15}{'someone alerted':>17}"
          f"{'false alarms':>14}")
    print("  " + "-" * 53)

    rows = []
    for seconds in [2.0, 4.0, 6.0, 8.0, 10.0, 15.0]:
        result = simulate(predictions, confidences, truths,
                          confidence, seconds)
        rows.append((seconds, result))
        marker = "  <-- current" if abs(seconds - 10.0) < 0.1 else ""
        print(f"  {seconds:>8.0f}s{result['dispatch_rate']:>14.0%}"
              f"{result['alerted_rate']:>16.0%}"
              f"{result['false_alarm']:>13.0%}{marker}")
    return rows


def recommend(rows):
    """
    Pick a setting that catches most real events without many false calls.

    Weighted deliberately: a missed collapse is worse than a wasted
    ambulance, so dispatch rate counts for more. But false alarms above
    5% are treated as unacceptable regardless.
    """
    best = None
    for threshold, result in rows:
        if result["false_alarm"] > 0.05:
            continue
        score = result["dispatch_rate"] - 3 * result["false_alarm"]
        if best is None or score > best[2]:
            best = (threshold, result, score)
    return best


def draw(conf_rows, save_to):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = [r[0] * 100 for r in conf_rows]
    dispatch = [r[1]["dispatch_rate"] * 100 for r in conf_rows]
    alerted = [r[1]["alerted_rate"] * 100 for r in conf_rows]
    false_alarm = [r[1]["false_alarm"] * 100 for r in conf_rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))

    ax.plot(thresholds, alerted, color="#B07219", lw=2,
            label="someone is alerted")
    ax.plot(thresholds, dispatch, color="#0B7285", lw=2,
            label="ambulance called automatically")
    ax.plot(thresholds, false_alarm, color="#B03A2E", lw=2, ls="--",
            label="false alarms")

    ax.axvline(90, color="#5A6B7D", lw=1, ls=":")
    ax.text(90.4, 92, "current", fontsize=9, color="#5A6B7D", style="italic")

    ax.set_xlabel("confidence needed for a RED alert (%)")
    ax.set_ylabel("percent of test clips")
    ax.set_title("The safety trade-off")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=9, loc="center left")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_to, dpi=160)
    plt.close()


def main():
    print("=" * 60)
    print("  Where should the thresholds sit?")
    print("=" * 60)

    predictions, confidences, truths = load_and_classify()
    print(f"\n  {len(predictions)} test clips")

    conf_rows = sweep_confidence(predictions, confidences, truths,
                                 C.RED_HOLD_SECONDS)
    dur_rows = sweep_duration(predictions, confidences, truths,
                              C.RED_CONFIDENCE)

    print("\n" + "-" * 60)
    print("  Reading this")
    print("-" * 60)
    print("  'someone alerted' counts YELLOW and RED together. If there is")
    print("  an operator watching, that's a caught event — they look and")
    print("  decide. If nobody is watching, only RED matters.")

    best = recommend(conf_rows)
    if best:
        threshold, result, _ = best
        print(f"\n  Best confidence within a 5% false alarm budget: "
              f"{threshold:.0%}")
        print(f"    auto-dispatch    {result['dispatch_rate']:.0%} of real events")
        print(f"    someone alerted  {result['alerted_rate']:.0%}")
        print(f"    false alarms     {result['false_alarm']:.0%} "
              f"({result['false_count']} of {result['normal_total']})")

        if abs(threshold - C.RED_CONFIDENCE) > 0.02:
            print(f"\n  To use it, set this in config.py:")
            print(f"    RED_CONFIDENCE = {threshold:.2f}")
        else:
            print(f"\n  That's what you're already using.")

    draw(conf_rows, C.LOGS / "threshold_tradeoff.png")

    print("\n" + "=" * 60)
    print("  saved logs/threshold_tradeoff.png")
    print()
    print("  Worth saying in your write-up: this is a deliberate choice")
    print("  between missing real emergencies and wasting ambulances, and")
    print("  the right answer depends on whether anyone is watching the")
    print("  screen. It isn't a number that can be optimised in isolation.")
    print("=" * 60)


if __name__ == "__main__":
    main()
