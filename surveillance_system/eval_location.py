"""
eval_location.py — do the cameras agree on where people are?

WHY THIS IS THE RIGHT QUESTION
------------------------------
Anyone can print a zone name next to a person. The honest test is
different: when two cameras both see the same person at the same moment,
do they place them in the same spot?

Without calibration they can't, because each camera only knows about its
own picture. With calibration they should, because both are describing
positions on the same floor.

That disagreement, measured in metres, is the number worth reporting.

Run:   python eval_location.py

Uses WildTrack, which ships with proper calibration and tells you which
person is which, so every answer can be checked.
"""
import json
import sys
from collections import defaultdict

import numpy as np

import config as C
from eval_wildtrack import find_wildtrack, load_annotations


# WildTrack watches a square roughly 12 by 36 metres, mapped onto 0-1
# coordinates. Multiplying by this turns disagreement into something
# meaningful rather than an abstract number.
SQUARE_WIDTH_M = 12.0
SQUARE_DEPTH_M = 36.0


def positions_per_frame(frames, zone_map, frame_shapes):
    """
    For every frame, work out where each camera thinks each person is.

    Returns {frame: {person: [(camera, x, y), ...]}}
    """
    result = defaultdict(lambda: defaultdict(list))

    for frame_no, sightings in frames.items():
        for s in sightings:
            camera = s["camera"]
            shape = frame_shapes.get(camera, (1080, 1920, 3))
            _, position, real = zone_map.locate(camera, s["box"], shape)
            result[frame_no][s["person"]].append(
                (camera, position[0], position[1], real))

    return result


def disagreement(per_frame):
    """
    How far apart two cameras place the same person.

    Only looks at moments where at least two cameras can see someone,
    since that's the only time they can disagree.
    """
    gaps = []
    checked = 0
    calibrated_pairs = 0

    for frame_no, people in per_frame.items():
        for person, seen in people.items():
            if len(seen) < 2:
                continue
            checked += 1

            for i in range(len(seen)):
                for j in range(i + 1, len(seen)):
                    _, x1, y1, real1 = seen[i]
                    _, x2, y2, real2 = seen[j]

                    dx = (x1 - x2) * SQUARE_WIDTH_M
                    dy = (y1 - y2) * SQUARE_DEPTH_M
                    gaps.append(np.hypot(dx, dy))

                    if real1 and real2:
                        calibrated_pairs += 1

    if not gaps:
        return None

    gaps = np.array(gaps)
    return {
        "people_seen_twice": checked,
        "pairs": len(gaps),
        "calibrated_pairs": calibrated_pairs,
        "average_m": float(gaps.mean()),
        "median_m": float(np.median(gaps)),
        "worst_m": float(gaps.max()),
        "within_1m": float((gaps < 1.0).mean()),
        "within_2m": float((gaps < 2.0).mean()),
        "within_5m": float((gaps < 5.0).mean()),
    }


def zone_agreement(frames, zone_map, frame_shapes):
    """
    Simpler question, same idea. When two cameras see one person, do they
    name the same zone? A guard reading the dashboard cares about this
    more than they care about metres.
    """
    per_frame = defaultdict(lambda: defaultdict(list))

    for frame_no, sightings in frames.items():
        for s in sightings:
            shape = frame_shapes.get(s["camera"], (1080, 1920, 3))
            zone, _, _ = zone_map.locate(s["camera"], s["box"], shape)
            per_frame[frame_no][s["person"]].append(zone)

    agreed = 0
    total = 0
    for people in per_frame.values():
        for zones in people.values():
            if len(zones) < 2:
                continue
            total += 1
            if len(set(zones)) == 1:
                agreed += 1

    return agreed / total if total else 0.0, total


def draw(uncalibrated, calibrated, save_to):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    # How far apart, by tolerance
    tolerances = ["within 1m", "within 2m", "within 5m"]
    keys = ["within_1m", "within_2m", "within_5m"]

    x = np.arange(len(tolerances))
    width = 0.36

    for i, (name, data, colour) in enumerate([
            ("frame position", uncalibrated, "#B03A2E"),
            ("calibrated", calibrated, "#0B7285")]):
        if not data:
            continue
        values = [data[k] * 100 for k in keys]
        bars = ax1.bar(x + (i - 0.5) * width, values, width,
                       label=name, color=colour)
        for bar, v in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                     f"{v:.0f}", ha="center", fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(tolerances)
    ax1.set_ylabel("percent of sightings")
    ax1.set_title("Do two cameras place a person in the same spot?")
    ax1.set_ylim(0, 108)
    ax1.legend(frameon=False)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_axisbelow(True)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)

    # Average disagreement
    names, values, colours = [], [], []
    for name, data, colour in [("frame position", uncalibrated, "#B03A2E"),
                               ("calibrated", calibrated, "#0B7285")]:
        if data:
            names.append(name)
            values.append(data["average_m"])
            colours.append(colour)

    bars = ax2.bar(names, values, color=colours, width=0.5)
    top = max(values) if values else 1.0
    ax2.set_ylim(0, top * 1.22)
    for bar, v in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + top * 0.04,
                 f"{v:.2f} m", ha="center", fontsize=10)

    ax2.set_ylabel("metres apart")
    ax2.set_title("Average disagreement between cameras")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_axisbelow(True)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_to, dpi=160)
    plt.close()


def main():
    print("=" * 58)
    print("  Do the cameras agree on where people are?")
    print("=" * 58)

    root = getattr(C, "WILDTRACK_FOLDER", None)
    found = find_wildtrack(root) if root else None
    if found is None:
        print(f"\n  couldn't find WildTrack at {root}")
        sys.exit(1)

    frames = load_annotations(
        found, max_frames=getattr(C, "WILDTRACK_FRAMES", 60))
    if not frames:
        print("\n  no annotations could be read")
        sys.exit(1)

    people = {s["person"] for ss in frames.values() for s in ss}
    cameras = sorted({s["camera"] for ss in frames.values() for s in ss})
    print(f"\n  {len(frames)} frames, {len(people)} people, "
          f"{len(cameras)} cameras")

    # Work out each camera's frame size from the boxes in it
    frame_shapes = {}
    for camera in cameras:
        boxes = [s["box"] for ss in frames.values()
                 for s in ss if s["camera"] == camera]
        if boxes:
            widest = max(b[2] for b in boxes)
            tallest = max(b[3] for b in boxes)
            frame_shapes[camera] = (max(tallest, 1080), max(widest, 1920), 3)

    from reid import ZoneMap

    # Before calibration
    print("\n" + "-" * 58)
    print("  Guessing from frame position")
    print("-" * 58)

    plain = ZoneMap(auto_load=False)
    per_frame = positions_per_frame(frames, plain, frame_shapes)
    uncalibrated = disagreement(per_frame)

    if uncalibrated:
        print(f"  people seen by 2+ cameras   "
              f"{uncalibrated['people_seen_twice']}")
        print(f"  average disagreement        "
              f"{uncalibrated['average_m']:.2f} m")
        print(f"  worst                       "
              f"{uncalibrated['worst_m']:.2f} m")
        print(f"  placed within 2m            "
              f"{uncalibrated['within_2m']:.1%}")

    agree_rate, checked = zone_agreement(frames, plain, frame_shapes)
    print(f"  named the same zone         {agree_rate:.1%} of {checked}")

    # After calibration
    print("\n" + "-" * 58)
    print("  Using calibration")
    print("-" * 58)

    calibrated_map = ZoneMap(auto_load=True)
    if not calibrated_map.calibrated:
        print("\n  Nothing calibrated yet, so there's nothing to compare.")
        print("\n  Calibrate a camera and run this again:")
        print("    python calibrate.py --camera 0")
        print("\n  For WildTrack you would calibrate from a saved frame")
        print("  of each camera view.")
        calibrated = None
    else:
        cams = sorted(calibrated_map.homographies.keys())
        print(f"  calibrated cameras: {cams}")

        per_frame_cal = positions_per_frame(frames, calibrated_map,
                                            frame_shapes)
        calibrated = disagreement(per_frame_cal)

        if calibrated:
            print(f"  average disagreement        "
                  f"{calibrated['average_m']:.2f} m")
            print(f"  worst                       "
                  f"{calibrated['worst_m']:.2f} m")
            print(f"  placed within 2m            "
                  f"{calibrated['within_2m']:.1%}")

        agree_cal, checked_cal = zone_agreement(frames, calibrated_map,
                                                frame_shapes)
        print(f"  named the same zone         "
              f"{agree_cal:.1%} of {checked_cal}")

        if uncalibrated and calibrated:
            better = (uncalibrated["average_m"] - calibrated["average_m"])
            print(f"\n  calibration closed the gap by {better:.2f} m "
                  f"({better / max(uncalibrated['average_m'], 1e-6):.0%})")

    draw(uncalibrated, calibrated, C.LOGS / "location_accuracy.png")

    with open(C.LOGS / "location_results.json", "w") as f:
        json.dump({"frames": len(frames), "people": len(people),
                   "cameras": cameras,
                   "uncalibrated": uncalibrated,
                   "calibrated": calibrated}, f, indent=2, default=float)

    print("\n" + "=" * 58)
    print("  saved logs/location_accuracy.png")
    print("  saved logs/location_results.json")
    print("=" * 58)


if __name__ == "__main__":
    main()
