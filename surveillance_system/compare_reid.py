"""
compare_reid.py — colour matching versus a real network.

This is the experiment, not just a demo. Same data, same measurements,
two different ways of describing what a person looks like. The gap
between them is the result.

Run:   python compare_reid.py

Takes a few minutes. Produces a table you can paste into your results
chapter and two plots for your slides.
"""
import json
import sys
import time
from collections import defaultdict

import numpy as np

import config as C
from eval_wildtrack import (find_wildtrack, load_annotations, load_image,
                            cross_camera_scores, threshold_behaviour,
                            run_live_gallery)


MIN_WIDTH = 20        # crops smaller than this are mostly noise
MIN_HEIGHT = 50


def gather_crops(found, frames, max_per_person=6):
    """
    Pull every labelled person out of every camera view, once.

    Both descriptors then work from the same crops, so the comparison
    is fair — any difference is the descriptor, not the data.
    """
    import cv2

    seen = defaultdict(int)
    crops, labels = [], []
    total = len(frames)
    skipped_small = 0

    print(f"\n  cutting people out of {total} frames")

    for n, (frame_no, sightings) in enumerate(sorted(frames.items()), 1):
        if n % 20 == 0 or n == total:
            print(f"    {n}/{total} frames, {len(crops)} crops, "
                  f"{skipped_small} too small", end="\r")

        cache = {}
        for s in sightings:
            key = (s["person"], s["camera"])
            if seen[key] >= max_per_person:
                continue

            cache_key = s.get("image") or s["camera"]
            if cache_key not in cache:
                cache[cache_key] = load_image(found, s, frame_no)
            image = cache[cache_key]
            if image is None:
                continue

            x1, y1, x2, y2 = s["box"]
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 - x1 < MIN_WIDTH or y2 - y1 < MIN_HEIGHT:
                skipped_small += 1
                continue

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crops.append(crop)
            labels.append({"person": s["person"], "camera": s["camera"],
                           "frame": frame_no})
            seen[key] += 1

    print(f"    {total}/{total} frames, {len(crops)} crops kept, "
          f"{skipped_small} too small to use" + " " * 8)
    return crops, labels


def describe_all(describer, crops, name, batch=48):
    """Run one descriptor over every crop and time it."""
    print(f"\n  describing with {name}")
    start = time.perf_counter()

    if hasattr(describer, "describe_many"):
        signatures = []
        for i in range(0, len(crops), batch):
            signatures.append(describer.describe_many(crops[i:i + batch]))
            done = min(i + batch, len(crops))
            print(f"    {done}/{len(crops)}", end="\r")
        signatures = np.concatenate(signatures)
    else:
        signatures = []
        for i, crop in enumerate(crops, 1):
            signatures.append(describer.describe(crop))
            if i % 200 == 0 or i == len(crops):
                print(f"    {i}/{len(crops)}", end="\r")
        signatures = np.stack(signatures)

    elapsed = time.perf_counter() - start
    per_crop = elapsed / max(len(crops), 1) * 1000
    print(f"    {len(crops)}/{len(crops)} done in {elapsed:.0f}s "
          f"({per_crop:.1f} ms each)" + " " * 8)
    return signatures, per_crop


def measure(signatures, labels):
    """All the numbers for one descriptor."""
    records = [{"person": l["person"], "camera": l["camera"],
                "frame": l["frame"], "signature": sig}
               for l, sig in zip(labels, signatures)]

    scores = cross_camera_scores(records)
    sweep = threshold_behaviour(records)
    best = max(sweep, key=lambda s: s["f1"]) if sweep else None
    gallery = run_live_gallery(records, best["threshold"] if best
                               else C.REID_THRESHOLD)

    return {"scores": scores, "sweep": sweep,
            "best": best, "gallery": gallery}


def draw_comparison(results, save_to):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results.keys())
    metrics = ["rank1", "rank5", "mAP"]
    pretty = ["rank-1", "rank-5", "mAP"]
    colours = ["#0B7285", "#B03A2E", "#B07219", "#5F3D7E", "#276749"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))

    # Accuracy bars
    x = np.arange(len(metrics))
    width = 0.8 / max(len(names), 1)
    for i, name in enumerate(names):
        s = results[name]["scores"]
        if not s:
            continue
        values = [s[m] * 100 for m in metrics]
        offset = (i - (len(names) - 1) / 2) * width
        bars = ax1.bar(x + offset, values, width,
                       label=name, color=colours[i % len(colours)])
        for bar, v in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                     f"{v:.0f}", ha="center", fontsize=7.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(pretty)
    ax1.set_ylabel("percent")
    ax1.set_title("Matching people across cameras")
    ax1.set_ylim(0, 105)
    ax1.legend(frameon=False, fontsize=8.5)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_axisbelow(True)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)

    # Threshold curves
    for i, name in enumerate(names):
        sweep = results[name]["sweep"]
        if not sweep:
            continue
        t = [s["threshold"] for s in sweep]
        f = [s["f1"] for s in sweep]
        ax2.plot(t, f, lw=2, label=name, color=colours[i % len(colours)])
        best = results[name]["best"]
        if best:
            ax2.plot(best["threshold"], best["f1"], "o",
                     color=colours[i % len(colours)], ms=7)

    ax2.set_xlabel("matching threshold")
    ax2.set_ylabel("f1")
    ax2.set_title("Best threshold for each")
    ax2.legend(frameon=False, fontsize=8.5)
    ax2.grid(alpha=0.3)
    ax2.set_axisbelow(True)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_to, dpi=160)
    plt.close()


def print_table(results, timings):
    """The table that goes in your dissertation."""
    names = list(results.keys())

    print("\n" + "=" * 80)
    print("  Results")
    print("=" * 80)

    rows = [
        ("rank-1 accuracy", lambda r: f"{r['scores']['rank1']:.1%}"
                            if r["scores"] else "-"),
        ("rank-5 accuracy", lambda r: f"{r['scores']['rank5']:.1%}"
                            if r["scores"] else "-"),
        ("mAP", lambda r: f"{r['scores']['mAP']:.1%}"
                if r["scores"] else "-"),
        ("best threshold", lambda r: f"{r['best']['threshold']:.2f}"
                           if r["best"] else "-"),
        ("precision there", lambda r: f"{r['best']['precision']:.1%}"
                            if r["best"] else "-"),
        ("recall there", lambda r: f"{r['best']['recall']:.1%}"
                         if r["best"] else "-"),
        ("people held together", lambda r: f"{r['gallery']['consistency']:.0%}"),
        ("ids created", lambda r: str(r["gallery"]["ids_created"])),
        ("people merged", lambda r: str(r["gallery"]["merged"])),
    ]

    width = 16
    header = f"  {'':<22}" + "".join(f"{n:>{width}}" for n in names)
    print(header)
    print("  " + "-" * (22 + width * len(names)))

    for label, get in rows:
        line = f"  {label:<22}"
        for name in names:
            try:
                line += f"{get(results[name]):>{width}}"
            except (KeyError, TypeError):
                line += f"{'-':>{width}}"
        print(line)

    line = f"  {'ms per person':<22}"
    for name in names:
        line += f"{timings[name]:>{width}.1f}"
    print(line)
    print("=" * (22 + width * len(names) + 2))


def main():
    print("=" * 66)
    print("  Colour matching versus a trained network")
    print("=" * 66)

    root = getattr(C, "WILDTRACK_FOLDER", None)
    found = find_wildtrack(root) if root else None
    if found is None:
        print(f"\n  couldn't find WildTrack at {root}")
        print("  check WILDTRACK_FOLDER in config.py")
        sys.exit(1)

    frames = load_annotations(
        found, max_frames=getattr(C, "WILDTRACK_FRAMES", 60))
    if not frames:
        print("\n  no annotations could be read")
        sys.exit(1)

    people = {s["person"] for ss in frames.values() for s in ss}
    cameras = {s["camera"] for ss in frames.values() for s in ss}
    print(f"\n  {len(frames)} frames, {len(people)} people, "
          f"{len(cameras)} cameras")

    crops, labels = gather_crops(found, frames)
    if len(crops) < 20:
        print("\n  too few usable crops")
        sys.exit(1)

    from reid import (ColourSignature, DeepSignature,
                      CombinedSignature, TrainedSignature)

    results, timings = {}, {}

    def add(name, describer):
        if hasattr(describer, "ready") and not describer.ready:
            return
        sigs, ms = describe_all(describer, crops, name)
        results[name] = measure(sigs, labels)
        timings[name] = ms

    # The baseline. Always works, nothing to install.
    add("colour", ColourSignature())

    # The obvious deep version. Usually disappointing, and the reason
    # is the interesting part.
    deep_final = DeepSignature(depth="final")
    if deep_final.ready:
        add("resnet final", deep_final)

        # Earlier layers keep texture and pattern instead of collapsing
        # to "this is a person", which every crop already is.
        add("resnet layer4", DeepSignature(depth="deep"))
        add("resnet layer3", DeepSignature(depth="mid"))
        add("resnet layer2", DeepSignature(depth="early"))

        # Colour and network fail differently, so both together
        # usually beats either alone.
        add("colour + layer3", CombinedSignature(depth="mid"))

    # The model actually trained for this job, if it exists
    trained = TrainedSignature()
    if trained.ready:
        add("trained on Market", trained)
    else:
        print("\n  torchvision isn't available, so only colour was measured")
        print("  pip install torchvision")

    print_table(results, timings)

    # Which one actually won
    ranked = [(name, r["scores"]["rank1"])
              for name, r in results.items() if r["scores"]]
    if len(ranked) > 1:
        ranked.sort(key=lambda x: x[1], reverse=True)
        winner, best_score = ranked[0]
        baseline = dict(ranked).get("colour", 0)

        print(f"\n  Best: {winner} at {best_score:.1%} rank-1")
        print(f"  That's {best_score - baseline:+.1%} against the colour baseline.")

        if winner != "colour":
            best = results[winner]["best"]
            print(f"\n  Put this in config.py:")
            print(f"    REID_USE_DEEP  = True")
            if "layer3" in winner:
                print(f'    REID_DEPTH     = "mid"')
            elif "layer4" in winner:
                print(f'    REID_DEPTH     = "deep"')
            elif "layer2" in winner:
                print(f'    REID_DEPTH     = "early"')
            elif "trained" in winner:
                print("    (the trained model is picked up automatically)")
            if best:
                print(f"    REID_THRESHOLD = {best['threshold']:.2f}")

        print("\n  Worth noting for your write-up: the final layer of an")
        print("  ImageNet network is trained to recognise object categories.")
        print("  Every crop here is already a person, so that layer says the")
        print("  same thing about all of them. Earlier layers still carry the")
        print("  texture and colour that actually separate one person from")
        print("  another, which is why they do better at this.")

    draw_comparison(results, C.LOGS / "reid_comparison.png")

    with open(C.LOGS / "reid_comparison.json", "w") as f:
        json.dump({
            "frames": len(frames), "people": len(people),
            "cameras": sorted(cameras), "crops": len(crops),
            "results": {k: {"scores": v["scores"], "best": v["best"],
                            "gallery": v["gallery"]}
                        for k, v in results.items()},
            "ms_per_person": timings,
        }, f, indent=2, default=float)

    print(f"\n  saved logs/reid_comparison.png")
    print(f"  saved logs/reid_comparison.json")


if __name__ == "__main__":
    main()
