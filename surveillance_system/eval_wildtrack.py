"""
eval_wildtrack.py — measuring whether re-identification actually works.

Running the system on a video shows you it does something. It doesn't
tell you how often it's right. WildTrack does, because seven cameras
watch the same square at the same time and every person is labelled,
so we know the correct answer for every match.

Run:
    python eval_wildtrack.py

What comes out:
    rank-1 accuracy    given a sighting, is the closest match the right person
    mAP                how well ranked the whole gallery is
    cross-camera rate  how often the same person is linked between views
    ID switches        how often two different people get merged

Those numbers go in your results chapter. The plots go in your slides.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import config as C


# ══════════════════════════════════════════════════════════════════════
#  Finding the dataset
# ══════════════════════════════════════════════════════════════════════
def find_wildtrack(root):
    """
    WildTrack comes in two shapes depending on where you got it.

    The original release:
        Image_subsets/C1 ... C7        images grouped by camera
        annotations_positions/*.json   one file per frame, all cameras

    The DatasetNinja re-release (Supervisely format):
        img/C1_00000000.png            camera baked into the filename
        ann/C1_00000000.png.json       one file per image
        meta/

    This works out which one you have.
    """
    root = Path(root)
    if not root.exists():
        return None

    def check(folder):
        try:
            inside = {p.name.lower(): p for p in folder.iterdir() if p.is_dir()}
        except (PermissionError, OSError):
            return None

        # DatasetNinja / Supervisely
        if "img" in inside and "ann" in inside:
            return {"style": "supervisely",
                    "images": inside["img"], "labels": inside["ann"]}

        # Original release
        images = next((inside[n] for n in
                       ("image_subsets", "images", "frames") if n in inside), None)
        labels = next((inside[n] for n in
                       ("annotations_positions", "annotations", "labels")
                       if n in inside), None)
        if images and labels:
            return {"style": "original", "images": images, "labels": labels}
        return None

    found = check(root)
    if found:
        return found

    # Look one level down too
    try:
        for child in root.iterdir():
            if child.is_dir():
                found = check(child)
                if found:
                    return found
    except (PermissionError, OSError):
        pass

    return None


# ── Original release ──────────────────────────────────────────────────
def load_original(labels_dir, max_frames=None):
    """
    One json per frame. Each person lists a box for every camera that
    can see them, and -1 for the cameras that can't.
    """
    files = sorted(labels_dir.glob("*.json"))
    if max_frames:
        files = files[:max_frames]

    frames = {}
    for path in files:
        try:
            with open(path) as f:
                people = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        frame_no = int(path.stem)
        sightings = []
        for entry in people:
            person = entry.get("personID")
            if person is None:
                continue
            for view in entry.get("views", []):
                if view.get("xmax", -1) < 0:
                    continue
                sightings.append({
                    "person": int(person),
                    "camera": int(view["viewNum"]),
                    "box": (int(view["xmin"]), int(view["ymin"]),
                            int(view["xmax"]), int(view["ymax"])),
                    "image": None,
                })
        if sightings:
            frames[frame_no] = sightings

    return frames


# ── DatasetNinja / Supervisely ────────────────────────────────────────
CAMERA_IN_NAME = re.compile(r"[Cc](\d+)")
FRAME_IN_NAME = re.compile(r"(\d{4,})")


def read_name(filename):
    """
    Pull the camera and frame out of something like C1_00000000.png.

    Returns (camera, frame). Camera comes back 0-based to match the
    original release, so both styles behave the same downstream.
    """
    stem = Path(filename).stem
    if stem.endswith(".png") or stem.endswith(".jpg"):
        stem = Path(stem).stem

    cam_match = CAMERA_IN_NAME.search(stem)
    camera = int(cam_match.group(1)) - 1 if cam_match else 0

    numbers = FRAME_IN_NAME.findall(stem)
    frame = int(numbers[-1]) if numbers else 0

    return max(camera, 0), frame


def person_from_tags(obj):
    """
    Find the person ID. Different exports name the tag differently,
    so several spellings are accepted.
    """
    for tag in obj.get("tags", []):
        name = str(tag.get("name", "")).lower().replace("_", " ")
        if "person" in name and "id" in name:
            value = tag.get("value")
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass

    # Some exports put it in the object id instead
    for key in ("personID", "person_id"):
        if key in obj:
            try:
                return int(obj[key])
            except (TypeError, ValueError):
                pass
    return None


def load_supervisely(labels_dir, images_dir, max_frames=None):
    """
    One json per image, named after the image it describes.
    Boxes are corner points rather than min/max fields.
    """
    files = sorted(labels_dir.glob("*.json"))
    if not files:
        return {}

    # Group by frame so the rest of the script sees the same shape
    frames = defaultdict(list)
    frames_seen = set()

    for path in files:
        camera, frame_no = read_name(path.name)

        if max_frames and frame_no not in frames_seen:
            if len(frames_seen) >= max_frames:
                continue
            frames_seen.add(frame_no)

        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        image_name = path.name
        if image_name.endswith(".json"):
            image_name = image_name[:-5]
        image_path = images_dir / image_name
        if not image_path.exists():
            found = list(images_dir.glob(Path(image_name).stem + ".*"))
            image_path = found[0] if found else None

        for obj in data.get("objects", []):
            if obj.get("geometryType") not in (None, "rectangle"):
                continue

            person = person_from_tags(obj)
            if person is None:
                continue

            corners = obj.get("points", {}).get("exterior", [])
            if len(corners) < 2:
                continue

            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            box = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

            frames[frame_no].append({
                "person": person,
                "camera": camera,
                "box": box,
                "image": image_path,
            })

    return {k: v for k, v in frames.items() if v}


def load_annotations(found, max_frames=None):
    """Read whichever format this dataset is in."""
    if found["style"] == "supervisely":
        return load_supervisely(found["labels"], found["images"], max_frames)
    return load_original(found["labels"], max_frames)


# ── Getting the pixels ────────────────────────────────────────────────
def camera_folder(images_dir, camera):
    """The original release numbers cameras from 1 on disk, 0 in labels."""
    for name in (f"C{camera + 1}", f"c{camera + 1}",
                 f"C{camera}", f"cam{camera + 1}"):
        path = images_dir / name
        if path.is_dir():
            return path
    return None


def load_image(found, sighting, frame_no):
    """Works for both styles."""
    import cv2

    # Supervisely already knows the file
    if sighting.get("image") is not None:
        return cv2.imread(str(sighting["image"]))

    folder = camera_folder(found["images"], sighting["camera"])
    if folder is None:
        return None
    for pattern in (f"{frame_no:08d}.png", f"{frame_no:08d}.jpg",
                    f"{frame_no:05d}.png", f"{frame_no}.png"):
        path = folder / pattern
        if path.exists():
            return cv2.imread(str(path))
    return None


# ══════════════════════════════════════════════════════════════════════
#  Building descriptions of everyone
# ══════════════════════════════════════════════════════════════════════
def collect_signatures(found, frames, describer, max_per_person=6):
    """
    Crop every labelled person out of every camera view and describe
    what they look like.

    Returns a list of {person, camera, frame, signature}.
    """
    import cv2

    seen_count = defaultdict(int)
    records = []
    total_frames = len(frames)

    print(f"\n  reading {total_frames} frames")

    for n, (frame_no, sightings) in enumerate(sorted(frames.items()), 1):
        if n % 20 == 0 or n == total_frames:
            print(f"    {n}/{total_frames} frames, "
                  f"{len(records)} sightings described", end="\r")

        # Load each camera's image once per frame, not once per person
        cache = {}

        for s in sightings:
            key = (s["person"], s["camera"])
            if seen_count[key] >= max_per_person:
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
            if x2 - x1 < 12 or y2 - y1 < 24:
                continue          # too small to describe usefully

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            records.append({
                "person": s["person"],
                "camera": s["camera"],
                "frame": frame_no,
                "signature": describer.describe(crop),
            })
            seen_count[key] += 1

    print(f"    {total_frames}/{total_frames} frames, "
          f"{len(records)} sightings described" + " " * 10)
    return records


# ══════════════════════════════════════════════════════════════════════
#  The measurements
# ══════════════════════════════════════════════════════════════════════
def cross_camera_scores(records):
    """
    The proper re-ID test.

    Take each sighting in turn. Search only the OTHER cameras for a match.
    If the closest one is the same person, that's a rank-1 hit.

    Searching the same camera would be cheating, since consecutive frames
    of one person look almost identical.
    """
    if len(records) < 2:
        return None

    signatures = np.stack([r["signature"] for r in records]).astype(np.float32)
    people = np.array([r["person"] for r in records])
    cameras = np.array([r["camera"] for r in records])

    # With a couple of thousand sightings and 2048 numbers each, building
    # the whole similarity matrix at once needs more memory than most
    # laptops have. One query at a time costs almost nothing.

    rank1_hits = 0
    rank5_hits = 0
    average_precisions = []
    usable = 0

    for i in range(len(records)):
        other_camera = cameras != cameras[i]
        if not other_camera.any():
            continue

        correct = people[other_camera] == people[i]
        if not correct.any():
            continue          # this person was never seen elsewhere

        usable += 1
        scores = signatures[other_camera] @ signatures[i]
        order = np.argsort(scores)[::-1]
        ranked = correct[order]

        if ranked[0]:
            rank1_hits += 1
        if ranked[:5].any():
            rank5_hits += 1

        # average precision for this query
        hits = 0
        precision_sum = 0.0
        for position, is_right in enumerate(ranked, 1):
            if is_right:
                hits += 1
                precision_sum += hits / position
        average_precisions.append(precision_sum / max(hits, 1))

    if usable == 0:
        return None

    return {
        "queries": usable,
        "rank1": rank1_hits / usable,
        "rank5": rank5_hits / usable,
        "mAP": float(np.mean(average_precisions)),
    }


def threshold_behaviour(records, thresholds=None):
    """
    How the matching threshold trades off correct links against wrong ones.

    Too low and different people get merged. Too high and the same person
    keeps being treated as a stranger. This finds where to set it.
    """
    if thresholds is None:
        thresholds = np.arange(0.40, 0.96, 0.02)

    signatures = np.stack([r["signature"] for r in records]).astype(np.float32)
    people = np.array([r["person"] for r in records])
    cameras = np.array([r["camera"] for r in records])

    # Every pair of sightings from different cameras.
    # With thousands of sightings that's millions of pairs, so it's
    # sampled down. The shape of the curve doesn't change.
    n = len(records)
    MAX_PAIRS = 400_000

    rows, cols = np.triu_indices(n, k=1)
    different_camera = cameras[rows] != cameras[cols]
    rows, cols = rows[different_camera], cols[different_camera]

    if len(rows) == 0:
        return None

    if len(rows) > MAX_PAIRS:
        rng = np.random.default_rng(C.SEED)
        keep = rng.choice(len(rows), MAX_PAIRS, replace=False)
        rows, cols = rows[keep], cols[keep]

    scores = np.einsum("ij,ij->i", signatures[rows], signatures[cols])
    same_person = people[rows] == people[cols]

    results = []
    for t in thresholds:
        linked = scores >= t
        true_pos = np.sum(linked & same_person)
        false_pos = np.sum(linked & ~same_person)
        false_neg = np.sum(~linked & same_person)

        precision = true_pos / max(true_pos + false_pos, 1)
        recall = true_pos / max(true_pos + false_neg, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        results.append({"threshold": float(t), "precision": float(precision),
                        "recall": float(recall), "f1": float(f1),
                        "wrong_links": int(false_pos)})

    return results


def run_live_gallery(records, threshold):
    """
    Run the actual PersonGallery from reid.py over the sightings, in
    order, the way it would run on live cameras. Then check how many
    global IDs it ended up with versus how many people there really were.
    """
    from reid import PersonGallery

    gallery = PersonGallery(match_threshold=threshold, forget_after=1e9)
    assigned = {}

    for r in sorted(records, key=lambda x: (x["frame"], x["camera"])):
        gid, matched, score = gallery.identify(
            r["signature"], r["camera"], 0)
        assigned.setdefault(r["person"], []).append(gid)

    real_people = len(assigned)
    ids_created = len(gallery.people)

    # A person is "held together" if most of their sightings share one ID
    consistent = 0
    fragmented = 0
    for person, gids in assigned.items():
        counts = defaultdict(int)
        for g in gids:
            counts[g] += 1
        biggest = max(counts.values())
        if biggest / len(gids) >= 0.7:
            consistent += 1
        if len(counts) > 1:
            fragmented += 1

    # Two real people sharing one global ID is a merge
    gid_owners = defaultdict(set)
    for person, gids in assigned.items():
        for g in gids:
            gid_owners[g].add(person)
    merges = sum(1 for owners in gid_owners.values() if len(owners) > 1)

    return {
        "real_people": real_people,
        "ids_created": ids_created,
        "held_together": consistent,
        "fragmented": fragmented,
        "merged": merges,
        "consistency": consistent / max(real_people, 1),
    }


# ══════════════════════════════════════════════════════════════════════
#  Plots
# ══════════════════════════════════════════════════════════════════════
def draw_threshold_curve(sweep, save_to):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = [s["threshold"] for s in sweep]
    p = [s["precision"] for s in sweep]
    r = [s["recall"] for s in sweep]
    f = [s["f1"] for s in sweep]

    best = max(sweep, key=lambda s: s["f1"])

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(t, p, color="#0B7285", lw=2, label="precision")
    ax.plot(t, r, color="#B07219", lw=2, label="recall")
    ax.plot(t, f, color="#5F3D7E", lw=2, ls="--", label="f1")

    ax.axvline(best["threshold"], color="#5A6B7D", lw=1, ls=":")
    ax.text(best["threshold"] + 0.008, 0.05,
            f"best  {best['threshold']:.2f}",
            fontsize=9, color="#5A6B7D", style="italic")
    ax.axvline(C.REID_THRESHOLD, color="#B03A2E", lw=1, ls=":")
    ax.text(C.REID_THRESHOLD + 0.008, 0.14,
            f"current  {C.REID_THRESHOLD:.2f}",
            fontsize=9, color="#B03A2E", style="italic")

    ax.set_xlabel("how alike two sightings must be to count as the same person")
    ax.set_ylabel("score")
    ax.set_title("Where to set the matching threshold")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.18))
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    plt.subplots_adjust(bottom=0.26)
    plt.savefig(save_to, dpi=160)
    plt.close()
    return best


def draw_camera_grid(records, save_to):
    """How often each pair of cameras sees the same people."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cameras = sorted({r["camera"] for r in records})
    seen_by = defaultdict(set)
    for r in records:
        seen_by[r["person"]].add(r["camera"])

    size = len(cameras)
    grid = np.zeros((size, size), dtype=int)
    index = {c: i for i, c in enumerate(cameras)}

    for person, cams in seen_by.items():
        for a in cams:
            for b in cams:
                grid[index[a], index[b]] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    img = ax.imshow(grid, cmap="Blues")
    ax.set_xticks(range(size)); ax.set_yticks(range(size))
    ax.set_xticklabels([f"cam {c}" for c in cameras], fontsize=9)
    ax.set_yticklabels([f"cam {c}" for c in cameras], fontsize=9)
    ax.set_title("People seen by both cameras")

    for i in range(size):
        for j in range(size):
            ax.text(j, i, str(grid[i, j]), ha="center", va="center",
                    fontsize=8.5,
                    color="white" if grid[i, j] > grid.max() * 0.55 else "#333")

    fig.colorbar(img, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(save_to, dpi=160)
    plt.close()


# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 52)
    print("  Measuring re-identification on WildTrack")
    print("=" * 52)

    root = getattr(C, "WILDTRACK_FOLDER", None)
    if not root:
        print("\n  WILDTRACK_FOLDER isn't set in config.py")
        sys.exit(1)

    found = find_wildtrack(root)
    if found is None:
        print(f"\n  couldn't find WildTrack under:\n    {root}")
        print("\n  expected either")
        print("    Image_subsets/ and annotations_positions/   (original)")
        print("    img/ and ann/                               (DatasetNinja)")
        print("\n  what's actually there:")
        p = Path(root)
        if p.exists():
            for item in list(p.iterdir())[:12]:
                print(f"    {item.name}")
        else:
            print("    (the folder doesn't exist)")
        sys.exit(1)

    print(f"\n  format       {found['style']}")
    print(f"  images from  {found['images']}")
    print(f"  labels from  {found['labels']}")

    max_frames = getattr(C, "WILDTRACK_FRAMES", 60)
    frames = load_annotations(found, max_frames=max_frames)
    if not frames:
        print("\n  no usable annotations found")
        sys.exit(1)

    people = {s["person"] for ss in frames.values() for s in ss}
    cameras = {s["camera"] for ss in frames.values() for s in ss}
    print(f"  {len(frames)} frames, {len(people)} people, "
          f"{len(cameras)} cameras")

    # Describe everyone
    from reid import ColourSignature, DeepSignature
    if C.REID_USE_DEEP:
        describer = DeepSignature()
        if not describer.ready:
            describer = ColourSignature()
    else:
        describer = ColourSignature()
        print("  describing appearance by colour")

    records = collect_signatures(found, frames, describer)
    if len(records) < 10:
        print("\n  too few sightings to measure anything")
        print("  check the image folder names match the cameras")
        sys.exit(1)

    # ── Results ───────────────────────────────────────────────────────
    print("\n" + "-" * 52)
    print("  Matching people across cameras")
    print("-" * 52)

    scores = cross_camera_scores(records)
    if scores:
        print(f"  queries tested     {scores['queries']}")
        print(f"  rank-1 accuracy    {scores['rank1']:.1%}"
              f"    closest match was the right person")
        print(f"  rank-5 accuracy    {scores['rank5']:.1%}"
              f"    right person in the top five")
        print(f"  mAP                {scores['mAP']:.1%}")
    else:
        print("  not enough people appear on more than one camera")

    print("\n" + "-" * 52)
    print("  Running the real gallery over the sightings")
    print("-" * 52)

    live = run_live_gallery(records, C.REID_THRESHOLD)
    print(f"  people in the data     {live['real_people']}")
    print(f"  ids the system made    {live['ids_created']}")
    print(f"  held together          {live['held_together']} "
          f"({live['consistency']:.0%})")
    print(f"  split across ids       {live['fragmented']}")
    print(f"  merged with someone    {live['merged']}")

    print("\n" + "-" * 52)
    print("  Choosing the threshold")
    print("-" * 52)

    sweep = threshold_behaviour(records)
    best = None
    if sweep:
        best = draw_threshold_curve(sweep, C.LOGS / "reid_threshold.png")
        print(f"  best f1 at             {best['threshold']:.2f}")
        print(f"    precision            {best['precision']:.1%}")
        print(f"    recall               {best['recall']:.1%}")
        print(f"  you are using          {C.REID_THRESHOLD:.2f}")
        if abs(best["threshold"] - C.REID_THRESHOLD) > 0.04:
            print(f"\n  worth changing REID_THRESHOLD to "
                  f"{best['threshold']:.2f} in config.py")

    draw_camera_grid(records, C.LOGS / "reid_cameras.png")

    # ── Save ──────────────────────────────────────────────────────────
    summary = {
        "frames": len(frames), "people": len(people),
        "cameras": sorted(cameras), "sightings": len(records),
        "cross_camera": scores, "live_gallery": live,
        "best_threshold": best, "threshold_used": C.REID_THRESHOLD,
    }
    with open(C.LOGS / "reid_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n" + "=" * 52)
    print("  saved to logs/")
    print("    reid_results.json")
    print("    reid_threshold.png")
    print("    reid_cameras.png")
    print("=" * 52)


if __name__ == "__main__":
    main()
