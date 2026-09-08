"""
prepare_data.py — turn NTU skeleton files into training data.

Run:   python prepare_data.py

Reads your NTU folder, keeps only the movements you care about, samples
evenly across people, and saves the result so you never have to do it again.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import config as C
import skeleton as S


def check_folder():
    """Make sure the path in config.py actually points somewhere real."""
    folder = Path(C.NTU_FOLDER)

    if folder.exists():
        files = sorted(folder.rglob("*.skeleton"))
        if files:
            print(f"Found {len(files)} skeleton files")
            print(f"  example: {files[0].name}")
            return files
        print(f"Folder exists but contains no .skeleton files:\n  {folder}")
        print("\nWhat's actually in there:")
        for item in list(folder.iterdir())[:10]:
            print("   ", item.name)
        return None

    # Walk the path and show exactly where it breaks
    print(f"Folder not found:\n  {folder}\n")
    print("Checking each level:")
    parts = folder.parts
    for i in range(1, len(parts) + 1):
        step = Path(*parts[:i])
        print(f"  [{'OK  ' if step.exists() else 'MISS'}] {step}")
    print("\nThe first MISS is where your path goes wrong.")
    print("Fix NTU_FOLDER at the top of config.py.")
    return None


def survey(files):
    """Group files by action and report which target classes you have."""
    by_action = defaultdict(list)
    for f in files:
        info = S.read_filename(f)
        if info:
            by_action[info["action"]].append((f, info))

    print(f"\n{len(by_action)} different action classes in your download\n")
    print("Your target movements:")
    print(f"  {'action':>7}  {'label':>9}  {'files':>6}  {'people':>7}")
    print("  " + "-" * 36)

    found = {}
    for action_id, label in sorted(C.ACTIONS.items()):
        entries = by_action.get(action_id, [])
        people  = len({info["person"] for _, info in entries})
        note    = "" if entries else "   <-- not in your download"
        print(f"  A{action_id:03d}    {label:>9}  {len(entries):>6}  {people:>7}{note}")
        if entries:
            found[action_id] = label

    if not found:
        print("\nNone of your target actions are here.")
        print("Action IDs you do have:", sorted(by_action.keys()))
        return None, None

    missing = set(C.ACTIONS) - set(found)
    if missing:
        names = [C.ACTIONS[m] for m in missing]
        print(f"\nMissing: {', '.join(names)}")
        print("Training will continue with the classes you do have.")

    return by_action, found


def build(by_action, found):
    """Sample evenly across people, parse, normalise, and stack."""
    rng = np.random.default_rng(C.SEED)
    X, y, info_rows = [], [], []

    print("\nReading files...")
    for action_id, label in found.items():
        entries = by_action[action_id]

        # Group by person so no single person dominates the class
        by_person = defaultdict(list)
        for f, info in entries:
            by_person[info["person"]].append((f, info))

        per_person = max(1, C.MAX_FILES_PER_CLASS // max(len(by_person), 1))
        chosen = []
        for person, items in by_person.items():
            n = min(per_person, len(items))
            idx = rng.choice(len(items), n, replace=False)
            chosen += [items[i] for i in idx]

        rng.shuffle(chosen)
        chosen = chosen[:C.MAX_FILES_PER_CLASS]

        ok = 0
        for n, (f, info) in enumerate(chosen, 1):
            if n % 50 == 0 or n == len(chosen):
                print(f"  {label:>9}  {n}/{len(chosen)}", end="\r")

            seq = S.read_skeleton_file(f)
            if seq is None:
                continue
            seq = S.normalise(seq)

            for w in S.to_windows(seq):
                X.append(w)
                y.append(C.LABEL_ID[label])
                info_rows.append({"file": f.name,
                                  "person": info["person"],
                                  "camera": info["camera"],
                                  "label": label})
            ok += 1

        print(f"  {label:>9}  {ok}/{len(chosen)} files usable" + " " * 12)

    if not X:
        print("\nNothing could be parsed. The files may be a different format.")
        return None

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)

    np.save(C.DATA_DIR / "X.npy", X)
    np.save(C.DATA_DIR / "y.npy", y)
    with open(C.DATA_DIR / "info.json", "w") as f:
        json.dump(info_rows, f)

    print("\n" + "=" * 46)
    print(f"  samples : {X.shape[0]}")
    print(f"  shape   : {X.shape[1]} frames x {X.shape[2]} joints x {X.shape[3]}")
    print(f"  size    : {X.nbytes / 1e6:.1f} MB")
    print()
    for cid, n in sorted(Counter(y.tolist()).items()):
        print(f"  {C.LABELS[cid]:>9} : {n:>6}")
    print("=" * 46)
    print("\nSaved to data/X.npy and data/y.npy")
    print("Next:  python train.py")
    return X, y


def main():
    print("=" * 46)
    print("  Preparing training data")
    print("=" * 46 + "\n")

    files = check_folder()
    if files is None:
        sys.exit(1)

    by_action, found = survey(files)
    if found is None:
        sys.exit(1)

    build(by_action, found)


if __name__ == "__main__":
    main()
