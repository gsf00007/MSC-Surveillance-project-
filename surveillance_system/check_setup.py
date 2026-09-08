"""
check_setup.py — is everything ready?

Run:   python check_setup.py

Checks packages, datasets, trained models and demo files, then tells you
exactly what to do about anything missing. Run it before a demo so you
find problems now rather than in front of your examiner.
"""
import sys
from pathlib import Path

import config as C


OK = "  [ok]  "
NO = "  [--]  "
WARN = "  [??]  "


def heading(text):
    print()
    print("-" * 56)
    print(f"  {text}")
    print("-" * 56)


def check_packages():
    heading("Packages")
    import importlib

    needed = {
        "numpy": "core",
        "torch": "the models",
        "cv2": "video",
        "mediapipe": "pose reading",
        "ultralytics": "person detection",
        "sklearn": "the classifier",
        "joblib": "loading models",
        "flask": "dashboard",
        "flask_socketio": "dashboard",
    }
    optional = {
        "shap": "explanations",
        "twilio": "phone calls",
        "torchvision": "the trained re-id model",
        "matplotlib": "plots",
    }

    missing = []
    for name, why in needed.items():
        try:
            importlib.import_module(name)
            print(f"{OK}{name:<16} {why}")
        except ImportError:
            print(f"{NO}{name:<16} {why}   MISSING")
            missing.append(name)

    for name, why in optional.items():
        try:
            importlib.import_module(name)
            print(f"{OK}{name:<16} {why}")
        except ImportError:
            print(f"{WARN}{name:<16} {why}   optional")

    if missing:
        fixes = {"cv2": "opencv-python", "sklearn": "scikit-learn",
                 "flask_socketio": "flask-socketio"}
        names = " ".join(fixes.get(m, m) for m in missing)
        print(f"\n  fix with:  pip install {names}")

    return not missing


def check_models():
    heading("Trained models")

    gesture_net = C.WEIGHTS / "motion_net.pth"
    gesture_svm = C.WEIGHTS / "classifier.pkl"
    reid_net = C.WEIGHTS / "reid_net.pth"

    have_gesture = gesture_net.exists() and gesture_svm.exists()
    have_reid = reid_net.exists()

    # Gesture classifier
    if have_gesture:
        size = gesture_net.stat().st_size / 1e6
        print(f"{OK}gesture classifier    ({size:.1f} MB)")
        try:
            import torch, joblib
            from train import MotionNet
            net = MotionNet()
            net.load_state_dict(torch.load(gesture_net, map_location="cpu"))
            svm, scaler = joblib.load(gesture_svm)
            print(f"{OK}  loads correctly, {len(svm.classes_)} classes")
        except Exception as e:
            print(f"{NO}  won't load: {e}")
            print("       retrain:  python prepare_data.py && python train.py")
            have_gesture = False
    else:
        print(f"{NO}gesture classifier    MISSING")
        print("       without this the system only detects and tracks,")
        print("       it can't tell a collapse from normal walking")
        print("       fix:  python prepare_data.py")
        print("             python train.py")

    # Re-ID model
    if have_reid:
        size = reid_net.stat().st_size / 1e6
        print(f"{OK}re-id model           ({size:.1f} MB)")
        try:
            import torch
            saved = torch.load(reid_net, map_location="cpu")
            state = saved["state"]

            # Read the shape from the weights rather than the metadata,
            # since older saves didn't record the architecture.
            width = None
            if "norm.weight" in state:
                width = state["norm.weight"].shape[0]
            elif "name_them.weight" in state:
                width = state["name_them.weight"].shape[1]

            backbone = ("resnet50" if width == 2048
                        else "resnet18" if width == 512
                        else saved.get("backbone", "unknown"))
            people = saved.get("people", "?")
            if "name_them.weight" in state:
                people = state["name_them.weight"].shape[0]

            print(f"{OK}  trained on {people} people ({backbone})")
        except Exception as e:
            print(f"{WARN}  odd format: {e}")
    else:
        print(f"{WARN}re-id model           missing (optional)")
        print("       the system still works, it just falls back to")
        print("       colour matching for recognising people")
        print("       fix:  python train_reid.py")

    return have_gesture


def check_data():
    heading("Prepared data")

    X = C.DATA_DIR / "X.npy"
    y = C.DATA_DIR / "y.npy"

    if X.exists() and y.exists():
        try:
            import numpy as np
            samples = np.load(X, mmap_mode="r")
            labels = np.load(y)
            print(f"{OK}training data         {samples.shape[0]} samples")
            from collections import Counter
            counts = Counter(labels.tolist())
            for cid, n in sorted(counts.items()):
                name = C.LABELS[cid] if cid < len(C.LABELS) else str(cid)
                print(f"         {name:>9}: {n}")
        except Exception as e:
            print(f"{WARN}training data         unreadable: {e}")
    else:
        print(f"{WARN}training data         not prepared")
        print("       only needed if you want to retrain")

    test = C.DATA_DIR / "X_test.npy"
    print(f"{OK if test.exists() else WARN}test split            "
          f"{'present' if test.exists() else 'missing (needed for evaluate.py)'}")


def check_datasets():
    heading("Datasets")

    for label, path, why in [
        ("NTU RGB+D", C.NTU_FOLDER, "training the gesture classifier"),
        ("WildTrack", getattr(C, "WILDTRACK_FOLDER", ""), "re-id evaluation"),
        ("Market-1501", getattr(C, "MARKET_FOLDER", ""), "training re-id"),
    ]:
        if path and Path(path).exists():
            print(f"{OK}{label:<14} {why}")
        else:
            print(f"{WARN}{label:<14} not found — {why}")
            print(f"         looked in: {path}")

    print("\n  These are only needed for retraining or re-running")
    print("  evaluation. The live demo doesn't touch them.")


def check_demo_bits():
    heading("Demo")

    cal = C.DATA_DIR / "calibration.json"
    if cal.exists():
        try:
            from calibrate import load_all
            everything = load_all()
            print(f"{OK}calibration           {len(everything)} camera(s)")
        except Exception:
            print(f"{WARN}calibration           file exists but won't read")
    else:
        print(f"{WARN}calibration           none")
        print("       zones will say '?' meaning they're guessed")
        print("       fix:  python calibrate.py --camera 0")

    plots = list(C.LOGS.glob("*.png")) if C.LOGS.exists() else []
    if plots:
        print(f"{OK}result plots          {len(plots)} in logs/")
        for p in sorted(plots)[:6]:
            print(f"         {p.name}")
    else:
        print(f"{WARN}result plots          none in logs/")
        print("       fix:  python evaluate.py")

    # Twilio
    if str(C.TWILIO_SID).startswith("AC"):
        mode = "test phone" if C.TEST_MODE else "REAL SERVICES"
        print(f"{OK}phone calls           configured, calling {mode}")
        if not C.TEST_MODE:
            print(f"{NO}  TEST_MODE IS OFF. Turn it back on unless this")
            print("       demo is supervised and intended.")
    else:
        print(f"{WARN}phone calls           not set up")
        print("       the countdown still shows, no call is placed")
        print("       fix:  python setup_calls.py")


def check_settings():
    heading("Current settings")
    print(f"  a RED alert needs {C.RED_CONFIDENCE:.0%} confidence "
          f"held for {C.RED_HOLD_SECONDS:.0f} seconds")
    print(f"  YELLOW needs {C.YELLOW_CONFIDENCE:.0%} "
          f"held for {C.YELLOW_HOLD_SECONDS:.0f} seconds")
    print(f"  processing every {C.PROCESS_EVERY} frame(s)")
    print(f"  {C.OVERRIDE_SECONDS} seconds to cancel before a call goes out")


def main():
    print("=" * 56)
    print("  Is everything ready?")
    print("=" * 56)

    packages_ok = check_packages()
    models_ok = check_models()
    check_data()
    check_datasets()
    check_demo_bits()
    check_settings()

    print()
    print("=" * 56)
    if packages_ok and models_ok:
        print("  Ready to demo.")
        print()
        print("  python run_live.py --dashboard --explain")
        print('  python run_live.py --video "yourfile.mp4" --dashboard')
    elif packages_ok:
        print("  Packages fine, but the gesture classifier is missing.")
        print()
        print("  python prepare_data.py     (~15 min)")
        print("  python train.py            (~30 min)")
    else:
        print("  Install the missing packages first, then run this again.")
    print("=" * 56)


if __name__ == "__main__":
    main()
