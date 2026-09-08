"""
calibrate.py — teach the system where things actually are.

THE PROBLEM THIS SOLVES
-----------------------
Right now the system says "Lobby" or "Stairwell" based purely on where
someone appears in the frame. Left third of the picture, near the bottom,
call it the Lobby. That is a guess dressed up as a location, and it falls
apart the moment you ask a fair question: if two cameras both see the same
person, do they agree on where that person is standing? They don't, because
neither one knows anything about the room.

A homography fixes that. Click four points on the floor in the camera view,
say where those points are on the floor plan, and every pixel below the
camera can then be converted into a real position. Two cameras calibrated
against the same plan will agree, because they are both talking about the
same floor rather than about their own pictures.

Run:
    python calibrate.py --camera 0            calibrate a webcam
    python calibrate.py --video clip.mp4      calibrate from a video
    python calibrate.py --image frame.png     calibrate from a still
    python calibrate.py --check               see what's already saved

Click four points on the floor that you can identify on a plan, going
clockwise from the top left. Corners of a floor tile, doorway edges, table
legs. Anything flat and findable.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

import config as C
from reid import DEFAULT_ZONES


CALIBRATION_FILE = C.DATA_DIR / "calibration.json"


# ══════════════════════════════════════════════════════════════════════
#  Picking points
# ══════════════════════════════════════════════════════════════════════
class PointPicker:
    """Click four spots on the floor. Backspace undoes."""

    LABELS = ["top left", "top right", "bottom right", "bottom left"]

    def __init__(self, frame):
        self.frame = frame
        self.points = []
        self.done = False

    def _clicked(self, event, x, y, flags, param):
        import cv2
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))

    def _draw(self):
        import cv2
        canvas = self.frame.copy()

        # Shade the area covered so far
        if len(self.points) >= 3:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [np.array(self.points, np.int32)],
                         (60, 140, 60))
            canvas = cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0)

        for i, (x, y) in enumerate(self.points):
            cv2.circle(canvas, (x, y), 7, (40, 200, 40), -1)
            cv2.circle(canvas, (x, y), 7, (255, 255, 255), 2)
            cv2.putText(canvas, str(i + 1), (x + 11, y - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            if i > 0:
                cv2.line(canvas, self.points[i - 1], (x, y),
                         (40, 200, 40), 2)
        if len(self.points) == 4:
            cv2.line(canvas, self.points[3], self.points[0],
                     (40, 200, 40), 2)

        # Instructions across the top
        bar = canvas.copy()
        cv2.rectangle(bar, (0, 0), (canvas.shape[1], 74), (25, 25, 30), -1)
        canvas = cv2.addWeighted(bar, 0.82, canvas, 0.18, 0)

        if len(self.points) < 4:
            message = (f"Click the {self.LABELS[len(self.points)]} corner "
                       f"of a floor area  ({len(self.points)}/4)")
            colour = (90, 220, 90)
        else:
            message = "Four points set.  ENTER to accept,  BACKSPACE to redo"
            colour = (90, 220, 220)

        cv2.putText(canvas, message, (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.66, colour, 2)
        cv2.putText(canvas,
                    "Pick spots ON THE FLOOR you can find on a plan   "
                    "BACKSPACE undo   Q quit",
                    (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (185, 185, 195), 1)
        return canvas

    def run(self):
        import cv2

        window = "click four floor points"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, min(1280, self.frame.shape[1]),
                         min(760, self.frame.shape[0]))
        cv2.setMouseCallback(window, self._clicked)

        while True:
            cv2.imshow(window, self._draw())
            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                cv2.destroyAllWindows()
                return None
            if key == 8 and self.points:            # backspace
                self.points.pop()
            if key in (13, 10) and len(self.points) == 4:   # enter
                cv2.destroyAllWindows()
                return self.points


# ══════════════════════════════════════════════════════════════════════
#  Where those points are in the real room
# ══════════════════════════════════════════════════════════════════════
def ask_floor_positions():
    """
    The four clicked points need real positions on a floor plan.

    Coordinates run 0 to 1 across the plan, so 0,0 is one corner of the
    building and 1,1 is the opposite one. Rough is fine. The point is
    that both cameras use the same plan.
    """
    print("\n  Now say where those four points are on your floor plan.")
    print("  Use 0 to 1 across the building, so 0,0 is one corner")
    print("  and 1,1 is the far one. Rough estimates are fine.")
    print("\n  Press enter on its own to accept a sensible default.\n")

    defaults = [(0.30, 0.30), (0.70, 0.30), (0.70, 0.70), (0.30, 0.70)]
    labels = PointPicker.LABELS
    positions = []

    for i, label in enumerate(labels):
        dx, dy = defaults[i]
        raw = input(f"    point {i+1} ({label:>12})  "
                    f"[{dx}, {dy}]: ").strip()
        if not raw:
            positions.append((dx, dy))
            continue
        try:
            parts = raw.replace(",", " ").split()
            positions.append((float(parts[0]), float(parts[1])))
        except (ValueError, IndexError):
            print(f"      didn't understand that, using {dx}, {dy}")
            positions.append((dx, dy))

    return positions


# ══════════════════════════════════════════════════════════════════════
#  Working out the homography
# ══════════════════════════════════════════════════════════════════════
def compute(image_points, floor_points):
    """
    Four pairs of points is exactly enough to work out how the camera
    sees the floor. More would be better but four is the minimum and
    what a person can reasonably click.
    """
    import cv2

    src = np.array(image_points, dtype=np.float32)
    dst = np.array(floor_points, dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


def check_accuracy(H, image_points, floor_points):
    """
    Push the clicked points back through and see how far off they land.
    A large error means the points weren't really on one flat plane,
    or two of them were nearly on top of each other.
    """
    errors = []
    for (px, py), (fx, fy) in zip(image_points, floor_points):
        point = np.array([px, py, 1.0])
        out = H @ point
        gx, gy = out[0] / out[2], out[1] / out[2]
        errors.append(np.hypot(gx - fx, gy - fy))
    return float(np.mean(errors)), float(np.max(errors))


# ══════════════════════════════════════════════════════════════════════
#  Saving
# ══════════════════════════════════════════════════════════════════════
def load_all():
    if not CALIBRATION_FILE.exists():
        return {}
    try:
        with open(CALIBRATION_FILE) as f:
            saved = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return {int(k): v for k, v in saved.items()}


def save_one(camera, H, image_points, floor_points, error):
    everything = load_all()
    everything[camera] = {
        "homography": H.tolist(),
        "image_points": [list(p) for p in image_points],
        "floor_points": [list(p) for p in floor_points],
        "error": error,
    }
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_FILE, "w") as f:
        json.dump({str(k): v for k, v in everything.items()}, f, indent=2)


def load_homographies():
    """What run_live.py calls to pick up any calibration that exists."""
    return {camera: np.array(entry["homography"], dtype=np.float64)
            for camera, entry in load_all().items()}


# ══════════════════════════════════════════════════════════════════════
#  Showing the result
# ══════════════════════════════════════════════════════════════════════
def preview(frame, H, camera):
    """
    Draw a grid on the floor. If the calibration is right the grid lines
    look like they're painted on the ground rather than floating in the air.
    """
    import cv2

    inverse = np.linalg.inv(H)
    canvas = frame.copy()

    def to_pixels(fx, fy):
        point = np.array([fx, fy, 1.0])
        out = inverse @ point
        return int(out[0] / out[2]), int(out[1] / out[2])

    steps = 6
    for i in range(steps + 1):
        t = i / steps
        try:
            cv2.line(canvas, to_pixels(t, 0), to_pixels(t, 1),
                     (70, 190, 70), 1)
            cv2.line(canvas, to_pixels(0, t), to_pixels(1, t),
                     (70, 190, 70), 1)
        except (ValueError, OverflowError, np.linalg.LinAlgError):
            pass

    for zone in DEFAULT_ZONES:
        cx = (zone["x"][0] + zone["x"][1]) / 2
        cy = (zone["y"][0] + zone["y"][1]) / 2
        try:
            x, y = to_pixels(cx, cy)
            if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
                cv2.putText(canvas, zone["name"], (x - 40, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 235, 120), 2)
        except (ValueError, OverflowError, np.linalg.LinAlgError):
            pass

    cv2.putText(canvas, f"camera {camera} - grid should look painted "
                        f"on the floor. any key to close",
                (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)

    cv2.namedWindow("check", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("check", min(1280, canvas.shape[1]),
                     min(760, canvas.shape[0]))
    cv2.imshow("check", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════
#  Getting a frame to click on
# ══════════════════════════════════════════════════════════════════════
def grab_frame(source):
    import cv2

    if isinstance(source, str) and Path(source).exists():
        if Path(source).suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
            return cv2.imread(source)

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        return None

    # Skip a little way in, the first frame of a video is often dark
    for _ in range(12):
        ok, frame = capture.read()
        if not ok:
            break
    capture.release()
    return frame if ok else None


# ══════════════════════════════════════════════════════════════════════
def show_saved():
    everything = load_all()
    if not everything:
        print("\n  Nothing calibrated yet.")
        print("  Run:  python calibrate.py --camera 0")
        return

    print(f"\n  {len(everything)} camera(s) calibrated\n")
    for camera, entry in sorted(everything.items()):
        error = entry.get("error", [0, 0])
        mean = error[0] if isinstance(error, (list, tuple)) else error
        quality = ("good" if mean < 0.02 else
                   "usable" if mean < 0.06 else "poor, worth redoing")
        print(f"    camera {camera}    average error {mean:.4f}   {quality}")
    print(f"\n  saved in {CALIBRATION_FILE}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=None,
                        help="which webcam, and which camera number to save as")
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--as-camera", type=int, default=None,
                        help="save under a different camera number")
    parser.add_argument("--check", action="store_true",
                        help="show what's already calibrated")
    args = parser.parse_args()

    if args.check:
        show_saved()
        return

    if args.video:
        source, camera = args.video, args.as_camera or 0
    elif args.image:
        source, camera = args.image, args.as_camera or 0
    elif args.camera is not None:
        source, camera = args.camera, args.as_camera or args.camera
    else:
        source, camera = 0, 0

    print("=" * 58)
    print(f"  Calibrating camera {camera}")
    print("=" * 58)

    frame = grab_frame(source)
    if frame is None:
        print(f"\n  couldn't read anything from {source}")
        print("  for a webcam try --camera 1")
        sys.exit(1)

    print(f"\n  got a {frame.shape[1]}x{frame.shape[0]} frame")
    print("\n  A window will open. Click four points ON THE FLOOR,")
    print("  clockwise from the top left. Pick things you can also")
    print("  find on a floor plan: tile corners, a doorway, table legs.")
    input("\n  press enter when ready ")

    picker = PointPicker(frame)
    image_points = picker.run()
    if image_points is None:
        print("\n  cancelled")
        sys.exit(0)

    print(f"\n  clicked: {image_points}")

    floor_points = ask_floor_positions()

    H = compute(image_points, floor_points)
    if H is None:
        print("\n  couldn't work out a homography from those points.")
        print("  They may be too close together or in a straight line.")
        sys.exit(1)

    mean_error, worst_error = check_accuracy(H, image_points, floor_points)

    print(f"\n  average error {mean_error:.4f}, worst {worst_error:.4f}")
    if mean_error < 0.02:
        print("  that's good")
    elif mean_error < 0.06:
        print("  usable, though the points may not be perfectly flat")
    else:
        print("  that's poor. Most likely the four points weren't all on")
        print("  the same flat floor, or two were very close together.")

    save_one(camera, H, image_points, floor_points, [mean_error, worst_error])
    print(f"\n  saved for camera {camera}")

    answer = input("\n  see the floor grid overlaid? (y/n) ").strip().lower()
    if answer == "y":
        preview(frame, H, camera)

    print("\n" + "=" * 58)
    print("  run_live.py will use this automatically now.")
    print("  Calibrate your other cameras with --camera 1, --camera 2")
    print("=" * 58)


if __name__ == "__main__":
    main()
