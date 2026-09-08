"""
check_classifier.py — see what the model is actually thinking.

WHY THIS EXISTS
---------------
"It isn't flagging collapses" has several possible causes and they need
different fixes:

  the pose reader can't see you        no keypoints, nothing to classify
  the model says something else        it's working, just wrong
  the model is right but not sure      confidence under the threshold
  it's sure but not for long enough    duration filter holding it back

Guessing between those wastes an afternoon. This shows all of them at
once, live, so you can see which one is happening.

Run:
    python check_classifier.py                webcam
    python check_classifier.py --video x.mp4  a video

The window shows every class score, the pose skeleton, and what the
threat engine would decide.
"""
import argparse
import sys
from collections import deque

import numpy as np

import config as C
import skeleton as S


BAR_COLOURS = {
    "collapse": (60, 60, 220),
    "assault":  (60, 100, 230),
    "stagger":  (60, 190, 240),
    "sos":      (80, 200, 200),
    "normal":   (90, 190, 90),
}


class Checker:

    def __init__(self, show_pose=True):
        self.show_pose = show_pose
        print("=" * 52)
        print("  Classifier diagnostic")
        print("=" * 52)

        self._load_pose()
        self._load_classifier()

        self.poses = deque(maxlen=C.WINDOW)
        self.recent = deque(maxlen=60)     # score history for the graph
        self.streak = 0
        self.held_since = None      # when the current label first appeared
        self.held = 0.0             # seconds
        self.last_label = ""
        self.no_pose_frames = 0
        self.total_frames = 0

        print("\n  ready. press Q to stop\n")

    def _load_pose(self):
        try:
            import mediapipe as mp
        except ImportError:
            print("\n  mediapipe missing:  pip install mediapipe==0.10.9")
            sys.exit(1)
        self.mp = mp
        self.pose = mp.solutions.pose.Pose(
            model_complexity=1,             # a bit better than the live one
            min_detection_confidence=0.4,   # deliberately forgiving here
            min_tracking_confidence=0.4)
        print("  pose reader ready")

    def _load_classifier(self):
        try:
            import torch, joblib
            from train import MotionNet
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            net = MotionNet().to(self.device)
            net.load_state_dict(torch.load(C.WEIGHTS / "motion_net.pth",
                                           map_location=self.device))
            net.eval()
            self.svm, self.scaler = joblib.load(C.WEIGHTS / "classifier.pkl")
            self.torch, self.net = torch, net
            print(f"  classifier ready ({self.device})")
        except FileNotFoundError:
            print("\n  no trained model found.")
            print("  run:  python prepare_data.py")
            print("        python train.py")
            sys.exit(1)

    # ── per frame ─────────────────────────────────────────────────────
    def process(self, frame):
        import cv2

        self.total_frames += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        found = self.pose.process(rgb)

        display = frame.copy()

        if found.pose_landmarks is None:
            self.no_pose_frames += 1
            self._draw_no_pose(display)
            return display

        if self.show_pose:
            self.mp.solutions.drawing_utils.draw_landmarks(
                display, found.pose_landmarks,
                self.mp.solutions.pose.POSE_CONNECTIONS)

        joints = S.from_mediapipe(found.pose_landmarks.landmark)
        self.poses.append(joints)

        # How many joints can it actually see?
        visible = sum(1 for lm in found.pose_landmarks.landmark
                      if lm.visibility > 0.5)

        if len(self.poses) < C.WINDOW:
            self._draw_filling(display, visible)
            return display

        clip = S.normalise(np.stack(self.poses))
        scores = self._classify(clip)
        self._draw_scores(display, scores, clip, visible)
        return display

    def _classify(self, clip):
        flat = S.flatten(clip[None])
        with self.torch.no_grad():
            x = self.torch.tensor(flat).to(self.device)
            summary = self.net.summarise(x).cpu().numpy()
        chances = self.svm.predict_proba(self.scaler.transform(summary))[0]

        best = int(np.argmax(chances))
        label = C.LABELS[best]

        import time as _t
        now = _t.time()
        if label == self.last_label:
            self.streak += 1
            self.held = now - (self.held_since or now)
        else:
            self.streak = 1
            self.last_label = label
            self.held_since = now
            self.held = 0.0

        self.recent.append(chances)
        return chances

    # ── drawing ───────────────────────────────────────────────────────
    def _panel(self, img, height=None):
        import cv2
        h = height or 230
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (400, h), (22, 24, 30), -1)
        return cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

    def _draw_no_pose(self, img):
        import cv2
        self._panel(img, 110)
        cv2.putText(img, "no pose detected", (16, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (70, 70, 235), 2)
        cv2.putText(img, "step back so your whole body is in frame",
                    (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (190, 190, 200), 1)
        missed = self.no_pose_frames / max(self.total_frames, 1)
        cv2.putText(img, f"missed {missed:.0%} of frames so far",
                    (16, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (150, 150, 165), 1)

    def _draw_filling(self, img, visible):
        import cv2
        self._panel(img, 110)
        have = len(self.poses)
        cv2.putText(img, f"collecting frames  {have}/{C.WINDOW}", (16, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 200, 90), 2)
        width = int(360 * have / C.WINDOW)
        cv2.rectangle(img, (16, 48), (16 + width, 62), (240, 200, 90), -1)
        cv2.rectangle(img, (16, 48), (376, 62), (110, 110, 120), 1)
        cv2.putText(img, f"{visible}/33 joints visible", (16, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (190, 190, 200), 1)

    def _draw_scores(self, img, chances, clip, visible):
        import cv2
        self._panel(img)

        order = np.argsort(chances)[::-1]
        best = int(order[0])
        label = C.LABELS[best]
        confidence = float(chances[best])

        y = 28
        for i in order:
            name = C.LABELS[i]
            value = float(chances[i])
            colour = BAR_COLOURS.get(name, (150, 150, 150))
            width = int(250 * value)

            cv2.rectangle(img, (108, y - 11), (108 + width, y + 3),
                          colour, -1)
            cv2.rectangle(img, (108, y - 11), (358, y + 3),
                          (70, 70, 80), 1)
            cv2.putText(img, name, (14, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                        (235, 235, 240) if i == best else (150, 150, 160), 1)
            cv2.putText(img, f"{value:.0%}", (364, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1)
            y += 24

        # Thresholds, drawn where they actually sit
        red_x = 108 + int(250 * C.RED_CONFIDENCE)
        yellow_x = 108 + int(250 * C.YELLOW_CONFIDENCE)
        cv2.line(img, (red_x, 12), (red_x, y - 18), (80, 80, 235), 1)
        cv2.line(img, (yellow_x, 12), (yellow_x, y - 18), (80, 200, 240), 1)

        y += 6
        cv2.line(img, (14, y), (386, y), (70, 70, 80), 1)
        y += 22

        # What the threat engine would say, and why
        verdict, reason = self._verdict(label, confidence)
        colour = {"RED": (60, 60, 225), "YELLOW": (70, 195, 240),
                  "GREEN": (90, 190, 90)}[verdict]
        cv2.putText(img, verdict, (14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, colour, 2)
        cv2.putText(img, reason, (86, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 210), 1)

        y += 22
        needed = (C.RED_HOLD_SECONDS if label in C.RED_ACTIONS
                  else C.YELLOW_HOLD_SECONDS)
        cv2.putText(img, f"held {self.held:.1f}s / {needed:.0f}s    "
                         f"{visible}/33 joints",
                    (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (170, 170, 180), 1)

        # A couple of the measurements the explanation uses
        y += 20
        try:
            from explain import describe_movement
            signs = describe_movement(clip)
            cv2.putText(img,
                        f"drop {signs['height_lost']:.2f}   "
                        f"still {signs['still_frames']}/{signs['total_frames']}   "
                        f"arms {signs['arms_raised']:.2f}",
                        (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (150, 150, 165), 1)
        except Exception:
            pass

    def _verdict(self, label, confidence):
        """Same logic as threat.py, but says which check stopped it."""
        if label == "normal":
            return "GREEN", "normal movement"

        if confidence < C.YELLOW_CONFIDENCE:
            return "GREEN", (f"only {confidence:.0%}, needs "
                             f"{C.YELLOW_CONFIDENCE:.0%}")

        if label in C.RED_ACTIONS:
            if confidence < C.RED_CONFIDENCE:
                return "YELLOW", (f"{confidence:.0%} < "
                                  f"{C.RED_CONFIDENCE:.0%} for red")
            if self.held < C.RED_HOLD_SECONDS:
                return "YELLOW", (f"held {self.held:.1f}s of "
                                  f"{C.RED_HOLD_SECONDS:.0f}s needed")
            return "RED", "would call for help"

        if self.held < C.YELLOW_HOLD_SECONDS:
            return "GREEN", (f"held {self.held:.1f}s of "
                             f"{C.YELLOW_HOLD_SECONDS:.0f}s")
        return "YELLOW", "operator would be alerted"

    # ── running ───────────────────────────────────────────────────────
    def run(self, source):
        import cv2

        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            print(f"  couldn't open {source}")
            print("  for a webcam try --camera 1")
            return

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            cv2.imshow("what the model sees  (Q to quit)",
                       self.process(frame))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        capture.release()
        cv2.destroyAllWindows()
        self._summary()

    def _summary(self):
        print("\n" + "=" * 52)
        print(f"  {self.total_frames} frames")

        missed = self.no_pose_frames / max(self.total_frames, 1)
        print(f"  no pose found in {missed:.0%} of them")
        if missed > 0.3:
            print("\n  That's high. The pose reader is the problem, not the")
            print("  classifier. Usually it means the body is cut off, too")
            print("  far away, or lying flat where MediaPipe struggles.")

        if self.recent:
            average = np.mean(np.stack(self.recent), axis=0)
            print("\n  average scores across the run:")
            for i in np.argsort(average)[::-1]:
                print(f"    {C.LABELS[i]:>9}  {average[i]:.1%}")

            top = C.LABELS[int(np.argmax(average))]
            if top == "normal":
                print("\n  It mostly saw normal movement. If you were acting")
                print("  something out, it didn't read it that way.")
        print("=" * 52)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--no-skeleton", action="store_true")
    args = parser.parse_args()

    checker = Checker(show_pose=not args.no_skeleton)
    checker.run(args.video if args.video else args.camera)


if __name__ == "__main__":
    main()
