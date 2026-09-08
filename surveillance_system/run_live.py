"""
run_live.py — the whole system, running on a camera.

    python run_live.py                       webcam
    python run_live.py --dashboard           with the web page
    python run_live.py --video clip.mp4      a video file
    python run_live.py --deep-reid           better appearance matching
    python run_live.py --explain             say why on every alert

Press Q in the window to stop.

Per frame:
    find people  ->  keep track of who's who  ->  recognise them across
    cameras  ->  read their pose  ->  classify the movement  ->  decide
    how serious  ->  act on it
"""
import argparse
import sys
from collections import defaultdict, deque

import numpy as np

import config as C
import skeleton as S
from threat import ThreatEngine
from dispatch import Dispatcher


COLOURS = {
    "RED":    (0, 0, 220),
    "YELLOW": (0, 180, 240),
    "GREEN":  (0, 170, 0),
}


class LiveSystem:

    def __init__(self, show_window=True, use_dashboard=False,
                 deep_reid=False, explain_alerts=False):
        self.show_window = show_window
        self.explain_alerts = explain_alerts

        print("=" * 46)
        print("  Starting up")
        print("=" * 46)

        self._load_detector()
        self._load_pose()
        self._load_classifier()
        self._load_reid(deep_reid)
        self._load_explainer(explain_alerts)

        print("\n  threat engine ready")
        self.threat = ThreatEngine()

        self.dashboard = None
        if use_dashboard:
            self._load_dashboard()

        self.dispatcher = Dispatcher(
            on_alert=self._on_alert,
            on_call=self._on_call,
            on_cancel=self._on_cancel,
        )

        # person -> their recent poses, enough for one classification
        self.poses = defaultdict(lambda: deque(maxlen=C.WINDOW))
        self.frames_seen = 0
        self._last_said = {}      # stops the same alert repeating each frame
        self._explained = set()   # explain each person once per level change

        print("\n" + "=" * 46)
        print("  Ready. Press Q to stop.")
        print("=" * 46 + "\n")

    # ── loading ───────────────────────────────────────────────────────────
    def _load_detector(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            print("\n  ultralytics isn't installed.")
            print("  pip install ultralytics")
            sys.exit(1)

        print(f"\n  loading {C.YOLO_MODEL} (downloads itself the first time)")
        self.yolo = YOLO(C.YOLO_MODEL)
        print("  detector ready")

    def _load_pose(self):
        try:
            import mediapipe as mp
        except ImportError:
            print("\n  mediapipe isn't installed.")
            print("  pip install mediapipe==0.10.9")
            sys.exit(1)

        self.pose = mp.solutions.pose.Pose(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        print("  pose reader ready")

    def _load_classifier(self):
        self.classifier = None
        try:
            import torch, joblib
            from train import MotionNet

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            net = MotionNet().to(self.device)
            net.load_state_dict(torch.load(C.WEIGHTS / "motion_net.pth",
                                           map_location=self.device))
            net.eval()
            svm, scaler = joblib.load(C.WEIGHTS / "classifier.pkl")

            self.torch = torch
            self.net, self.svm, self.scaler = net, svm, scaler
            self.classifier = True
            print(f"  movement classifier ready ({self.device})")

        except FileNotFoundError:
            print("\n  no trained model found, so movement classification")
            print("  is switched off. Detection and tracking still work.")
            print("  train one with:  python prepare_data.py && python train.py")
        except Exception as e:
            print(f"\n  couldn't load the classifier: {e}")

    def _load_reid(self, deep):
        try:
            from reid import ReIdentifier
            self.reid = ReIdentifier(use_deep=deep or C.REID_USE_DEEP)
        except Exception as e:
            print(f"  re-identification unavailable: {e}")
            self.reid = None

    def _load_explainer(self, wanted):
        self.explainer = None
        if not wanted:
            return
        try:
            from explain import Explainer
            ex = Explainer()
            self.explainer = ex if ex.ready else None
        except Exception as e:
            print(f"  explanations unavailable: {e}")

    def _load_dashboard(self):
        try:
            from dashboard import Dashboard
            self.dashboard = Dashboard(on_cancel=self._cancel_from_page)
            self.dashboard.start_background()
        except ImportError:
            print("\n  flask-socketio isn't installed, so no dashboard")
            print("  pip install flask flask-socketio")
        except Exception as e:
            print(f"  dashboard failed to start: {e}")

    # ── per-frame work ────────────────────────────────────────────────────
    def process(self, frame, camera=0, timestamp=None):
        # Tell the threat engine what time it is. For a video that's the
        # timestamp inside the video, so "10 seconds" means 10 seconds of
        # footage rather than 10 seconds of your laptop grinding.
        if timestamp is not None:
            self.threat.set_time(timestamp)

        results = self.yolo.track(
            frame, persist=True, tracker="bytetrack.yaml",
            conf=C.YOLO_CONF, classes=[0], verbose=False)

        display = frame.copy()
        people_now = 0

        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            ids   = result.boxes.id.cpu().numpy().astype(int)
            people_now = len(ids)

            for box, local_id in zip(boxes, ids):
                x1, y1, x2, y2 = box
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size == 0:
                    continue

                # Who is this, and where are they?
                if self.reid:
                    person, zone, matched, score = self.reid.identify(
                        crop, box, camera, int(local_id), frame.shape)
                    self._located_properly = self.reid.last_position_real
                else:
                    person, zone = int(local_id), "unknown"
                    matched, score = False, 0.0
                    self._located_properly = False

                joints = self._read_pose(crop)
                if joints is not None:
                    self.poses[person].append(joints)

                action, confidence, summary, clip = self._classify(person)

                alert = self.threat.update(
                    person=person, action=action, confidence=confidence,
                    camera=camera, zone=zone)
                self.dispatcher.handle(alert)

                if (self.explain_alerts and alert.level in ("RED", "YELLOW")
                        and clip is not None):
                    mark = (person, alert.level, alert.action)
                    if mark not in self._explained:
                        self._explained.add(mark)
                        self._explain(alert, clip, summary)

                self._draw(display, box, person, alert, matched)

        if self.dashboard:
            self.dashboard.show_frame(display, people=people_now)

        self.frames_seen += 1
        return display

    def _read_pose(self, crop):
        import cv2
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        found = self.pose.process(rgb)
        if found.pose_landmarks is None:
            return None
        return S.from_mediapipe(found.pose_landmarks.landmark)

    def _classify(self, person):
        """Returns (action, confidence, network summary, the clip used)."""
        if not self.classifier or len(self.poses[person]) < C.WINDOW:
            return "normal", 0.5, None, None

        clip = S.normalise(np.stack(self.poses[person]))
        flat = S.flatten(clip[None])

        with self.torch.no_grad():
            x = self.torch.tensor(flat).to(self.device)
            summary = self.net.summarise(x).cpu().numpy()

        chances = self.svm.predict_proba(self.scaler.transform(summary))[0]
        best = int(np.argmax(chances))
        return C.LABELS[best], float(chances[best]), summary[0], clip

    def _explain(self, alert, clip, summary):
        from explain import full_explanation
        try:
            why = full_explanation(clip, alert, summary, self.explainer)
            print(f"      {why['plain']}")
            if why["counterfactual"]:
                print(f"      would change if: {why['counterfactual']}")
        except Exception as e:
            print(f"      couldn't explain: {e}")

    def _draw(self, img, box, person, alert, matched):
        import cv2
        x1, y1, x2, y2 = box
        colour = COLOURS.get(alert.level, (160, 160, 160))

        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)

        seen_before = "*" if matched else ""
        held = f" {alert.held:.0f}s" if alert.held >= 1 else ""
        label = (f"#{person}{seen_before} {alert.action} "
                 f"{alert.confidence:.0%}{held} {alert.level}")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), colour, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # A "?" means the zone is guessed from frame position rather
        # than worked out from a calibrated floor plan.
        guessed = "" if getattr(self, "_located_properly", False) else " ?"
        cv2.putText(img, alert.zone + guessed, (x1 + 3, y2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1)

    # ── things that happen ────────────────────────────────────────────────
    def _on_alert(self, alert):
        # Only speak up when something actually changes. Otherwise a
        # person standing still generates the same line thirty times a
        # second and buries everything useful.
        key = (alert.person, alert.level, alert.action)
        if key == self._last_said.get(alert.person):
            return
        self._last_said[alert.person] = key

        print(f"  {alert.level:<6} person {alert.person}  "
              f"{alert.action} {alert.confidence:.0%}  {alert.zone}")
        if self.dashboard:
            self.dashboard.show_alert(alert)
            if alert.level == "RED":
                self.dashboard.show_countdown(alert)

    def _on_call(self, alert, service, ref):
        if self.dashboard:
            self.dashboard.show_called(alert, service, ref)

    def _on_cancel(self, person, note):
        self.threat.forget(person)

    def _cancel_from_page(self, person, note):
        """Someone pressed cancel in the browser."""
        stopped = self.dispatcher.cancel(person, note)
        if stopped:
            print(f"  cancelled from the dashboard: person {person}")

    # ── main loop ─────────────────────────────────────────────────────────
    def run(self, source=0):
        import cv2

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"\n  couldn't open {source}")
            print("  for a webcam try --camera 1 instead of 0")
            return

        # A video knows its own frame rate. A webcam usually reports 0,
        # in which case fall back to wall-clock time.
        fps = cap.get(cv2.CAP_PROP_FPS)
        is_video = isinstance(source, str) and fps and fps > 1
        if is_video:
            print(f"  {fps:.0f} fps video, "
                  f"processing every {C.PROCESS_EVERY} frame(s)")
        print(f"  reading from {source}\n")

        frame_number = 0
        last_display = None

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_number += 1

                # Skip frames for speed. Bodies barely move in a thirtieth
                # of a second, so this costs almost no accuracy.
                if frame_number % C.PROCESS_EVERY != 0:
                    if self.show_window and last_display is not None:
                        cv2.imshow("surveillance  (Q to quit)", last_display)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    continue

                timestamp = (frame_number / fps) if is_video else None
                display = self.process(frame, timestamp=timestamp)
                last_display = display

                if self.show_window:
                    if is_video:
                        # Show how far into the video we are, so the hold
                        # times on screen make sense.
                        cv2.putText(display, f"{timestamp:5.1f}s",
                                    (display.shape[1] - 90, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (255, 255, 255), 2)
                    cv2.imshow("surveillance  (Q to quit)", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except KeyboardInterrupt:
            print("\n  stopped")
        finally:
            cap.release()
            if self.show_window:
                cv2.destroyAllWindows()
            self._wrap_up()

    def _wrap_up(self):
        print("\n" + "=" * 46)
        print(f"  {self.frames_seen} frames processed")
        if self.reid and not self.reid.zones.calibrated:
            print("  locations were guessed from frame position.")
            print("  run calibrate.py to make them real.")
        if self.reid:
            s = self.reid.stats()
            print(f"  {s['people_known']} people known")
            print(f"  {s['matches']} re-identifications "
                  f"({s['match_rate']:.0%} of sightings)")
            print(f"  {s['on_multiple_cameras']} seen on more than one camera")
        print("=" * 46)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0,
                    help="webcam number, usually 0")
    ap.add_argument("--video", type=str, default=None,
                    help="path to a video file instead of a webcam")
    ap.add_argument("--no-window", action="store_true",
                    help="don't open a display window")
    ap.add_argument("--dashboard", action="store_true",
                    help="also serve the web page on localhost:5000")
    ap.add_argument("--deep-reid", action="store_true",
                    help="use a network for appearance instead of colour")
    ap.add_argument("--explain", action="store_true",
                    help="print why on every red and yellow alert")
    args = ap.parse_args()

    system = LiveSystem(
        show_window=not args.no_window,
        use_dashboard=args.dashboard,
        deep_reid=args.deep_reid,
        explain_alerts=args.explain,
    )
    system.run(args.video if args.video else args.camera)


if __name__ == "__main__":
    main()
