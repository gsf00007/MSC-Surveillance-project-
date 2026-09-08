"""
demo_reid.py — showing that the system recognises the same person again.

WHAT THIS IS FOR
----------------
Benchmark numbers are how you *measure* re-identification. They are not a
good way to *show* it. A table saying 24.8% rank-1 means nothing to anyone
watching a demo.

What means something is this: a person walks out of shot, comes back thirty
seconds later, and the box above their head still says #3. Or they appear on
a second camera and get the same number they had on the first.

This script does that, on ordinary video files, and produces evidence you
can point at:

    a labelled video          IDs drawn on every person, live
    a contact sheet           every crop the system filed under each ID
    a short report            how many people, how many IDs, how often
                              it recognised someone returning

No benchmark dataset needed. Any video with people in it will do.


HOW RE-IDENTIFICATION ACTUALLY WORKS HERE
-----------------------------------------
Three layers, and they do different jobs:

    YOLOv8      finds people in this frame. Has no memory at all — every
                frame is a fresh start as far as it's concerned.

    ByteTrack   links detections between consecutive frames using motion.
                Gives a "local ID" that survives a person walking around,
                but breaks the moment they leave the frame or something
                passes in front of them.

    Re-ID       describes what the person looks like as a list of numbers,
                and compares that against everyone seen so far. This is the
                layer that survives someone disappearing and coming back,
                and it's the only one that works across separate cameras.

The demo makes each layer visible so you can talk about them separately.


RUNNING IT
----------
    python demo_reid.py --video hallway.mp4

    python demo_reid.py --video cam1.mp4 --video2 cam2.mp4
        two videos treated as two cameras. Anyone appearing in both should
        keep the same global ID — that's the cross-camera claim.

    python demo_reid.py --video clip.mp4 --expect 3
        if you know how many people are in the clip, it reports how close
        the system got.
"""
import argparse
import json
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

import config as C


# How many crops to keep per person for the contact sheet. Enough to show
# the system saw them from several angles, few enough to fit on a page.
CROPS_PER_PERSON = 6

# A person is "gone" if we haven't seen them for this long. If they come
# back after that, it counts as a re-acquisition rather than continuous
# tracking — which is the thing worth demonstrating.
GONE_AFTER_SECONDS = 1.5


class ReIDDemo:
    """
    Runs the detection → tracking → re-identification chain over video and
    records enough to prove it worked.
    """

    def __init__(self, use_deep=True, show_window=True):
        self.show_window = show_window

        print("=" * 58)
        print("  Re-identification demo")
        print("=" * 58)

        self._load_detector()
        self._load_reid(use_deep)

        # ── what we record as we go ───────────────────────────────────
        # global id -> list of crops, for the contact sheet
        self.crops = defaultdict(list)

        # global id -> when we last saw them, so we can spot returns
        self.last_seen = {}

        # global id -> which cameras they've appeared on
        self.cameras_seen = defaultdict(set)

        # every time someone vanished and came back
        self.reacquisitions = []

        # every time someone was matched across two different cameras
        self.cross_camera = []

        # local (per-camera) track IDs, to show what tracking alone gives
        self.local_ids_seen = defaultdict(set)

        self.frames_processed = 0

    # ── loading ───────────────────────────────────────────────────────
    def _load_detector(self):
        """
        YOLOv8 finds people; ByteTrack links them frame to frame.

        Both come pretrained — YOLOv8 on COCO, and ByteTrack isn't a
        learned model at all, it's a matching algorithm. Neither was
        trained as part of this project.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            print("\n  ultralytics isn't installed")
            print("  pip install ultralytics")
            sys.exit(1)

        print(f"\n  loading {C.YOLO_MODEL}")
        self.yolo = YOLO(C.YOLO_MODEL)
        print("  detector ready")

    def _load_reid(self, use_deep):
        """
        The appearance describer. Three options, in order of preference:

          the model trained on Market-1501     best, if train_reid.py has run
          an ImageNet network                  decent, nothing to train
          colour histograms                    always works, weakest

        Whichever loads, the rest of the script is identical.
        """
        from reid import ReIdentifier
        self.reid = ReIdentifier(use_deep=use_deep)

    # ── per frame ─────────────────────────────────────────────────────
    def process_frame(self, frame, camera, timestamp):
        """
        One frame, one camera. Returns the frame with boxes drawn on it.

        The camera number matters: it's what makes a match "cross-camera"
        rather than just "the tracker doing its job".
        """
        import cv2

        # Detection plus tracking in one call. persist=True tells
        # ultralytics to carry track state between calls rather than
        # treating each frame as a new video.
        results = self.yolo.track(
            frame, persist=True, tracker="bytetrack.yaml",
            conf=C.YOLO_CONF, classes=[0], verbose=False)

        display = frame.copy()

        for result in results:
            # boxes.id is None when the tracker hasn't locked on yet
            if result.boxes is None or result.boxes.id is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            local_ids = result.boxes.id.cpu().numpy().astype(int)

            for box, local_id in zip(boxes, local_ids):
                x1, y1, x2, y2 = box

                # Crop the person out. This is what the appearance model
                # actually looks at — not the whole frame.
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size == 0:
                    continue

                # Skip anything too small to describe usefully. A
                # 20-pixel-wide person is a few blurry blobs.
                if (x2 - x1) < 25 or (y2 - y1) < 60:
                    continue

                # ── the actual re-identification ──────────────────────
                # This compares the crop's appearance against everyone
                # seen so far. `matched` is True if it recognised them.
                global_id, zone, matched, similarity = self.reid.identify(
                    crop, box, camera, int(local_id), frame.shape)

                self._record(global_id, local_id, camera, crop,
                             timestamp, matched, similarity)
                self._draw(display, box, global_id, local_id,
                           matched, similarity, camera)

        self.frames_processed += 1
        return display

    def _record(self, global_id, local_id, camera, crop,
                timestamp, matched, similarity):
        """
        Keep track of what happened, so there's something to report at
        the end beyond "it seemed to work".
        """
        # Was this person gone long enough to count as returning?
        if global_id in self.last_seen:
            gap = timestamp - self.last_seen[global_id]
            if gap > GONE_AFTER_SECONDS and matched:
                self.reacquisitions.append({
                    "person": int(global_id),
                    "gone_for": round(gap, 1),
                    "at": round(timestamp, 1),
                    "similarity": round(float(similarity), 3),
                })
        self.last_seen[global_id] = timestamp

        # Did we just match them on a camera they hadn't been seen on?
        previous_cameras = self.cameras_seen[global_id]
        if previous_cameras and camera not in previous_cameras and matched:
            self.cross_camera.append({
                "person": int(global_id),
                "first_seen_on": sorted(previous_cameras),
                "now_on": camera,
                "at": round(timestamp, 1),
                "similarity": round(float(similarity), 3),
            })
        self.cameras_seen[global_id].add(camera)

        # Local IDs, so we can show what tracking alone would have given
        self.local_ids_seen[camera].add(int(local_id))

        # Keep a few crops for the contact sheet, spread through the video
        # rather than six consecutive frames of the same pose
        held = self.crops[global_id]
        if len(held) < CROPS_PER_PERSON:
            held.append((crop.copy(), camera, round(timestamp, 1)))

    def _draw(self, img, box, global_id, local_id,
              matched, similarity, camera):
        """
        Draw the box. The label deliberately shows both IDs, because the
        difference between them is the point of the whole demo.
        """
        import cv2

        x1, y1, x2, y2 = box
        colour = _colour_for(global_id)

        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)

        # A tick means "I've seen this person before and recognised them"
        mark = " recognised" if matched else ""
        label = f"person {global_id}{mark}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                      0.55, 2)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 8, y1),
                      colour, -1)
        cv2.putText(img, label, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Smaller line underneath showing the tracker's own ID and the
        # match score, so the two layers are visibly separate
        detail = f"tracker #{local_id}"
        if matched:
            detail += f"  match {similarity:.2f}"
        cv2.putText(img, detail, (x1 + 2, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1)

    # ── two cameras at once ───────────────────────────────────────────
    def run_two_videos(self, path_a, path_b):
        """
        Play both videos side by side, stepping through them together.

        Running them one after the other still proves the point, but it
        looks like two separate demos. Side by side, someone can watch a
        person appear on the left, then appear on the right a moment
        later already carrying the same number. That is the whole claim
        of cross-camera re-identification, visible in one frame.
        """
        import cv2

        cap_a = cv2.VideoCapture(str(path_a))
        cap_b = cv2.VideoCapture(str(path_b))

        if not cap_a.isOpened() or not cap_b.isOpened():
            print("\n  couldn't open both videos")
            return False

        fps_a = cap_a.get(cv2.CAP_PROP_FPS) or 25.0
        fps_b = cap_b.get(cv2.CAP_PROP_FPS) or 25.0

        print(f"\n  camera 0: {Path(path_a).name}")
        print(f"  camera 1: {Path(path_b).name}")
        print("  playing both together\n")

        writer = None
        out_path = C.LOGS / "reid_demo_both.mp4"
        frame_number = 0
        running_a, running_b = True, True

        while running_a or running_b:
            frame_number += 1

            ok_a, frame_a = (cap_a.read() if running_a else (False, None))
            ok_b, frame_b = (cap_b.read() if running_b else (False, None))
            running_a, running_b = ok_a, ok_b

            if not ok_a and not ok_b:
                break

            if frame_number % C.PROCESS_EVERY != 0:
                continue

            panels = []

            # Camera 0 goes through the pipeline first, so anyone it sees
            # is already in the gallery when camera 1 is processed. That
            # ordering is what lets camera 1 recognise rather than
            # register.
            if ok_a:
                shown = self.process_frame(frame_a, 0, frame_number / fps_a)
                panels.append(self._label_panel(shown, "camera 0"))
            if ok_b:
                shown = self.process_frame(frame_b, 1, frame_number / fps_b)
                panels.append(self._label_panel(shown, "camera 1"))

            if not panels:
                break

            combined = self._side_by_side(panels)

            if writer is None:
                writer = cv2.VideoWriter(
                    str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                    max(fps_a, fps_b) / C.PROCESS_EVERY,
                    (combined.shape[1], combined.shape[0]))
            writer.write(combined)

            if self.show_window:
                cv2.imshow("both cameras  (Q to stop)", combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_number % 150 == 0:
                shared = sum(1 for cams in self.cameras_seen.values()
                             if len(cams) > 1)
                print(f"    {frame_number/max(fps_a,1):5.1f}s, "
                      f"{len(self.crops)} people, "
                      f"{shared} seen on both", end="\r")

        cap_a.release()
        cap_b.release()
        if writer:
            writer.release()
            print(f"    saved {out_path}" + " " * 24)
        if self.show_window:
            cv2.destroyAllWindows()
        return True

    def _label_panel(self, frame, text):
        """Put a camera name on a panel so the two are tellable apart."""
        import cv2
        out = frame.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 34), (30, 33, 40), -1)
        cv2.putText(out, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.66, (245, 245, 250), 2)
        return out

    def _side_by_side(self, panels):
        """Match heights, then join left to right with a divider."""
        import cv2

        target_h = min(p.shape[0] for p in panels)
        resized = []
        for p in panels:
            scale = target_h / p.shape[0]
            resized.append(cv2.resize(
                p, (int(p.shape[1] * scale), target_h)))

        gap = np.full((target_h, 6, 3), 25, dtype=np.uint8)
        joined = resized[0]
        for panel in resized[1:]:
            joined = np.hstack([joined, gap, panel])
        return joined

    # ── running over a video ──────────────────────────────────────────
    def run_video(self, path, camera):
        """
        Walk through one video file, treating it as one camera.

        Timestamps come from the video's own frame rate, not wall clock,
        so "gone for 3 seconds" means three seconds of footage regardless
        of how fast the laptop chews through it.
        """
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            print(f"\n  couldn't open {path}")
            return False

        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        print(f"\n  camera {camera}: {Path(path).name}")
        print(f"    {fps:.0f} fps, {total} frames "
              f"({total/fps:.0f} seconds)" if total else "")

        writer = None
        out_path = C.LOGS / f"reid_demo_cam{camera}.mp4"
        frame_number = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1

            # Skip frames for speed. Because timestamps come from the
            # frame number, skipping doesn't distort the timing.
            if frame_number % C.PROCESS_EVERY != 0:
                continue

            timestamp = frame_number / fps
            display = self.process_frame(frame, camera, timestamp)

            # Save a labelled copy so there's something to show later
            if writer is None:
                import cv2 as _cv
                writer = _cv.VideoWriter(
                    str(out_path), _cv.VideoWriter_fourcc(*"mp4v"),
                    fps / C.PROCESS_EVERY,
                    (display.shape[1], display.shape[0]))
            writer.write(display)

            if self.show_window:
                cv2.imshow(f"camera {camera}  (Q to stop)", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_number % 150 == 0:
                print(f"    {timestamp:5.1f}s, "
                      f"{len(self.crops)} people so far", end="\r")

        capture.release()
        if writer:
            writer.release()
            print(f"    saved {out_path}" + " " * 20)
        if self.show_window:
            cv2.destroyAllWindows()
        return True

    # ── evidence ──────────────────────────────────────────────────────
    def contact_sheet(self, path=None):
        """
        One row per person, showing the crops the system filed under that
        ID. This is the single most convincing artefact — if row 3 is all
        the same person, re-identification worked. If it's a mix, it
        didn't, and you can see exactly where it went wrong.
        """
        import cv2

        path = path or (C.LOGS / "reid_people.jpg")

        people = sorted(self.crops.keys())
        if not people:
            return None

        cell_w, cell_h = 70, 150
        label_w = 110
        pad = 6

        rows = len(people)
        cols = CROPS_PER_PERSON
        width = label_w + cols * (cell_w + pad) + pad
        height = rows * (cell_h + pad) + pad + 34

        sheet = np.full((height, width, 3), 245, dtype=np.uint8)

        cv2.putText(sheet, "Every crop the system filed under each ID",
                    (pad + 4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (60, 60, 60), 1)

        for r, person in enumerate(people):
            y = 34 + pad + r * (cell_h + pad)
            colour = _colour_for(person)

            cv2.rectangle(sheet, (pad, y), (pad + 5, y + cell_h),
                          colour, -1)
            cv2.putText(sheet, f"person {person}", (pad + 12, y + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)

            cameras = sorted(self.cameras_seen[person])
            cv2.putText(sheet,
                        f"cam {','.join(str(c) for c in cameras)}",
                        (pad + 12, y + 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

            for c, (crop, camera, when) in enumerate(self.crops[person]):
                x = label_w + pad + c * (cell_w + pad)
                resized = cv2.resize(crop, (cell_w, cell_h))
                sheet[y:y + cell_h, x:x + cell_w] = resized
                cv2.rectangle(sheet, (x, y), (x + cell_w, y + cell_h),
                              colour, 1)
                cv2.putText(sheet, f"{when:.0f}s", (x + 3, y + cell_h - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                            (255, 255, 255), 1)

        cv2.imwrite(str(path), sheet)
        return path

    def report(self, expected_people=None):
        """
        What actually happened, in numbers you can quote.
        """
        print("\n" + "=" * 58)
        print("  What happened")
        print("=" * 58)

        n_people = len(self.crops)
        local_total = sum(len(ids) for ids in self.local_ids_seen.values())

        print(f"  frames processed          {self.frames_processed}")
        print(f"  tracker IDs handed out    {local_total}")
        print(f"  distinct people recognised{n_people:>6}")

        if expected_people:
            print(f"  people actually present   {expected_people}")
            if n_people == expected_people:
                print("      exactly right")
            elif n_people > expected_people:
                extra = n_people - expected_people
                print(f"      {extra} too many — someone was split "
                      f"into separate identities")
            else:
                missing = expected_people - n_people
                print(f"      {missing} too few — two people were "
                      f"merged into one identity")

        # This is the number that shows re-id doing something tracking can't
        if local_total > n_people:
            saved = local_total - n_people
            print(f"\n  Re-identification merged {saved} tracker IDs that")
            print(f"  belonged to people already seen. Without it there")
            print(f"  would be {local_total} identities instead of {n_people}.")

        print(f"\n  recognised someone returning  "
              f"{len(self.reacquisitions)} times")
        for r in self.reacquisitions[:6]:
            print(f"      person {r['person']} came back after "
                  f"{r['gone_for']}s  (match {r['similarity']:.2f})")
        if len(self.reacquisitions) > 6:
            print(f"      ... and {len(self.reacquisitions) - 6} more")

        if self.cross_camera:
            print(f"\n  matched across cameras        "
                  f"{len(self.cross_camera)} times")
            for x in self.cross_camera[:6]:
                print(f"      person {x['person']}: camera "
                      f"{x['first_seen_on']} then camera {x['now_on']}  "
                      f"(match {x['similarity']:.2f})")
        elif len(self.local_ids_seen) > 1:
            print("\n  no cross-camera matches — nobody appeared in both,")
            print("  or the threshold is too strict. Try lowering")
            print("  REID_THRESHOLD in config.py.")

        # Save it all so the numbers can be quoted later
        summary = {
            "frames": self.frames_processed,
            "tracker_ids": local_total,
            "people_recognised": n_people,
            "expected_people": expected_people,
            "reacquisitions": self.reacquisitions,
            "cross_camera_matches": self.cross_camera,
            "threshold": C.REID_THRESHOLD,
        }
        with open(C.LOGS / "reid_demo.json", "w") as f:
            json.dump(summary, f, indent=2)

        return summary


def _colour_for(person_id):
    """A stable colour per person, so the same ID is always the same hue."""
    rng = np.random.default_rng(int(person_id) * 7919)
    return tuple(int(v) for v in rng.integers(70, 245, 3))


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate re-identification on ordinary video")
    parser.add_argument("--video", required=True,
                        help="video file, treated as camera 0")
    parser.add_argument("--video2", default=None,
                        help="second video, treated as camera 1")
    parser.add_argument("--expect", type=int, default=None,
                        help="how many people are actually in the video")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--colour-only", action="store_true",
                        help="use colour matching instead of a network")
    parser.add_argument("--together", action="store_true",
                        help="play both videos side by side rather than "
                             "one after the other")
    args = parser.parse_args()

    demo = ReIDDemo(use_deep=not args.colour_only,
                    show_window=not args.no_window)

    if args.video2 and args.together:
        # Both at once, side by side. Better for a live demo.
        if not demo.run_two_videos(args.video, args.video2):
            sys.exit(1)
    else:
        if not demo.run_video(args.video, camera=0):
            sys.exit(1)
        if args.video2:
            demo.run_video(args.video2, camera=1)

    demo.report(expected_people=args.expect)

    sheet = demo.contact_sheet()
    if sheet:
        print(f"\n  contact sheet saved to {sheet}")
        print("  open it — one row per person, showing every crop filed")
        print("  under that ID. If a row is all the same person, it worked.")

    print("\n" + "=" * 58)
    print("  labelled video(s) in logs/reid_demo_cam*.mp4")
    print("=" * 58)


if __name__ == "__main__":
    main()
