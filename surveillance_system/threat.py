"""
threat.py — deciding how serious something is.

WHY THIS COUNTS SECONDS, NOT FRAMES
-----------------------------------
The first version counted processed frames. "A collapse must persist for 15
frames before it counts." That sounds reasonable and it is quietly broken.

Fifteen frames is not a length of time. It depends entirely on how fast the
pipeline happens to be running:

    30 fps video, everything processed     15 frames = 0.5 seconds
    same video, every 3rd frame skipped    15 frames = 1.5 seconds
    live webcam struggling at 4 fps        15 frames = 3.75 seconds

So the same setting means three different things depending on hardware and
video. On a normal video it fires almost instantly, which is exactly the
problem you'd see as "it alarms after a second".

Now it measures actual elapsed time. For a video file that means the
timestamp inside the video, not how long your laptop took to grind through
it. Ten seconds means ten seconds either way.
"""
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

import config as C


@dataclass
class Alert:
    """One classified moment for one person."""
    person: int
    action: str
    confidence: float
    level: str                  # RED / YELLOW / GREEN
    camera: int = 0
    zone: str = "unknown"
    why: str = ""
    at: float = field(default_factory=time.time)
    held: float = 0.0           # seconds this has been going on


# Severity score for sorting / display
SEVERITY_SCORE = {"RED": 3, "YELLOW": 2, "GREEN": 1}

RED_CLASSES    = set(C.RED_ACTIONS)
YELLOW_CLASSES = set(C.YELLOW_ACTIONS)
NORMAL_CLASSES = {"normal"}

EVENT_TO_SERVICE = dict(C.SERVICE_FOR)


class ThreatEngine:
    """
    Tracks what each person has been doing and decides when something
    crosses from odd into serious.

    Every decision is based on elapsed seconds, so it behaves the same
    on a fast machine, a slow machine, and a video file.
    """

    def __init__(self):
        # person -> recent alerts
        self._history = defaultdict(lambda: deque(maxlen=3000))

        # person -> when the current action first appeared, and what it is
        self._started_at = {}
        self._current = defaultdict(str)

        # The clock. Wall time for a live camera, video timestamp for a file.
        self._now = time.time()
        self._using_video_clock = False

        self.log = deque(maxlen=5000)

    # ── the clock ─────────────────────────────────────────────────────────
    def set_time(self, seconds):
        """
        Tell the engine what time it is, in seconds.

        run_live.py calls this with the video's own timestamp when reading
        a file. That way ten seconds means ten seconds of footage, however
        long your laptop takes to process it.
        """
        self._now = float(seconds)
        self._using_video_clock = True

    def _clock(self):
        return self._now if self._using_video_clock else time.time()

    # ── main entry point ──────────────────────────────────────────────────
    def update(self, person, action, confidence, camera=0, zone="unknown"):
        """
        Feed in one classification for one person. Returns an Alert.
        """
        now = self._clock()

        # How long has this action been going on?
        if action != self._current[person]:
            self._current[person] = action
            self._started_at[person] = now
        held = now - self._started_at.get(person, now)

        # Forget anything older than the history window
        recent = self._history[person]
        while recent and now - recent[0].at > C.HISTORY_SECONDS:
            recent.popleft()

        level, why = self._judge(person, action, confidence, held, now)

        alert = Alert(person=person, action=action, confidence=confidence,
                      level=level, camera=camera, zone=zone,
                      why=why, at=now, held=held)
        recent.append(alert)
        self.log.append(alert)
        return alert

    # ── the rules ─────────────────────────────────────────────────────────
    def _judge(self, person, action, confidence, held, now):
        """All the threshold logic in one place, so it's easy to tune."""

        # Nothing unusual, or the model isn't sure enough to care
        if action in NORMAL_CLASSES or confidence < C.YELLOW_CONFIDENCE:
            return "GREEN", (f"nothing to act on "
                             f"({action}, {confidence:.0%} confident)")

        # Has this person been setting things off repeatedly?
        warnings = [a for a in self._history[person]
                    if now - a.at <= C.HISTORY_SECONDS
                    and a.level in ("YELLOW", "RED")]

        # Count separate episodes, not individual frames. Thirty frames of
        # one continuous stagger is one event, not thirty warnings.
        episodes = self._count_episodes(warnings)
        repeat_offender = episodes >= C.ESCALATE_AFTER

        # ── serious kinds of movement ─────────────────────────────────
        if action in RED_CLASSES:
            if confidence >= C.RED_CONFIDENCE and held >= C.RED_HOLD_SECONDS:
                return "RED", (f"{action} for {held:.1f}s "
                               f"at {confidence:.0%} confidence")

            if repeat_offender:
                return "RED", (f"{episodes} separate warnings in the last "
                               f"{C.HISTORY_SECONDS:.0f}s")

            if held < C.RED_HOLD_SECONDS:
                return "YELLOW", (f"possible {action} at {confidence:.0%}, "
                                  f"but only {held:.1f}s of "
                                  f"{C.RED_HOLD_SECONDS:.0f}s so far")

            return "YELLOW", (f"possible {action} for {held:.1f}s, but "
                              f"{confidence:.0%} is under the "
                              f"{C.RED_CONFIDENCE:.0%} needed")

        # ── moderate kinds of movement ────────────────────────────────
        if action in YELLOW_CLASSES:
            if held >= C.YELLOW_HOLD_SECONDS:
                if repeat_offender:
                    return "RED", (f"{episodes} separate warnings for this "
                                   f"person in {C.HISTORY_SECONDS:.0f}s")
                return "YELLOW", (f"{action} for {held:.1f}s "
                                  f"at {confidence:.0%}")
            return "GREEN", (f"{action} but only {held:.1f}s of "
                             f"{C.YELLOW_HOLD_SECONDS:.0f}s")

        return "GREEN", f"below the threshold ({action}, {confidence:.0%})"

    def _count_episodes(self, warnings, gap=2.0):
        """
        Group warnings into episodes.

        Without this, a person staggering continuously produces one warning
        per processed frame, and the "three warnings means escalate" rule
        fires within a fraction of a second. That's not what the rule is for.

        A gap of more than `gap` seconds starts a new episode.
        """
        if not warnings:
            return 0

        episodes = 1
        previous = warnings[0].at
        for alert in warnings[1:]:
            if alert.at - previous > gap:
                episodes += 1
            previous = alert.at
        return episodes

    # ── queries ───────────────────────────────────────────────────────────
    def get_person_history(self, person):
        return list(self._history[person])

    def get_active_reds(self, within=5.0):
        now = self._clock()
        reds = []
        for person, history in self._history.items():
            if history and history[-1].level == "RED":
                if now - history[-1].at < within:
                    reds.append(history[-1])
        return reds

    def service_for(self, action):
        return EVENT_TO_SERVICE.get(action)

    def forget(self, person):
        """Clear someone's state, e.g. after a confirmed false alarm."""
        self._history[person].clear()
        self._current[person] = ""
        self._started_at.pop(person, None)
