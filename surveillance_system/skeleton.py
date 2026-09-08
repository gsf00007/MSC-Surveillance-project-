"""
skeleton.py — converting body joints into numbers the model can learn from.

Both training and the live camera use the functions in here, which is what
keeps them speaking the same language. If you change normalise(), it changes
for both at once — that's deliberate.
"""
import re
import numpy as np
import config as C


# ── Filenames ─────────────────────────────────────────────────────────────
# S001C001P001R001A043.skeleton
#  ^setup ^cam ^person ^rep ^action
_NAME = re.compile(r"S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})")


def read_filename(path):
    """Pull the setup / camera / person / rep / action numbers out of a name."""
    from pathlib import Path
    m = _NAME.match(Path(path).stem)
    if not m:
        return None
    setup, cam, person, rep, action = (int(g) for g in m.groups())
    return {"setup": setup, "camera": cam, "person": person,
            "rep": rep, "action": action}


# ── Reading NTU files ─────────────────────────────────────────────────────
def read_skeleton_file(path):
    """
    Read one .skeleton file.

    The format is plain text:
        how many frames
        then for each frame:
            how many people in view
            for each person:
                a line of tracking info we ignore
                how many joints (always 25)
                25 lines, first three numbers are x y z

    Returns (frames, 13, 3) or None if the file is too short or broken.
    Only the first person is kept, which is right for these action classes.
    """
    try:
        with open(path, "r") as f:
            lines = f.read().split("\n")
    except OSError:
        return None

    pos = 0

    def take():
        nonlocal pos
        while pos < len(lines) and lines[pos].strip() == "":
            pos += 1
        if pos >= len(lines):
            raise IndexError
        value = lines[pos].strip()
        pos += 1
        return value

    try:
        n_frames = int(take())
    except (ValueError, IndexError):
        return None

    frames = []
    for _ in range(n_frames):
        try:
            n_people = int(take())
        except (ValueError, IndexError):
            break

        first_person = None
        for person_i in range(n_people):
            try:
                take()                       # tracking info line, skipped
                n_joints = int(take())
            except (ValueError, IndexError):
                break

            joints = np.zeros((n_joints, 3), dtype=np.float32)
            for j in range(n_joints):
                try:
                    bits = take().split()
                    joints[j] = (float(bits[0]), float(bits[1]), float(bits[2]))
                except (ValueError, IndexError):
                    pass                     # leave as zeros

            if person_i == 0:
                first_person = joints

        if first_person is not None and len(first_person) >= 25:
            frames.append(first_person[C.NTU_TO_COMMON])

    if len(frames) < C.WINDOW:
        return None
    return np.stack(frames).astype(np.float32)


# ── Live camera input ─────────────────────────────────────────────────────
def from_mediapipe(landmarks):
    """
    Turn MediaPipe's 33 landmarks into the same 13 joints NTU gave us.
    landmarks: result.pose_landmarks.landmark
    """
    full = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
    return full[C.MEDIAPIPE_TO_COMMON]


# ── Making the numbers comparable ─────────────────────────────────────────
def normalise(sequence):
    """
    NTU measures in metres from a Kinect. MediaPipe gives 0-1 image
    coordinates. Straight out of the box those numbers mean nothing
    to each other.

    Two fixes:
      1. Move everything so the hips sit at zero, which throws away
         where the person is standing.
      2. Divide by torso length, which throws away how far they are
         from the camera.

    What's left is the shape of the movement, which is the only part
    that should matter.

    sequence: (frames, 13, 3)
    """
    seq = np.asarray(sequence, dtype=np.float32).copy()

    L_HIP, R_HIP = 7, 8
    L_SHO, R_SHO = 1, 2

    hips      = (seq[:, L_HIP] + seq[:, R_HIP]) / 2.0
    shoulders = (seq[:, L_SHO] + seq[:, R_SHO]) / 2.0

    seq -= hips[:, None, :]

    torso = np.linalg.norm(shoulders - hips, axis=1)
    good  = torso[torso > 1e-6]
    scale = np.median(good) if good.size else 1.0

    return (seq / (scale + 1e-8)).astype(np.float32)


# ── Cutting into training samples ─────────────────────────────────────────
def to_windows(sequence, window=None, stride=None):
    """Slide a fixed window over a clip. Returns a list of (window, 13, 3)."""
    window = window or C.WINDOW
    stride = stride or C.STRIDE
    return [sequence[i:i + window]
            for i in range(0, len(sequence) - window + 1, stride)]


def flatten(batch):
    """(N, frames, 13, 3) -> (N, frames, 39) for the LSTM."""
    batch = np.asarray(batch)
    return batch.reshape(batch.shape[0], batch.shape[1], -1)
