"""
config.py — every setting for the project lives here.

The ONLY line you must change is NTU_FOLDER below.
"""
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════
#  1.  WHERE YOUR NTU DATA IS          
# ══════════════════════════════════════════════════════════════════════════
NTU_FOLDER = "C:/Users/fgund/OneDrive/Desktop/datasets/ntu-rgbd/ntu-rgbd/nturgb+d_skeletons"




# ══════════════════════════════════════════════════════════════════════════
#  2.  PROJECT FOLDERS  (created automatically, don't change)
# ══════════════════════════════════════════════════════════════════════════
HERE     = Path(__file__).parent.resolve()
DATA_DIR = HERE / "data"
WEIGHTS  = HERE / "weights"
LOGS     = HERE / "logs"

for d in (DATA_DIR, WEIGHTS, LOGS):
    d.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
#  3.  WHAT MOVEMENTS TO RECOGNISE
# ══════════════════════════════════════════════════════════════════════════
# NTU action ID  ->  your label
ACTIONS = {
    43: "collapse",   # A043  fall down
    27: "stagger",    # A027  staggering
    24: "assault",    # A024  punching / slapping
    40: "sos",        # A040  cross hands / wave for help
     1: "normal",     # A001  drink water  (normal-motion proxy)
}

LABELS  = ["collapse", "stagger", "assault", "sos", "normal"]
LABEL_ID = {name: i for i, name in enumerate(LABELS)}
N_CLASSES = len(LABELS)

MAX_FILES_PER_CLASS = 500     # lower to 100 for a quick first run


# ══════════════════════════════════════════════════════════════════════════
#  4.  SKELETON FORMAT
# ══════════════════════════════════════════════════════════════════════════
# NTU gives 25 joints, MediaPipe gives 33. They share these 13.
# Training and live inference BOTH use this 13-joint set.
JOINTS = ["head", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
          "l_wrist", "r_wrist", "l_hip", "r_hip",
          "l_knee", "r_knee", "l_ankle", "r_ankle"]
N_JOINTS = 13

NTU_TO_COMMON       = [3, 4, 8, 5, 9, 6, 10, 12, 16, 13, 17, 14, 18]
MEDIAPIPE_TO_COMMON = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

WINDOW = 30      # frames per sample (about 1 second)
STRIDE = 15      # 50% overlap between samples


# ══════════════════════════════════════════════════════════════════════════
#  5.  TRAINING
# ══════════════════════════════════════════════════════════════════════════
HIDDEN_SIZE  = 256
NUM_LAYERS   = 2
FEATURE_SIZE = 128
DROPOUT      = 0.3
EPOCHS       = 30
BATCH_SIZE   = 32
LEARNING_RATE = 1e-3
PATIENCE     = 7
SEED         = 42


# ══════════════════════════════════════════════════════════════════════════
#  6.  LIVE SYSTEM
# ══════════════════════════════════════════════════════════════════════════
YOLO_MODEL = "yolov8n.pt"     # downloads itself on first run
YOLO_CONF  = 0.45

# Threat levels
RED_CONFIDENCE    = 0.75      # min confidence for a RED alert
YELLOW_CONFIDENCE = 0.65

# How long something must persist, in SECONDS of real time.
#
# These used to be frame counts, which was quietly broken: 15 frames is
# half a second on a 30fps video and nearly four seconds on a webcam
# struggling at 4fps. The same number meant different things depending
# on hardware. Seconds mean seconds.
RED_HOLD_SECONDS    = 10.0    # a collapse must last this long to alarm
YELLOW_HOLD_SECONDS = 3.0

# Escalation: several separate warnings for one person means trouble even
# if no single one was convincing.
HISTORY_SECONDS  = 60.0       # how far back to look
ESCALATE_AFTER   = 3          # separate episodes, not individual frames

# Processing speed. Skipping frames costs almost nothing in accuracy
# because bodies don't move much in a thirtieth of a second, and it makes
# video playback several times faster.
#   1 = every frame, 3 = every third frame
PROCESS_EVERY = 3

RED_ACTIONS    = {"collapse", "assault"}
YELLOW_ACTIONS = {"stagger", "sos"}

# Which service each action calls
SERVICE_FOR = {
    "collapse": "AMBULANCE",
    "assault":  "POLICE",
    "stagger":  None,
    "sos":      None,
    "normal":   None,
}


# ══════════════════════════════════════════════════════════════════════════
#  7.  EMERGENCY CALLS  (leave TEST_MODE = True until your final demo)
# ══════════════════════════════════════════════════════════════════════════
TEST_MODE = True              # True = call YOUR phone, never real services

TWILIO_SID   = "PASTE_YOUR_ACCOUNT_SID"
TWILIO_TOKEN = "PASTE_YOUR_AUTH_TOKEN"
TWILIO_FROM  = "+1XXXXXXXXXX"
YOUR_PHONE   = "+971XXXXXXXXX"       # used when TEST_MODE is True

REAL_NUMBERS = {
    "POLICE":    "+971999",
    "AMBULANCE": "+971999",
    "FIRE":      "+971997",
}

OVERRIDE_SECONDS = 10         # time to cancel before the call goes out


# ══════════════════════════════════════════════════════════════════════════
#  8.  RECOGNISING PEOPLE ACROSS CAMERAS
# ══════════════════════════════════════════════════════════════════════════
REID_THRESHOLD = 0.72     # how alike two sightings must be to count as
                          # the same person. Higher = stricter.
                          # Too many people merged together? Raise it.
                          # Same person keeps getting new IDs? Lower it.

REID_USE_DEEP = False     # True uses a ResNet (slower, better with
                          # lighting changes). False uses colour.

REID_FORGET_AFTER = 120   # seconds before someone unseen is forgotten


# ══════════════════════════════════════════════════════════════════════════
#  9.  DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
DASHBOARD_PORT = 5000
DASHBOARD_FPS  = 8        # how often to push camera frames to the browser
JPEG_QUALITY   = 70


# ══════════════════════════════════════════════════════════════════════════
#  10.  WILDTRACK  (only needed for eval_wildtrack.py)
# ══════════════════════════════════════════════════════════════════════════
WILDTRACK_FOLDER = "C:/Users/fgund/OneDrive/Desktop/ds"
WILDTRACK_FRAMES = 60     # how many frames to measure on. 60 is plenty
                          # and keeps the run down to a few minutes.
REID_DEPTH = "mid"        # which part of the network to read features from.
                          # "mid"   = layer3, texture and pattern. Best for
                          #           telling different people apart.
                          # "deep"  = layer4, the conventional choice.
                          # "final" = after pooling. Trained to say "person",
                          #           which every crop already is, so it can't
                          #           tell them apart. Run compare_reid.py to
                          #           see this for yourself.


# ══════════════════════════════════════════════════════════════════════════
#  11.  TRAINING A RE-ID MODEL  (train_reid.py)
# ══════════════════════════════════════════════════════════════════════════
MARKET_FOLDER = "C:/Users/fgund/OneDrive/Desktop/datasets/Market-1501-v15.09.15"

REID_BACKBONE = "resnet18"   # resnet18 is about 4x faster than resnet50 on
                             # a CPU and loses very little here.
REID_HEIGHT   = 128          # WildTrack crops are 30-55 pixels wide, so
REID_WIDTH    = 64           # training small matches the real thing better
                             # than training at 256x128 and upscaling.

REID_EPOCHS   = 20           # about a minute an epoch on CPU with the
                             # settings above. If the run says it was still
                             # improving at the end, raise this.

REID_PEOPLE_PER_BATCH = 8    # triplet loss needs several photos of the
REID_SHOTS_EACH       = 4    # same person together in a batch, so batches
                             # are built as 8 people x 4 photos = 32.
