"""
reid.py — recognising the same person on a different camera.

The tracker gives each person an ID, but only within one camera. Walk out
of shot and back in, or into another camera's view, and you become a new
person as far as it's concerned.

This fixes that. Every time someone is seen, we describe what they look
like as a list of numbers. If a new sighting looks close enough to someone
we've seen before, it's the same person.

Two ways of describing appearance:
  - a colour signature, which needs nothing installed and works surprisingly
    well when people are wearing different colours
  - a ResNet network, which is slower but handles lighting changes better

There's also zone mapping here, which turns a position in the frame into
a place in the building like "Lobby" or "Stairwell".
"""
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import config as C


# ══════════════════════════════════════════════════════════════════════
#  Describing what someone looks like
# ══════════════════════════════════════════════════════════════════════
class ColourSignature:
    """
    Splits the body into horizontal bands and records the colours in each.

    Head, torso, legs get described separately, so a person in a red top
    and dark jeans produces a very different signature to someone in a
    white shirt and light trousers.

    Uses HSV rather than RGB because it copes better with brightness
    changes between cameras.

    Honest limitation: on a crowded outdoor scene where half the people
    are in dark coats, this struggles. It's here because it needs nothing
    installed and it's a fair baseline to measure a real model against.
    """

    def __init__(self, bands=6, hue_bins=12, sat_bins=4,
                 weight_torso=True):
        self.bands = bands
        self.hue_bins = hue_bins
        self.sat_bins = sat_bins
        self.weight_torso = weight_torso
        self.size = bands * hue_bins * sat_bins

    def describe(self, crop_bgr):
        import cv2

        if crop_bgr is None or crop_bgr.size == 0:
            return np.zeros(self.size, dtype=np.float32)

        crop = cv2.resize(crop_bgr, (64, 128))
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Even out brightness differences between cameras before
        # measuring colour, otherwise a sunlit camera and a shaded one
        # describe the same coat completely differently.
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

        band_height = hsv.shape[0] // self.bands
        pieces = []

        for b in range(self.bands):
            band = hsv[b * band_height:(b + 1) * band_height]

            # Ignore near-black and near-white pixels. They're usually
            # shadow or blown-out sky rather than clothing, and they
            # make everybody look alike.
            mask = cv2.inRange(band, (0, 30, 30), (180, 255, 245))

            hist = cv2.calcHist([band], [0, 1], mask,
                                [self.hue_bins, self.sat_bins],
                                [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            # The torso is where the distinctive clothing usually is.
            # Head and feet are mostly skin, hair and ground.
            if self.weight_torso and 1 <= b <= 3:
                hist = hist * 1.6

            pieces.append(hist)

        signature = np.concatenate(pieces).astype(np.float32)
        length = np.linalg.norm(signature)
        return signature / (length + 1e-6)


class DeepSignature:
    """
    Uses a pretrained network to describe appearance.

    WHY THE OBVIOUS VERSION DOESN'T WORK
    ------------------------------------
    The first thing anyone tries is ResNet's final layer. It barely beats
    colour, and the reason is worth understanding.

    ResNet was trained on ImageNet to answer "what kind of object is this".
    Its last layer has learned to say "person" — and every single crop here
    IS a person. So it says the same thing about all 75 of them. The feature
    that makes it good at its original job makes it useless at this one.

    Earlier layers haven't collapsed to a category yet. They still carry
    texture, colour patterns, the shape of a jacket. That's what tells two
    people apart.

    So this pulls from layer3 instead of the end, splits the body into
    horizontal stripes so a red top and dark trousers stay distinguishable,
    and averages over a horizontal flip so which way someone faces matters
    less.
    """

    def __init__(self, depth="mid", stripes=4):
        """
        depth:
            "early"  layer2, mostly edges and colour.
            "mid"    layer3, texture and pattern.  Usually the best of these.
            "deep"   layer4, the conventional choice.
            "final"  after pooling.                What most people try first.

        Going earlier trades semantic meaning for raw appearance. Since
        every crop is already a person, semantic meaning is the part we
        don't need.
        """
        self.ready = False
        self.depth = depth
        self.stripes = stripes

        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms

            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            parts = list(net.children())

            if depth == "early":
                self.net = torch.nn.Sequential(*parts[:6])    # up to layer2
                channels = 512
            elif depth == "mid":
                self.net = torch.nn.Sequential(*parts[:7])    # up to layer3
                channels = 1024
            elif depth == "deep":
                self.net = torch.nn.Sequential(*parts[:8])    # up to layer4
                channels = 2048
            else:
                self.net = torch.nn.Sequential(*parts[:9])    # through avgpool
                channels = 2048
                stripes = 1

            self.net.eval().to(self.device)

            self.prepare = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])

            self.size = channels * stripes
            self.ready = True
            print(f"  appearance: ResNet50 {depth} layer, "
                  f"{stripes} stripes ({self.device})")

        except ImportError:
            print("  appearance: torchvision missing, using colour instead")
            print("              pip install torchvision")
        except Exception as e:
            print(f"  appearance: couldn't load the network ({type(e).__name__})")
            print("              check your internet, the weights download once")

    def _pool(self, maps):
        """
        maps: (batch, channels, height, width)

        Average within horizontal stripes rather than over the whole image.
        Pooling everything together throws away the fact that the colour
        was at the top and not the bottom, which is exactly what separates
        one person from another.
        """
        if maps.dim() == 2:
            return maps

        batch, channels, height, width = maps.shape
        if self.stripes <= 1 or height < self.stripes:
            return maps.mean(dim=(2, 3))

        band = height // self.stripes
        pieces = []
        for i in range(self.stripes):
            top = i * band
            bottom = (i + 1) * band if i < self.stripes - 1 else height
            pieces.append(maps[:, :, top:bottom, :].mean(dim=(2, 3)))
        return self.torch.cat(pieces, dim=1)

    def _run(self, batch):
        """Average the features over the image and its mirror image."""
        with self.torch.no_grad():
            normal = self._pool(self.net(batch).flatten(1)
                                if self.depth == "final"
                                else self.net(batch))
            mirrored = self._pool(self.net(batch.flip(3)).flatten(1)
                                  if self.depth == "final"
                                  else self.net(batch.flip(3)))
            combined = (normal + mirrored) / 2

        out = combined.cpu().numpy()
        lengths = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / (lengths + 1e-6)).astype(np.float32)

    def describe(self, crop_bgr):
        if crop_bgr is None or crop_bgr.size == 0:
            return np.zeros(self.size, dtype=np.float32)
        batch = self.prepare(crop_bgr[:, :, ::-1].copy()).unsqueeze(0)
        return self._run(batch.to(self.device))[0]

    def describe_many(self, crops):
        """A batch at once. Much faster with a couple of thousand crops."""
        if not crops:
            return np.zeros((0, self.size), dtype=np.float32)
        batch = self.torch.stack([
            self.prepare(c[:, :, ::-1].copy()) for c in crops
        ]).to(self.device)
        return self._run(batch)


class TrainedSignature:
    """
    The model from train_reid.py, taught on Market-1501.

    Unlike the ImageNet versions, this one was shown the actual task:
    put two photos of the same stranger close together, and two photos
    of different strangers far apart.
    """

    def __init__(self, weights_path=None):
        self.ready = False
        path = Path(weights_path or (C.WEIGHTS / "reid_net.pth"))

        try:
            import torch
            import torch.nn as nn
            import torchvision.models as models
            import torchvision.transforms as transforms

            if not path.exists():
                print("  appearance: no trained model yet")
                print("              run train_reid.py to make one")
                return

            saved = torch.load(path, map_location="cpu")
            state = saved["state"]

            # Work out the architecture from the weights themselves rather
            # than trusting the metadata. Older files were saved before those
            # fields existed, and building the wrong network then loading with
            # strict=False fails silently — you get a model full of random
            # weights that still runs and produces confident nonsense.
            width = None
            for key in ("norm.weight", "name_them.weight"):
                if key in state:
                    shape = state[key].shape
                    width = shape[0] if key == "norm.weight" else shape[1]
                    break
            if width is None:
                width = saved.get("width", 512)

            backbone_name = "resnet50" if width == 2048 else "resnet18"

            n_people = saved.get("people", 751)
            if "name_them.weight" in state:
                n_people = state["name_them.weight"].shape[0]

            height = saved.get("height", 128)
            img_width = saved.get("img_width", 64)

            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            if backbone_name == "resnet50":
                backbone = models.resnet50(weights=None)
            else:
                backbone = models.resnet18(weights=None)
            backbone.fc = nn.Identity()

            class ReIDNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.backbone = backbone
                    self.norm = nn.BatchNorm1d(width)
                    self.drop = nn.Dropout(0.5)
                    self.name_them = nn.Linear(width, n_people, bias=False)

                def describe(self, x):
                    return self.norm(self.backbone(x))

            self.net = ReIDNet()
            missing, unexpected = self.net.load_state_dict(state, strict=False)

            # A handful of mismatches is fine. Dozens means the architecture
            # is wrong and the model would be mostly random.
            if len(missing) > 10:
                print(f"  appearance: saved model doesn't fit "
                      f"({len(missing)} layers missing)")
                print("              retrain with:  python train_reid.py")
                return

            self.net.eval().to(self.device)

            self.prepare = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((height, img_width)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])

            self.size = width
            self.ready = True
            print(f"  appearance: trained on {n_people} people "
                  f"({backbone_name}, {self.device})")

        except ImportError:
            print("  appearance: torchvision missing")
        except Exception as e:
            print(f"  appearance: couldn't load the trained model ({e})")

    def _run(self, batch):
        with self.torch.no_grad():
            normal = self.net.describe(batch)
            mirrored = self.net.describe(batch.flip(3))
            out = ((normal + mirrored) / 2).cpu().numpy()
        lengths = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / (lengths + 1e-6)).astype(np.float32)

    def describe(self, crop_bgr):
        if crop_bgr is None or crop_bgr.size == 0:
            return np.zeros(self.size, dtype=np.float32)
        batch = self.prepare(crop_bgr[:, :, ::-1].copy()).unsqueeze(0)
        return self._run(batch.to(self.device))[0]

    def describe_many(self, crops):
        if not crops:
            return np.zeros((0, self.size), dtype=np.float32)
        batch = self.torch.stack([
            self.prepare(c[:, :, ::-1].copy()) for c in crops
        ]).to(self.device)
        return self._run(batch)


class CombinedSignature:
    """
    Colour and network features stuck together.

    They fail in different ways. Colour is thrown by lighting but knows
    exactly what shade someone's coat is. The network handles lighting
    but is vaguer about colour. Using both usually beats either.
    """

    def __init__(self, colour_weight=0.5, depth="mid"):
        self.colour = ColourSignature()
        self.deep = DeepSignature(depth=depth)
        self.colour_weight = colour_weight
        self.ready = self.deep.ready
        self.size = self.colour.size + (self.deep.size if self.deep.ready else 0)
        if self.ready:
            print(f"  appearance: colour + network together")

    def _join(self, colour_part, deep_part):
        joined = np.concatenate([
            colour_part * self.colour_weight,
            deep_part * (1 - self.colour_weight),
        ], axis=-1)
        if joined.ndim == 1:
            return joined / (np.linalg.norm(joined) + 1e-6)
        lengths = np.linalg.norm(joined, axis=1, keepdims=True)
        return joined / (lengths + 1e-6)

    def describe(self, crop_bgr):
        c = self.colour.describe(crop_bgr)
        if not self.deep.ready:
            return c
        return self._join(c, self.deep.describe(crop_bgr)).astype(np.float32)

    def describe_many(self, crops):
        c = np.stack([self.colour.describe(x) for x in crops])
        if not self.deep.ready:
            return c
        return self._join(c, self.deep.describe_many(crops)).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
#  Remembering who we've seen
# ══════════════════════════════════════════════════════════════════════
class PersonGallery:
    """
    Keeps a description of everyone seen so far and matches new sightings
    against them.

    A person's description is updated slowly over time rather than replaced,
    so one bad frame doesn't ruin their entry.
    """

    def __init__(self, match_threshold=0.72, forget_after=120, blend=0.25):
        self.threshold = match_threshold
        self.forget_after = forget_after     # seconds
        self.blend = blend                   # how fast descriptions update
        self.people = {}                     # global id -> record
        self._next_id = 1
        self.matches_made = 0
        self.new_people = 0

    def identify(self, signature, camera, local_id):
        """
        Work out who this is.
        Returns (global_id, was_matched, similarity).
        """
        now = time.time()
        self._forget_old(now)

        if not self.people:
            return self._add(signature, camera, local_id, now), False, 0.0

        ids = list(self.people.keys())
        known = np.stack([self.people[i]["signature"] for i in ids])
        scores = known @ signature          # cosine, both are unit length

        best = int(np.argmax(scores))
        score = float(scores[best])

        if score >= self.threshold:
            gid = ids[best]
            record = self.people[gid]
            record["signature"] = self._blend(record["signature"], signature)
            record["last_seen"] = now
            record["cameras"].add(camera)
            record["sightings"] += 1
            self.matches_made += 1
            return gid, True, score

        return self._add(signature, camera, local_id, now), False, score

    def _add(self, signature, camera, local_id, now):
        gid = self._next_id
        self._next_id += 1
        self.people[gid] = {
            "signature": signature.copy(),
            "first_seen": now,
            "last_seen": now,
            "cameras": {camera},
            "sightings": 1,
        }
        self.new_people += 1
        return gid

    def _blend(self, old, new):
        mixed = (1 - self.blend) * old + self.blend * new
        return mixed / (np.linalg.norm(mixed) + 1e-6)

    def _forget_old(self, now):
        stale = [gid for gid, r in self.people.items()
                 if now - r["last_seen"] > self.forget_after]
        for gid in stale:
            del self.people[gid]

    def seen_on_multiple_cameras(self):
        """Anyone who has appeared on more than one camera."""
        return {gid: r for gid, r in self.people.items()
                if len(r["cameras"]) > 1}

    def summary(self):
        total = self.matches_made + self.new_people
        rate = self.matches_made / total if total else 0.0
        return {
            "people_known": len(self.people),
            "matches": self.matches_made,
            "new": self.new_people,
            "match_rate": rate,
            "on_multiple_cameras": len(self.seen_on_multiple_cameras()),
        }

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"people": self.people, "next_id": self._next_id}, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.people = data["people"]
        self._next_id = data["next_id"]


# ══════════════════════════════════════════════════════════════════════
#  Where in the building
# ══════════════════════════════════════════════════════════════════════
class ZoneMap:
    """
    Turns a position in the camera frame into a named place.

    Two modes. Without calibration it splits the frame into a rough grid,
    which is fine for a demo. With calibration it uses a homography, which
    maps the floor properly and lets two cameras agree on where someone is.

    To calibrate: pick four points on the floor visible in the camera,
    measure roughly where they are on your floor plan, and pass both sets in.
    """

    def __init__(self, zones=None, homographies=None, auto_load=True):
        self.zones = zones or DEFAULT_ZONES
        self.homographies = homographies or {}

        # Pick up anything calibrate.py has saved
        if auto_load and not self.homographies:
            try:
                from calibrate import load_homographies
                self.homographies = load_homographies()
            except Exception:
                self.homographies = {}

        self.calibrated = bool(self.homographies)

    def where(self, camera, box, frame_shape):
        """box is (x1, y1, x2, y2). Returns a zone name."""
        return self.locate(camera, box, frame_shape)[0]

    def locate(self, camera, box, frame_shape):
        """
        Returns (zone name, floor position, is it real).

        The third value matters. With calibration the position is a real
        point on the floor plan that another camera would agree with.
        Without it, it's just where they happen to sit in the picture,
        which is a guess. Worth being honest about which you have.
        """
        height, width = frame_shape[:2]

        # Feet are the part touching the floor, so use the bottom centre
        foot_x = (box[0] + box[2]) / 2
        foot_y = box[3]

        real = camera in self.homographies
        if real:
            x, y = self._project(camera, foot_x, foot_y)
        else:
            x, y = foot_x / width, foot_y / height

        for zone in self.zones:
            x0, x1 = zone["x"]
            y0, y1 = zone["y"]
            if x0 <= x < x1 and y0 <= y < y1:
                return zone["name"], (x, y), real
        return "unknown area", (x, y), real

    def _project(self, camera, px, py):
        H = self.homographies[camera]
        point = np.array([px, py, 1.0])
        out = H @ point
        return float(out[0] / out[2]), float(out[1] / out[2])

    @staticmethod
    def calibrate(image_points, floor_points):
        """
        Work out the homography for one camera.

        image_points: four (x, y) spots on the floor as seen by the camera
        floor_points: where those same spots are on your floor plan, 0-1

        Pick corners of a tile, doorway edges, anything flat and identifiable.
        """
        import cv2
        src = np.array(image_points, dtype=np.float32)
        dst = np.array(floor_points, dtype=np.float32)
        H, _ = cv2.findHomography(src, dst)
        return H


DEFAULT_ZONES = [
    {"name": "Entrance",      "x": (0.00, 0.33), "y": (0.00, 0.50)},
    {"name": "Lobby",         "x": (0.33, 0.67), "y": (0.00, 0.50)},
    {"name": "Corridor",      "x": (0.67, 1.00), "y": (0.00, 0.50)},
    {"name": "Stairwell",     "x": (0.00, 0.33), "y": (0.50, 1.00)},
    {"name": "Central Area",  "x": (0.33, 0.67), "y": (0.50, 1.00)},
    {"name": "Fire Exit",     "x": (0.67, 1.00), "y": (0.50, 1.00)},
]


# ══════════════════════════════════════════════════════════════════════
#  Putting it together
# ══════════════════════════════════════════════════════════════════════
class ReIdentifier:
    """
    What run_live.py actually uses. Give it a crop and a box, get back
    a person ID that stays the same across cameras, plus a zone name.
    """

    def __init__(self, use_deep=False, match_threshold=None):
        threshold = match_threshold or C.REID_THRESHOLD

        if use_deep:
            # A model trained on the actual task beats ImageNet features,
            # so use it if train_reid.py has been run.
            trained = TrainedSignature()
            if trained.ready:
                self.describer = trained
            else:
                self.describer = DeepSignature(
                    depth=getattr(C, "REID_DEPTH", "mid"))
                if not self.describer.ready:
                    self.describer = ColourSignature()
        else:
            self.describer = ColourSignature()
            print("  appearance: colour signature")

        self.gallery = PersonGallery(match_threshold=threshold)
        self.zones = ZoneMap()
        self.last_position = (0.0, 0.0)
        self.last_position_real = False

        print(f"  re-identification ready (match above {threshold})")
        if self.zones.calibrated:
            cams = sorted(self.zones.homographies.keys())
            print(f"  location: calibrated for camera(s) {cams}")
        else:
            print("  location: not calibrated, so zones are guessed from")
            print("            frame position. run calibrate.py to fix that.")

    def identify(self, crop, box, camera, local_id, frame_shape):
        """Returns (global_id, zone, was_matched, similarity)."""
        signature = self.describer.describe(crop)
        gid, matched, score = self.gallery.identify(signature, camera, local_id)
        zone, position, real = self.zones.locate(camera, box, frame_shape)
        self.last_position = position
        self.last_position_real = real
        return gid, zone, matched, score

    def locate(self, box, camera, frame_shape):
        """Just the position, without touching the gallery."""
        return self.zones.locate(camera, box, frame_shape)

    def stats(self):
        return self.gallery.summary()
