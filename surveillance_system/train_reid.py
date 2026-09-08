"""
train_reid.py — teach a network to tell people apart.

WHY THE FIRST ATTEMPT UNDERPERFORMED
------------------------------------
Three epochs of ResNet50 on a CPU took over an hour and produced a model
that scored 15.3%, below plain ImageNet layer3 features at 23.1%.

The training curve explains it. Loss went 6.0 -> 3.8 -> 2.8, still falling
by a full point per epoch when it stopped. Training accuracy was 58.9% and
climbing by 23 points an epoch. That model was stopped less than halfway
through learning anything.

WHAT CHANGED
------------
Speed, so it can actually finish:
  ResNet18 instead of ResNet50, and 128x64 input instead of 256x128.
  Together about 19 times faster per epoch. Twenty epochs now costs less
  than three did before.

  The smaller input is not only a speed trick. WildTrack crops are 30 to
  55 pixels wide. Training at 256x128 and testing on a 40-pixel crop
  blown up sixfold is a mismatch. 128x64 sits much closer to reality.

How it learns:
  Classification alone teaches the network to name 736 specific people.
  That is not quite the job. The job is to place two photos of the same
  stranger near each other, and two photos of different strangers far
  apart, for people it has never seen.

  Triplet loss trains that directly. Within each batch it finds the
  hardest positive and the hardest negative for every photo and pushes
  them apart. Both losses together is the standard recipe.

Run:   python train_reid.py
"""
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import config as C


# ══════════════════════════════════════════════════════════════════════
#  Reading Market-1501
# ══════════════════════════════════════════════════════════════════════
def find_market(root):
    """Locate bounding_box_train, however deeply it got nested."""
    root = Path(root)
    if not root.exists():
        return None

    if (root / "bounding_box_train").is_dir():
        return root / "bounding_box_train"

    for candidate in root.iterdir():
        if not candidate.is_dir():
            continue
        if (candidate / "bounding_box_train").is_dir():
            return candidate / "bounding_box_train"
        for sub in candidate.iterdir():
            if sub.is_dir() and (sub / "bounding_box_train").is_dir():
                return sub / "bounding_box_train"
    return None


def read_market(folder, min_images=4):
    """
    Filenames look like 0002_c1s1_000451_03.jpg
                        ^person  ^camera

    Person -1 is a detector mistake, not a person, so those go.
    Anyone with fewer than four photos can't teach much either.
    """
    by_person = defaultdict(list)

    for path in folder.glob("*.jpg"):
        parts = path.stem.split("_")
        if len(parts) < 2:
            continue
        try:
            person = int(parts[0])
        except ValueError:
            continue
        if person < 0:
            continue
        by_person[person].append(path)

    usable = {p: paths for p, paths in by_person.items()
              if len(paths) >= min_images}

    people = sorted(usable.keys())
    index = {person: i for i, person in enumerate(people)}

    by_index = {index[person]: paths for person, paths in usable.items()}
    samples = [(path, index[person])
               for person, paths in usable.items()
               for path in paths]

    return samples, by_index, len(people)


# ══════════════════════════════════════════════════════════════════════
#  Batches that triplet loss can use
# ══════════════════════════════════════════════════════════════════════
class PeopleSampler:
    """
    Triplet loss needs several photos of the same person in one batch,
    otherwise there is nothing to pull together.

    So each batch is built as P people with K photos each, rather than
    a random scoop of images.
    """

    def __init__(self, by_person, people_per_batch=8, shots_each=4, seed=42):
        self.by_person = by_person
        self.people_per_batch = people_per_batch
        self.shots_each = shots_each
        self.rng = random.Random(seed)
        self.people = list(by_person.keys())

        total = sum(len(v) for v in by_person.values())
        self.batches = max(1, total // (people_per_batch * shots_each))

    def __iter__(self):
        for _ in range(self.batches):
            chosen = self.rng.sample(
                self.people, min(self.people_per_batch, len(self.people)))
            batch = []
            for person in chosen:
                photos = self.by_person[person]
                if len(photos) >= self.shots_each:
                    picked = self.rng.sample(photos, self.shots_each)
                else:
                    picked = [self.rng.choice(photos)
                              for _ in range(self.shots_each)]
                batch += [(p, person) for p in picked]
            yield batch

    def __len__(self):
        return self.batches


# ══════════════════════════════════════════════════════════════════════
#  Losses
# ══════════════════════════════════════════════════════════════════════
def batch_hard_triplet(features, people, margin=0.3):
    """
    For every photo in the batch, find the same person who looks least
    like it, and the different person who looks most like it. Push those
    apart by at least the margin.

    Picking the hardest cases rather than random ones is what makes this
    work. Easy triplets are already correct and teach nothing.
    """
    import torch

    distances = torch.cdist(features, features, p=2)

    same = people.unsqueeze(0) == people.unsqueeze(1)
    different = ~same
    same = same.fill_diagonal_(False)          # not itself

    # Hardest positive: same person, furthest away
    hardest_positive = (distances * same.float()).max(dim=1)[0]

    # Hardest negative: different person, closest
    big = distances.max() + 1.0
    masked = distances + (~different).float() * big
    hardest_negative = masked.min(dim=1)[0]

    loss = torch.clamp(hardest_positive - hardest_negative + margin, min=0)
    return loss.mean()


# ══════════════════════════════════════════════════════════════════════
#  The model
# ══════════════════════════════════════════════════════════════════════
def build_model(n_people, device, backbone_name="resnet18"):
    """
    A backbone, a normalisation layer, and a naming layer.

    The naming layer is scaffolding. What gets used afterwards is what
    comes out of the normalisation layer, which is where the network has
    written down what makes this person look like themselves.
    """
    import torch
    import torch.nn as nn
    import torchvision.models as models

    if backbone_name == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        width = 2048
    else:
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        width = 512

    net.fc = nn.Identity()

    class ReIDNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = net
            self.norm = nn.BatchNorm1d(width)
            self.norm.bias.requires_grad_(False)   # standard for this setup
            self.drop = nn.Dropout(0.5)
            self.name_them = nn.Linear(width, n_people, bias=False)
            self.width = width

        def describe(self, x):
            return self.norm(self.backbone(x))

        def forward(self, x):
            raw = self.backbone(x)               # triplet loss uses this
            normalised = self.norm(raw)
            return self.name_them(self.drop(normalised)), raw

    return ReIDNet().to(device)


# ══════════════════════════════════════════════════════════════════════
#  Loading photos
# ══════════════════════════════════════════════════════════════════════
def make_transforms(height, width):
    import torchvision.transforms as T

    train = T.Compose([
        T.Resize((height, width)),
        T.RandomHorizontalFlip(),
        T.Pad(8),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.5),
    ])
    plain = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train, plain


def load_batch(items, transform):
    import torch
    from PIL import Image

    images, people = [], []
    for path, person in items:
        try:
            image = Image.open(path).convert("RGB")
        except OSError:
            continue
        images.append(transform(image))
        people.append(person)

    if not images:
        return None, None
    return torch.stack(images), torch.tensor(people, dtype=torch.long)


# ══════════════════════════════════════════════════════════════════════
#  Checking it learned something useful
# ══════════════════════════════════════════════════════════════════════
def check_matching(model, by_person, transform, device, n_people=60,
                   shots=4, seed=7):
    """
    Naming accuracy is not the point. What matters is whether two photos
    of the same person end up close together.

    So this holds out some people, describes their photos, and asks: for
    each photo, is the nearest other photo the same person?
    """
    import torch

    rng = random.Random(seed)
    people = rng.sample(list(by_person.keys()),
                        min(n_people, len(by_person)))

    items = []
    for person in people:
        photos = by_person[person]
        picked = rng.sample(photos, min(shots, len(photos)))
        items += [(p, person) for p in picked]

    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for i in range(0, len(items), 32):
            images, tags = load_batch(items[i:i + 32], transform)
            if images is None:
                continue
            out = model.describe(images.to(device))
            out = torch.nn.functional.normalize(out, dim=1)
            features.append(out.cpu().numpy())
            labels.append(tags.numpy())

    if not features:
        return 0.0

    features = np.concatenate(features)
    labels = np.concatenate(labels)

    similarity = features @ features.T
    np.fill_diagonal(similarity, -2)
    nearest = similarity.argmax(axis=1)

    return float((labels[nearest] == labels).mean())


# ══════════════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════════════
def train(model, sampler, by_person, transforms, device,
          epochs=20, lr=3e-4, triplet_weight=1.0):
    import torch
    import torch.nn as nn

    train_tf, plain_tf = transforms
    naming_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    backbone_params = list(model.backbone.parameters())
    head_params = (list(model.norm.parameters()) +
                   list(model.name_them.parameters()))
    optimiser = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": head_params, "lr": lr},
    ], weight_decay=5e-4)

    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=[lr * 0.1, lr],
        total_steps=epochs * len(sampler), pct_start=0.25)

    best = 0.0
    save_to = C.WEIGHTS / "reid_net.pth"
    history = []

    print(f"\n  {len(sampler)} batches an epoch, {epochs} epochs\n")

    for epoch in range(1, epochs + 1):
        model.train()
        started = time.perf_counter()
        totals = {"naming": 0.0, "triplet": 0.0}
        right, seen, batches = 0, 0, 0

        for n, items in enumerate(sampler, 1):
            images, people = load_batch(items, train_tf)
            if images is None or len(images) < 4:
                continue

            images, people = images.to(device), people.to(device)

            optimiser.zero_grad()
            guesses, raw = model(images)

            loss_naming = naming_loss(guesses, people)
            loss_triplet = batch_hard_triplet(raw, people)
            loss = loss_naming + triplet_weight * loss_triplet

            loss.backward()
            optimiser.step()
            schedule.step()

            totals["naming"] += loss_naming.item()
            totals["triplet"] += loss_triplet.item()
            right += (guesses.argmax(1) == people).sum().item()
            seen += len(people)
            batches += 1

            if n % 20 == 0:
                print(f"    epoch {epoch}  batch {n}/{len(sampler)}  "
                      f"naming {totals['naming']/batches:.2f}  "
                      f"triplet {totals['triplet']/batches:.3f}", end="\r")

        matching = check_matching(model, by_person, plain_tf, device)
        took = time.perf_counter() - started

        print(f"    epoch {epoch:>2}  naming {totals['naming']/max(batches,1):.2f}  "
              f"triplet {totals['triplet']/max(batches,1):.3f}  "
              f"train {right/max(seen,1):.1%}  "
              f"matching {matching:.1%}  {took:.0f}s" + " " * 8)

        history.append({"epoch": epoch, "matching": matching,
                        "train": right / max(seen, 1)})

        if matching > best:
            best = matching
            torch.save({"state": model.state_dict(),
                        "people": model.name_them.out_features,
                        "width": model.width,
                        "backbone": ("resnet50" if model.width == 2048
                                     else "resnet18"),
                        "height": C.REID_HEIGHT,
                        "img_width": C.REID_WIDTH},
                       save_to)

    print(f"\n  best matching accuracy: {best:.1%}")
    print(f"  saved to {save_to}")

    # A quick look at whether it was still improving
    if len(history) >= 4:
        recent = [h["matching"] for h in history[-4:]]
        climbing = recent[-1] - recent[0]
        if climbing > 0.02:
            print(f"\n  Still improving ({climbing:+.1%} over the last four).")
            print(f"  More epochs would help. Raise REID_EPOCHS in config.py.")
        else:
            print(f"\n  It has levelled off, so more epochs won't add much.")

    return best


# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 58)
    print("  Teaching a network to tell people apart")
    print("=" * 58)

    try:
        import torch
        import torchvision
        from PIL import Image
    except ImportError as e:
        print(f"\n  missing something: {e}")
        print("  pip install torch torchvision pillow")
        sys.exit(1)

    root = getattr(C, "MARKET_FOLDER", None)
    folder = find_market(root) if root else None
    if folder is None:
        print(f"\n  couldn't find Market-1501 under:\n    {root}")
        print("\n  expected a folder called bounding_box_train")
        sys.exit(1)

    print(f"\n  reading {folder}")
    samples, by_person, n_people = read_market(folder)
    if n_people < 20:
        print(f"\n  only {n_people} people found, not enough")
        sys.exit(1)

    print(f"  {len(samples)} photos of {n_people} people")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = getattr(C, "REID_BACKBONE", "resnet18")
    height = getattr(C, "REID_HEIGHT", 128)
    width = getattr(C, "REID_WIDTH", 64)
    epochs = getattr(C, "REID_EPOCHS", 20)

    print(f"\n  {backbone} at {height}x{width} on {device}")
    if device == "cpu":
        print(f"  roughly a minute or two per epoch")

    transforms = make_transforms(height, width)
    sampler = PeopleSampler(by_person,
                            people_per_batch=getattr(C, "REID_PEOPLE_PER_BATCH", 8),
                            shots_each=getattr(C, "REID_SHOTS_EACH", 4),
                            seed=C.SEED)

    model = build_model(n_people, device, backbone)
    train(model, sampler, by_person, transforms, device, epochs=epochs)

    print("\n" + "=" * 58)
    print("  Now run:  python compare_reid.py")
    print("=" * 58)


if __name__ == "__main__":
    main()
