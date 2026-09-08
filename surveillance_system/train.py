"""
train.py — train the movement classifier.

Run:   python train.py

Two models work together here. An LSTM reads the motion over time and
boils each clip down to 128 numbers. An SVM then makes the actual
decision from those numbers.

Splitting the job like this is deliberate: the SVM is easy to explain
with SHAP, and for a system that can call an ambulance, being able to
say why it decided something matters.
"""
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
import joblib

import config as C
import skeleton as S


# ── The network ───────────────────────────────────────────────────────────
class MotionNet(nn.Module):
    """
    Reads a sequence of body positions and summarises the movement.

    in :  (batch, 30 frames, 39 numbers)
    out:  (batch, 128) summary, or class scores during training
    """

    def __init__(self, n_classes=C.N_CLASSES):
        super().__init__()
        input_size = C.N_JOINTS * 3          # 13 joints x (x,y,z) = 39

        self.lstm = nn.LSTM(
            input_size, C.HIDDEN_SIZE, C.NUM_LAYERS,
            batch_first=True,
            dropout=C.DROPOUT if C.NUM_LAYERS > 1 else 0.0,
        )
        self.squeeze = nn.Sequential(
            nn.Linear(C.HIDDEN_SIZE, C.FEATURE_SIZE),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.decide = nn.Linear(C.FEATURE_SIZE, n_classes)

    def summarise(self, x):
        out, _ = self.lstm(x)
        return self.squeeze(out[:, -1, :])   # last time step

    def forward(self, x):
        return self.decide(self.summarise(x))


# ── Splitting ─────────────────────────────────────────────────────────────
def split_by_person(X, y, info):
    """
    Split by PERSON, not by clip.

    If the same person appears in training and testing, the model can
    memorise how they move and your accuracy looks better than it is.
    An examiner will ask about this.
    """
    people = np.array([r["person"] for r in info])
    unique = np.unique(people)

    rng = np.random.default_rng(C.SEED)
    rng.shuffle(unique)

    n = len(unique)
    print(f"  {n} different people in the data")

    if n < 3:
        print("  Need at least 3 people to split properly.")
        print("  Raise MAX_FILES_PER_CLASS in config.py.")
        sys.exit(1)

    n_val  = max(1, round(n * 0.15))
    n_test = max(1, round(n * 0.15))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train, n_val, n_test = n - 2, 1, 1

    train_p = set(unique[:n_train])
    val_p   = set(unique[n_train:n_train + n_val])
    test_p  = set(unique[n_train + n_val:])

    pick = lambda group: np.array([p in group for p in people])
    parts = {
        "train": (X[pick(train_p)], y[pick(train_p)]),
        "val":   (X[pick(val_p)],   y[pick(val_p)]),
        "test":  (X[pick(test_p)],  y[pick(test_p)]),
    }

    print(f"  train {len(train_p)} people | "
          f"val {len(val_p)} | test {len(test_p)}\n")
    for name, (Xa, ya) in parts.items():
        counts = {C.LABELS[c]: n for c, n in sorted(Counter(ya.tolist()).items())}
        print(f"  {name:>5}: {len(ya):>6} samples  {counts}")
        if len(ya) == 0:
            print(f"\n  The {name} split came out empty.")
            print("  Raise MAX_FILES_PER_CLASS in config.py.")
            sys.exit(1)

    return parts


# ── Training the network ──────────────────────────────────────────────────
def train_network(parts, device):
    X_train, y_train = parts["train"]
    X_val,   y_val   = parts["val"]

    def make_loader(Xa, ya, shuffle):
        ds = TensorDataset(torch.tensor(S.flatten(Xa)), torch.tensor(ya))
        return DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=shuffle)

    train_dl = make_loader(X_train, y_train, True)
    val_dl   = make_loader(X_val,   y_val,   False)

    model = MotionNet().to(device)
    optimiser = torch.optim.Adam(model.parameters(),
                                 lr=C.LEARNING_RATE, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_score, stale = 0.0, 0
    save_to = C.WEIGHTS / "motion_net.pth"

    print(f"\nTraining on {device}\n")

    for epoch in range(1, C.EPOCHS + 1):
        model.train()
        running, right, seen = 0.0, 0, 0

        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            scores = model(xb)
            loss = loss_fn(scores, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            running += loss.item()
            right   += (scores.argmax(1) == yb).sum().item()
            seen    += len(yb)

        model.eval()
        guesses, truth = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                guesses += model(xb.to(device)).argmax(1).cpu().tolist()
                truth   += yb.tolist()
        score = f1_score(truth, guesses, average="macro")

        print(f"  epoch {epoch:>2}   loss {running/len(train_dl):.4f}   "
              f"train {right/seen:.3f}   val {score:.3f}")

        if score > best_score:
            best_score, stale = score, 0
            torch.save(model.state_dict(), save_to)
        else:
            stale += 1
            if stale >= C.PATIENCE:
                print(f"  no improvement for {C.PATIENCE} epochs, stopping")
                break

    model.load_state_dict(torch.load(save_to, map_location=device))
    print(f"\n  best validation score: {best_score:.3f}")
    return model


# ── Training the classifier ───────────────────────────────────────────────
def train_classifier(model, parts, device):
    model.eval()

    def summarise_all(Xa):
        chunks = []
        with torch.no_grad():
            for i in range(0, len(Xa), C.BATCH_SIZE):
                batch = torch.tensor(S.flatten(Xa[i:i + C.BATCH_SIZE])).to(device)
                chunks.append(model.summarise(batch).cpu().numpy())
        return np.concatenate(chunks)

    X_train, y_train = parts["train"]
    X_val,   y_val   = parts["val"]

    print("\nSummarising clips for the classifier...")
    F_train = summarise_all(X_train)
    F_val   = summarise_all(X_val)

    scaler = StandardScaler().fit(F_train)

    print("Fitting SVM...")
    svm = SVC(kernel="rbf", C=10.0, gamma="scale",
              probability=True, class_weight="balanced",
              random_state=C.SEED)
    svm.fit(scaler.transform(F_train), y_train)

    guesses = svm.predict(scaler.transform(F_val))
    present = sorted(set(y_train.tolist()) | set(y_val.tolist()))
    print()
    print(classification_report(
        y_val, guesses, labels=present,
        target_names=[C.LABELS[i] for i in present],
        zero_division=0))

    joblib.dump((svm, scaler), C.WEIGHTS / "classifier.pkl")

    # Keep a small sample for SHAP explanations later
    keep = min(100, len(F_train))
    idx = np.random.default_rng(C.SEED).choice(len(F_train), keep, replace=False)
    np.save(C.DATA_DIR / "shap_background.npy", F_train[idx])

    return svm, scaler


def main():
    print("=" * 46)
    print("  Training the movement classifier")
    print("=" * 46)

    try:
        X = np.load(C.DATA_DIR / "X.npy")
        y = np.load(C.DATA_DIR / "y.npy")
        import json
        with open(C.DATA_DIR / "info.json") as f:
            info = json.load(f)
    except FileNotFoundError:
        print("\nNo training data found.")
        print("Run this first:   python prepare_data.py")
        sys.exit(1)

    print(f"\nLoaded {len(X)} samples of shape {X.shape[1:]}\n")

    parts = split_by_person(X, y, info)

    np.save(C.DATA_DIR / "X_test.npy", parts["test"][0])
    np.save(C.DATA_DIR / "y_test.npy", parts["test"][1])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_network(parts, device)
    train_classifier(model, parts, device)

    print("\n" + "=" * 46)
    print("  Saved weights/motion_net.pth")
    print("  Saved weights/classifier.pkl")
    print("=" * 46)
    print("\nNext:  python evaluate.py")


if __name__ == "__main__":
    main()
