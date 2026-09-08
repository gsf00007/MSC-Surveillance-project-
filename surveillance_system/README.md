# Multi-Camera Surveillance with Threat Detection and Emergency Dispatch

MSc project. The system watches camera feeds, follows people between them,
reads how they're moving, decides how serious it looks, and calls for help —
with a ten-second window for a person to stop it.

---

## What it does

```
camera  →  find people  →  follow them  →  recognise them on other cameras
        →  read their pose  →  classify the movement
        →  decide how serious  →  call for help, or don't
```

Seven stages, each measurable on its own.

---

## Results

**Movement classification** — trained on NTU RGB+D, tested on people the model
had never seen.

| Movement | Precision | Recall | F1 |
|---|---|---|---|
| collapse | 1.00 | 0.84 | **0.91** |
| normal | 0.93 | 0.80 | 0.86 |
| sos | 0.65 | 0.69 | 0.67 |
| stagger | 0.45 | 0.59 | 0.51 |
| assault | 0.36 | 0.53 | 0.43 |

Macro F1 0.677. Zero serious events were labelled as normal.

**As a complete system**, with the confidence and duration filters applied:

| | Classifier alone | Full system |
|---|---|---|
| false alarms | 10.2% | 4.1% |
| collapses auto-dispatched | — | 76% (29 of 38) |
| collapses alerting someone | — | 89% (34 of 38) |

**Re-identification on WildTrack**, evaluated but never trained on:

| Method | Rank-1 |
|---|---|
| colour histogram | 13.3% |
| ResNet50 final layer | 15.0% |
| ResNet50 layer4 | 18.7% |
| ResNet50 layer3 | 23.1% |
| trained on Market-1501 | 24.8% |

The same trained model scores 87.1% on Market-1501 itself.

---

## Findings

**ImageNet features get worse the deeper you read them.** The final layer was
trained to answer "what kind of object is this" and outputs "person". Every
crop here is already a person, so it says the same thing about all of them.
Earlier layers still carry texture and clothing, which is what actually
separates one person from another.

**Re-identification does not transfer between datasets.** 87.1% on Market-1501,
24.8% on WildTrack, from the same model. Market was shot at head height with
large crops; WildTrack looks down on a crowded square with 30–55 pixel people.

**A gesture classifier trained on lab footage does transfer to real CCTV.**
Trained on Kinect recordings of volunteers, it correctly identified a collapse
in a doorbell camera video at 85–86% confidence.

**The dispatch threshold is a choice, not an optimisation.** At 90% confidence
only 17% of real collapses call an ambulance. At 75%, 76% do, for one extra
false alarm across 49 normal clips. The right answer depends on whether an
operator is watching, since YELLOW alerts reach a human either way.

---

## Running it

```
conda create -n msc python=3.10
conda activate msc
pip install -r requirements.txt
```

Set the three dataset paths at the top of `config.py`, then:

```
python check_setup.py         is everything in place
python prepare_data.py        build training data from NTU
python train.py               train the classifier
python evaluate.py            per-class results and plots
python run_live.py --dashboard --explain
```

Then open `http://localhost:5000`.

---

## The files

### Core pipeline

| File | |
|---|---|
| `config.py` | every setting in one place |
| `skeleton.py` | reads NTU files, converts MediaPipe output, normalises both |
| `prepare_data.py` | NTU skeletons into training data |
| `train.py` | trains the LSTM and SVM |
| `run_live.py` | the whole system on a camera |

### Deciding and responding

| File | |
|---|---|
| `threat.py` | duration, confidence and history checks; RED / YELLOW / GREEN |
| `dispatch.py` | countdown, cancel, Twilio call, audit log |
| `dashboard.py` | operator console at localhost:5000 |
| `explain.py` | plain English, SHAP values, counterfactuals |

### Recognising people across cameras

| File | |
|---|---|
| `reid.py` | appearance descriptors and the person gallery |
| `train_reid.py` | trains a re-id model on Market-1501 |
| `calibrate.py` | click four floor points to get a real floor position |

### Measuring

| File | |
|---|---|
| `evaluate.py` | classifier performance, confusion matrix |
| `eval_system.py` | what the whole system does, filters included |
| `tune_thresholds.py` | where the thresholds should sit, and why |
| `compare_reid.py` | every appearance descriptor, side by side |
| `eval_wildtrack.py` | re-identification accuracy on WildTrack |
| `eval_location.py` | do two cameras agree where someone is standing |

### Setup and diagnosis

| File | |
|---|---|
| `check_setup.py` | verifies packages, models, datasets before a demo |
| `check_classifier.py` | live view of what the classifier is thinking |
| `setup_calls.py` | Twilio credentials and a safe test call |

---

## Datasets

| Dataset | Used for | Trained on it |
|---|---|---|
| NTU RGB+D 120 | movement classification | yes, from scratch |
| Market-1501 | re-identification | yes, fine-tuned from ImageNet |
| WildTrack | re-identification evaluation | **no** |

WildTrack was never trained on. Every number from it comes from a model that
has not seen a single WildTrack image.

---

## Design decisions worth knowing

**Thirteen joints, not thirty-three.** NTU's Kinect reports 25 joints,
MediaPipe reports 33, and they are different joints in a different order. Both
are reduced to the 13 they share. Without this the model would train on one
layout and run on another, producing confident nonsense without ever crashing.

**An SVM after the LSTM.** A neural network's decision is hard to interrogate;
an SVM's is not. For a system that can dial 999, being able to explain a
decision is worth more than a marginal accuracy gain.

**Split by person, not by clip.** If the same person appeared in training and
test, the model could learn how they personally move rather than what the
movement is. Every test subject is someone the model has never seen.

**Duration measured in seconds, not frames.** Fifteen frames is half a second
on a 30fps video and nearly four seconds on a struggling webcam. Seconds mean
the same thing on any hardware.

**A ten-second countdown before any call.** Long enough for a watching operator
to stop a false alarm, short enough not to matter in a real emergency. The
point is that no automatic call goes out that a human could not have prevented.

---

## Known limitations

**The "normal" class is too narrow.** NTU's A001 is "drink water" — a person
standing still. That is not the same as walking around a building, and it is
the most likely source of false alarms in deployment.

**Stagger and assault are weak** (F1 0.51 and 0.43). Both involve sudden
upper-body movement and only ~68 training clips each were available.

**Re-id accuracy is low in absolute terms** because the model was never trained
on WildTrack. Fine-tuning on WildTrack identities, with a person-level split,
is the obvious next step.

**Demonstrated on one camera.** The pipeline supports several and
re-identification was evaluated across WildTrack's seven, but the live system
was shown on one.

**Trained on acted movements.** NTU participants were told to fall over. The
doorbell camera test partly addresses this; one video is not evidence.
