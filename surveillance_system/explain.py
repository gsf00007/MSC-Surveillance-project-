"""
explain.py — why did it decide that?

A system that can call an ambulance shouldn't be a black box. When it
flags someone, whoever's watching needs to see the reasoning, not just
a percentage.

Three kinds of answer here:

  plain English    "barely moving for 2 seconds, and much lower than
                    they started"  — what an operator reads

  SHAP             which of the 128 learned features pushed the decision
                    — what goes in your dissertation

  counterfactual   "if they'd got up within 8 frames this would be yellow"
                    — what would have had to be different
"""
import warnings

import numpy as np

import config as C

# SHAP's internal regression prints a lot of harmless warnings.
# They bury the actual output, so they're switched off here.
warnings.filterwarnings("ignore", category=UserWarning, module="shap")
warnings.filterwarnings("ignore", message=".*Regressors in active set.*")
warnings.filterwarnings("ignore", message=".*singular.*")


# ══════════════════════════════════════════════════════════════════════
#  Plain English
# ══════════════════════════════════════════════════════════════════════
def describe_movement(clip):
    """
    Look at the actual body positions and pull out things a person
    would notice. No model involved, just measurements.

    clip: (frames, 13, 3) already normalised
    """
    clip = np.asarray(clip)

    HEAD, L_SHO, R_SHO = 0, 1, 2
    L_WRI, R_WRI = 5, 6
    L_HIP, R_HIP = 7, 8
    L_ANK, R_ANK = 11, 12

    frame_to_frame = np.diff(clip, axis=0)
    speed = np.abs(frame_to_frame).mean()

    height = clip[:, HEAD, 1]
    height_lost = float(height[0] - height.min())

    shoulders = (clip[:, L_SHO] + clip[:, R_SHO]) / 2
    wrists = (clip[:, L_WRI] + clip[:, R_WRI]) / 2
    arms_up = float((shoulders[:, 1] - wrists[:, 1]).mean())

    still_frames = int((np.abs(frame_to_frame).mean(axis=(1, 2)) < 0.01).sum())

    sideways = clip[:, [L_HIP, R_HIP], 0].mean(axis=1)
    wobble = float(np.std(np.diff(sideways))) if len(sideways) > 1 else 0.0

    feet = clip[:, [L_ANK, R_ANK], 1].mean(axis=1)
    foot_movement = float(np.abs(np.diff(feet)).mean()) if len(feet) > 1 else 0.0

    return {
        "speed": float(speed),
        "height_lost": height_lost,
        "arms_raised": arms_up,
        "still_frames": still_frames,
        "wobble": wobble,
        "foot_movement": foot_movement,
        "total_frames": len(clip),
    }


def to_sentences(signs, action, confidence):
    """Turn those measurements into something readable."""
    lines = []

    if signs["height_lost"] > 0.35:
        lines.append(f"dropped noticeably from where they started "
                     f"({signs['height_lost']:.2f})")
    elif signs["height_lost"] > 0.18:
        lines.append("lost some height")

    still = signs["still_frames"]
    total = signs["total_frames"]
    if still > total * 0.6:
        lines.append(f"barely moved for {still} of {total} frames")
    elif signs["speed"] > 0.045:
        lines.append("moving quickly and erratically")

    if signs["arms_raised"] > 0.28:
        lines.append("arms held up above the shoulders")

    if signs["wobble"] > 0.028:
        lines.append("unsteady side to side")

    if signs["foot_movement"] < 0.006 and still > total * 0.4:
        lines.append("feet not moving at all")

    if not lines:
        lines.append("nothing unusual in the movement")

    return (f"Looks like {action} ({confidence:.0%} confident). "
            + "; ".join(lines) + ".")


# ══════════════════════════════════════════════════════════════════════
#  SHAP
# ══════════════════════════════════════════════════════════════════════
class Explainer:
    """
    Works out which learned features drove a decision.

    Needs the background sample that train.py saved. Without it there's
    nothing to compare against and SHAP can't run.
    """

    def __init__(self, quiet=False):
        self.ready = False
        try:
            import shap
            import joblib

            svm, scaler = joblib.load(C.WEIGHTS / "classifier.pkl")
            background = np.load(C.DATA_DIR / "shap_background.npy")

            self.svm = svm
            self.scaler = scaler
            self.shap = shap

            sample = background[:40]
            self.explainer = shap.KernelExplainer(
                svm.predict_proba, scaler.transform(sample))
            self.ready = True
            if not quiet:
                print(f"  explanations ready ({len(sample)} reference clips)")

        except ImportError:
            if not quiet:
                print("  shap not installed, so no feature breakdown")
                print("  pip install shap")
        except FileNotFoundError:
            if not quiet:
                print("  no reference data yet, run train.py first")
        except Exception as e:
            if not quiet:
                print(f"  explanations unavailable: {e}")

    def why(self, summary, predicted_class, samples=60):
        """
        summary: the (128,) output from the network for one clip
        Returns the SHAP values for the predicted class, or None.
        """
        if not self.ready:
            return None

        scaled = self.scaler.transform(summary.reshape(1, -1))
        values = self.explainer.shap_values(scaled, nsamples=samples,
                                            silent=True)

        # shap returns different shapes depending on version
        if isinstance(values, list):
            return np.asarray(values[predicted_class][0])
        values = np.asarray(values)
        if values.ndim == 3:
            return values[0, :, predicted_class]
        return values[0]

    def top_features(self, shap_values, n=6):
        """The n features that mattered most, biggest first."""
        if shap_values is None:
            return []
        order = np.argsort(np.abs(shap_values))[::-1][:n]
        return [(int(i), float(shap_values[i])) for i in order]

    def chart(self, shap_values, save_to=None, n=10):
        """Bar chart of the most influential features."""
        if shap_values is None:
            return None

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        order = np.argsort(np.abs(shap_values))[::-1][:n]
        values = shap_values[order]
        names = [f"feature {i}" for i in order]

        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        colours = ["#B03A2E" if v > 0 else "#276749" for v in values]
        ax.barh(range(len(values)), values, color=colours)
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("pushed towards (red) or away from (green) this class")
        ax.set_title("What drove the decision")
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        plt.tight_layout()
        path = save_to or (C.LOGS / "why.png")
        plt.savefig(path, dpi=150)
        plt.close()
        return str(path)


# ══════════════════════════════════════════════════════════════════════
#  Counterfactual
# ══════════════════════════════════════════════════════════════════════
def what_would_change_it(alert, signs):
    """
    What would have needed to be different for a lower alert level.

    More useful to an operator than a feature ranking, because it tells
    them what to look for when deciding whether to cancel.
    """
    if alert.level == "GREEN":
        return None

    reasons = []

    if alert.level == "RED":
        gap = C.RED_CONFIDENCE - alert.confidence
        if gap > 0:
            reasons.append(f"confidence was {alert.confidence:.0%}; below "
                           f"{C.RED_CONFIDENCE:.0%} it stays yellow")
        else:
            over = alert.confidence - C.RED_CONFIDENCE
            reasons.append(f"confidence {over:.0%} above the "
                           f"{C.RED_CONFIDENCE:.0%} line")

        if signs["still_frames"] > 5:
            reasons.append(f"if they had moved within "
                           f"{signs['still_frames']} frames instead of "
                           f"staying still, this would be yellow")

        if signs["height_lost"] > 0.3:
            reasons.append(f"the drop of {signs['height_lost']:.2f} is what "
                           f"separates this from sitting down")

    elif alert.level == "YELLOW":
        needed = C.RED_CONFIDENCE - alert.confidence
        if needed > 0:
            reasons.append(f"another {needed:.0%} of confidence would make "
                           f"this red")
        reasons.append("holding longer would also escalate it")

    return "  |  ".join(reasons) if reasons else None


# ══════════════════════════════════════════════════════════════════════
#  Everything at once
# ══════════════════════════════════════════════════════════════════════
def full_explanation(clip, alert, summary=None, explainer=None):
    """
    The complete picture for one alert.

    clip     : (frames, 13, 3) normalised body positions
    alert    : the Alert from threat.py
    summary  : the network's (128,) output, if you want SHAP
    explainer: a shared Explainer, so it isn't rebuilt every time
    """
    signs = describe_movement(clip)

    result = {
        "plain": to_sentences(signs, alert.action, alert.confidence),
        "measurements": signs,
        "counterfactual": what_would_change_it(alert, signs),
        "shap": None,
        "top_features": [],
    }

    if summary is not None and explainer is not None and explainer.ready:
        class_id = C.LABEL_ID.get(alert.action)
        if class_id is not None:
            values = explainer.why(summary, class_id)
            result["shap"] = values
            result["top_features"] = explainer.top_features(values)

    return result
