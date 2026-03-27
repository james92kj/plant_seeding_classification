# Label Smoothing — Why Soft Targets Beat Hard Targets

## The Problem with Hard Targets

In classification, we normally train with hard targets (one-hot encoding):
- Class 3 out of 12: `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]`

The model tries to push the probability of the correct class to **exactly 1.0** and everything else to **exactly 0.0**. To do this, it pushes logits toward infinity.

**Why is this bad?**
- The model becomes **overconfident** — it says "I'm 100% sure" even when it shouldn't be
- Pushes weights to extreme values → overfitting
- Poor generalization to unseen data
- Badly calibrated probabilities (says 99% when it's really 70% accurate)

## The Solution: Label Smoothing

Instead of hard targets, use soft targets with smoothing factor `ε = 0.1`:

```
smooth_target = (1 - ε) × hard_target + ε / num_classes
```

For 12 classes with ε = 0.1:
- Hard target:  [0,      0,      0,      1,      0,      ...]
- Soft target:  [0.0083, 0.0083, 0.0083, 0.9917, 0.0083, ...]

The correct class gets `1 - 0.1 + 0.1/12 = 0.9917` instead of 1.0.
Wrong classes get `0.1/12 = 0.0083` instead of 0.0.

## Why This Works

- Tells the model "be 99% sure, not 100% sure"
- Prevents overfitting by keeping weights from going to extremes
- Acts as a **regularizer** (similar effect to dropout or weight decay)
- Produces better calibrated probabilities
- Standard value of 0.1 works well for most Kaggle image competitions

## Example Walkthrough

Say the model predicts class probabilities: `[0.01, 0.02, 0.05, 0.85, 0.02, ...]`

- **With hard targets:** Loss punishes the model for not being at 1.0 for class 3. Model keeps pushing, overfits.
- **With soft targets:** Loss is satisfied when the model is around 0.99. Model relaxes, generalizes better.

## Config Field

| Field | Value | Meaning |
|-------|-------|---------|
| `label_smoothing` | `0.1` | 10% of the probability mass is spread across all classes. 90% stays on the correct class. |

## What Happens When You Increase Label Smoothing?

Think of it as a **confidence dial**:

| `label_smoothing` | Correct class target | Wrong class target | Effect |
|---|---|---|---|
| `0.0` | 1.0 | 0.0 | No smoothing. Model tries to be 100% certain. Overfits. |
| `0.1` | 0.9917 | 0.0083 | Sweet spot. Slight uncertainty. Good regularization. |
| `0.3` | 0.725 | 0.025 | Heavy smoothing. Model becomes very unsure. |
| `0.5` | 0.542 | 0.042 | Extreme. Correct class barely above wrong classes. |
| `1.0` | 0.083 | 0.083 | All classes equal. Model learns nothing! |

- **Too low (0.0):** Overfits, overconfident, poor generalization.
- **Just right (0.1):** The model is "almost sure" — enough confidence to classify correctly, enough uncertainty to generalize.
- **Too high (0.3+):** The target for the correct class is so close to the wrong classes that the model gets confused. It can't distinguish signal from noise. Accuracy drops.
- **At 1.0:** Every class has the same target (1/12 = 0.083). The model has literally no signal about which class is correct. Training is useless.

**Rule:** 0.1 is standard for almost all image classification tasks. Rarely go above 0.2.
