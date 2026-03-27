# Learning Rate Scheduler & Warmup — How They Work Together

## Two Phases of Learning Rate Control

Your training has **two phases** back-to-back:

---

## Phase 1: Warmup (Epochs 0-2) — Linear Ramp

Simple straight line going up:

```
Epoch 0: 1e-6
Epoch 1: 5e-5
Epoch 2: 1e-4
```

This is just basic math — linearly increasing from `warmup_lr` to `lr`. No cosine involved yet.

**Why warmup?** The classifier head starts with random weights. If you hit it with a high LR immediately, the huge random gradients flow backwards and **damage the pretrained backbone features**. Warmup says "go slow until the head stabilizes."

---

## Phase 2: Cosine Annealing with Warm Restarts (Epoch 3+)

Once warmup is done, the **cosine scheduler** takes control. It follows a cosine (half-wave) curve:

```
LR
1e-4 |*                              *  (restart)
     | **                          **
     |   **                      **
     |     ***                ***
     |        ****        ****
1e-7 |            ********
     +----------------------------------------→ Epochs
      3  4  5  6  7  8  9  10 11 12 13 (restart!)
```

It's called "cosine" because the shape is literally the `cos()` math function mapped to LR values.

- **Top** of the curve = `lr` (1e-4) — where each cycle starts
- **Bottom** of the curve = `eta_min` (1e-7) — the lowest the LR goes before restarting
- Why not go all the way to 0? Because at LR=0, the model literally stops learning. 1e-7 is effectively zero but still allows tiny weight updates.

---

## The Full Picture Across 30 Epochs

```
Phase:  |--WARMUP--|--------COSINE CYCLE 1--------|--------COSINE CYCLE 2--------|...
Epoch:   0   1   2   3  4  5  6  7  8  9  10  11  12  13  14 ... 22  23 ...
LR:     1e-6→→→1e-4  ↘↘↘↘↘↘↘↘↘1e-7  RESET→1e-4  ↘↘↘↘↘↘↘↘↘1e-7  RESET...
```

- Warmup handles epochs 0-2 (linear ramp up)
- Cosine handles epoch 3 onwards (smooth curve down, restart every T_0=10 epochs)
- They **don't run simultaneously** — warmup finishes, then hands off to cosine

**Mental picture:** Imagine a ball rolling down a hilly landscape (loss surface). The LR is how hard you push it. Warmup = gentle start. Cosine = gradually slow down. Restart = give it another push to escape valleys.

---

## Scheduler Config Fields Explained

| Field | Value | Meaning |
|-------|-------|---------|
| `scheduler` | `"CosineAnnealingWarmRestarts"` | Cosine curve with periodic restarts |
| `T_0` | `10` | Length of first cosine cycle in epochs. After 10 epochs, the LR resets back |
| `T_mult` | `1` | Multiplier for subsequent cycle lengths. 1 = every cycle is 10 epochs. If 2, cycles would be 10, 20, 40... |
| `eta_min` | `1e-7` | The floor — LR never goes below this value |
| `warmup_epochs` | `2` | First 2 epochs use a low LR that gradually ramps up |
| `warmup_lr` | `1e-6` | Starting LR during warmup. Ramps from 1e-6 up to `lr` (1e-4) over 2 epochs |

---

## Why Warm Restarts?

After the LR drops low, the model may be stuck in a local minimum. Restarting the LR lets it "jump out" and explore again. This works synergistically with SWA (Stochastic Weight Averaging), which we'll add later.
