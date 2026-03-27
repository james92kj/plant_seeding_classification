# Config Q&A — Test Your Knowledge

## Training Hyperparameters

**Q: What is `seed` and why do we set it to 42?**
<details>
<summary>Answer</summary>
Random seed for reproducibility. Everything (data splits, weight init, augmentations) uses this. The value 42 is arbitrary — what matters is that it's fixed so experiments are reproducible.
</details>

**Q: What is `n_folds` and why 5?**
<details>
<summary>Answer</summary>
Number of cross-validation folds. 5 is standard — gives a good balance between reliable CV scores and training time. More folds = more reliable estimate but more training time.
</details>

**Q: What is `epochs` and why 30?**
<details>
<summary>Answer</summary>
How many full passes through the training data. 30 is enough for fine-tuning pretrained models on small datasets. The model has already learned general features from ImageNet — it just needs to adapt to our specific classes.
</details>

**Q: What is `batch_size` and why 32?**
<details>
<summary>Answer</summary>
Number of images processed per training step. Depends on GPU memory. 32 works for EfficientNet-B3 at 384px on most GPUs. Larger batches give more stable gradients but use more memory.
</details>

**Q: What is `accumulation_steps` and when would you change it from 1?**
<details>
<summary>Answer</summary>
Gradient accumulation. If you want effective batch size of 64 but only 32 fits in memory, set this to 2. It accumulates gradients over 2 steps before updating weights. Effective batch size = batch_size × accumulation_steps.
</details>

**Q: What is `num_workers` and why 4?**
<details>
<summary>Answer</summary>
Number of CPU processes loading data in parallel. Keeps the GPU fed so it's never waiting for data. 4 is a good default for most machines.
</details>

**Q: What is `pin_memory` and why True?**
<details>
<summary>Answer</summary>
Speeds up CPU→GPU data transfer by using pinned (page-locked) memory. The OS guarantees this memory won't be swapped to disk, making the transfer faster. Always set True when training on GPU.
</details>

---

## Optimizer

**Q: Why AdamW over regular Adam?**
<details>
<summary>Answer</summary>
Standard Adam has a flaw: weight decay is coupled with the adaptive learning rate scaling. AdamW decouples them, leading to better generalization. It's the de facto standard for fine-tuning pretrained models in competitions.
</details>

**Q: Why is the learning rate set to 1e-4 for fine-tuning?**
<details>
<summary>Answer</summary>
For fine-tuning pretrained models, 1e-4 is the sweet spot. Too high (1e-2) destroys pretrained features. Too low (1e-6) trains too slowly. The pretrained weights are already good — we just want to gently adjust them.
</details>

**Q: What is `weight_decay` and why 1e-4?**
<details>
<summary>Answer</summary>
Penalizes large weights to prevent overfitting. Think of it as telling the model "keep your weights small unless you really need them big." 1e-4 is a mild regularization that works well for most fine-tuning tasks.
</details>

**Q: What is `max_grad_norm` (gradient clipping) and why 1.0?**
<details>
<summary>Answer</summary>
If gradients explode during training (a bad batch, for example), this caps them at 1.0. Prevents one bad step from ruining the model. It's a safety net — most of the time it doesn't activate, but when it does, it saves your training run.
</details>

---

## Image Settings

**Q: Why is `img_size` set to 384 and not the original EfficientNet-B3 size of 300?**
<details>
<summary>Answer</summary>
Using 384 gives slightly higher resolution than the pretrained input, helping capture fine textures (critical for distinguishing similar grass species). Going higher (512) gives diminishing returns and increases memory/compute. 384 is the sweet spot.
</details>

**Q: Why do we use ImageNet mean and std for normalization instead of computing our own?**
<details>
<summary>Answer</summary>
Since we're using a model pretrained on ImageNet, those weights expect inputs normalized with ImageNet stats. If you use different stats, it's like the model learned to read English but you're feeding it French — the patterns don't match. Rule of thumb: Pretrained model → use that dataset's stats. Training from scratch → compute your own.
</details>

**Q: What does normalization actually do to a pixel value? Walk through an example.**
<details>
<summary>Answer</summary>
A pixel's red value might be 178 (range 0-255). After dividing by 255: 0.698 (range 0-1). After normalization: (0.698 - 0.485) / 0.229 = 0.930. This centers the data around zero with unit variance, which helps neural networks train faster and more stably.
</details>

---

## Scheduler

**Q: Why do we need warmup before cosine scheduling?**
<details>
<summary>Answer</summary>
The classifier head starts with random weights. If you hit it with a high LR immediately, the huge random gradients flow backwards and damage the pretrained backbone features. Warmup says "go slow until the head stabilizes." It linearly ramps from warmup_lr (1e-6) to lr (1e-4) over 2 epochs.
</details>

**Q: What is `T_0` in CosineAnnealingWarmRestarts?**
<details>
<summary>Answer</summary>
Length of the first cosine cycle in epochs. With T_0=10, the LR follows a cosine curve from 1e-4 down to eta_min (1e-7) over 10 epochs, then resets back to 1e-4.
</details>

**Q: What is `T_mult` and what happens if you change it from 1 to 2?**
<details>
<summary>Answer</summary>
Multiplier for subsequent cycle lengths. With T_mult=1, every cycle is 10 epochs. With T_mult=2, cycles would be 10, 20, 40... epochs — increasingly longer cycles that allow finer convergence as training progresses.
</details>

**Q: What is `eta_min` and why not set it to 0?**
<details>
<summary>Answer</summary>
The floor — LR never goes below this value. At LR=0, the model literally stops learning. 1e-7 is effectively zero but still allows tiny weight updates. It's the bottom of each cosine curve before a restart.
</details>

**Q: Describe the full LR schedule across 30 epochs with warmup + cosine restarts.**
<details>
<summary>Answer</summary>
Epochs 0-2: Warmup — linear ramp from 1e-6 to 1e-4. Epochs 3-12: Cosine cycle 1 — smooth curve from 1e-4 down to 1e-7. Epoch 13: Restart — LR jumps back to 1e-4. Epochs 13-22: Cosine cycle 2. Epoch 23: Another restart. And so on. Warmup finishes first, then hands off to cosine. They don't run simultaneously.
</details>

**Q: Why warm restarts? Why not just let the LR decay to zero?**
<details>
<summary>Answer</summary>
After the LR drops low, the model may be stuck in a local minimum. Restarting the LR lets it "jump out" and explore again. Think of a ball rolling down a hilly landscape — the restart gives it another push to escape valleys and potentially find a better solution.
</details>

---

## Model

**Q: What does `tf_efficientnet_b3_ns` mean? Break down each part.**
<details>
<summary>Answer</summary>
tf = TensorFlow-ported weights (slightly better than PyTorch-trained). efficientnet_b3 = EfficientNet at B3 scale (~12M params). ns = Noisy Student pretraining — trained on 300M pseudo-labeled images, transfers exceptionally well to small datasets.
</details>

**Q: Why B3 and not B0 or B7?**
<details>
<summary>Answer</summary>
B0-B2 are too small for 384px input. B4-B7 would overfit on just 4750 images — too many parameters for too little data. B3 (~12M params) is the right capacity for this dataset size.
</details>

**Q: What is `drop_rate` (dropout) and how does it work?**
<details>
<summary>Answer</summary>
During training, randomly sets 30% of neurons to zero. Forces the network to not rely on any single neuron, reducing overfitting. At inference time, all neurons are active but scaled down.
</details>

**Q: What is `drop_path_rate` (stochastic depth) and how is it different from dropout?**
<details>
<summary>Answer</summary>
Randomly skips entire layers during training (20% chance). Dropout skips individual neurons; stochastic depth skips whole layers. Both are regularization techniques but work at different granularities. Stochastic depth is especially effective for deep networks.
</details>

---

## Loss

**Q: What is label smoothing and why use it?**
<details>
<summary>Answer</summary>
Instead of hard targets (0 or 1), we use soft targets. With smoothing=0.1 and 12 classes, the correct class gets 0.9917 instead of 1.0, and wrong classes get 0.0083 instead of 0.0. This prevents overconfidence, reduces overfitting, and produces better calibrated probabilities.
</details>

**Q: What is the formula for label smoothing?**
<details>
<summary>Answer</summary>
smooth_target = (1 - ε) × hard_target + ε / num_classes. With ε=0.1 and 12 classes: correct class = 1 - 0.1 + 0.1/12 = 0.9917. Wrong classes = 0.1/12 = 0.0083.
</details>

**Q: Why does training with hard targets cause overfitting?**
<details>
<summary>Answer</summary>
Hard targets force the model to push logits toward infinity to achieve exactly 1.0 probability. This drives weights to extreme values, causes overconfidence, and hurts generalization. Soft targets say "be 99% sure, not 100% sure" — the model relaxes and generalizes better.
</details>
