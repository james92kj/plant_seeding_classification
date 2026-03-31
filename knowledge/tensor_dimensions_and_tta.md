# Tensor Dimensions & Test-Time Augmentation (TTA)

## Understanding Image Tensor Dimensions

A batch of images in PyTorch has shape `(B, C, H, W)`:

```
dim=0    dim=1    dim=2    dim=3
  |        |        |        |
Batch   Channels  Height   Width
(32)      (3)     (384)    (384)
          RGB      rows     cols
```

### Visual Example (one 4x4 image)

```
Original image:
+---------------+
| 1   2   3   4 |  <- row 0
| 5   6   7   8 |  <- row 1
| 9  10  11  12 |  <- row 2
| 13 14  15  16 |  <- row 3
+---------------+
  ^   ^   ^   ^
 c0  c1  c2  c3   (columns)
```

### torch.flip(image, dims=[3]) - Horizontal flip (flip columns)

dim=3 is Width (columns), so we reverse left <-> right

```
+---------------+
| 4   3   2   1 |
| 8   7   6   5 |
| 12 11  10   9 |
| 16 15  14  13 |
+---------------+
```

### torch.flip(image, dims=[2]) - Vertical flip (flip rows)

dim=2 is Height (rows), so we reverse top <-> bottom

```
+---------------+
| 13 14  15  16 |
| 9  10  11  12 |
| 5   6   7   8 |
| 1   2   3   4 |
+---------------+
```

### torch.flip(image, dims=[2, 3]) - Both flips

Flip rows AND columns (like rotating 180 degrees)

```
+---------------+
| 16 15  14  13 |
| 12 11  10   9 |
| 8   7   6   5 |
| 4   3   2   1 |
+---------------+
```

### Quick Reference Table

| What you want   | Code                              | Why                       |
|-----------------|-----------------------------------|---------------------------|
| Horizontal flip | `torch.flip(images, dims=[3])`    | dim 3 = Width = columns   |
| Vertical flip   | `torch.flip(images, dims=[2])`    | dim 2 = Height = rows     |
| Both flips      | `torch.flip(images, dims=[2, 3])` | flip both axes            |

---

## Test-Time Augmentation (TTA)

### What is TTA?

Instead of predicting each test image once, predict it multiple times with different
augmentations (flips, rotations), then average the results. This is like getting a
"second opinion" from the same model by showing it different views of the same image.

### Why does this help?

A seedling photo could be taken from any angle. By showing the model 4 views and
averaging, we cancel out orientation bias:

```
Original    -> model says: 80% Charlock, 15% Maize
H-Flip      -> model says: 85% Charlock, 10% Maize
V-Flip      -> model says: 70% Charlock, 20% Maize
Both-Flip   -> model says: 82% Charlock, 12% Maize
-----------------------------------------------------
Average     -> 79% Charlock, 14% Maize  <- more confident & stable!
```

### How TTA works in code

For each batch of test images:

1. Run original images through the model -> logits_1
2. Flip horizontally, run through model  -> logits_2
3. Flip vertically, run through model    -> logits_3
4. Flip both ways, run through model     -> logits_4
5. Average all 4 sets of logits
6. This averaged result is more robust than any single prediction

### Combining TTA with Fold Ensembling

With 5 folds x 4 TTA views = 20 predictions per test image, all averaged together.
This is a very powerful combination for competition submissions.

```
Fold 0: [original, h_flip, v_flip, both_flip] -> avg_logits_fold0
Fold 1: [original, h_flip, v_flip, both_flip] -> avg_logits_fold1
Fold 2: [original, h_flip, v_flip, both_flip] -> avg_logits_fold2
Fold 3: [original, h_flip, v_flip, both_flip] -> avg_logits_fold3
Fold 4: [original, h_flip, v_flip, both_flip] -> avg_logits_fold4

Final = average(avg_logits_fold0 ... avg_logits_fold4)
Prediction = argmax(Final)
```

### Key PyTorch operations for TTA

- `torch.flip(tensor, dims)` - flip tensor along specified dimensions
- `torch.stack(list_of_tensors)` - combine list into tensor with new dim at front
- `.mean(dim=0)` - average across the first dimension (the stacked TTA views)

### Expected accuracy boost

- Fold ensemble alone: ~1-2% improvement
- TTA alone: ~0.5-1% improvement
- Both combined: ~1.5-3% improvement over single fold, single view
