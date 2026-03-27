# Image Normalization — Why and How

## The Problem

Neural networks are sensitive to input scale. If one feature has values 0-255 and another has values 0-1, the network struggles — the large-valued feature dominates gradient updates.

Images have pixel values from 0 to 255. That's a big range. Training is unstable and slow on raw pixel values.

## Step 1: Scale to 0-1

First, divide by 255. Now pixels are in range 0-1. Better, but still not ideal because the distribution is not centered around zero.

## Step 2: Normalize to zero mean, unit variance

This is standard statistics. For any dataset, you can compute:
- **Mean**: the average value (the "center" of the data)
- **Std**: how spread out the values are

Then normalize: `(value - mean) / std`

After this, your data has mean ~0 and std ~1. Neural networks train **much faster and more stably** with this.

## Simple Example

- A pixel's red value might be 178 (range 0-255)
- After dividing by 255: 0.698 (range 0-1)
- After normalization: (0.698 - 0.485) / 0.229 = 0.930

## What "someone computed" means

ImageNet has 1.2 million images. Someone literally:

1. Loaded all 1.2M images
2. For the **Red** channel: averaged all red pixel values across all images → got **0.485**
3. For the **Green** channel: → got **0.456**
4. For the **Blue** channel: → got **0.406**
5. Did the same for standard deviation → got **0.229, 0.224, 0.225**

It's just `np.mean()` and `np.std()` across a massive image collection.

## Why these specific numbers?

The EfficientNet model was pretrained on ImageNet (millions of images). Someone computed the average mean and std across ALL ImageNet images and got:

- **Mean**: `(0.485, 0.456, 0.406)` — one value per RGB channel
- **Std**: `(0.229, 0.224, 0.225)` — one value per RGB channel

Since our model learned with these stats, **we must use the same stats**. Otherwise the model sees inputs it wasn't trained on.

## Why not compute our own stats?

You could! But since we're using a model pretrained on ImageNet, those weights expect inputs normalized with ImageNet stats. If you use different stats, it's like the model learned to read English but you're feeding it French — the patterns don't match.

**Rule of thumb:** Pretrained model → use that dataset's stats. Training from scratch → compute your own.
