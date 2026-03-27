# Transfer Learning & Model Architecture

## What is Transfer Learning?
Instead of training a model from scratch (needs millions of images), we take a model already trained on ImageNet (1.2M images, 1000 classes) and replace the last layer to output our number of classes (12 seedling types).

The backbone already knows how to detect edges, textures, and shapes. We just teach the new head "which combination of features = which seedling."

```
EfficientNet-B3 (pretrained on ImageNet)
├── Backbone (feature extractor) ← KEEP these learned weights
│   ├── Conv layers that detect edges
│   ├── Conv layers that detect textures
│   └── Conv layers that detect shapes
└── Classifier head: 1000 classes ← REPLACE with 12 classes
```

## PlantModel Architecture

### Class Structure
- Inherits from `nn.Module` (base class for all PyTorch models)
- `__init__` accepts: `model_name`, `num_classes`, `pretrained`

### Inside `__init__`:
1. `super().__init__()` — required to initialize the parent nn.Module
2. `self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)` — loads the pretrained model
3. `self.n_features = self.backbone.num_features` — gets the output feature dimension from the backbone
4. `self.head = nn.Linear(self.n_features, num_classes)` — our custom classification layer

### Key Detail: `num_classes=0`
When you pass `num_classes=0` to `timm.create_model`, it tells timm to **remove the original classification head** and return only the feature extractor. Then we attach our own `nn.Linear` head with our number of classes.

### The `forward` Method:
1. `features = self.backbone(x)` — extract features from image (e.g., 1536-dim vector for EfficientNet-B3)
2. `output = self.head(features)` — classify features into 12 classes
3. Return `output`

### Why Separate Backbone and Head?
- **Freeze backbone**: Train only the head (useful for small datasets)
- **Different learning rates**: Lower LR for backbone (don't destroy pretrained features), higher LR for head (learn fast from scratch)
- **Swap the head**: Replace with a more complex one (e.g., Dropout + Linear for regularization)

## Why `timm`?
- Provides hundreds of pretrained models with one line of code
- Consistent API across all architectures (EfficientNet, ResNet, ViT, etc.)
- Easy to swap models — just change `model_name` in config

## Q&A

**Q: Why not train from scratch?**
A: Our dataset has ~4,750 images. Models have millions of parameters. Training from scratch would massively overfit. Pretrained weights give us a huge head start.

**Q: What does `nn.Linear(n_features, num_classes)` do?**
A: It's a single fully-connected layer. Takes a feature vector of size `n_features` (e.g., 1536 for EfficientNet-B3) and outputs `num_classes` scores (12). The highest score = the predicted class.

**Q: Why inherit from `nn.Module`?**
A: It gives your class PyTorch superpowers — automatic parameter tracking, GPU transfer with `.to(device)`, `model.train()`/`model.eval()` modes, and `state_dict()` for saving/loading.

**Q: What happens during `self.backbone(x)`?**
A: The image tensor (batch_size, 3, 384, 384) passes through all convolutional layers and gets reduced to a feature vector (batch_size, 1536). Think of it as the model "describing" each image as 1536 numbers.
