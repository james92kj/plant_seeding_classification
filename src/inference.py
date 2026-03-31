import torch


def predict(model, loader, device):
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for images in loader:
            images = images.to(device)              # (B, 3, 384, 384)
            predictions = model(images)             # (B, 12) raw logits
            all_predictions.extend(predictions.cpu().numpy())

    return all_predictions                          # list of (12,) arrays, total length = num_test_images


# Dimensions key:
#   B = batch size (e.g. 32)
#   3 = RGB channels
#   384 = image height & width
#   12 = number of plant species (classes)
#   4 = number of TTA views (original + 3 flips)

def predict_tta(model, loader, device):
    all_predictions = []
    model.eval()
    with torch.no_grad():
        for images in loader:
            images = images.to(device)              # (B, 3, 384, 384)

            tta_logits = []                         # will hold 4 tensors, each (B, 12)

            # collect the logits of raw image
            tta_logits.append(model(images))        # (B, 12)

            # do a horizontal flip (flip columns / width)
            flip_h = torch.flip(images, dims=[3])   # (B, 3, 384, 384) columns reversed
            tta_logits.append(model(flip_h))         # (B, 12)

            # do a vertical flip (flip rows / height)
            flip_v = torch.flip(images, dims=[2])    # (B, 3, 384, 384) rows reversed
            tta_logits.append(model(flip_v))          # (B, 12)

            # do a complete flip (rows & columns)
            flip = torch.flip(images, dims=[2,3])    # (B, 3, 384, 384) rows & cols reversed
            tta_logits.append(model(flip))            # (B, 12)

            # tta_logits = list of 4 tensors, each (B, 12)
            # torch.stack  -> (4, B, 12)   stack 4 views into one tensor
            # .mean(dim=0) -> (B, 12)      average across the 4 TTA views
            avg_logits = torch.stack(tta_logits).mean(dim=0)  # (B, 12)
            all_predictions.append(avg_logits.cpu().numpy())  # append (B, 12) numpy array

    return all_predictions  # list of (B, 12) arrays, one per batch
                            # when converted to np.array in main.py -> (num_test_images, 12)




