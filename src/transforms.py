from albumentations.pytorch import ToTensorV2
import albumentations as A
from typing import Tuple

def get_train_transform(img_size : int, img_mean: Tuple, img_std: Tuple) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(mean=img_mean, std=img_std),
            ToTensorV2()
        ]
    )


def get_valid_transform(img_size : int, img_mean: Tuple, img_std: Tuple) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=img_mean, std=img_std),
            ToTensorV2()
        ]
    )

