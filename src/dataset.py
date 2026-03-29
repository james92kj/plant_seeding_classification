# How to load images in Pytorch

from typing import List, Tuple, Optional, Dict

from torch.utils.data import Dataset
import cv2
import torch


class PlantDataset(Dataset):

    def __init__(self,image_paths : List[str], labels: Optional[List[int]] = None, transform=None):
        """
        :param image_paths: List of full paths to images.
        :param labels: List of labels.
        """

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Read the image
        img = cv2.imread(self.image_paths[idx])

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform: # Albumentations return image and mask {'image':...., 'mask':....}
            img = self.transform(image=img)['image']

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return img, label

        return img