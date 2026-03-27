import torch
import torch.nn as nn
import timm

class PlantModel(nn.Module):

    def __init__(self,  model_name: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0) # Removes original classification head
        self.n_features = self.backbone.num_features

        self.head = nn.Linear(in_features=self.n_features, out_features=num_classes)


    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits



