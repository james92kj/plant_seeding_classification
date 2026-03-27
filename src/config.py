from dataclasses import dataclass, field
from typing import List, Tuple
import os


@dataclass
class CFG:
    """ Master configuration for the entire pipeline"""
    project_dir:str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = ""
    train_dir: str = ""
    test_dir: str = ""
    output_dir: str = ""
    model_dir: str = ""
    oof_dir: str = ""
    submission_dir: str = ""
    log_dir: str = ""
    classes : List[str] = field(default_factory=lambda: ["Black-grass",
                                                         "Charlock",
                                                         "Cleavers",
                                                         "Common Chickweed",
                                                         "Common wheat",
                                                         "Fat Hen",
                                                         "Loose Silky-bent",
                                                         "Maize",
                                                         "Scentless Mayweed",
                                                         "Shepherds Purse",
                                                         "Small-flowered Cranesbill",
                                                         "Sugar beet"])
    num_classes: int = 12
    img_size: int = 384
    in_channels: int = 3
    img_mean: Tuple[int] = (0.485, 0.456, 0.406)
    img_std: Tuple[int] = (0.229, 0.224, 0.225)

    seed: int = 42
    n_folds: int = 5
    epochs: int = 30
    batch_size: int = 32
    accumulation_steps: int = 1
    num_workers: int = 4
    pin_memory: bool = True

    optimizer: str = "AdamW"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    scheduler: str = "CosineAnnealingWarmRestarts"
    T_0: int = 10
    T_mult: int = 1
    eta_min: float = 1e-7
    warmup_epochs: int = 2
    warmup_lr: float = 1e-6

    model_name: str = "tf_efficientnet_b3_ns"
    pretrained: bool = True
    drop_rate: float = 0.3 # skips neurons
    drop_path_rate: float = 0.2 # skips layers

    label_smoothing: float = 0.1


    def __post_init__(self):
        self.data_dir = os.path.join(self.project_dir, "data")
        self.train_dir = os.path.join(self.data_dir, "train")
        self.test_dir = os.path.join(self.data_dir, "test")
        self.output_dir = os.path.join(self.project_dir, "output")
        self.model_dir = os.path.join(self.output_dir, "model")
        self.oof_dir = os.path.join(self.output_dir, "oof")
        self.submission_dir = os.path.join(self.output_dir, "submission")
        self.log_dir = os.path.join(self.output_dir, "log")

