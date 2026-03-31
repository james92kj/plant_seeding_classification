import numpy as np
from torch.utils.data import DataLoader

from src.config import CFG
from src.inference import predict_tta
from src.submission import create_submission
from src.train import run_fold
from src.utils import seed_everything

from src.dataset import PlantDataset
from src.transforms import get_valid_transform

from src.model import PlantModel
from src.utils import load_checkpoint, get_device

from sklearn.model_selection import StratifiedKFold
import os


def prepare_data(cfg):
    image_paths, labels = [], []

    label_to_idx = {name: idx for idx, name in enumerate(cfg.classes)}
    idx_to_label = {idx: name for idx, name in enumerate(cfg.classes)}

    for cls_idx, cls_name in enumerate(cfg.classes):
        cls_path = os.path.join(cfg.train_dir, cls_name)
        paths = os.listdir(cls_path)
        for path in paths:
            image_paths.append(os.path.join(cls_path, path))
            labels.append(cls_idx)

    return image_paths, labels, label_to_idx, idx_to_label


def main():
    cfg = CFG()
    seed_everything(cfg.seed)
    skip_training = True

    # create directories
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.submission_dir, exist_ok=True)
    os.makedirs(cfg.oof_dir, exist_ok=True)

    image_paths, labels, label_to_idx, idx_to_label = prepare_data(cfg)

    skfold = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    np_img_paths, np_labels = np.array(image_paths), np.array(labels)

    if not skip_training:

        for fold, (train_idx, val_idx) in enumerate(skfold.split(np_img_paths, np_labels)):
            train_paths = np_img_paths[train_idx]
            train_labels = np_labels[train_idx]

            valid_paths = np_img_paths[val_idx]
            valid_labels = np_labels[val_idx]

            run_fold(fold, train_paths.tolist(), train_labels.tolist(), valid_paths.tolist(), valid_labels.tolist(), cfg)

    # Run prediction

    test_paths = [os.path.join(cfg.test_dir, f) for f in os.listdir(cfg.test_dir) if not f.startswith(".")]

    device = get_device()
    # we are loading our weights

    predictions = []

    test_transform = get_valid_transform(cfg.img_size, cfg.img_mean, cfg.img_std)
    test_dataset = PlantDataset(test_paths, transform=test_transform, labels=None)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    for i in range(5):
        model = PlantModel(cfg.model_name, cfg.num_classes, pretrained=False)
        model.to(device)

        filepath = os.path.join(cfg.model_dir, f'fold_{i}_best.pth')
        load_checkpoint(filepath, model, device=device)
        predictions.append(predict_tta(model, test_loader, device))

    predictions = np.array(predictions)
    avg_predictions = np.average(predictions, axis=0)
    final_predictions = np.argmax(avg_predictions, axis=1)
    # create submission
    output_path = os.path.join(cfg.submission_dir, 'submission.csv')
    create_submission(test_paths, final_predictions, idx_to_label, output_path)



if __name__ == '__main__':
    main()
