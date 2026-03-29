import numpy as np
from torch.utils.data import DataLoader

from plant_seeding_classification.src.config import CFG
from plant_seeding_classification.src.inference import predict
from plant_seeding_classification.src.submission import create_submission
from plant_seeding_classification.src.train import run_fold
from plant_seeding_classification.src.utils import seed_everything

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
            image_paths.append(os.path.join(cls_path,path))
            labels.append(cls_idx)

    return image_paths, labels, label_to_idx, idx_to_label



def main():

    cfg = CFG()
    seed_everything(cfg.seed)

    # create directories
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.submission_dir, exist_ok=True)
    os.makedirs(cfg.oof_dir, exist_ok=True)

    image_paths, labels, label_to_idx, idx_to_label = prepare_data(cfg)

    skfold = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    np_img_paths, np_labels = np.array(image_paths), np.array(labels)

    for fold, (train_idx, val_idx) in enumerate(skfold.split(np_img_paths, np_labels)):
        train_paths = np_img_paths[train_idx]
        train_labels = np_labels[train_idx]

        valid_paths = np_img_paths[val_idx]
        valid_labels = np_labels[val_idx]

        run_fold(fold, train_paths, train_labels, valid_paths, valid_labels,cfg)

    # Run prediction

    test_paths =[os.path.join(cfg.test_dir, f) for f in os.listdir(cfg.test_dir)]

    from plant_seeding_classification.src.model import PlantModel
    from plant_seeding_classification.src.utils import load_checkpoint, get_device

    device = get_device()
    # we are loading our weights
    model = PlantModel(cfg.model_name,cfg.num_classes,pretrained=False)
    model.to(device)

    filepath = os.path.join(cfg.model_dir, 'fold_0_best.pth')
    load_checkpoint(filepath,model)

    from plant_seeding_classification.src.dataset import PlantDataset
    from plant_seeding_classification.src.transforms import get_valid_transform


    test_transform = get_valid_transform(cfg.img_size, cfg.img_mean, cfg.img_std)
    test_dataset = PlantDataset(test_paths,transform=test_transform, labels= None)
    test_loader = DataLoader(test_dataset,batch_size=cfg.batch_size,shuffle=False)

    predictions = predict(model, test_loader, device)

    # create submission
    output_path = os.path.join(cfg.submission_dir, 'submission.csv')
    create_submission(test_paths, predictions, idx_to_label,output_path)

if __name__ == '__main__':
    main()

