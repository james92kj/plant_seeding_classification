from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
import torch
import os

from plant_seeding_classification.src.dataset import PlantDataset
from plant_seeding_classification.src.model import PlantModel
from plant_seeding_classification.src.transforms import get_train_transform, get_valid_transform
from plant_seeding_classification.src.utils import get_device, save_checkpoint, setup_logging


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, scheduler = None):

    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # output - (batch_size, 12). For each image, we predict the raw scores for each class

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if scheduler is not None:
            scheduler.step(epoch + batch_idx / len(loader))

        _, predictions = torch.max(outputs, dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    avg_loss = total_loss / len(loader) # oss per batch
    accuracy = correct / total

    return avg_loss, accuracy

def validate_one_epoch(model, loader, criterion, device):

    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total += labels.size(0)
            _, predictions = torch.max(outputs, dim=1)
            correct += (predictions == labels).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / len(loader)

    return avg_loss, accuracy

def run_fold(fold, train_paths, train_labels, val_paths, val_labels, cfg):

    os.makedirs(cfg.model_dir, exist_ok=True)
    device = get_device()
    logger = setup_logging(cfg.log_dir, f"{cfg.name}_fold{fold}")

    logger.info(f"{'=' * 50}")
    logger.info(f"FOLD {fold} — Training started")
    logger.info(f"Train: {len(train_paths)} images | Val: {len(val_paths)} images")

    train_transform = get_train_transform(cfg.img_size, cfg.img_mean, cfg.img_std)
    val_transform = get_valid_transform(cfg.img_size, cfg.img_mean, cfg.img_std)

    train_dataset = PlantDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = PlantDataset(val_paths, val_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    model = PlantModel(cfg.model_name, cfg.num_classes, cfg.pretrained)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=cfg.T_0, T_mult=cfg.T_mult, eta_min=cfg.eta_min)

    best_score = 0
    for epoch in range(cfg.epochs):
        train_avg_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device , epoch, scheduler)
        valid_avg_loss, valid_accuracy = validate_one_epoch(model, val_loader, criterion, device)

        if valid_accuracy > best_score:
            best_score = valid_accuracy

            filepath = os.path.join(cfg.model_dir, f'fold_{fold}_best.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, valid_accuracy, fold, cfg, filepath)

        logger.info(f"Epoch {epoch} | Train loss {train_avg_loss:.4f} | valid loss {valid_avg_loss:.4f} | Valid accuracy {valid_accuracy:.4f}")

    logger.info(f"FOLD {fold} — Best validation accuracy: {best_score:.4f}")







