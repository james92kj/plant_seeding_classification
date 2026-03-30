import torch

def predict(model, loader, device):
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for images,_ in loader:
            images = images.to(device)
            predictions = model(images)
            _, predictions = torch.max(predictions, 1)
            all_predictions.extend(predictions.cpu().numpy())

    return all_predictions


