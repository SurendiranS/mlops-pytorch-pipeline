import json
import os

import torch
import torch.nn as nn
import yaml

from dataset import get_dataloaders
from model import FashionMNISTCNN


def log_json(event, **kwargs):
    record = {"event": event, **kwargs}
    print(json.dumps(record), flush=True)


def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def train(config):
    # Configurations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_json("training_start", device=str(device))
    data_path = config["data"]["path"]
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    num_workers = config["training"]["num_workers"]
    patience = config["early_stopping"]["patience"]
    checkpoint_path = config["output"]["checkpoint_path"]

    # Data
    train_loader, test_loader = get_dataloaders(
        data_dir=data_path, batch_size=batch_size, num_workers=num_workers
    )

    # Model
    model = FashionMNISTCNN(num_classes=10)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_loss = float("inf")
    epochs_without_improvement = 0
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        # Metrics
        train_loss = running_loss / total
        train_accuracy = correct / total
        val_loss, val_accuracy = evaluate(model, test_loader, criterion, device)
        log_json(
            "epoch",
            epoch=epoch,
            train_loss=round(train_loss, 4),
            train_accuracy=round(train_accuracy, 4),
            val_loss=round(val_loss, 4),
            val_accuracy=round(val_accuracy, 4),
        )

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": 10,
                },
                checkpoint_path,
            )
            log_json(
                "checkpoint_saved", path=checkpoint_path, val_loss=round(val_loss, 4)
            )
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            log_json("early_stopping", epoch=epoch, patience=patience)
            break

    log_json("training_complete", checkpoint=checkpoint_path)


if __name__ == "__main__":
    with open("configs/training_config.yaml", "r") as file:
        config = yaml.safe_load(file)
    train(config)
