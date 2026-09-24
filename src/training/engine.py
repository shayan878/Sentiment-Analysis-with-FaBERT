from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class TrainingHistory:
    train_loss: list
    valid_loss: list
    train_accuracy: list
    valid_accuracy: list

def _accuracy(logits, targets):
    predictions = logits.argmax(dim=1)
    return (predictions == targets).sum().item()

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    losses = []
    correct = 0
    total = 0

    for input_ids, attention_mask, targets in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        correct += _accuracy(outputs, targets)
        total += targets.size(0)

    return float(np.mean(losses)), correct / total

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    losses = []
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    for input_ids, attention_mask, targets in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, targets)

        losses.append(loss.item())
        correct += _accuracy(outputs, targets)
        total += targets.size(0)
        y_true.append(targets.detach().cpu())
        y_pred.append(outputs.argmax(dim=1).detach().cpu())

    return {
        "loss": float(np.mean(losses)),
        "accuracy": correct / total,
        "y_true": torch.cat(y_true).numpy(),
        "y_pred": torch.cat(y_pred).numpy(),
    }

def fit(
    model,
    train_loader,
    valid_loader,
    epochs,
    learning_rate,
    device,
):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    history = TrainingHistory([], [], [], [])

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        valid_metrics = evaluate(model, valid_loader, criterion, device)

        history.train_loss.append(train_loss)
        history.train_accuracy.append(train_acc)
        history.valid_loss.append(valid_metrics["loss"])
        history.valid_accuracy.append(valid_metrics["accuracy"])

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"train_accuracy: {train_acc:.4f}, "
            f"train_loss: {train_loss:.4f}, "
            f"valid_accuracy: {valid_metrics['accuracy']:.4f}, "
            f"valid_loss: {valid_metrics['loss']:.4f}"
        )

    return history
