import torch
from torch import nn
from tqdm import tqdm
from src import config


class Training:

    # TODO: Scheduler never used, we need to use it
    def __init__(self):
        pass

    def optimizers(self, custom_model, train_loader):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(params = custom_model.parameters(), lr = 0.01)
        # learning rate scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(train_loader), epochs=config.EPOCHS)

        return loss_fn, optimizer, scheduler

    def train_loop(train_loader, model, loss_fn, optimizer, device):

        model.train()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        progress_bar = tqdm(train_loader, total=len(train_loader), desc="Training", unit="batch")

        for batch, (X, y) in enumerate(progress_bar):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            # compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # backpropagation
            loss.backward()
            optimizer.step()

            # accumulate training loss and accuracy
            total_loss += loss.item() * X.size(0)
            predicted = torch.argmax(pred, dim=1)
            total_correct += (predicted == y).sum().item()
            total_examples += X.size(0)

        average_loss = total_loss / total_examples
        accuracy = total_correct / total_examples * 100
        return average_loss, accuracy


    def test_loop(validation_loader, model, loss_fn, description, device):

        model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(validation_loader, total=len(validation_loader), desc=description, unit="batch")
        with torch.no_grad():
            for X, y in progress_bar:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = loss_fn(pred, y)
                total_loss += loss.item() * X.size(0)
                predicted = torch.argmax(pred, dim=1)
                correct += (predicted == y).sum().item()
                total += X.size(0)
        average_loss = total_loss / total
        accuracy = correct / total * 100
        return average_loss, accuracy