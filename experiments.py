import os

import torch
import wandb
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from model.custom_cnn import CustomCNN
from src.dataset import Dataset
from src.training import Training
import src.config as config
def experiments():

    #wandb.login(key=os.environ.get("WANDB_API_KEY"))

    # Initialize wandb with your project name and configuration settings
    #wandb.init(project="food-project-mac", config={
    #    "epochs": config.EPOCHS,
    #    "learning_rate": config.LEARNING_RATE,  # update with your learning rate
        # add any other hyperparameters you want to track
    #})


    # Setup device-agnostic code
    if torch.cuda.is_available():
        device = "cuda" # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = "mps" # Apple GPU
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Pre-trained model
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, config.NUM_CLASSES)
    model.to(device)

    #train classifer and freeze other layers
    for p in model.features.parameters():
        p.requires_grad = False

    # Custom model
    #model = CustomCNN()

    # init dataset
    dataset = Dataset()
    train_loader, validation_loader, test_loader = dataset.data_loading()

    # optimizers
    training = Training()
    loss_fn, optimizer, scheduler = training.optimizers(custom_model=model, train_loader=train_loader)

    print(model.classifier)
    print("Training head only, backbone frozen")
    for epoch in range(config.EPOCHS):
        train_loss, train_accuracy = training.train_loop(train_loader=train_loader, model=model, loss_fn=loss_fn,
                                                         optimizer=optimizer, scheduler=scheduler, device=device)

        valid_loss, valid_accuracy = training.test_loop(validation_loader=validation_loader, model=model,
                                                        loss_fn=loss_fn, device=device, description="Validation")

        # Log metrics to wandb
        #wandb.log({
        #    "epoch": epoch + 1,
        #    "train_loss": train_loss,
        #    "train_accuracy": train_accuracy,
        #    "valid_loss": valid_loss,
        #    "valid_accuracy": valid_accuracy
        #})

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% | "
              f"Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.2f}%")


        if epoch + 1 == config.UNFREEZE_EPOCHS:
            print("Unfreezing last backbone blocks at epoch {}".format(epoch+1))
            for name, p in model.features.named_parameters():
                if "stage4" in name or "stage5" in name:
                    p.requires_grad = True

                    optimizer = torch.optim.SGD(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=config.FINE_TUNE_LR,
                        momentum=0.9,
                        weight_decay=config.FINE_TUNE_LR
                    )

                    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer,
                        max_lr=config.FINE_TUNE_LR,
                        steps_per_epoch=len(train_loader),
                        epochs=config.EPOCHS - epoch - 1
                    )


    # Testing the Model
    test_loss, test_accuracy = training.test_loop(validation_loader=test_loader, model=model, loss_fn=loss_fn, device=device, description="Test")

    #wandb.log({
    #    "test_loss": test_loss,
    #    "test_accuracy": test_accuracy
    #})

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    scripted = torch.jit.script(model.cpu())
    scripted.save("model.pt")

    # Save the model after training and testing
    torch.save(model.state_dict(), "fine_tuned_model.pth")
    print("Model saved as custom_model.pth")

if __name__ == "__main__":
    experiments()






