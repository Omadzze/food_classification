import torch
from model.custom_cnn import CustomCNN
from src.dataset import Dataset
def experiments():

    # Setup device-agnostic code
    if torch.cuda.is_available():
        device = "cuda" # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = "mps" # Apple GPU
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Our model
    custom_cnn = CustomCNN()

    train_loader, validation_loader, test_loader = Dataset.data_loading()

    loss_fn,







