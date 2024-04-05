from data.datasets import CharacterDataset

from torchvision.io import read_image
from PIL import Image

from models.cnn import ObjectDetectionCNN
from functions.losses import BinaryCrossEntropyMeanSquareLoss
from functions.trainers import Trainer

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import torch
import config

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

training_data = CharacterDataset(
    data_dir="datasets/train",
    transform=None, 
    target_transform=None
)

print("Training data loaded, size: ", len(training_data))

validation_data = CharacterDataset(
    data_dir="datasets/val",
    transform=None, 
    target_transform=None
)

print("Validation data loaded, size: ", len(validation_data))

if __name__ == "__main__":
    model = ObjectDetectionCNN()
    
    criterion = BinaryCrossEntropyMeanSquareLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 

    trainer = Trainer(
        model=model, 
        optimizer=optimizer, 
        criterion=criterion, 
        num_epochs=config.NUM_EPOCHS, 
        batch_size=config.BATCH_SIZE, 
        train_dataset=training_data, 
        device=device
    )
    
    model = trainer.train()
    

    
    