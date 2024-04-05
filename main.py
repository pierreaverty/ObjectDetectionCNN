from data.datasets import CharacterDataset

from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.io import read_image
from PIL import Image
from torch import nn

from models.cnn import ObjectDetectionCNN

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import torch


BATCH_SIZE = 16

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
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
    validatation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE)
