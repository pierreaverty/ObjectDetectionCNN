from torchvision.io import read_image

from models.cnn import ObjectDetectionCNN

from functions.losses import BinaryCrossEntropyMeanSquareLoss
from functions.trainers import Trainer
from utils.plots import plot_bbox

import torch
import config

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
        train_dataset=config.training_data, 
        device=config.device
    )
    
    model = trainer.train()
    
    model.eval()
    
    test_image_path = '/home/omilab-gpu/ObjectDetectionCNN/datasets/train/images/woman1_front_50.jpg'
    test_image = read_image(test_image_path).float()
    test_image = test_image.unsqueeze(0)  
    
    with torch.no_grad():
        class_pred, bbox_pred = model(test_image)

    image_width, image_height = 640, 480 
    bbox_pred = bbox_pred.squeeze().cpu()  
    bbox_pred = bbox_pred * torch.tensor([image_width, image_height, image_width, image_height])  
    
    plot_bbox(test_image_path, bbox_pred)

    
    