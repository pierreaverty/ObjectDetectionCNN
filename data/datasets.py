import os
import torch
from torchvision.io import read_image

class CharacterDataset(torch.utils.data.Dataset):
    """
    A custom dataset class for character data.

    Args:
        data_dir (str): The directory path where the data is stored.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version. Default is None.
        target_transform (callable, optional): A function/transform that takes in the target and returns a transformed version. Default is None.
    """

    def __init__(self, data_dir, transform=None, target_transform=None):
        self.img_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
            """
            Returns the number of images in the dataset.

            Returns:
                int: The number of images in the dataset.
            """
            return len([name for name in os.listdir(self.img_dir+"/images") if name.endswith(".png") or name.endswith('.jpg')])

    def __getitem__(self, idx):
        """
        Retrieves the image, class label, and bounding box label for the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image and the labels (class label and bbox label).
        Raises:
            FileNotFoundError: If no image file or label file is found for the given index.
        """
        img_path = None
        for i, file in enumerate(os.listdir(self.img_dir+"/images")):
            if (file.endswith(".png") or file.endswith('.jpg')) and i == idx:
                img_path = os.path.join(self.img_dir+"/images", file)
                break  
        if img_path is None:
            raise FileNotFoundError(f"No image file found for index {idx}")
        image = read_image(img_path).float()

        class_label = None
        bbox_label = None
        for i, file in enumerate(os.listdir(self.img_dir+"/labels")):
            if (file.endswith(".txt")) and i == idx:
                with open(os.path.join(self.img_dir+"/labels", file), 'r') as f:
                    class_label, bbox_str = f.read().strip().split(" ", 1)
                class_label = float(class_label)
                bbox_label = torch.tensor(list(map(float, bbox_str.split(" "))), dtype=torch.float)
                break
        if class_label is None or bbox_label is None:
            raise FileNotFoundError(f"No label file found for index {idx}")

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            class_label = self.target_transform(class_label)
            bbox_label = self.target_transform(bbox_label)
            
        labels = (class_label, bbox_label)
        
        return image, labels
