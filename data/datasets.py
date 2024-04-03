from data import *

class CharacterDataset(Dataset):
    """
    A custom dataset class for character data.

    Args:
        data_dir (str): The directory containing the data.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version. Default: None.
        target_transform (callable, optional): A function/transform that takes in the target and returns a transformed version. Default: None.
    """

    def __init__(self, data_dir, transform=None, target_transform=None):
        self.img_labels = data_dir
        self.img_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(os.listdir(self.img_dir+"/images"))

    def __getitem__(self, idx):
        """
        Retrieves the image and label at the given index.

        Args:
            idx (int): The index of the image and label to retrieve.

        Returns:
            tuple: A tuple containing the image and label.
        """
        for i, file in enumerate(os.listdir(self.img_dir+"/images")):
            if (file.endswith(".png") or file.endswith('.jpg')) and i == idx:
                img_path = os.path.join(self.img_dir+"/images", file)
        image = read_image(img_path)

        for i, file in enumerate(os.listdir(self.img_dir+"/labels")):
            if (file.endswith(".txt")) and i == idx:
                label = open(os.path.join(self.img_dir+"/labels", file)).read().strip()

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label