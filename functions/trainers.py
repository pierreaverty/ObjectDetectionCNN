from torch.utils.data import DataLoader

class Trainer():
    """
    A class to train a model using a given dataset.

    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (callable): The loss function used for training.
        num_epochs (int): The number of training epochs.
        batch_size (int): The batch size for training.
        train_dataset (torch.utils.data.Dataset): The training dataset.
        device (str, optional): The device to be used for training. Defaults to 'cuda'.

    Attributes:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (callable): The loss function used for training.
        dataset (torch.utils.data.Dataset): The training dataset.
        num_epochs (int): The number of training epochs.
        device (str): The device to be used for training.
        batch_size (int): The batch size for training.

    Methods:
        train(): Trains the model using the provided dataset.
        print_step(epoch, i, loss, bbox_loss, class_loss, train_dataloader): Prints the training step information.
        print_info(train_dataloader): Prints the training information.

    """

    def __init__(self, model, optimizer, criterion, num_epochs, batch_size, train_dataset, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataset = train_dataset
        self.num_epochs = num_epochs
        self.device = device 
        self.batch_size = batch_size   
        
    def train(self):
        """
        Trains the model using the provided dataset.

        Returns:
            torch.nn.Module: The trained model.

        """
        train_dataloader = DataLoader(self.dataset, batch_size=self.batch_size)
        
        self.print_info(train_dataloader)

        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(train_dataloader):
                preds = self.model(images)
            
                loss, bbox_loss, class_loss = self.criterion(preds, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                self.print_step(epoch, i, loss, bbox_loss, class_loss, train_dataloader)
                

        return self.model
    
    def print_step(self, epoch, i, loss, bbox_loss, class_loss, train_dataloader):
        """
        Prints the training step information.

        Args:
            epoch (int): The current epoch.
            i (int): The current step.
            loss (torch.Tensor): The total loss.
            bbox_loss (torch.Tensor): The bounding box loss.
            class_loss (torch.Tensor): The classification loss.
            train_dataloader (torch.utils.data.DataLoader): The training dataloader.

        """
        if (i+1) % len(train_dataloader) == 0:
            print(f"Epoch {epoch+1}, Step {i+1}, Loss: {loss.item()}, Bbox Loss: {bbox_loss.item()}, Class Loss: {class_loss.item()}")
    
    def print_info(self, train_dataloader):
        """
        Prints the training information.

        Args:
            train_dataloader (torch.utils.data.DataLoader): The training dataloader.

        """
        print("\n")
        print(f"Training on {self.device}")
        print(f"Number of epochs: {self.num_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of batches: {len(train_dataloader)}")
        print(f"Number of samples: {len(self.dataset)}")
        print(f"Model: {self.model}")
        print(f"Optimizer: {self.optimizer}")
        print(f"Criterion: {self.criterion}")
        print("\n")