from functions import *

class BinaryCrossEntropyMeanSquareLoss(nn.Module):
    """
    Custom loss function that combines binary cross entropy loss and mean square loss.
    
    Args:
        alpha (float): The weight given to the binary cross entropy loss. Default is 0.5.
    """
    def __init__(self, alpha=0.5):
        super(BinaryCrossEntropyMeanSquareLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred, y_val):
        """
        Forward pass of the loss function.
        
        Args:
            y_pred (tuple): Tuple containing predicted values for classification and bounding box regression.
            y_val (tuple): Tuple containing ground truth values for classification and bounding box regression.
        
        Returns:
            tuple: A tuple containing the combined loss, bounding box loss, and classification loss.
        """
        class_loss = F.binary_cross_entropy(y_pred[0].squeeze(), y_val[0].float())
        bbox_loss = F.mse_loss(y_pred[1], y_val[1].float())

        return self.alpha * class_loss + (1 - self.alpha) * bbox_loss, bbox_loss, class_loss