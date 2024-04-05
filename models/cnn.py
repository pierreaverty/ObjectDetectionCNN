from models import * 

class ObjectDetectionCNN(nn.Module):
    """
    A convolutional neural network for object detection.

    Args:
        None

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        pool (nn.MaxPool2d): Max pooling layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        dropout (nn.Dropout): Dropout layer.
        fc1 (nn.Linear): First fully connected layer.
        fc_class (nn.Linear): Fully connected layer for class prediction.
        fc_bbox (nn.Linear): Fully connected layer for bounding box prediction.

    Methods:
        forward(x): Forward pass of the network.

    """

    def __init__(self, ):
        super(ObjectDetectionCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=16, 
            kernel_size=3,
            padding=1
        )
                
        self.pool = nn.MaxPool2d(
            kernel_size=2, 
            stride=2, 
            padding=0
        )
        
        self.conv2 = nn.Conv2d(
            in_channels=16, 
            out_channels=32, 
            kernel_size=3,
            padding=1
        )
        
        self.dropout = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(32*120*160, 128) 
        
        self.fc_class = nn.Linear(128, 1) 
        self.fc_bbox = nn.Linear(128, 4) 
        

    def forward(self, x):
        """
        Forward pass of the CNN model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: A tuple containing the class output tensor and the bounding box output tensor.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 120 * 160)
        x = F.relu(self.fc1(x))
        
        class_output = torch.sigmoid(self.fc_class(x)) 
        bbox_output = self.fc_bbox(x)
        
        return class_output, bbox_output