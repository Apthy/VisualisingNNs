import torch.nn.functional as F
from torch import nn


class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 40, (5, 5), padding=2)
        self.conv2 = nn.Conv2d(40, 40, (5, 5), padding=2)
        self.conv3 = nn.Conv2d(40, 40, (3, 3), padding=1)
        self.fc1 = nn.Linear(40 * 14 ** 2, 128)
        self.fc2 = nn.Linear(128, 50)
        self.fc3 = nn.Linear(50, 10)

    # Convolutional layer with 40 feature maps of size 5×5 and ReLU activation.
    # Pooling layer taking the max over 2×2 patches.
    # Convolutional layer with 20 feature maps of size 3×3 and ReLU activation.
    # Pooling layer taking the max over 2×2 patches.
    # Dropout layer with a probability of 20%.
    # Flatten layer.
    # Fully connected layer with 128 neurons and ReLU activation.
    # Fully connected layer with 50 neurons and ReLU activation.
    # Linear output layer.
    def forward(self, x):
        #print(x.shape)
        out = self.conv1(x)  # Convolutional layer with 40 feature maps of size 5×5 and ReLU activation.
        out = F.relu(out)
        #print(x.shape)
        #out = self.conv2(out)  # Convolutional layer with 40 feature maps of size 5×5 and ReLU activation.
        #out = F.relu(out)

        #print(out.shape)
        #out = F.max_pool2d(out, (2, 2))  # Pooling layer taking the max over 2×2 patches.
        out = self.conv3(out)  # Convolutional layer with 20 feature maps of size 3×3 and ReLU activation.
        out = F.relu(out)
        #print(out.shape)
        out = F.max_pool2d(out, (2, 2))  # Pooling layer taking the max over 2×2 patches.
        #print(out.shape)
        out = F.dropout(out, 0.2)  # Dropout layer with a probability of 20%.
        out = out.view(out.shape[0], -1)  # Flatten layer.
        out = self.fc1(out)  # Fully connected layer with 128 neurons and ReLU activation.
        out = F.relu(out)
        out = self.fc2(out)  # Fully connected layer with 50 neurons and ReLU activation.
        out = F.relu(out)
        out = self.fc3(out)  # Fully connected layer with 50 neurons and ReLU activation.
        return out

            # build the model and load state