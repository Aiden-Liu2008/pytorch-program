import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader

# Set the number of intra-op threads at the very beginning to prevent threading warnings
torch.set_num_threads(1)  # You can adjust this number as needed

# Define data transformations for training data
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to CIFAR-10 size
    transforms.RandomHorizontalFlip(),  # Data augmentation: random horizontal flip
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR-10 training dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Define an enhanced neural network for CIFAR-10 classification
class EnhancedNet(nn.Module):
    def __init__(self):
        super(EnhancedNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)  # Increased channels from 32 to 64
        self.bn1 = nn.BatchNorm2d(64)  # Batch normalization
        self.conv2 = nn.Conv2d(64, 128, 3)  # Additional convolutional layer
        self.bn2 = nn.BatchNorm2d(128)  # Batch normalization for second conv layer
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
        self.fc1 = nn.Linear(128 * 6 * 6, 256)  # Adjusted input size for fully connected layer
        self.dropout = nn.Dropout(0.3)  # Reduced dropout rate
        self.fc2 = nn.Linear(256, 10)  # 10 output classes for CIFAR-10

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))  # Apply batch normalization after conv1
        x = self.pool(x)  # Apply pooling after first conv layer
        x = torch.relu(self.bn2(self.conv2(x)))  # Apply batch normalization after conv2
        x = self.pool(x)  # Apply pooling after second conv layer
        x = x.view(-1, 128 * 6 * 6)  # Flatten the tensor for the fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after fc1
        x = self.fc2(x)
        return x

# Define data transformations for test data
transform_test = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to a consistent size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize image data
])

# Load the CIFAR-10 test dataset
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

# Load the trained model
model_path = 'enhanced_trained_model0.pth'
if os.path.exists(model_path):
    net = EnhancedNet()
    net.load_state_dict(torch.load(model_path))
    net.eval()  # Set the model to evaluation mode

    # Perform inference on the test data
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # Print the overall accuracy of the model
    print(f'Overall accuracy of the model on the CIFAR-10 test set: {100 * correct / total:.2f}%')

    # Print accuracy for each class
    for i in range(10):
        print(f'Accuracy of {testset.classes[i]} : {100 * class_correct[i] / class_total[i]:.2f}%')
else:
    print(f'File not found: {model_path}')

