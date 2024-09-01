import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import prune

# Define data transformations for test data
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to a consistent size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize image data
])

# Load the CIFAR-10 test dataset
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create a DataLoader for the test dataset
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

# Define the AdvancedNet model class (same as the one used for training)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Add the shortcut connection
        return torch.relu(out)

class AdvancedNet(nn.Module):
    def __init__(self):
        super(AdvancedNet, self).__init__()
        self.layer1 = ResidualBlock(3, 64)
        self.layer2 = ResidualBlock(64, 128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = x.view(-1, 128 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the trained model
model_path = '/Users/aiden/Desktop/pytorch AI/trained_model1.pth'  # Update this path if necessary
if os.path.exists(model_path):
    net = AdvancedNet()
    net.load_state_dict(torch.load(model_path))
    print("Model loaded successfully.")
    net.eval()  # Set the model to evaluation mode

    # Perform inference on the test data
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Track accuracy for each class
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    # Print the overall accuracy of the model
    print(f'Overall Accuracy of the model: {100 * correct / total:.2f}%')

    # Print the accuracy for each class
    for i in range(10):
        print(f'Accuracy of {testset.classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')
else:
    print(f'File not found: {model_path}')