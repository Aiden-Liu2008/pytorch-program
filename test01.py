import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
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

# Filter the test dataset to include only car images (label 1 for 'car')
car_indices = [i for i, (_, label) in enumerate(testset) if label == 1]
car_dataset = Subset(testset, car_indices)

# Create a DataLoader for the test dataset
testloader = DataLoader(car_dataset, batch_size=64, shuffle=False, num_workers=4)

# Define the model class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization
        self.fc1 = nn.Linear(32 * 30 * 30, 128)
        self.fc2 = nn.Linear(128, 2)

        # Apply pruning to the convolutional layer (e.g., 20% pruning)
        prune.random_unstructured(self.conv1, name="weight", amount=0.2)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))  # Apply batch normalization after convolution
        x = x.view(-1, 32 * 30 * 30)  # Flatten the tensor for the fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
model_path = '/Users/aiden/Desktop/pytorch AI/trained_model.pth'  # Update this path for your own image dataset
if os.path.exists(model_path):
    net = Net()
    net.load_state_dict(torch.load(model_path))
    net.eval()  # Set the model to evaluation mode

    # Perform inference on the test data
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print the accuracy of the model
    print(f'Accuracy of the model on the car images: {100 * correct / total:.2f}%')
else:
    print(f'File not found: {model_path}')
