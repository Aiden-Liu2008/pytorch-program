import torch

# Set the number of intra-op threads at the very beginning to prevent threading warnings
torch.set_num_threads(1)  # You can adjust this number as needed

import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

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


# Define the model class to match the saved model's architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.fc1 = nn.Linear(32 * 30 * 30, 128)
        self.fc2 = nn.Linear(128, 10)  # Modify the output layer to match CIFAR-10's 10 classes

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(-1, 32 * 30 * 30)  # Flatten the tensor for the fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load the trained model
model_path = '/Users/aiden/Desktop/pytorch AI/trained_model0.pth'  # Update this path for your own model
if os.path.exists(model_path):
    net = Net()
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


