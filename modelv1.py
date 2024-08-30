import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import prune

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to a consistent size
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.RandomRotation(10),  # Data augmentation
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize image data
])

# Load the CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Filter the dataset to include only car images (label 1 for 'car')
car_indices = [i for i, (_, label) in enumerate(trainset) if label == 1]
car_dataset = Subset(trainset, car_indices)

# Define a simple neural network with batch normalization
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

# Initialize the model, loss function, and optimizer
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)  # Use Adam optimizer

# Create a DataLoader with multiple workers to speed up data loading
trainloader = DataLoader(car_dataset, batch_size=64, shuffle=True, num_workers=4)

# Train the model
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update the weights
        running_loss += loss.item()

    # Print the average loss for this epoch
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')

print('Finished Training')

# Save the trained model
torch.save(net.state_dict(), 'trained_model2.pth')
print('Model saved to trained_model.pth')
