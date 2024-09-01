import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to CIFAR-10 size
    transforms.RandomHorizontalFlip(),  # Data augmentation: random horizontal flip
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR-10 dataset
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

# Initialize the enhanced model, loss function, and optimizer
net = EnhancedNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)  # Increased learning rate

# Train the model
for epoch in range(5):  # Increased number of epochs for better performance
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')

print('Finished Training')
torch.save(net.state_dict(), 'enhanced_trained_model0.pth')
print('Model saved to enhanced_trained_model0.pth')

