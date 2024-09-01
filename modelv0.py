import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to CIFAR-10 size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Define a simple neural network for CIFAR-10 classification
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.fc1 = nn.Linear(32 * 30 * 30, 128)
        self.fc2 = nn.Linear(128, 10)  # Updated to 10 output classes

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(-1, 32 * 30 * 30)  # Flatten the tensor for the fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train the model
for epoch in range(5):  # You may want to increase the number of epochs for better performance
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
torch.save(net.state_dict(), 'trained_model0.pth')
print('Model saved to trained_model0.pth')