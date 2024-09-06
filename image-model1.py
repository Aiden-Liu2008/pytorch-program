import cv2
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
import torch.nn as nn

# Define the ResidualBlock used in the model
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

# Define the model architecture used during training
class AdvancedNet(nn.Module):
    def __init__(self):
        super(AdvancedNet, self).__init__()
        self.layer1 = ResidualBlock(3, 64)
        self.layer2 = ResidualBlock(64, 128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Adjust the input size for fc1 to match the saved model's size
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # Matches the size used during training
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)  # Matches the size used during training

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = x.view(-1, 128 * 8 * 8)  # This size matches the trained model
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize the model and load the trained weights
net = AdvancedNet()
net.load_state_dict(torch.load('trained_model1.pth'))
net.eval()  # Set the model to evaluation mode

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 class names
class_names = datasets.CIFAR10(root='./data', train=True, download=True).classes

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to PIL Image for transformation
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    # Apply transformations
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        outputs = net(image)
        _, predicted = torch.max(outputs, 1)
        probability = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()

    # Display the results
    label = class_names[predicted.item()]
    cv2.putText(frame, f'Label: {label}, Probability: {probability:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Real-Time Object Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
