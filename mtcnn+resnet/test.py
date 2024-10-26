import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Define transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

# Load the training dataset (or validation dataset if you have one)
test_data = datasets.ImageFolder(root='eyes/', transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

# Load the fine-tuned ResNet-18 model
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Two output classes: open/closed

# Load the trained model weights
model.load_state_dict(torch.load('eye_classifier_resnet18.pth'))
model.eval()  # Set the model to evaluation mode
model = model.to('cuda')  # If using GPU

# Define the loss function (same as used during training)
criterion = nn.CrossEntropyLoss()

# Test the model
correct = 0
total = 0
running_loss = 0.0

with torch.no_grad():  # No need to calculate gradients during evaluation
    for inputs, labels in test_loader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        
        # Get the predicted class (open/closed)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate accuracy
accuracy = 100 * correct / total
average_loss = running_loss / len(test_loader)

print(f'Accuracy: {accuracy:.2f}%')
print(f'Average Loss: {average_loss:.4f}')
