import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader

# Step 1: Define transforms for the training dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

# Step 2: Load the dataset (you need a directory with 'open' and 'closed' subfolders)
train_data = datasets.ImageFolder(root='eyes/', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Step 3: Load pre-trained ResNet-18
model = models.resnet18(pretrained=True)

# Step 4: Replace the last layer with a new layer for 2 classes (open/closed)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

'''
for param in model.parameters():
    param.requires_grad = False
'''
# Step 5: Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 6: Train the model
num_epochs = 10
model = model.to('cuda')  # If using GPU
for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

# Step 7: Save the trained model
torch.save(model.state_dict(), 'eye_classifier_resnet18.pth')
