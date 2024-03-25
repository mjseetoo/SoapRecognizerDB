import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 50 * 50, 128)  # Adjusted input size to match the output of conv3
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Read labels from the text file
        with open(os.path.join(self.root_dir, "labels.txt"), 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                image_name = parts[0]  # First part is the image name
                
                # Join remaining parts to form label (handling cases like "not soap")
                label = " ".join(parts[1:])
                
                # Append file extension to the image path
                image_path = os.path.join(self.root_dir, f"{image_name}.jpg")
                self.image_paths.append(image_path)
                
                # Map string labels to integers
                if label == "soap":
                    self.labels.append(0)
                elif label == "not_soap":
                    self.labels.append(1)
                else:
                    raise ValueError(f"Unknown label: {label}")

                # Print image paths and labels for debugging
                print(f"Image path: {self.image_paths[-1]}, Label: {self.labels[-1]}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
def train():
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(data_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
            optimizer.zero_grad()

        # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 mini-batches
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(data_loader)}], Loss: {running_loss / 10:.3f}")
                running_loss = 0.0
            
            epoch_loss = running_loss / len(data_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss:.3f}")
            
    torch.save(model.state_dict(), "path_to_save_model.pth")
    print("Training finished!")

    
def opt():
        # Specify device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Specify dataset directory
    dataset_dir = "output"

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
    ])

    # Load dataset
    dataset = CustomDataset(root_dir=dataset_dir, transform=transform)

    # Define hyperparameters
    batch_size = 1
    num_epochs = 10
    learning_rate = 0.001

    # Create data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define number of classes
    num_classes = 2 # assuming 2 classes: soap and not soap

    # Initialize model
    model = SimpleCNN(num_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)