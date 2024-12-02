import os
import cv2
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from torch.nn.functional import one_hot

# Configuration
DATASET_PATH = "/home/rogelio/scanner/dataset" 
LABEL_MAPPING_FILE = "label_mapping.json"
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 0.001

# Dataset paths
TRAIN_PATH = "/home/rogelio/scanner/dataset/train" 
VAL_PATH = "/home/rogelio/scanner/dataset/val" 
TEST_PATH = "/home/rogelio/scanner/dataset/test" 

# Step 1: Initialize or Load Label Mapping
def initialize_label_mapping():
    if os.path.exists(LABEL_MAPPING_FILE):
        with open(LABEL_MAPPING_FILE, "r") as f:
            label_mapping = json.load(f)
    else:
        label_mapping = {}
    return label_mapping

def save_label_mapping(label_mapping):
    with open(LABEL_MAPPING_FILE, "w") as f:
        json.dump(label_mapping, f)

# Step 2: Custom Dataset Class
class FaceDataset(Dataset):
    def __init__(self, dataset_path, label_mapping, transform=None):
        self.dataset_path = dataset_path
        self.label_mapping = label_mapping
        self.transform = transform
        self.data = []
        self.labels = []
        self._prepare_data()

    def _prepare_data(self):
        current_label = len(self.label_mapping)

        for category in ["positive", "negative"]:
            category_path = os.path.join(self.dataset_path, category)
            if not os.path.exists(category_path):
                continue

            # Handle "positive" class: assign unique labels to each person
            if category == "positive":
                for person in os.listdir(category_path):
                    person_path = os.path.join(category_path, person)
                    rgb_path = os.path.join(person_path, "rgb")
                    depth_path = os.path.join(person_path, "depth")

                    # Assign a label to each person if not already labeled
                    if person not in self.label_mapping:
                        self.label_mapping[person] = current_label
                        current_label += 1

                    rgb_files = sorted(os.listdir(rgb_path))
                    depth_files = sorted(os.listdir(depth_path))

                    for rgb_file in rgb_files:
                        rgb_id = rgb_file.replace("rgb_", "").replace(".png", "")
                        matching_depth_file = f"depth_{rgb_id}.png"

                        depth_file_path = os.path.join(depth_path, matching_depth_file)
                        rgb_file_path = os.path.join(rgb_path, rgb_file)

                        if os.path.exists(depth_file_path):
                            self.data.append((rgb_file_path, depth_file_path))
                            self.labels.append(self.label_mapping[person])

            # Handle "negative" class
            elif category == "negative":
                if category not in self.label_mapping:
                    self.label_mapping[category] = current_label
                    current_label += 1

                for subfolder in os.listdir(category_path):
                    subfolder_path = os.path.join(category_path, subfolder)
                    rgb_path = os.path.join(subfolder_path, "rgb")
                    depth_path = os.path.join(subfolder_path, "depth")

                    rgb_files = sorted(os.listdir(rgb_path))
                    depth_files = sorted(os.listdir(depth_path))

                    for rgb_file in rgb_files:
                        rgb_id = rgb_file.replace("rgb_", "").replace(".png", "")
                        matching_depth_file = f"depth_{rgb_id}.png"

                        depth_file_path = os.path.join(depth_path, matching_depth_file)
                        rgb_file_path = os.path.join(rgb_path, rgb_file)

                        if os.path.exists(depth_file_path):
                            self.data.append((rgb_file_path, depth_file_path))
                            self.labels.append(self.label_mapping["negative"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rgb_file, depth_file = self.data[idx]
        label = self.labels[idx]

        # Load RGB and depth images
        rgb_img = cv2.imread(rgb_file)  # Load RGB image
        depth_img = cv2.imread(depth_file, cv2.IMREAD_GRAYSCALE)  # Load depth as grayscale (2D)

        if self.transform:
            # Apply transformations to RGB image
            rgb_img = self.transform(rgb_img)

            # Resize and normalize the depth image
            depth_img = cv2.resize(depth_img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))  # Resize depth to match RGB
            depth_img = torch.tensor(depth_img, dtype=torch.float32).unsqueeze(0)  # Add channel dimension (1, H, W)

        return rgb_img, depth_img, label

# Step 3: Define the Model
class DualInputModel(nn.Module):
    def __init__(self, num_classes):
        super(DualInputModel, self).__init__()
        self.rgb_backbone = resnet18(pretrained=True)
        self.rgb_backbone.fc = nn.Identity()

        self.depth_backbone = resnet18(pretrained=True)
        self.depth_backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.depth_backbone.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(512 + 512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, rgb, depth):
        rgb_features = self.rgb_backbone(rgb)  # Expected: [batch_size, 512]
        depth_features = self.depth_backbone(depth)  # Expected: [batch_size, 512]
        combined_features = torch.cat((rgb_features, depth_features), dim=1)
        output = self.fc(combined_features)
        return output

# Step 4: Training Function
def train_model(model, train_loader, val_loader, num_classes, num_epochs=10, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for rgb, depth, labels in train_loader:
            rgb, depth, labels = rgb.to(device), depth.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(rgb, depth)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}, Accuracy: {train_acc:.2f}%")

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for rgb, depth, labels in val_loader:
                rgb, depth, labels = rgb.to(device), depth.to(device), labels.to(device)
                outputs = model(rgb, depth)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        if total == 0:
            val_acc = 0
        else:
            val_acc = 100. * correct / total

        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

    print("Training Complete!")
    return model

# Step 5: Main Function
def main():
    # Load or initialize label mapping
    label_mapping = initialize_label_mapping()

    # Define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    # Create Dataset and DataLoader
    train_dataset = FaceDataset(TRAIN_PATH, label_mapping, transform=transform)
    val_dataset = FaceDataset(VAL_PATH, label_mapping, transform=transform)

    save_label_mapping(label_mapping)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for rgb, depth, labels in train_loader:
        print(f"RGB Shape: {rgb.shape}")      # Expected: [batch_size, 3, 256, 256]
        print(f"Depth Shape: {depth.shape}")  # Expected: [batch_size, 1, 256, 256]
        print(f"Labels Shape: {labels.shape}")  # Expected: [batch_size]
        break

    # Initialize and train model
    num_classes = len(label_mapping)
    model = DualInputModel(num_classes=num_classes)
    trained_model = train_model(model, train_loader, val_loader, num_classes=num_classes, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE)

    # Save the trained model
    torch.save(trained_model.state_dict(), "dual_input_model.pth")
    print("Model saved!")

# Run the script
if __name__ == "__main__":
    main()
