import os
import cv2
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18

# Configuration
DATASET_PATH = "/home/rogelio/scanner/dataset/train"
LABEL_MAPPING_FILE = "label_mapping.json"
MODEL_SAVE_PATH = "dual_input_model2.pth"
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
DEPTH_WEIGHT = 0.5  # Ensure a minimum weight on depth contribution

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

        # Process positive samples
        positive_path = os.path.join(self.dataset_path, "positive")
        if os.path.exists(positive_path):
            for person in os.listdir(positive_path):
                person_path = os.path.join(positive_path, person)
                if not os.path.isdir(person_path):
                    continue

                rgb_path = os.path.join(person_path, "rgb")
                depth_path = os.path.join(person_path, "depth")

                # Assign a label to each person
                if person not in self.label_mapping:
                    self.label_mapping[person] = current_label
                    current_label += 1

                self._load_images(rgb_path, depth_path, self.label_mapping[person])

    def _load_images(self, rgb_path, depth_path, label):
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            return

        rgb_files = sorted(os.listdir(rgb_path))
        depth_files = sorted(os.listdir(depth_path))

        for rgb_file in rgb_files:
            rgb_id = rgb_file.replace("rgb_", "").replace(".png", "")
            matching_depth_file = f"depth_{rgb_id}.png"

            rgb_file_path = os.path.join(rgb_path, rgb_file)
            depth_file_path = os.path.join(depth_path, matching_depth_file)

            if os.path.exists(depth_file_path):
                self.data.append((rgb_file_path, depth_file_path))
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rgb_file, depth_file = self.data[idx]
        label = self.labels[idx]

        # Load RGB and depth images
        rgb_img = cv2.imread(rgb_file)
        depth_img = cv2.imread(depth_file, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            rgb_img = self.transform(rgb_img)
            depth_img = cv2.resize(depth_img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
            depth_img = torch.tensor(depth_img, dtype=torch.float32).unsqueeze(0)

        return rgb_img, depth_img, label

# Step 3: Define the Model
class DualInputModel(nn.Module):
    def __init__(self, num_classes, depth_weight=DEPTH_WEIGHT):
        super(DualInputModel, self).__init__()
        self.rgb_backbone = resnet18(pretrained=True)
        self.rgb_backbone.fc = nn.Identity()

        self.depth_backbone = resnet18(pretrained=True)
        self.depth_backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.depth_backbone.fc = nn.Identity()

        self.fc_rgb = nn.Linear(512, 256)
        self.fc_depth = nn.Linear(512, 256)

        self.fc_combined = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        self.depth_weight = depth_weight

    def forward(self, rgb, depth):
        rgb_features = self.rgb_backbone(rgb)
        depth_features = self.depth_backbone(depth)

        rgb_output = self.fc_rgb(rgb_features)
        depth_output = self.fc_depth(depth_features)

        # Ensure minimum weight contribution of depth features
        combined_features = torch.cat((self.depth_weight * depth_output, rgb_output), dim=1)
        output = self.fc_combined(combined_features)
        return output

# Step 4: Training Function
def train_model(model, train_loader, val_loader, num_classes, num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        # Validation
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

        val_acc = 100. * correct / total
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

    print("Training Complete!")
    return model

# Step 5: Main Function
def main():
    # Initialize label mapping
    label_mapping = initialize_label_mapping()

    # Define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    # Create datasets and data loaders
    train_dataset = FaceDataset(DATASET_PATH, label_mapping, transform=transform)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    save_label_mapping(label_mapping)

    # Initialize and train model
    num_classes = len(label_mapping)
    model = DualInputModel(num_classes)
    trained_model = train_model(model, train_loader, val_loader, num_classes, NUM_EPOCHS, LEARNING_RATE)

    # Save the trained model
    torch.save(trained_model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved!")

if __name__ == "__main__":
    main()
