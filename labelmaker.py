import os
import cv2
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.functional import one_hot

# Configuration
DATASET_PATH = "/home/rogelio/scanner/dataset"  # Replace with the actual dataset path
LABEL_MAPPING_FILE = "label_mapping.json"
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16

# Step 1: Initialize or Load Label Mapping
def initialize_label_mapping():
    if os.path.exists(LABEL_MAPPING_FILE):
        # Load existing mapping
        with open(LABEL_MAPPING_FILE, "r") as f:
            label_mapping = json.load(f)
    else:
        # Create a new mapping
        label_mapping = {}
        # Add "negative" class if not already present
        if "negative" not in label_mapping:
            label_mapping["negative"] = len(label_mapping)
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

        # Iterate through "positive" and "negative" folders
        for category in ["positive", "negative"]:
            category_path = os.path.join(self.dataset_path, category)
            if not os.path.exists(category_path):
                continue

            # For positive, assign unique labels to each individual
            if category == "positive":
                for person in os.listdir(category_path):
                    person_path = os.path.join(category_path, person)
                    rgb_path = os.path.join(person_path, "rgb")
                    depth_path = os.path.join(person_path, "depth")

                    # Assign a label to each individual
                    if person not in self.label_mapping:
                        self.label_mapping[person] = current_label
                        current_label += 1

                    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                        continue

                    rgb_files = sorted(os.listdir(rgb_path))
                    depth_files = sorted(os.listdir(depth_path))

                    for rgb_file in rgb_files:
                        rgb_id = rgb_file.replace("rgb_", "").replace(".png", "")
                        matching_depth_file = None

                        for depth_file in depth_files:
                            depth_id = depth_file.replace("depth_", "").replace(".png", "")
                            if depth_id.startswith(rgb_id):  # Match base names
                                matching_depth_file = depth_file
                                break

                        if matching_depth_file:
                            rgb_file_path = os.path.join(rgb_path, rgb_file)
                            depth_file_path = os.path.join(depth_path, matching_depth_file)
                            self.data.append((rgb_file_path, depth_file_path))
                            self.labels.append(self.label_mapping[person])

            # For negative, assign a single label
            elif category == "negative":
                if category not in self.label_mapping:
                    self.label_mapping[category] = current_label
                    current_label += 1

                for subfolder in os.listdir(category_path):
                    subfolder_path = os.path.join(category_path, subfolder)
                    rgb_path = os.path.join(subfolder_path, "rgb")
                    depth_path = os.path.join(subfolder_path, "depth")

                    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                        continue

                    rgb_files = sorted(os.listdir(rgb_path))
                    depth_files = sorted(os.listdir(depth_path))

                    for rgb_file in rgb_files:
                        rgb_id = rgb_file.replace("rgb_", "").replace(".png", "")
                        matching_depth_file = None

                        for depth_file in depth_files:
                            depth_id = depth_file.replace("depth_", "").replace(".png", "")
                            if depth_id.startswith(rgb_id):  # Match base names
                                matching_depth_file = depth_file
                                break

                        if matching_depth_file:
                            rgb_file_path = os.path.join(rgb_path, rgb_file)
                            depth_file_path = os.path.join(depth_path, matching_depth_file)
                            self.data.append((rgb_file_path, depth_file_path))
                            self.labels.append(self.label_mapping["negative"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rgb_file, depth_file = self.data[idx]
        label = self.labels[idx]

        # Load images
        rgb_img = cv2.imread(rgb_file)
        depth_img = cv2.imread(depth_file, cv2.IMREAD_GRAYSCALE)

        # Preprocess images
        if self.transform:
            rgb_img = self.transform(rgb_img)
            depth_img = self.transform(depth_img).unsqueeze(0)  # Add channel for depth

        return rgb_img, depth_img, label

# Step 3: Prepare Data and Transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

# Step 4: Main Function
def main():
    # Load or initialize label mapping
    label_mapping = initialize_label_mapping()

    # Create Dataset and DataLoader
    dataset = FaceDataset(DATASET_PATH, label_mapping, transform=transform)

    # Save updated mapping
    save_label_mapping(label_mapping)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Example: Iterate through training data
    for rgb, depth, labels in train_loader:
        print(f"RGB Shape: {rgb.shape}, Depth Shape: {depth.shape}, Labels: {labels}")
        # Convert labels to one-hot
        one_hot_labels = one_hot(torch.tensor(labels), num_classes=len(label_mapping))
        print(f"One-Hot Labels: {one_hot_labels}")
        break

# Run the script
if __name__ == "__main__":
    main()
