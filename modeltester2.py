import os
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from model2 import DualInputModel  # Import the trained model

# Configuration
TEST_PATH = "/home/rogelio/scanner/dataset/test"  # Path to your test dataset
LABEL_MAPPING_FILE = "label_mapping.json"  # Path to label mapping
MODEL_PATH = "dual_input_model2.pth"  # Path to the trained model
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset class
class TestDataset(Dataset):
    def __init__(self, dataset_path, label_mapping, transform=None):
        self.dataset_path = dataset_path
        self.label_mapping = label_mapping
        self.transform = transform
        self.data = []
        self.labels = []
        self._prepare_data()

    def _prepare_data(self):
        for category in ["positive", "negative"]:
            category_path = os.path.join(self.dataset_path, category)
            if not os.path.exists(category_path):
                print(f"Warning: Skipping category {category} as it does not exist.")
                continue

            if category == "positive":
                for person in os.listdir(category_path):
                    person_path = os.path.join(category_path, person)
                    rgb_path = os.path.join(person_path, "rgb")
                    depth_path = os.path.join(person_path, "depth")
                    
                    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                        print(f"Skipping {person} due to missing RGB or Depth folder.")
                        continue

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

            elif category == "negative":
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

# Load label mapping
def load_label_mapping():
    with open(LABEL_MAPPING_FILE, "r") as f:
        return json.load(f)

# Load the trained model
def load_model(num_classes):
    model = DualInputModel(num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()  # Set model to evaluation mode
    return model

# Test the model
def test_model(model, test_loader):
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for rgb, depth, labels in test_loader:
            rgb, depth, labels = rgb.to(DEVICE), depth.to(DEVICE), labels.to(DEVICE)
            outputs = model(rgb, depth)
            _, predicted = outputs.max(1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    return true_labels, predicted_labels

# Plot confusion matrix
def plot_confusion_matrix(true_labels, predicted_labels, class_names):
    cm = confusion_matrix(true_labels, predicted_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

# Main function
def main():
    # Load label mapping
    label_mapping = load_label_mapping()
    num_classes = len(label_mapping)

    print(f"Label Mapping: {label_mapping}")

    # Define data transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    # Prepare test dataset and loader
    test_dataset = TestDataset(TEST_PATH, label_mapping, transform=transform)
    if len(test_dataset) == 0:
        print("Test dataset is empty. Please check the TEST_PATH and dataset structure.")
        return

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load the trained model
    model = load_model(num_classes)
    print("Model loaded successfully.")

    # Test the model
    true_labels, predicted_labels = test_model(model, test_loader)

    # Check for valid labels
    if len(true_labels) == 0 or len(predicted_labels) == 0:
        print("No valid predictions. Please check the dataset and model.")
        return

    # Classification report and confusion matrix
    class_names = list(label_mapping.keys())
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=class_names))

    # Plot the confusion matrix
    plot_confusion_matrix(true_labels, predicted_labels, class_names)

if __name__ == "__main__":
    main()
