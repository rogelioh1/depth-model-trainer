import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from model import DualInputModel  # Import your DualInputModel

# Configuration
TEST_PATH = "/home/rogelio/scanner/dataset/test"  # Path to your test dataset
LABEL_MAPPING_FILE = "label_mapping.json"  # Path to label mapping
MODEL_PATH = "dual_input_model.pth"  # Path to the trained model
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset class
class TestDataset:
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
                print(f"Skipping {category_path}: Folder does not exist.")
                continue

            for person in os.listdir(category_path):
                person_path = os.path.join(category_path, person)
                rgb_path = os.path.join(person_path, "rgb")
                depth_path = os.path.join(person_path, "depth")

                if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                    print(f"Skipping {person_path}: Missing 'rgb' or 'depth' folder.")
                    continue

                print(f"Processing {person} in {category}: {rgb_path} and {depth_path}")

                rgb_files = sorted(os.listdir(rgb_path))
                depth_files = sorted(os.listdir(depth_path))

                for rgb_file in rgb_files:
                    rgb_id = rgb_file.replace("rgb_", "").replace(".png", "")
                    matching_depth_file = f"depth_{rgb_id}.png"

                    depth_file_path = os.path.join(depth_path, matching_depth_file)
                    rgb_file_path = os.path.join(rgb_path, rgb_file)

                    if os.path.exists(depth_file_path):
                        self.data.append((rgb_file_path, depth_file_path))
                        if category == "positive":
                            self.labels.append(self.label_mapping.get(person, -1))
                        else:
                            self.labels.append(self.label_mapping["negative"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rgb_file, depth_file = self.data[idx]
        label = self.labels[idx]

        rgb_img = cv2.imread(rgb_file)
        depth_img = cv2.imread(depth_file, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            rgb_img = self.transform(rgb_img)
            # Correctly reshape depth image
            depth_img = transforms.ToTensor()(depth_img)  # Converts to [1, height, width]

        return rgb_img, depth_img, label

# Load label mapping
def load_label_mapping():
    with open(LABEL_MAPPING_FILE, "r") as f:
        return json.load(f)

# Load the trained model
def load_model(num_classes):
    model = DualInputModel(num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)
    model.eval()
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

    plt.figure(figsize=(6, 6))
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
    print(f"Loaded {len(test_dataset)} samples for testing.")
    
    if len(test_dataset) == 0:
        print("Test dataset is empty. Please check your TEST_PATH and data structure.")
        return

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load trained model
    model = load_model(num_classes)
    print("Model loaded successfully.")

    # Test the model
    true_labels, predicted_labels = test_model(model, test_loader)

    # Check for valid labels
    if not true_labels or not predicted_labels:
        print("No valid labels found in true_labels or predicted_labels.")
        return

    # Calculate accuracy
    accuracy = np.mean(np.array(true_labels) == np.array(predicted_labels)) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Classification report
    unique_classes = sorted(set(true_labels + predicted_labels))
    print(f"Unique Classes in Test Data: {unique_classes}")

    if len(unique_classes) == 1:
        class_name = list(label_mapping.keys())[list(label_mapping.values()).index(unique_classes[0])]
        print(f"Only one class present in test data: {class_name}")
    else:
        valid_target_names = [list(label_mapping.keys())[i] for i in unique_classes]
        print("\nClassification Report:")
        print(classification_report(true_labels, predicted_labels, labels=unique_classes, target_names=valid_target_names))

        # Plot confusion matrix
        plot_confusion_matrix(true_labels, predicted_labels, valid_target_names)


if __name__ == "__main__":
    main()
