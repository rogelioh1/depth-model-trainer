import sys
import numpy as np
import torch
from torchvision import transforms
from freenect import sync_get_video, sync_get_depth
import json
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
import cv2
from model2 import DualInputModel  # Import the DualInputModel from your training script

# Configuration
MODEL_PATH = "dual_input_model2.pth"
LABEL_MAPPING_FILE = "label_mapping.json"
IMAGE_SIZE = (256, 256)
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to recognize a face
DEPTH_VARIANCE_THRESHOLD = 100.0  # Minimum variance in depth values to consider valid

# Load the trained model
def load_model():
    with open(LABEL_MAPPING_FILE, "r") as f:
        label_mapping = json.load(f)

    num_classes = len(label_mapping)
    model = DualInputModel(num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()  # Set model to evaluation mode
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model, label_mapping

# Preprocessing function for live feed
def preprocess_frame(rgb_frame, depth_frame):
    depth_frame_normalized = (depth_frame / depth_frame.max() * 255).astype(np.uint8)

    transform_rgb = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    transform_depth = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    rgb_tensor = transform_rgb(rgb_frame).unsqueeze(0)  # Add batch dimension
    depth_tensor = transform_depth(depth_frame_normalized).unsqueeze(0)  # Add batch dimension

    return rgb_tensor, depth_tensor

# Depth consistency check
def is_valid_depth(depth_frame):
    depth_variance = np.var(depth_frame)
    return depth_variance >= DEPTH_VARIANCE_THRESHOLD

# Predict function with confidence threshold
def predict(model, rgb_tensor, depth_tensor, label_mapping):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb_tensor, depth_tensor = rgb_tensor.to(device), depth_tensor.to(device)

    with torch.no_grad():
        outputs = model(rgb_tensor, depth_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        max_confidence, predicted_class = torch.max(probabilities, dim=1)

    if max_confidence.item() >= CONFIDENCE_THRESHOLD:
        reverse_label_mapping = {v: k for k, v in label_mapping.items()}
        predicted_label = reverse_label_mapping[predicted_class.item()]
        return predicted_label, max_confidence.item(), probabilities.cpu().numpy()
    else:
        return None, None, probabilities.cpu().numpy()

# PyQt5 Application
class KinectApp(QMainWindow):
    def __init__(self, model, label_mapping):
        super().__init__()
        self.model = model
        self.label_mapping = label_mapping

        # Set up the main window
        self.setWindowTitle("Kinect RGB Feed")
        self.setStyleSheet("background-color: black;")  # Set background to black

        # Create a QLabel to display the video
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        # Set up a QTimer to update the video feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30ms (~33 FPS)

    def resizeEvent(self, event):
        """Resize QLabel to fill the entire window."""
        self.video_label.setGeometry(self.rect())

    def update_frame(self):
        # Get RGB and Depth frames
        rgb_frame, _ = sync_get_video()
        depth_frame, _ = sync_get_depth()

        if rgb_frame is None or depth_frame is None:
            print("Error: Could not get frames from Kinect.")
            return

        rgb_frame = np.array(rgb_frame)
        depth_frame = np.array(depth_frame)

        # Detect face before making predictions
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            x, y, w, h = faces[0]  # Use the first detected face
            face_rgb = rgb_frame[y:y + h, x:x + w]
            face_depth = depth_frame[y:y + h, x:x + w]

            # Check depth consistency
            if not is_valid_depth(face_depth):
                cv2.putText(rgb_frame, "Flat image detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                # Preprocess the frames
                rgb_tensor, depth_tensor = preprocess_frame(face_rgb, face_depth)

                # Perform prediction
                predicted_label, confidence, probabilities = predict(self.model, rgb_tensor, depth_tensor, self.label_mapping)

                if predicted_label:
                    label_text = f"Recognized: {predicted_label} ({confidence:.2f})"
                    cv2.putText(rgb_frame, label_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(rgb_frame, "Face detected, but recognition confidence too low", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(rgb_frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Convert frame to QImage for PyQt5 display
        rgb_image = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Update QLabel with the scaled image
        self.video_label.setPixmap(QPixmap.fromImage(q_image).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio))

# Main function
def main():
    model, label_mapping = load_model()

    app = QApplication(sys.argv)
    window = KinectApp(model, label_mapping)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
