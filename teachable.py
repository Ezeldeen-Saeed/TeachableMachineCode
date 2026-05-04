# teachable.py
import os
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import cv2
import numpy as np
import platform
import tkinter as tk
from tkinter import filedialog

# Only show errors from TF logger
tf.get_logger().setLevel('ERROR')

def get_path_via_gui():
    """Opens a GUI dialog to select a file or directory, centered on the screen."""
    root = tk.Tk()
    root.withdraw()
    
    # Calculate center position
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = screen_width // 2
    y = screen_height // 2
    
    # Position the hidden root window at the center
    root.geometry(f"+{x}+{y}")
    root.attributes('-topmost', True)
    
    print("Please select a sample file...")
    path = filedialog.askopenfilename(parent=root, title="Select Sample Image")
    
    root.destroy()
    return path

np.set_printoptions(suppress=True)

class TeachableModel:
    def __init__(self, model_path, labels_path, confidence_threshold=0.9, input_size=(224, 224)):
        self.model = self._load_model(model_path)
        self.labels = self._load_labels(labels_path)
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size

    def _load_model(self, model_path):
        return tf.saved_model.load(model_path).signatures['serving_default']

    def _load_labels(self, labels_path):
        with open(labels_path, "r") as f:
            return [label.strip() for label in f.readlines()]

    def preprocess_image(self, image):
        width, height = self.input_size
        image_resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        image_normalized = np.asarray(image_resized, dtype=np.float32).reshape(1, width, height, 3) / 255.0
        return tf.convert_to_tensor(image_normalized)

    def get_prediction(self, image_tensor):
        outputs = self.model(image_tensor)
        # Dynamically get the first output key
        output_key = list(outputs.keys())[0]
        return outputs[output_key].numpy()

    def get_classification(self, predictions):
        class_idx = np.argmax(predictions)
        confidence = predictions[0][class_idx]
        if confidence >= self.confidence_threshold:
            return self.labels[class_idx], confidence
        return None, None

    @staticmethod
    def display_prediction(image, class_name, position=(10, 30)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 255, 0)
        thickness = 2
        cv2.putText(image, class_name, position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)

class CameraHandler:
    @staticmethod
    def get_camera_path():
        system = platform.system()

        if system == "Windows":
            for i in range(10):
                camera = cv2.VideoCapture(i)
                if camera.isOpened():
                    camera.release()
                    return i
                camera.release()
            return None

        elif system == "Darwin":
            camera_path = 0
            camera = cv2.VideoCapture(camera_path)
            if camera.isOpened():
                camera.release()
                return camera_path
            camera.release()
            return None

        elif system == "Linux":
            for i in range(11):
                camera_path = i
                camera = cv2.VideoCapture(camera_path)
                if camera.isOpened():
                    camera.release()
                    return camera_path
                camera.release()
            return None

        else:
            print("Unsupported operating system.")
            return None

    @staticmethod
    def capture_video(camera_path, model):
        cap = cv2.VideoCapture(camera_path)
        detections = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess the frame and get predictions
            image_tensor = model.preprocess_image(frame)
            predictions = model.get_prediction(image_tensor)

            # Find the class with the highest confidence
            class_name, confidence = model.get_classification(predictions)
            if class_name:
                model.display_prediction(frame, f"{class_name}: {confidence:.2f}")
                detections.append({"label": class_name, "confidence": confidence})

            # Show the frame with predictions
            try:
                cv2.imshow("Real-Time Object Detection", frame)
                # Quit when 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except cv2.error:
                # In headless mode, we just break to avoid infinite loop of errors.
                break

        # Release the video capture and close the window
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        
        return detections

class SampleHandler:
    @staticmethod
    def test_from_directory(path, model):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        results = []
        
        if not os.path.exists(path):
            return results

        if os.path.isfile(path):
            files = [os.path.basename(path)]
            directory_path = os.path.dirname(path) or "."
        else:
            files = os.listdir(path)
            directory_path = path

        for filename in files:
            if filename.lower().endswith(valid_extensions):
                image_path = os.path.join(directory_path, filename)
                frame = cv2.imread(image_path)
                if frame is None:
                    continue

                image_tensor = model.preprocess_image(frame)
                predictions = model.get_prediction(image_tensor)
                class_name, confidence = model.get_classification(predictions)

                if class_name:
                    results.append({"file": filename, "label": class_name, "confidence": float(confidence)})
                    model.display_prediction(frame, f"{class_name}: {confidence:.2f}")
                else:
                    results.append({"file": filename, "label": None, "confidence": 0.0})

                try:
                    cv2.imshow("Sample Test (Press any key for next)", frame)
                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        break
                except cv2.error:
                    pass
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        
        return results

def run_object_detection(
    mode='camera', 
    samples_path=None, 
    model_path="./converted_savedmodel/converted_savedmodel/model.savedmodel/", 
    labels_path="./converted_savedmodel/converted_savedmodel/labels.txt", 
    confidence_threshold=0.8
):
    # If in samples mode and no path provided, open the GUI
    if mode == 'samples' and not samples_path:
        samples_path = get_path_via_gui()
        if not samples_path:
            return None

    model = TeachableModel(model_path, labels_path, confidence_threshold)

    if mode == 'camera':
        camera_path = CameraHandler.get_camera_path()
        if camera_path is None:
            return None
        return CameraHandler.capture_video(camera_path, model)
    elif mode == 'samples':
        if not samples_path:
            return None
        return SampleHandler.test_from_directory(samples_path, model)
    else:
        return None
