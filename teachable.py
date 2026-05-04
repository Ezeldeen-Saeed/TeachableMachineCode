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
import threading
import time

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
            label = self.labels[class_idx].lower().strip()
            # Remove leading numeric index if present (e.g., "0 happy" -> "happy")
            parts = label.split(maxsplit=1)
            if len(parts) > 1 and parts[0].isdigit():
                label = parts[1]
            return label, confidence
        return None, None

    @staticmethod
    def display_prediction(image, class_name, position=(10, 30)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 255, 0)
        thickness = 2
        cv2.putText(image, class_name, position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)

class CameraHandler:
    _active_stream = None

    @staticmethod
    def get_camera_path():
        system = platform.system()
        paths = range(10) if system in ["Windows", "Linux"] else [0]
        for i in paths:
            camera = cv2.VideoCapture(i)
            if camera.isOpened():
                camera.release()
                return i
            camera.release()
        return None

    @staticmethod
    def start_preview():
        if CameraHandler._active_stream is None:
            path = CameraHandler.get_camera_path()
            if path is None:
                print("No camera found!")
                return None
            CameraHandler._active_stream = CameraStream(path)
            CameraHandler._active_stream.start()
            # Wait for first frame
            timeout = 50
            while CameraHandler._active_stream.frame is None and timeout > 0:
                time.sleep(0.1)
                timeout -= 1
        return CameraHandler._active_stream

    @staticmethod
    def get_latest_frame():
        if CameraHandler._active_stream and CameraHandler._active_stream.running:
            return CameraHandler._active_stream.frame
        return None

class CameraStream(threading.Thread):
    def __init__(self, camera_path):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(camera_path)
        self.frame = None
        self.running = True

    def run(self):
        print("Camera Preview started. Press 'q' in the window to stop.")
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame = frame
            cv2.imshow("Camera Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
        
        self.cap.release()
        cv2.destroyAllWindows()
        self.running = False

def preview_camera():
    """Starts a non-blocking camera preview in a background thread."""
    return CameraHandler.start_preview()

def run_object_detection(
    mode='camera', 
    samples_path=None, 
    model_path="./converted_savedmodel/converted_savedmodel/model.savedmodel/", 
    labels_path="./converted_savedmodel/converted_savedmodel/labels.txt", 
    confidence_threshold=0.8
):
    model = TeachableModel(model_path, labels_path, confidence_threshold)

    if mode == 'camera':
        # Check if preview is already running
        frame = CameraHandler.get_latest_frame()
        
        if frame is None:
            # Fallback to one-shot capture if preview isn't running
            camera_path = CameraHandler.get_camera_path()
            if camera_path is None: return None
            cap = cv2.VideoCapture(camera_path)
            ret, frame = cap.read()
            cap.release()
            if not ret: return None
            
        image_tensor = model.preprocess_image(frame)
        predictions = model.get_prediction(image_tensor)
        class_name, confidence = model.get_classification(predictions)
        
        return [{"label": class_name, "confidence": float(confidence) if confidence else 0.0}]

    elif mode == 'samples':
        if not samples_path:
            samples_path = get_path_via_gui()
            if not samples_path: return None
        
        # We'll use SampleHandler logic but simplified for brevity here
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        results = []
        if os.path.isfile(samples_path):
            files = [samples_path]
        else:
            files = [os.path.join(samples_path, f) for f in os.listdir(samples_path) if f.lower().endswith(valid_extensions)]
            
        for img_path in files:
            frame = cv2.imread(img_path)
            if frame is None: continue
            image_tensor = model.preprocess_image(frame)
            predictions = model.get_prediction(image_tensor)
            class_name, confidence = model.get_classification(predictions)
            results.append({"file": os.path.basename(img_path), "label": class_name, "confidence": float(confidence) if confidence else 0.0})
        return results
    
    return None

class SampleHandler:
    # Kept for compatibility if needed elsewhere, but logic moved to run_object_detection
    @staticmethod
    def test_from_directory(path, model):
        return run_object_detection(mode='samples', samples_path=path, model_path=model.model_path, labels_path=model.labels_path)
