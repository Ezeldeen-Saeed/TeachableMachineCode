# teachable.py
import tensorflow as tf
import cv2
import numpy as np
import platform

np.set_printoptions(suppress=True)

class TeachableModel:
    def __init__(self, model_path, labels_path, confidence_threshold=0.9):
        self.model = self._load_model(model_path)
        self.labels = self._load_labels(labels_path)
        self.confidence_threshold = confidence_threshold

    def _load_model(self, model_path):
        return tf.saved_model.load(model_path).signatures['serving_default']

    def _load_labels(self, labels_path):
        with open(labels_path, "r") as f:
            return [label.strip() for label in f.readlines()]

    def preprocess_image(self, image):
        image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image_normalized = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3) / 255.0
        return tf.convert_to_tensor(image_normalized)

    def get_prediction(self, image_tensor):
        predictions = self.model(image_tensor)['sequential_3'].numpy()
        return predictions

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

            # Show the frame with predictions
            cv2.imshow("Real-Time Object Detection", frame)

            # Quit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture and close the window
        cap.release()
        cv2.destroyAllWindows()

def run_object_detection(model_path, labels_path, confidence_threshold):
    print("Initializing model...")
    model = TeachableModel(model_path, labels_path, confidence_threshold)
    print("Model initialized successfully.")

    camera_path = CameraHandler.get_camera_path()
    if camera_path is None:
        print("No camera found. Exiting...")
        return

    CameraHandler.capture_video(camera_path, model)
