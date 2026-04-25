import os
# Suppress TensorFlow logging before importing it
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from teachable import run_object_detection

def main():
    # --- SETTINGS ---
    MODE = "samples"  # "camera" or "samples"
    
    # Paths are now relative to the project root based on your folder structure
    MODEL = "./Model/model.savedmodel/"
    LABELS = "./Model/labels.txt"
    THRESHOLD = 0.8
    # ----------------

    # If samples_path is None, teachable.py will automatically open the GUI
    run_object_detection(
        mode=MODE,
        model_path=MODEL,
        labels_path=LABELS,
        confidence_threshold=THRESHOLD
    )

if __name__ == "__main__":
    main()
