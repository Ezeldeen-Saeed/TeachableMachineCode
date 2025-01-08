import argparse
from teachable import run_object_detection, TeachableModel

confidence_threshold = 0.9

model_path = "./converted_savedmodel2/converted_savedmodel/model.savedmodel/"
labels_path = "./converted_savedmodel2/converted_savedmodel/labels.txt"


run_object_detection(model_path, labels_path, confidence_threshold)
