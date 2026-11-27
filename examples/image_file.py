"""Classify one image file and act only when the model is confident enough."""
from teachable_machine_lite import TeachableMachineLite

model = TeachableMachineLite(model_path="model.tflite", labels_file_path="labels.txt")

result = model.classify_image("photo.jpg", top_k=2)

if result["confidence"] > 80:
    print("Detected:", result["label"])
else:
    print("Not sure. Top guesses:", result["predictions"])
