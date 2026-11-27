"""Classify webcam frames with any exported model, chosen on the command line."""
import argparse

import cv2 as cv

from teachable_machine_lite import TeachableMachineLite

parser = argparse.ArgumentParser(
    description="Classify webcam frames with a Teachable Machine model.")
parser.add_argument("--model", default="model.tflite", help="path to the .tflite model file")
parser.add_argument("--labels", default="labels.txt", help="path to the labels.txt file")
parser.add_argument("--camera", type=int, default=0,
                    help="camera index (try 1 for an external webcam)")
parser.add_argument("--threads", type=int, default=None, help="CPU threads for inference, e.g. 4")
parser.add_argument("--top-k", type=int, default=1,
                    help="how many of the most likely classes to print")
args = parser.parse_args()

model = TeachableMachineLite(
    model_path=args.model, labels_file_path=args.labels, num_threads=args.threads)

cap = cv.VideoCapture(args.camera)
if not cap.isOpened():
    raise SystemExit(f"Could not open camera {args.camera}.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    result, annotated = model.classify_and_show(frame, top_k=args.top_k)
    print(", ".join(f"{p['label']}: {p['confidence']}%" for p in result["predictions"]))

    cv.imshow("Camera", annotated)
    if cv.waitKey(1) == 27:  # Esc
        break

cap.release()
cv.destroyAllWindows()
