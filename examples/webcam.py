"""Classify webcam frames and show the result on the video. Press Esc to quit."""
import cv2 as cv

from teachable_machine_lite import TeachableMachineLite

cap = cv.VideoCapture(0)
model = TeachableMachineLite(model_path="model.tflite", labels_file_path="labels.txt")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    result, annotated = model.classify_and_show(frame)
    print("result:", result)

    cv.imshow("Camera", annotated)
    if cv.waitKey(1) == 27:  # Esc
        break

cap.release()
cv.destroyAllWindows()
