"""Classify and draw in two separate steps. Press Esc to quit."""
import cv2 as cv

from teachable_machine_lite import TeachableMachineLite

cap = cv.VideoCapture(0)
model = TeachableMachineLite(model_path="model.tflite", labels_file_path="labels.txt")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    result = model.classify_image(frame, calc_time=True)
    annotated = model.show_prediction_on_image(frame, result)
    print(f"{result['label']} ({result['confidence']}%) in {result['time']}s")

    cv.imshow("Camera", annotated)
    if cv.waitKey(1) == 27:  # Esc
        break

cap.release()
cv.destroyAllWindows()
