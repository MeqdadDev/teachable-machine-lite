"""Classify frames from the Raspberry Pi camera module. Press Esc to quit."""
import cv2 as cv
from picamera2 import Picamera2

from teachable_machine_lite import TeachableMachineLite

model = TeachableMachineLite(
    model_path="model.tflite", labels_file_path="labels.txt", num_threads=4)

picam2 = Picamera2()
# "RGB888" frames are stored in BGR order, the package's default for NumPy arrays.
picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}))
picam2.start()

while True:
    frame = picam2.capture_array()

    result, annotated = model.classify_and_show(frame)
    print("result:", result["label"], result["confidence"])

    cv.imshow("Camera", annotated)
    if cv.waitKey(1) == 27:  # Esc
        break

picam2.stop()
cv.destroyAllWindows()
