## Code Examples

Before running any of the code examples, ensure that the package is installed successfully.
You can find the installation instructions [here](how-to-guide.md#how-to-install-the-package)
(and [here for Raspberry Pi OS](how-to-guide.md#how-to-install-on-raspberry-pi-os)).

You also need an exported TensorFlow Lite model (`.tflite`, quantized or floating point) with its
`labels.txt` file. You get both from the **TensorFlow Lite** tab when exporting your model in the
Teachable Machine platform.

Expected structure before running the code examples:

```
test-directory/
├── model.tflite
├── labels.txt
└── app.py (your code example file)
```

### Example 1: OpenCV camera

In this example we classify frames coming from the camera and display the classification results
on the camera view itself, using `classify_and_show`. Frames are passed to the model directly, with no
need to save them to a file first.

```python
from teachable_machine_lite import TeachableMachineLite
import cv2 as cv

cap = cv.VideoCapture(0)

model_path = "model.tflite"
labels_path = "labels.txt"

tm_model = TeachableMachineLite(model_path=model_path, labels_file_path=labels_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results, result_image = tm_model.classify_and_show(frame)
    print("results:", results)

    cv.imshow("Camera", result_image)
    k = cv.waitKey(1)
    if k == 27:  # Press ESC to close the camera view
        break

cap.release()
cv.destroyAllWindows()
```

The values of `results` are assigned based on the content of the `labels.txt` file. For example:

```python
{
    "id": 0,
    "label": "Class 1",           # class name from labels.txt, without its index
    "time": None,                 # seconds, when calc_time=True
    "confidence": 97.66,          # percentage
    "highest_class_id": 0,        # same as "id"
    "highest_class_prob": 97.66,  # same as "confidence"
    "predictions": [              # top_k most likely classes, best first
        {"id": 0, "label": "Class 1", "confidence": 97.66},
    ],
}
```

### Example 2: Separate classification and drawing

Here we use `classify_image` to classify the frame, then `show_prediction_on_image` to draw those
results on the frame.

Note that `classify_image` returns only the prediction results (a Python `dict`), not an image.

```python
from teachable_machine_lite import TeachableMachineLite
import cv2 as cv

cap = cv.VideoCapture(0)

tm_model = TeachableMachineLite(model_path="model.tflite", labels_file_path="labels.txt")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = tm_model.classify_image(frame, calc_time=True)
    result_image = tm_model.show_prediction_on_image(frame, results)
    print(f"{results['label']} ({results['confidence']}%) in {results['time']}s")

    cv.imshow("Camera", result_image)
    k = cv.waitKey(1)
    if k == 27:  # Press ESC to close the camera view
        break

cap.release()
cv.destroyAllWindows()
```

### Example 3: Raspberry Pi camera module (Picamera2)

The Raspberry Pi camera module (CSI ribbon cable) is used through Picamera2;
`cv.VideoCapture(0)` does not work with it on recent Raspberry Pi OS.

```python
from teachable_machine_lite import TeachableMachineLite
from picamera2 import Picamera2
import cv2 as cv

tm_model = TeachableMachineLite(
    model_path="model.tflite", labels_file_path="labels.txt", num_threads=4)

picam2 = Picamera2()
# "RGB888" frames are stored in BGR order, the package's default for NumPy arrays.
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.start()

while True:
    frame = picam2.capture_array()

    results, result_image = tm_model.classify_and_show(frame)
    print("results:", results["label"], results["confidence"])

    cv.imshow("Camera", result_image)
    if cv.waitKey(1) == 27:  # Press ESC to quit
        break

picam2.stop()
cv.destroyAllWindows()
```

### Example 4: Classify an image file and react to the result

A typical robotics pattern: act on the label only when the model is confident enough.

```python
from teachable_machine_lite import TeachableMachineLite

tm_model = TeachableMachineLite(model_path="model.tflite", labels_file_path="labels.txt")

result = tm_model.classify_image("photo.jpg", top_k=2)

if result["confidence"] > 80:
    print("Detected:", result["label"])
else:
    print("Not sure. Top guesses:", result["predictions"])
```
