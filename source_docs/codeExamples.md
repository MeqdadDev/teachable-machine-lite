## Code Examples

Before running any of the code examples, ensure that the package is installed successfully.
You can find the installation instructions [here](https://meqdaddev.github.io/teachable-machine-lite/how-to-guide/#how-to-install-the-package).

You also need to have an exported model (and also **quantized**) (with `.tflite` file extension) with an associated labels text file for the package to use in annotations.

**Note:** You can get an exported and quantized model from the TensorFlow Lite tab while exporting your model (in the Teachable Machine platform).

Expected structure before running the code examples:

```
test-directory/
├── model.tflite
├── labels.txt
└── app.py (your code example file)
```

### Example 1

In this example for the teachable machine lite package with OpenCV, we will classify frames coming from the camera view and display 
the classification results on the camera view itself. We use `classify_and_show` to achieve this. See the code example below:

```python
from teachable_machine_lite import TeachableMachineLite
import cv2 as cv

cap = cv.VideoCapture(0)

model_path = 'model.tflite'
labels_path = "labels.txt"
image_file_name = "screenshot.jpg"

tm_model = TeachableMachineLite(model_path=model_path, labels_file_path=labels_path)

while True:
    ret, img = cap.read()
    cv.imwrite(image_file_name, img)

    results, resultImage = tm_model.classify_and_show(image_file_name, convert_to_bgr=True)
    print("results:", results)

    cv.imshow("Camera", resultImage)
    k = cv.waitKey(1)
    if k == 27:  # Press ESC to close the camera view
        break

cap.release()
cv.destroyAllWindows()
```

The values of `result` are assigned based on the content of the `labels.txt` file.

### Example 2

In this example for the teachable machine package with OpenCV, we will classify frames coming from the camera view and display 
the classification results on the camera view itself, but with separated methods for each task.

We use `classify_image` to classify the captured image but without showing the results on the camera view. Also, we use `show_prediction_on_image` to
get a frame with previous classification results.

Note that the `classify_image` method returns only the prediction results (a Python `dict`), not an image (numpy array or PIL.Image).

See the code example below:

```python
from teachable_machine_lite import TeachableMachineLite
import cv2 as cv

cap = cv.VideoCapture(0)

model_path = 'model.tflite'
labels_path = "labels.txt"
image_file_name = "screenshot.jpg"

tm_model = TeachableMachineLite(model_path=model_path, labels_file_path=labels_path)

while True:
    ret, img = cap.read()
    cv.imwrite(image_file_name, img)

    results = tm_model.classify_image(image_file_name, False)
    resultImage = tm_model.show_prediction_on_image(image_file_name, results)
    print("results:", results)

    cv.imshow("Camera", resultImage)
    k = cv.waitKey(1)
    if k == 27:  # Press ESC to close the camera view
        break

cap.release()
cv.destroyAllWindows()
```
