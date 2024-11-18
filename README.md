# Teachable Machine Lite
_By: [Meqdad Darwish](https://github.com/MeqdadDev)_

<p align="center">
<picture>
  <img alt="Teachable Machine Lite Package Logo" src="https://meqdaddev.github.io/teachable-machine-lite/logo.png" width="80%" height="80%" >
</picture>
</p>

[![Downloads](https://static.pepy.tech/badge/teachable-machine-lite)](https://pepy.tech/project/teachable-machine-lite)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![PyPI](https://img.shields.io/pypi/v/teachable-machine-lite)](https://pypi.org/project/teachable-machine-lite/)

## Description

A lightweight Python package optimized for integrating exported models from Google's [Teachable Machine Platform](https://teachablemachine.withgoogle.com/) into robotics and embedded systems environments. This streamlined version of [Teachable Machine Package](https://github.com/MeqdadDev/teachable-machine) is specifically designed for resource-constrained devices, making it easier to deploy and use your trained models in embedded applications. With a focus on efficiency and minimal dependencies, this tool maintains the core functionality while being more suitable for robotics and IoT projects.

Source Code is published on [GitHub](https://github.com/MeqdadDev/teachable-machine-lite/)

Read more about the project (requirements, installation, examples and more) in the [Documentation Website](https://meqdaddev.github.io/teachable-machine-lite/) 

## Supported Classifiers

**Image Classification**: Use exported and quantized TensorFlow Lite model from [Teachable Machine Platform](https://teachablemachine.withgoogle.com/) (a model file with `tflite` extension).


## Requirements

```
Python >= 3.7
```

## How to install Teachable Machine Lite Package

```bash
pip install teachable-machine-lite
```

## Dependencies

```bash
numpy
tflite-runtime
Pillow
```

## Example

An example for teachable machine lite package with OpenCV:

```python
from teachable_machine_lite import TeachableMachineLite
import cv2 as cv

cap = cv.VideoCapture(0)

model_path = "model.tflite"
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
Values of `results` are assigned based on the content of `labels.txt` file.

For more; take a look on [these examples](https://meqdaddev.github.io/teachable-machine-lite/codeExamples/)

## Links:

### Links:

- [Documentation](https://meqdaddev.github.io/teachable-machine-lite)

- [PyPI](https://pypi.org/project/teachable-machine-lite/)

- [Source Code](https://github.com/MeqdadDev/teachable-machine-lite)

- [Teachable Machine Platform](https://teachablemachine.withgoogle.com/)