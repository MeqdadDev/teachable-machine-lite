# Teachable Machine Lite
_By: [Meqdad Darwish](https://github.com/MeqdadDev)_

<p align="center">
<picture>
  <img alt="Teachable Machine Lite Package Logo" src="https://raw.githubusercontent.com/MeqdadDev/teachable-machine-lite/main/logo.png" width="80%" height="80%" >
</picture>
</p>

[![Downloads](https://static.pepy.tech/badge/teachable-machine-lite)](https://pepy.tech/project/teachable-machine-lite)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![PyPI](https://img.shields.io/pypi/v/teachable-machine-lite)](https://pypi.org/project/teachable-machine-lite/)
[![Tests](https://github.com/MeqdadDev/teachable-machine-lite/actions/workflows/tests.yml/badge.svg)](https://github.com/MeqdadDev/teachable-machine-lite/actions/workflows/tests.yml)

## Description

**Run your Google Teachable Machine image models on Raspberry Pi, robots and other edge devices in a few lines of Python. Lightweight, powered by LiteRT, no TensorFlow required.**

Teachable Machine Lite lets you take an image model you trained in Google's [Teachable Machine](https://teachablemachine.withgoogle.com/), with no coding needed, and run it on a Raspberry Pi, a robot, or any small computer with just a few lines of Python. It runs on LiteRT (formerly TensorFlow Lite) instead of the full TensorFlow, so it installs quickly, starts fast and works comfortably on resource-constrained hardware. Pass it a camera frame from OpenCV or Picamera2 and get back the predicted class and its confidence, ready to draw on the image or act on. That makes it a good fit for robotics, IoT and other edge AI projects. It's the lightweight sibling of the [Teachable Machine](https://github.com/MeqdadDev/teachable-machine) package.

Source Code is published on [GitHub](https://github.com/MeqdadDev/teachable-machine-lite/)

Read more about the project (requirements, installation, examples and more) in the [Documentation Website](https://meqdaddev.github.io/teachable-machine-lite/) 

## Supported Classifiers

**Image Classification**: Use an exported TensorFlow Lite model from [Teachable Machine Platform](https://teachablemachine.withgoogle.com/) (a model file with `tflite` extension). Both the **Quantized** and the **Floating point** exports are supported.

## Requirements

For detailed information about package requirements and supported platforms, please visit our [documentation](https://meqdaddev.github.io/teachable-machine-lite/requirements/)

```
Python >= 3.9
```

Works with NumPy 1.x and 2.x. On Raspberry Pi, use the **64-bit** Raspberry Pi OS.

## How to install Teachable Machine Lite Package

```bash
pip install teachable-machine-lite
```

On Raspberry Pi OS, install inside a virtual environment; see the [installation guide](https://meqdaddev.github.io/teachable-machine-lite/how-to-guide/#how-to-install-on-raspberry-pi-os).

## Dependencies

```bash
numpy
Pillow
ai-edge-litert   # LiteRT (TensorFlow Lite); tflite-runtime on 32-bit Raspberry Pi OS
```

## Example

An example for teachable machine lite package with OpenCV:

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
Values of `results` are assigned based on the content of `labels.txt` file.

For more, including the Raspberry Pi camera module (Picamera2), take a look at [these examples](https://meqdaddev.github.io/teachable-machine-lite/codeExamples/)

## Links:

- [Documentation](https://meqdaddev.github.io/teachable-machine-lite)

- [Changelog](https://meqdaddev.github.io/teachable-machine-lite/changelog/)

- [PyPI](https://pypi.org/project/teachable-machine-lite/)

- [Source Code](https://github.com/MeqdadDev/teachable-machine-lite)

- [Teachable Machine Platform](https://teachablemachine.withgoogle.com/)
