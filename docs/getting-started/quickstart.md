# Quick Start

This page takes you from an exported model to live classification on a camera.

## Before you start

- [Install the package](installation.md).
- [Export your model](export-model.md) as TensorFlow Lite.
- For the camera program, install OpenCV (`pip install opencv-python`, or `apt` on Raspberry Pi).

Your folder should look like this:

```text
my-project/
├── model.tflite
├── labels.txt
└── app.py
```

## Your first classifier

Save this as `app.py`:

```python
--8<-- "examples/webcam.py"
```

Run it:

```bash
python app.py
```

Show your camera one of the things you trained the model on. The class name and confidence appear
at the bottom of the video. To quit, click the video window and press ++esc++.

!!! note "Using the Raspberry Pi camera module?"

    `cv.VideoCapture(0)` does not work with it on recent Raspberry Pi OS. Use the
    [Raspberry Pi camera example](../examples/picamera2.md) instead.

## What each part does

| Code | Purpose |
|---|---|
| `TeachableMachineLite(...)` | Loads the model and its labels once. |
| `cv.VideoCapture(0)` | Opens the first camera. |
| `cap.read()` | Reads one frame. `ok` is `False` if that failed. |
| `model.classify_and_show(frame)` | Classifies the frame and returns the result and an annotated copy. |
| `cv.imshow(...)` | Shows the annotated frame. |

## Act on the result

The result is a dictionary. Check the confidence before you act on the label:

```python
result = model.classify_image(frame)

if result["label"] == "Cat" and result["confidence"] > 80:
    print("There's a cat!")
```

`confidence` is a percentage from 0 to 100. See [Results](../reference/results.md) for every key.

## Next steps

- Learn what you can pass in with [Images and Camera Frames](../guides/images-and-frames.md).
- Make it faster with [Speed and Accuracy](../guides/performance.md).
- Try the other [Examples](../examples/index.md).
