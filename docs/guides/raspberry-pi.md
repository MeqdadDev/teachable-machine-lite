# Raspberry Pi

## Set up

Use the **64-bit** Raspberry Pi OS if you can, and follow the
[Raspberry Pi installation steps](../getting-started/installation.md#how-to-install-on-raspberry-pi-os).
Then check which runtime was installed:

```bash
python -c "import teachable_machine_lite as t; print(t.RUNTIME)"
```

## Choose a camera

| Camera | How to read it | Example |
|---|---|---|
| USB webcam | OpenCV, `cv.VideoCapture(0)` | [Webcam Classifier](../examples/webcam.md) |
| Camera module (CSI ribbon cable) | Picamera2 | [Raspberry Pi Camera](../examples/picamera2.md) |

`cv.VideoCapture(0)` does not work with the camera module on recent Raspberry Pi OS.

## Use all the CPU cores

```python
model = TeachableMachineLite(
    model_path="model.tflite", labels_file_path="labels.txt", num_threads=4)
```

`num_threads=4` matches the four cores of a Raspberry Pi 4 or 5.

## Spare the SD card

Pass camera frames to the model directly. There's no need to save each frame as an image file
first. See [Images and Camera Frames](images-and-frames.md).

## 32-bit Raspberry Pi OS

On 32-bit Raspberry Pi OS the package uses `tflite-runtime`, which supports Python 3.11 or earlier
and needs NumPy below 2. See [Requirements](../getting-started/requirements.md).
