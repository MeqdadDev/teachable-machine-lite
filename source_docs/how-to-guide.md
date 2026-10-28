# How-To Guide

## How to install the package

```bash
pip install teachable-machine-lite
```

## How to install on Raspberry Pi OS

Raspberry Pi OS Bookworm and later don't allow `pip install` into the system Python
(you'll see `error: externally-managed-environment`). Create a virtual environment first.
`--system-site-packages` lets it use the Picamera2 and OpenCV packages installed with `apt`:

```bash
sudo apt install -y python3-picamera2 python3-opencv
python -m venv --system-site-packages ~/tm-env
source ~/tm-env/bin/activate
pip install teachable-machine-lite
```

Run `source ~/tm-env/bin/activate` again in every new terminal before running your code.

Check which runtime was installed:

```bash
python -c "import teachable_machine_lite as t; print(t.RUNTIME)"
```

## How to export a model from Teachable Machine

1. Train your image model on [Teachable Machine](https://teachablemachine.withgoogle.com/).
2. Click **Export Model**, then open the **TensorFlow Lite** tab.
3. Choose **Quantized** (smaller and faster, recommended for Raspberry Pi) or **Floating point**
   (slightly more accurate). Both are supported.
4. Download and unzip. Put the model file and `labels.txt` next to your code. The quantized model
   is named `model.tflite`; the floating point one is `model_unquant.tflite`, so pass that name as
   `model_path`.

Each line of `labels.txt` looks like `0 Class 1`. The package strips the leading index, so results
contain just the class name (`"Class 1"`). Class names can be in any language, e.g. Arabic, since the
file is read as UTF-8.

## How to classify camera frames without saving them to disk

`classify_image`, `show_prediction_on_image` and `classify_and_show` accept a file path,
a PIL image, or a NumPy array. Passing the camera frame directly is faster than writing it to a
file first and avoids wearing out the Raspberry Pi's SD card:

```python
results, annotated = model.classify_and_show(frame)  # frame from OpenCV or Picamera2
```

NumPy arrays are treated as **BGR** by default, which matches OpenCV frames and Picamera2's
`"RGB888"` / `"XBGR8888"` formats. Pass `color_order="RGB"` for RGB arrays
(e.g. Picamera2's `"BGR888"` format).

## How to make inference faster on Raspberry Pi

Let the interpreter use all four CPU cores of a Raspberry Pi 4/5:

```python
model = TeachableMachineLite(model_path="model.tflite", labels_file_path="labels.txt", num_threads=4)
```

The quantized export is also noticeably faster than the floating point one.

## How to get the top predictions

```python
result = model.classify_image(frame, top_k=3)
for prediction in result["predictions"]:
    print(prediction["label"], prediction["confidence"])
```

## How to see log messages

The package reports through Python's `logging` module instead of printing. That includes which
runtime loaded the model, and a warning if `labels.txt` doesn't match the model's number of classes.
To see them, add this at the top of your script:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## How to upgrade from version 1.2

```bash
pip install --upgrade teachable-machine-lite
```

Your existing code keeps working. Things to know:

- **NumPy:** you no longer need `numpy==1.26.4`; NumPy 2 works. The old `tflite-runtime` package can
  stay installed. The package prefers the newer `ai-edge-litert`, which is installed automatically.
- **Labels:** `result["label"]` is now `"Class 1"` instead of `"0 Class 1"`. If your code compared
  labels including the number (e.g. `if result["label"] == "0 Cat":`), change it to `"Cat"`.
- **`model_type`:** this unused constructor parameter was removed. Delete `model_type="tflite"` if
  your code passes it.
- **No more file saving:** you can drop the `cv.imwrite("screenshot.jpg", img)` step and pass the
  camera frame directly, e.g. `model.classify_and_show(frame)`. Passing a file path still works.
- **Floating point models:** if a floating point export gave wrong predictions before, it now works
  correctly.

See the [Changelog](changelog.md) for the full list of changes.
