# Troubleshooting

## `error: externally-managed-environment`

Raspberry Pi OS Bookworm and later (and some other Linux systems) block `pip install` into the
system Python. [Install in a virtual environment](../getting-started/installation.md#how-to-install-on-raspberry-pi-os).

## `needs a TensorFlow Lite runtime`

The package couldn't import any runtime. Install the one for your system:

```bash
pip install ai-edge-litert    # 64-bit systems
pip install tflite-runtime    # 32-bit Raspberry Pi OS, Python 3.11 or earlier
```

If you use a virtual environment, check that it's activated.

## NumPy errors on 32-bit Raspberry Pi OS

`tflite-runtime` is compiled against NumPy 1.x and crashes with NumPy 2:

```bash
pip install "numpy<2"
```

## `FileNotFoundError`

Check `model_path` and `labels_file_path`. Relative paths start from the folder you run the
program in, which isn't always the folder the script is in. A floating point export is named
`model_unquant.tflite`, not `model.tflite`.

## `RuntimeError: Error loading model`

The file exists but the runtime can't load it. Make sure it's the `.tflite` file from the
**TensorFlow Lite** export tab and that the download is complete. A Keras `.h5` file needs the
[Teachable Machine](https://meqdaddev.github.io/teachable-machine/) package instead.

## The labels are wrong

Use the `labels.txt` from the same download as the model. The package logs a warning when the
number of labels doesn't match the model's classes.

## Predictions are poor

- With a NumPy array, check the [color order](images-and-frames.md#color-order).
- Check the [frame shape](images-and-frames.md#frame-shape).
- With a floating point model, make sure you have version 1.3.0 or later: earlier versions gave
  wrong predictions for these models. Run `pip install --upgrade teachable-machine-lite`.

## `TypeError: Expected a uint8 image array`

The array holds another type, often floats from 0 to 1. Pass the camera frame as it is, or convert
an array that uses the 0–255 range with `array.astype("uint8")`.

## `ValueError: top_k must be between 1 and ...`

`top_k` can't be larger than the number of classes in your model.

## The camera doesn't open

Try another camera index, such as `cv.VideoCapture(1)`, and close other programs that use the
camera. For the Raspberry Pi camera module, use [Picamera2](../examples/picamera2.md).

## See log messages

The package reports through Python's `logging` module instead of printing. That includes which
runtime loaded the model, and a warning if `labels.txt` doesn't match the model. To see the
messages, add this at the top of your script:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Still stuck?

[Open an issue](https://github.com/MeqdadDev/teachable-machine-lite/issues) with the full error
message, your system, your Python version, and the runtime name from
`teachable_machine_lite.RUNTIME`.
