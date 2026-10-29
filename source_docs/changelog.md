# Changelog

All notable changes to this project are documented in this file.

## [1.3.0] - 2026-10-29

### Changed

- **Runtime switched from `tflite-runtime` to LiteRT (`ai-edge-litert`)**. `tflite-runtime` has not been released since 2023 and has no wheels for Python 3.12+, so the package could not be installed on current systems such as Raspberry Pi OS Trixie (Python 3.13). LiteRT, Google's renamed TensorFlow Lite runtime, provides wheels for Python 3.9–3.14 on 64-bit Raspberry Pi OS, Linux, Windows and macOS. On 32-bit Raspberry Pi OS (`armv7l`), where LiteRT has no wheels, `tflite-runtime` is still installed (Python ≤ 3.11). The package imports whichever runtime is available, falling back to `tensorflow.lite`, and exposes the one in use as `teachable_machine_lite.RUNTIME`.
- **NumPy 2 supported**: the `numpy==1.26.4` pin is replaced by `numpy>=1.23.2`. The pin was only needed because `tflite-runtime` is compiled against NumPy 1.x, not because of Teachable Machine models.
- Labels: the `"<index> "` prefix Teachable Machine writes on each line of `labels.txt` (e.g. `0 Class 1`) is now stripped, so `result["label"]` is `"Class 1"` instead of `"0 Class 1"`. Labels are read as UTF-8, so non-English class names (e.g. Arabic) work on Windows too.
- `python_requires` is now `>=3.9` (it said 3.7 while the docs said 3.9). Requires `Pillow>=10.1`.
- Removed the unused `model_type` constructor parameter.
- Messages go through `logging` (logger `teachable_machine_lite`) only; the model-loaded message is no longer printed.

### Added

- `classify_image`, `show_prediction_on_image` and `classify_and_show` accept a PIL image or a NumPy array (OpenCV / Picamera2 frame) as well as a file path, so camera frames no longer need to be written to disk first. NumPy arrays are BGR by default; pass `color_order="RGB"` for RGB arrays. Grayscale and 4-channel arrays are accepted.
- `num_threads` constructor parameter, passed to the interpreter (e.g. `num_threads=4` on Raspberry Pi 4/5).
- `top_k` parameter on `classify_image` and `classify_and_show`; results now include a `"predictions"` list with the `top_k` most likely classes.
- A warning is logged when the number of labels doesn't match the model's outputs.
- Test suite with small generated `.tflite` models (float, uint8 and int8), plus a CI workflow on Linux x86_64/ARM64, Windows and macOS.
- Docs: Raspberry Pi installation guide (virtual environment, Picamera2), Picamera2 example, command-line webcam example, upgrade guide, changelog and releasing pages. The docs site is now deployed automatically by GitHub Actions.

### Fixed

- **Floating point models gave wrong predictions**: raw 0–255 pixels were copied into the float input tensor. Float inputs are now normalized to [-1, 1] as in Teachable Machine's training, so both the "Floating point" and "Quantized" exports work.
- **int8 quantized outputs** were not dequantized (only uint8 was); uint8 dequantization could also overflow with a non-zero zero point.
- **`show_prediction_on_image` crashed on Windows and macOS** with `OSError: cannot open resource`, because the `DejaVuSans-Bold.ttf` font can't be found by name there. It now falls back to Pillow's built-in font.
- The label background is now actually translucent (it was drawn opaque black) and covers letters with descenders.
- Results contain plain Python `int`/`float` instead of NumPy scalars, so they can be serialized with `json.dumps`.
- Image files are closed after reading, so the same file (e.g. `screenshot.jpg`) can be overwritten every frame on Windows.

### Build

- Version is defined only in `src/__init__.py`; `setup.py` reads it from there (`MANIFEST.in` includes it in the sdist).
- The PyPI publish workflow runs when a GitHub Release is published and builds with `python -m build` (instead of the deprecated `setup.py sdist bdist_wheel`).

## [1.2.0.2]

- Pinned `numpy==1.26.4` to work around NumPy 2 incompatibility; restructured the documentation.

## [1.2.0.1]

- Added the PyPI publishing workflow.

## [1.1]

- Initial releases of the `TeachableMachineLite` class for image classification with TensorFlow Lite models.
