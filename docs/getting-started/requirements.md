# Requirements

## Python version

Python 3.9 or later.

## Dependencies

| Package | Version | When it's installed |
|---|---|---|
| `numpy` | 1.23.2 or later (NumPy 1.x and 2.x both work) | Always |
| `Pillow` | 10.1 or later | Always |
| `ai-edge-litert` (LiteRT, Google's TensorFlow Lite runtime) | Latest | Everywhere except 32-bit ARM (`armv7l`) |
| `tflite-runtime` | Latest | 32-bit Raspberry Pi OS with Python 3.11 or earlier |

`pip install teachable-machine-lite` picks the right runtime for your system automatically.

## Supported platforms

| Platform | Runtime installed | Python |
|---|---|---|
| Raspberry Pi OS **64-bit** (Pi 3/4/5, Zero 2 W) | `ai-edge-litert` | 3.9 – 3.14 |
| Raspberry Pi OS **32-bit** | `tflite-runtime` | 3.9 – 3.11 only |
| Linux x86_64 / aarch64 | `ai-edge-litert` | 3.9 – 3.14 |
| Windows x86_64 | `ai-edge-litert` | 3.9 – 3.14 |
| macOS (Apple silicon) | `ai-edge-litert` | 3.9 – 3.14 |

On Raspberry Pi, the **64-bit** Raspberry Pi OS is recommended: it is supported on every current
Python version. On 32-bit Raspberry Pi OS Trixie (Python 3.13) there is no TensorFlow Lite runtime
available from PyPI.

If no runtime can be installed with pip on your system but TensorFlow is installed, the package
falls back to `tensorflow.lite`.

## About NumPy 2

Versions up to 1.2.0.2 pinned `numpy==1.26.4`. That pin was needed because `tflite-runtime`
(last released in 2023) is compiled against NumPy 1.x and crashes on import with NumPy 2.
It was never a limitation of Teachable Machine models. Since version 1.3.0 the package uses
LiteRT (`ai-edge-litert`), which works with NumPy 2, so no NumPy pin is needed.

If you are still on 32-bit Raspberry Pi OS with `tflite-runtime`, keep NumPy below 2:

```bash
pip install "numpy<2"
```
