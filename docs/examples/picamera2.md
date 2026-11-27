# Raspberry Pi Camera

Classifies frames from the Raspberry Pi camera module (CSI ribbon cable) through Picamera2. First follow the [Raspberry Pi installation steps](../getting-started/installation.md#how-to-install-on-raspberry-pi-os).

## Code

```python title="examples/picamera2_camera.py"
--8<-- "examples/picamera2_camera.py"
```

## How it works

- Picamera2's `"RGB888"` format stores pixels in BGR order, which is what the package expects by
  default. See [Color order](../guides/images-and-frames.md#color-order).
- `num_threads=4` lets the model use all four cores of a Raspberry Pi 4 or 5.
