# Installation

## Install the package { #how-to-install-the-package }

```bash
pip install teachable-machine-lite
```

pip picks the right LiteRT runtime for your system automatically. See
[Requirements](requirements.md) for supported platforms and Python versions.

The webcam examples also need OpenCV:

```bash
pip install opencv-python
```

## Install on Raspberry Pi OS { #how-to-install-on-raspberry-pi-os }

Raspberry Pi OS Bookworm and later don't allow `pip install` into the system Python
(you'll see `error: externally-managed-environment`). Create a virtual environment first.
`--system-site-packages` lets it use the Picamera2 and OpenCV packages installed with `apt`,
so you don't need `opencv-python` from pip:

```bash
sudo apt install -y python3-picamera2 python3-opencv
python -m venv --system-site-packages ~/tm-env
source ~/tm-env/bin/activate
pip install teachable-machine-lite
```

Run `source ~/tm-env/bin/activate` again in every new terminal before running your code.

## Check the installation

This prints the TensorFlow Lite runtime the package is using:

```bash
python -c "import teachable_machine_lite as t; print(t.RUNTIME)"
```

On most systems it prints `ai-edge-litert`.

## Next step

[Export your model](export-model.md), then follow the [Quick Start](quickstart.md).
