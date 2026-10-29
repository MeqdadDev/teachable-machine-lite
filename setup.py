import re
from pathlib import Path

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def _read_version():
    init_contents = (Path(__file__).parent / "src" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_contents, re.M)
    if not match:
        raise RuntimeError("Unable to find __version__ in src/__init__.py")
    return match.group(1)


setup(
    name="teachable_machine_lite",
    version=_read_version(),
    description="Run your Google Teachable Machine image models on Raspberry Pi, robots and other edge devices in a few lines of Python. Lightweight, powered by LiteRT, no TensorFlow required.",
    py_modules=["teachable_machine_lite"],
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy>=1.23.2",
        "Pillow>=10.1",
        # LiteRT (formerly TensorFlow Lite) has no 32-bit ARM wheels; tflite-runtime still
        # covers 32-bit Raspberry Pi OS, but only up to Python 3.11.
        'ai-edge-litert; platform_machine != "armv7l"',
        'tflite-runtime; platform_machine == "armv7l" and python_version < "3.12"',
    ],
    python_requires=">=3.9",
    url="https://github.com/MeqdadDev/teachable-machine-lite/",
    project_urls={
        "Documentation": "https://meqdaddev.github.io/teachable-machine-lite/",
        "Changelog": "https://meqdaddev.github.io/teachable-machine-lite/changelog/",
        "Source": "https://github.com/MeqdadDev/teachable-machine-lite/",
    },
    author="Meqdad Dev",
    author_email="meqdad.darweesh@gmail.com",
)
