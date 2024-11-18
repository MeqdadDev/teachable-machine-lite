from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="teachable_machine_lite",
    version="1.2.0.1",
    description="A lightweight Python package optimized for integrating exported models from Google's Teachable Machine Platform into robotics and embedded systems environments. This streamlined version of Teachable Machine Package is specifically designed for resource-constrained devices, making it easier to deploy and use your trained models in embedded applications. With a focus on efficiency and minimal dependencies, this tool maintains the core functionality while being more suitable for robotics and IoT projects.",
    py_modules=["teachable_machine_lite"],
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
        "tflite-runtime",
        "Pillow"
    ],
    python_requires=">=3.7",
    url="https://github.com/MeqdadDev/teachable-machine-lite/",
    author="Meqdad Dev",
    author_email="meqdad.darweesh@gmail.com",
)