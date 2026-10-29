"""
Teachable Machine Lite
======================

Run your Google Teachable Machine image models on Raspberry Pi, robots and other edge
devices in a few lines of Python. Lightweight, powered by LiteRT, no TensorFlow required.

Example
-------
>>> from teachable_machine_lite import TeachableMachineLite
>>> model = TeachableMachineLite(model_path="model.tflite", labels_file_path="labels.txt")
>>> result = model.classify_image("photo.jpg")
>>> print(result["label"], result["confidence"])

Documentation: https://meqdaddev.github.io/teachable-machine-lite/
Source code: https://github.com/MeqdadDev/teachable-machine-lite
"""
import functools
import logging
import os
import re
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from ai_edge_litert.interpreter import Interpreter

    RUNTIME = "ai-edge-litert"
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter

        RUNTIME = "tflite-runtime"
    except ImportError:
        try:
            import tensorflow as tf

            Interpreter = tf.lite.Interpreter
            RUNTIME = "tensorflow"
        except ImportError as e:
            raise ImportError(
                "teachable_machine_lite needs a TensorFlow Lite runtime. Install one of:\n"
                "  pip install ai-edge-litert   (64-bit OS: Raspberry Pi OS 64-bit, Linux, Windows, macOS)\n"
                "  pip install tflite-runtime   (32-bit Raspberry Pi OS, Python <= 3.11 only)"
            ) from e

logger = logging.getLogger(__name__)

_LABEL_INDEX_PREFIX = re.compile(r"^(\d+)\s+(.+)$")


@functools.lru_cache(maxsize=8)
def _load_font(size: int):
    try:
        # Not resolvable by bare name on every platform (notably Windows/macOS).
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


class TeachableMachineLite:
    """Classify images with a TensorFlow Lite model exported from Google's Teachable Machine."""

    def __init__(
        self,
        model_path: str = "model.tflite",
        labels_file_path: str = "labels.txt",
        num_threads: int = None,
    ) -> None:
        """
        Load an exported Teachable Machine TensorFlow Lite model and its labels.

        Both the quantized and the floating point TensorFlow Lite exports are supported.

        Args:
            model_path (str): Path to the `.tflite` model file. Default is 'model.tflite'.
            labels_file_path (str): Path to the `labels.txt` file exported with the model.
                Default is 'labels.txt'.
            num_threads (int, optional): Number of CPU threads the interpreter may use.
                Setting it to the number of cores (4 on Raspberry Pi 4/5) speeds up
                inference. Default is None (the runtime's default).

        Raises:
            FileNotFoundError: If the model or labels file is not found.
            RuntimeError: If the model cannot be loaded by the TensorFlow Lite runtime.
        """
        self.model_path = model_path
        self.label_path = labels_file_path

        self._labels = self._load_labels(labels_file_path)

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            self.interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
            self.interpreter.allocate_tensors()
        except Exception as e:
            raise RuntimeError(f"Error loading model '{model_path}': {e}") from e

        self._input_details = self.interpreter.get_input_details()[0]
        self._output_details = self.interpreter.get_output_details()[0]
        _, height, width, _ = self._input_details["shape"]
        self.height, self.width = int(height), int(width)

        num_outputs = int(np.prod(self._output_details["shape"]))
        if num_outputs != len(self._labels):
            logger.warning(
                "Model '%s' has %d outputs but '%s' has %d labels.",
                model_path, num_outputs, labels_file_path, len(self._labels),
            )

        logger.info("TeachableMachineLite model '%s' loaded using %s.", model_path, RUNTIME)

    def classify_image(self, image_path, calc_time: bool = False, top_k: int = 1, color_order: str = "BGR"):
        """
        Classify an image and return the classification results.

        Args:
            image_path (str, os.PathLike, PIL.Image.Image or np.ndarray): The image to classify:
                a file path, a PIL image, or a NumPy array such as an OpenCV frame or a Picamera2
                `capture_array()` result. Passing frames directly avoids writing every frame to disk.
            calc_time (bool, optional): If True, measures and returns the time taken for the
                classification. Default is False.
            top_k (int, optional): How many of the most likely classes to include in
                `"predictions"`. Default is 1.
            color_order (str, optional): Channel order of a NumPy array input, `"BGR"` (OpenCV,
                Picamera2 `"RGB888"`/`"XBGR8888"`) or `"RGB"`. Ignored for file paths and PIL
                images. Default is `"BGR"`.

        Returns:
            dict: The classification results, with these keys:

                - `"id"` (int): index of the predicted label.
                - `"label"` (str): the predicted label.
                - `"time"` (float or None): classification time in seconds (3 decimals),
                  or None if `calc_time` is False.
                - `"confidence"` (float): confidence of the prediction as a percentage.
                - `"highest_class_id"` (int): same as `"id"`.
                - `"highest_class_prob"` (float): same as `"confidence"`.
                - `"predictions"` (list of dict): the `top_k` most likely classes, best first,
                  each with `"id"`, `"label"` and `"confidence"`.

        Raises:
            ValueError: If `top_k` is out of range, `color_order` is invalid, or a NumPy array
                has an unsupported shape.
            TypeError: If a NumPy array is not of dtype uint8.
        """
        image = self._to_pil(image_path, color_order)

        start_time = time.perf_counter()
        predictions = self._predict(self._prepare_input(image), top_k)
        classification_time = round(time.perf_counter() - start_time, 3) if calc_time else None

        predictions = [
            {"id": class_id, "label": self._labels[class_id], "confidence": round(prob * 100, 2)}
            for class_id, prob in predictions
        ]
        best = predictions[0]

        return {
            "id": best["id"],
            "label": best["label"],
            "time": classification_time,
            "confidence": best["confidence"],
            "highest_class_id": best["id"],
            "highest_class_prob": best["confidence"],
            "predictions": predictions,
        }

    def show_prediction_on_image(
        self,
        image_path,
        classification_result=None,
        calc_time: bool = False,
        convert_to_bgr: bool = True,
        color_order: str = "BGR",
    ):
        """
        Draw the prediction results on the input image and return the modified image.

        Args:
            image_path (str, os.PathLike, PIL.Image.Image or np.ndarray): The input image, as a
                file path, a PIL image, or a NumPy array (see `classify_image`).
            classification_result (dict, optional): Pre-computed classification result. If not
                provided, the image is classified with `classify_image`.
            calc_time (bool, optional): Whether to calculate the time taken to classify the image.
                Default is False.
            convert_to_bgr (bool, optional): Whether to convert the output image from RGB to BGR
                format, for use with OpenCV. Default is True.
            color_order (str, optional): Channel order of a NumPy array input, `"BGR"` or `"RGB"`.
                Default is `"BGR"`.

        Returns:
            np.ndarray or PIL.Image.Image: The image with the prediction results drawn on it: a NumPy
                array in BGR format if `convert_to_bgr` is True, otherwise a PIL image in RGB format.
        """
        image = self._to_pil(image_path, color_order)

        if classification_result is None:
            classification_result = self.classify_image(image, calc_time)

        text = f"{classification_result['label']}: {classification_result['confidence']:.2f}%"
        image = self._draw_label(image, text)

        if convert_to_bgr:
            return np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
        return image

    def classify_and_show(self, image_path, convert_to_bgr: bool = True, top_k: int = 1, color_order: str = "BGR"):
        """
        Classify an image and show the prediction results on the image.

        Args:
            image_path (str, os.PathLike, PIL.Image.Image or np.ndarray): The input image, as a
                file path, a PIL image, or a NumPy array (see `classify_image`).
            convert_to_bgr (bool, optional): Whether to convert the image to BGR format for OpenCV.
                If False, the image is returned in RGB format. Default is True.
            top_k (int, optional): How many of the most likely classes to include in the
                result's `"predictions"`. Default is 1.
            color_order (str, optional): Channel order of a NumPy array input, `"BGR"` or `"RGB"`.
                Default is `"BGR"`.

        Returns:
            tuple: `(classification_result, image_with_prediction)`, where `classification_result`
                is the dict returned by `classify_image` and `image_with_prediction` is the image
                with the results drawn on it (see `show_prediction_on_image`).
        """
        image = self._to_pil(image_path, color_order)
        classification_result = self.classify_image(image, top_k=top_k)
        image_with_prediction = self.show_prediction_on_image(
            image, classification_result, convert_to_bgr=convert_to_bgr)
        return classification_result, image_with_prediction

    def _prepare_input(self, image):
        """
        Resize an RGB PIL image to the model's input size and convert it to the model's input type.

        Quantized models take raw 0-255 pixels; floating point models take pixels
        normalized to [-1, 1], as in Teachable Machine's training pipeline.
        """
        pixels = np.asarray(image.resize((self.width, self.height)), dtype=np.uint8)
        input_dtype = self._input_details["dtype"]

        if input_dtype == np.uint8:
            data = pixels
        elif input_dtype == np.int8:
            data = (pixels.astype(np.int16) - 128).astype(np.int8)
        else:
            data = (pixels.astype(np.float32) / 127.5 - 1.0).astype(input_dtype)

        return np.expand_dims(data, axis=0)

    def _predict(self, input_tensor, top_k: int = 1):
        """
        Run inference on a prepared input tensor.

        Returns:
            list of (int, float): The `top_k` (class_id, probability) pairs, most likely first.

        Raises:
            ValueError: If `top_k` is not between 1 and the number of classes.
        """
        self.interpreter.set_tensor(self._input_details["index"], input_tensor)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self._output_details["index"]).reshape(-1)
        scale, zero_point = self._output_details["quantization"]
        if np.issubdtype(output.dtype, np.integer) and scale:
            output = scale * (output.astype(np.float32) - zero_point)
        output = output.astype(np.float32)

        if not 1 <= top_k <= output.size:
            raise ValueError(f"top_k must be between 1 and {output.size}, got {top_k}")

        top_indices = np.argsort(-output, kind="stable")[:top_k]
        return [(int(i), float(output[i])) for i in top_indices]

    @staticmethod
    def _to_pil(image, color_order: str = "BGR"):
        """
        Convert a file path, PIL image, or NumPy array into an RGB PIL image.

        Raises:
            ValueError: If `color_order` is not "BGR"/"RGB" or the array shape is unsupported.
            TypeError: If a NumPy array is not of dtype uint8.
        """
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, np.ndarray):
            order = color_order.upper()
            if order not in ("BGR", "RGB"):
                raise ValueError(f"color_order must be 'BGR' or 'RGB', got {color_order!r}")
            if image.dtype != np.uint8:
                raise TypeError(f"Expected a uint8 image array, got dtype {image.dtype}")
            if image.ndim == 2:
                return Image.fromarray(image).convert("RGB")
            if image.ndim == 3 and image.shape[2] in (3, 4):
                # A 4th channel (alpha/padding, e.g. Picamera2 "XBGR8888") is dropped.
                pixels = image[:, :, :3]
                if order == "BGR":
                    pixels = pixels[:, :, ::-1]
                return Image.fromarray(np.ascontiguousarray(pixels))
            raise ValueError(f"Unsupported image array shape {image.shape}")

        with Image.open(image) as img:
            return img.convert("RGB")

    @staticmethod
    def _draw_label(image, text: str):
        """Draw `text` on a translucent box in the bottom-left corner of an RGB PIL image."""
        font = _load_font(max(12, int(image.height * 0.04)))
        margin, padding = 10, 4

        draw = ImageDraw.Draw(image)
        _, _, _, text_bottom = draw.textbbox((0, 0), text, font=font)
        position = (margin + padding, image.height - text_bottom - margin - padding)
        # Measured at the draw position so the box covers descenders ("g", "%").
        left, top, right, bottom = draw.textbbox(position, text, font=font)

        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        ImageDraw.Draw(overlay).rectangle(
            [left - padding, top - padding, right + padding, bottom + padding], fill=(0, 0, 0, 160))
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

        ImageDraw.Draw(image).text(position, text, font=font, fill=(255, 255, 255))
        return image

    @staticmethod
    def _load_labels(labels_file_path: str):
        """
        Load class labels, one per line, from a Teachable Machine `labels.txt` file.

        Teachable Machine writes each line as "<index> <class name>"; the index prefix
        is stripped when it matches the line's position.

        Raises:
            FileNotFoundError: If the labels file does not exist.
        """
        with open(labels_file_path, "r", encoding="utf-8-sig") as file:
            lines = [line.strip() for line in file if line.strip()]

        labels = []
        for index, line in enumerate(lines):
            match = _LABEL_INDEX_PREFIX.match(line)
            labels.append(match.group(2) if match and int(match.group(1)) == index else line)
        return labels
