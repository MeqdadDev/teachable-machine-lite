"""
Regenerate the tiny `.tflite` test models in this directory (requires `pip install tensorflow`):

    python tests/fixtures/make_models.py

Each model mimics a Teachable Machine image export (1x224x224x3 input, softmax output)
and classifies by average colour into the classes in labels.txt: Red, Green, Blue, Dark.
Inputs are expected normalized to [-1, 1], so a black image is "Dark" only when the
package normalizes float inputs correctly.
"""
from pathlib import Path

import numpy as np
import tensorflow as tf

OUT_DIR = Path(__file__).parent
SIZE = 224

# Rows: mean R, G, B of the normalized input. Columns: Red, Green, Blue, Dark.
WEIGHTS = tf.constant(
    [
        [4.0, 0.0, 0.0, -4.0 / 3],
        [0.0, 4.0, 0.0, -4.0 / 3],
        [0.0, 0.0, 4.0, -4.0 / 3],
    ]
)


@tf.function(input_signature=[tf.TensorSpec([1, SIZE, SIZE, 3], tf.float32)])
def classify(image):
    mean_rgb = tf.reduce_mean(image, axis=[1, 2])
    return tf.nn.softmax(tf.matmul(mean_rgb, WEIGHTS))


def representative_dataset():
    rng = np.random.default_rng(0)
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0), (255, 255, 255)]
    for colour in colours:
        pixels = np.full((1, SIZE, SIZE, 3), colour, dtype=np.float32)
        yield [pixels / 127.5 - 1.0]
    for _ in range(20):
        yield [rng.uniform(-1.0, 1.0, (1, SIZE, SIZE, 3)).astype(np.float32)]


def convert(io_type=None):
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [classify.get_concrete_function()], classify)
    if io_type is not None:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = io_type
        converter.inference_output_type = io_type
    return converter.convert()


if __name__ == "__main__":
    for name, io_type in [("model_float", None), ("model_uint8", tf.uint8), ("model_int8", tf.int8)]:
        path = OUT_DIR / f"{name}.tflite"
        path.write_bytes(convert(io_type))
        print(f"wrote {path} ({path.stat().st_size} bytes)")
