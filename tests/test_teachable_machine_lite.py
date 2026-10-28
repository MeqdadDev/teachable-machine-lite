import json
import logging
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFont

import teachable_machine_lite
from teachable_machine_lite import TeachableMachineLite

FIXTURES = Path(__file__).parent / "fixtures"
LABELS = FIXTURES / "labels.txt"
MODELS = ["model_float.tflite", "model_uint8.tflite", "model_int8.tflite"]

RED, GREEN, BLUE, BLACK = (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)


def solid_rgb(colour, size=(64, 48)):
    return Image.new("RGB", size, colour)


@pytest.fixture(params=MODELS)
def model(request):
    return TeachableMachineLite(model_path=str(FIXTURES / request.param), labels_file_path=str(LABELS))


@pytest.fixture
def float_model():
    return TeachableMachineLite(model_path=str(FIXTURES / "model_float.tflite"), labels_file_path=str(LABELS))


@pytest.mark.parametrize(
    "colour, expected", [(RED, "Red"), (GREEN, "Green"), (BLUE, "Blue"), (BLACK, "Dark")])
def test_classifies_pil_images(model, colour, expected):
    assert model.classify_image(solid_rgb(colour))["label"] == expected


def test_classifies_image_files(model, tmp_path):
    path = tmp_path / "green.jpg"
    solid_rgb(GREEN).save(path)

    assert model.classify_image(str(path))["label"] == "Green"
    assert model.classify_image(path)["label"] == "Green"


def test_numpy_arrays_default_to_bgr(model):
    red_bgr = np.zeros((48, 64, 3), dtype=np.uint8)
    red_bgr[:, :, 2] = 255

    assert model.classify_image(red_bgr)["label"] == "Red"
    assert model.classify_image(red_bgr, color_order="RGB")["label"] == "Blue"


def test_accepts_four_channel_and_grayscale_arrays(model):
    xbgr = np.zeros((48, 64, 4), dtype=np.uint8)
    xbgr[:, :, 1] = 255
    xbgr[:, :, 3] = 255

    assert model.classify_image(xbgr)["label"] == "Green"
    assert model.classify_image(np.zeros((48, 64), dtype=np.uint8))["label"] == "Dark"


def test_result_shape_and_plain_python_types(model):
    result = model.classify_image(solid_rgb(BLUE))

    assert result["id"] == result["highest_class_id"] == 2
    assert result["confidence"] == result["highest_class_prob"]
    assert 50 < result["confidence"] <= 100
    assert result["time"] is None
    assert type(result["id"]) is int and type(result["confidence"]) is float
    assert result["predictions"] == [{"id": 2, "label": "Blue", "confidence": result["confidence"]}]
    json.dumps(result)


def test_quantized_outputs_are_dequantized_to_probabilities(model):
    result = model.classify_image(solid_rgb(RED), top_k=4)

    total = sum(p["confidence"] for p in result["predictions"])
    assert total == pytest.approx(100, abs=2)


def test_top_k_returns_sorted_predictions(model):
    predictions = model.classify_image(solid_rgb(RED), top_k=3)["predictions"]

    assert len(predictions) == 3
    assert predictions[0]["label"] == "Red"
    confidences = [p["confidence"] for p in predictions]
    assert confidences == sorted(confidences, reverse=True)


@pytest.mark.parametrize("top_k", [0, 5])
def test_top_k_out_of_range_raises(float_model, top_k):
    with pytest.raises(ValueError):
        float_model.classify_image(solid_rgb(RED), top_k=top_k)


def test_calc_time(float_model):
    result = float_model.classify_image(solid_rgb(RED), calc_time=True)

    assert isinstance(result["time"], float) and result["time"] >= 0


def test_invalid_array_inputs(float_model):
    with pytest.raises(TypeError):
        float_model.classify_image(np.zeros((48, 64, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        float_model.classify_image(np.zeros((48, 64, 2), dtype=np.uint8))
    with pytest.raises(ValueError):
        float_model.classify_image(np.zeros((48, 64, 3), dtype=np.uint8), color_order="HSV")


def test_show_prediction_returns_bgr_array_of_original_size(float_model):
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[:, :, 0] = 255  # blue, in BGR
    original = frame.copy()

    annotated = float_model.show_prediction_on_image(frame)

    assert annotated.shape == frame.shape
    assert annotated.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(frame, original)
    # Top-right is untouched and still blue in BGR; the label sits bottom-left.
    assert tuple(annotated[0, -1]) == (255, 0, 0)
    assert not np.array_equal(annotated[-20:, :40], frame[-20:, :40])


def test_show_prediction_can_return_pil_without_mutating_input(float_model):
    image = solid_rgb(RED, size=(160, 120))
    result = float_model.classify_image(image)

    annotated = float_model.show_prediction_on_image(image, result, convert_to_bgr=False)

    assert isinstance(annotated, Image.Image) and annotated.mode == "RGB"
    assert annotated.size == image.size
    assert image.getpixel((5, 115)) == RED
    assert annotated.getpixel((159, 0)) == RED
    assert annotated.getpixel((12, 105)) != RED


def test_label_background_is_translucent(float_model):
    annotated = float_model.show_prediction_on_image(solid_rgb(RED, size=(400, 300)), convert_to_bgr=False)

    # Corner of the label box, outside the glyphs: darkened red, not opaque black.
    r, g, b = annotated.getpixel((11, 300 - 11))
    assert 0 < r < 255 and g == 0 and b == 0


def test_font_falls_back_when_dejavu_is_missing(float_model, monkeypatch):
    real_truetype = ImageFont.truetype

    def missing_font(font, *args, **kwargs):
        # load_default() also calls truetype() with an in-memory font; only fail named fonts.
        if isinstance(font, str):
            raise OSError("cannot open resource")
        return real_truetype(font, *args, **kwargs)

    teachable_machine_lite._load_font.cache_clear()
    monkeypatch.setattr(ImageFont, "truetype", missing_font)
    try:
        annotated = float_model.show_prediction_on_image(solid_rgb(GREEN, size=(160, 120)))
    finally:
        teachable_machine_lite._load_font.cache_clear()

    assert annotated.shape == (120, 160, 3)


def test_classify_and_show(model, tmp_path):
    path = tmp_path / "red.png"
    solid_rgb(RED, size=(160, 120)).save(path)

    result, image = model.classify_and_show(str(path), top_k=2)

    assert result["label"] == "Red"
    assert len(result["predictions"]) == 2
    assert image.shape == (120, 160, 3)


def test_input_size_comes_from_model(float_model):
    assert (float_model.width, float_model.height) == (224, 224)


def test_num_threads_is_accepted():
    model = TeachableMachineLite(
        model_path=str(FIXTURES / "model_float.tflite"), labels_file_path=str(LABELS), num_threads=2)
    assert model.classify_image(solid_rgb(GREEN))["label"] == "Green"


def test_missing_files_raise_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        TeachableMachineLite(model_path=str(FIXTURES / "model_float.tflite"), labels_file_path=str(tmp_path / "nope.txt"))
    with pytest.raises(FileNotFoundError):
        TeachableMachineLite(model_path=str(tmp_path / "nope.tflite"), labels_file_path=str(LABELS))


def test_invalid_model_raises_runtime_error(tmp_path):
    bad_model = tmp_path / "bad.tflite"
    bad_model.write_bytes(b"not a model")

    with pytest.raises(RuntimeError):
        TeachableMachineLite(model_path=str(bad_model), labels_file_path=str(LABELS))


def test_label_count_mismatch_logs_warning(tmp_path, caplog):
    labels = tmp_path / "labels.txt"
    labels.write_text("0 Red\n1 Green\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="teachable_machine_lite"):
        TeachableMachineLite(model_path=str(FIXTURES / "model_float.tflite"), labels_file_path=str(labels))

    assert "4 outputs" in caplog.text


@pytest.mark.parametrize(
    "content, expected",
    [
        ("0 Class 1\n1 Class 2\n", ["Class 1", "Class 2"]),
        ("0 Class 1\r\n1 Class 2\r\n\r\n", ["Class 1", "Class 2"]),
        ("﻿0 قطة\n1 كلب\n", ["قطة", "كلب"]),
        ("cat\ndog\n", ["cat", "dog"]),
        ("3 apples\n7 pears\n", ["3 apples", "7 pears"]),
    ],
)
def test_load_labels(tmp_path, content, expected):
    labels = tmp_path / "labels.txt"
    labels.write_bytes(content.encode("utf-8"))

    assert TeachableMachineLite._load_labels(str(labels)) == expected
