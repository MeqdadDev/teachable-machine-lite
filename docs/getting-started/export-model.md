# Export Your Model

The package runs **image classification** models exported from Google's Teachable Machine in
TensorFlow Lite format.

## Steps

1. Train your image model on [Teachable Machine](https://teachablemachine.withgoogle.com/).
2. Click **Export Model**, then open the **TensorFlow Lite** tab.
3. Choose **Quantized** or **Floating point**. Both are supported.
4. Download and unzip. Put the model file and `labels.txt` next to your code.

## Which type should you choose?

| Export | Model file | Good for |
|---|---|---|
| **Quantized** | `model.tflite` | Raspberry Pi and small devices: smaller and faster |
| **Floating point** | `model_unquant.tflite` | Slightly more accurate |

`model.tflite` is the package's default `model_path`. For a floating point export, pass its name:

```python
model = TeachableMachineLite(model_path="model_unquant.tflite", labels_file_path="labels.txt")
```

## The labels file

Each line of `labels.txt` holds a class number and a class name:

```text
0 Cat
1 Dog
```

The package strips the leading number, so results contain just the name (`"Cat"`). The number is
removed only when it matches the line's position, starting from 0. Class names can be in any
language, such as Arabic, because the file is read as UTF-8.

!!! warning "Keep the files together"

    Always use the `labels.txt` that came in the same download as the model. If the number of labels
    doesn't match the model's classes, the package [logs a warning](../guides/troubleshooting.md#see-log-messages).
