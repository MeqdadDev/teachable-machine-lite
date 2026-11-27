# Upgrading from 1.2

```bash
pip install --upgrade teachable-machine-lite
```

Most code keeps working. Check these points:

| What changed | What to do |
|---|---|
| **Labels** no longer include the number: `"Class 1"`, not `"0 Class 1"`. | If your code compares labels with the number, such as `result["label"] == "0 Cat"`, change it to `"Cat"`. |
| The unused **`model_type`** constructor parameter was removed. | Delete `model_type="tflite"` if your code passes it. |
| **NumPy 2** works. | Remove your `numpy==1.26.4` pin (except on 32-bit Raspberry Pi OS). |
| The runtime is now **LiteRT** (`ai-edge-litert`). | Nothing: it's installed automatically. The old `tflite-runtime` can stay installed. |
| **Frames can be passed directly.** | Optional: drop the `cv.imwrite("screenshot.jpg", img)` step. Passing a file path still works. |
| **Floating point models** now work correctly. | Nothing, if they gave wrong predictions before. |

See the [Changelog](../project/changelog.md) for the full list.
