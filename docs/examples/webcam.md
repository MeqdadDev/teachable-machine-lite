# Webcam Classifier

Classifies frames from a webcam and shows the result on the video, using `classify_and_show`.

## Code

```python title="examples/webcam.py"
--8<-- "examples/webcam.py"
```

## How it works

- The model is loaded once, before the loop.
- Each frame goes straight to the model, without being saved to a file.
- `classify_and_show` returns the [result](../reference/results.md) and an annotated copy of the frame.
