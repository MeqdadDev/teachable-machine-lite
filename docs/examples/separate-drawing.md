# Separate Drawing

Classifies first and draws afterwards, so your code can use the result in between.

## Code

```python title="examples/separate_drawing.py"
--8<-- "examples/separate_drawing.py"
```

## How it works

- `classify_image` returns only the result dictionary, not an image.
- `calc_time=True` fills in the result's `time` key, in seconds.
- `show_prediction_on_image` draws that result on the frame without classifying it again.
