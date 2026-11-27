# Results

`classify_image` returns a dictionary:

```python
{
    "id": 0,
    "label": "Class 1",           # class name from labels.txt, without its number
    "time": None,                 # seconds, when calc_time=True
    "confidence": 97.66,          # percentage
    "highest_class_id": 0,        # same as "id"
    "highest_class_prob": 97.66,  # same as "confidence"
    "predictions": [              # top_k most likely classes, best first
        {"id": 0, "label": "Class 1", "confidence": 97.66},
    ],
}
```

| Key | Type | Meaning |
|---|---|---|
| `id` | `int` | Position of the predicted class in `labels.txt`, starting at 0 |
| `label` | `str` | Name of the predicted class |
| `time` | `float` or `None` | Classification time in seconds (3 decimals), or `None` unless `calc_time=True` |
| `confidence` | `float` | Confidence as a percentage, rounded to 2 decimals |
| `highest_class_id` | `int` | Same as `id` |
| `highest_class_prob` | `float` | Same as `confidence` |
| `predictions` | `list` | The `top_k` most likely classes (1 by default), each with `id`, `label` and `confidence` |

All values are plain Python types, so `json.dumps(result)` works.

`classify_and_show` returns a tuple of this dictionary and the annotated image.
