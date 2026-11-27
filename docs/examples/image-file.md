# Image File

Classifies one photo and acts on the label only when the model is confident enough, a common pattern in robotics. Put a `photo.jpg` next to the model files.

## Code

```python title="examples/image_file.py"
--8<-- "examples/image_file.py"
```

## How it works

- `top_k=2` puts the two most likely classes in `result["predictions"]`, so it needs a model with
  at least two classes.
- `confidence` is a percentage, so `80` means 80%.
