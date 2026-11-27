# Speed and Accuracy

## Make it faster

**Use the quantized export.** It is noticeably faster than the floating point one on Raspberry Pi,
and the file is smaller.

**Give the runtime more threads.** Set `num_threads` to the number of CPU cores:

```python
model = TeachableMachineLite(
    model_path="model.tflite", labels_file_path="labels.txt", num_threads=4)
```

**Create the model once.** Load it before your loop, then reuse it for every frame.

**Pass frames directly** instead of saving them to image files first.

## Measure it

With `calc_time=True`, the result's `time` key holds the classification time in seconds:

```python
result = model.classify_image(frame, calc_time=True)
print(result["time"])
```

It covers preparing the image and running the model, not reading the camera or drawing.

## Improve accuracy

- Train with images from the same camera, background and lighting you'll use.
- Try the **Floating point** export, which is slightly more accurate.
- Match the shape of your training images. See [Frame shape](images-and-frames.md#frame-shape).
- Act only on confident results, for example `result["confidence"] > 80`.
- Check the runners-up with `top_k` to see which classes get confused:

```python
result = model.classify_image(frame, top_k=3)
for prediction in result["predictions"]:
    print(prediction["label"], prediction["confidence"])
```
