# Images and Camera Frames

`classify_image`, `show_prediction_on_image` and `classify_and_show` all accept:

| Input | Example |
|---|---|
| A file path (`str` or `pathlib.Path`) | `model.classify_image("photo.jpg")` |
| A PIL image | `model.classify_image(Image.open("photo.jpg"))` |
| A NumPy array of type `uint8` | `model.classify_image(frame)` |

Passing a camera frame directly is faster than writing it to a file first, and avoids wearing out
a Raspberry Pi's SD card.

## Color order

NumPy arrays are treated as **BGR** by default. That matches OpenCV frames and Picamera2's
`"RGB888"` and `"XBGR8888"` formats.

For RGB arrays, such as Picamera2's `"BGR888"` format, pass `color_order="RGB"`:

```python
result = model.classify_image(rgb_array, color_order="RGB")
```

`color_order` has no effect on file paths and PIL images.

Grayscale arrays work too, and the fourth channel of a 4-channel array is ignored.

## Get the annotated image

`classify_and_show` returns the result and a copy of the image with the label drawn in the bottom
left corner. Your original frame isn't changed.

```python
result, annotated = model.classify_and_show(frame)
```

The annotated image is a **BGR NumPy array**, ready for `cv.imshow` or `cv.imwrite`. To get a PIL
image in RGB instead, pass `convert_to_bgr=False`:

```python
result, picture = model.classify_and_show(frame, convert_to_bgr=False)
picture.save("labelled.jpg")
```

## Frame shape

The package resizes the whole image to the model's input size. It doesn't crop first, so a wide
frame is squeezed into the model's square input. If your training images were square and results
on wide frames are poor, crop the centre square before classifying:

```python
height, width = frame.shape[:2]
side = min(height, width)
top, left = (height - side) // 2, (width - side) // 2
square = frame[top:top + side, left:left + side]

result = model.classify_image(square)
```
