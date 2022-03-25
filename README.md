# Teachable Machine Lite

A Python package to simplify the deployment process of exported [Teachable Machine](https://teachablemachine.withgoogle.com/) models into different embedded environments like Raspberry Pi and other SBCs using TensorFlowLite.

Links:

[PyPI](https://pypi.org/project/teachable-machine-lite/)

[Source Code](https://github.com/MeqdadDev/teachable-machine-lite)


## Requirements

Python >= 3.8

## How to install package

```bash
pip install teachable-machine-lite
```

## Dependencies

```numpy, tflite-runtime```

## How to use teachable machine lite package

```py
from teachable_machine_lite import TeachableMachineLite
import cv2
from tflite_runtime.interpreter import Interpreter

model_path = 'models/model.tflite'
interpreter = Interpreter(model_path)

my_model = TeachableMachineLite(model_type='tflite', model_path=model_path)

img_path = 'images/my_image.jpg'

dim = my_model.get_image_dimensions(interpreter)
height, width = dim['height'], dim['width']

interpreter.allocate_tensors()

img = cv2.imread(img_path)
img = cv2.resize(img, (width, height))
my_model.transform_image(interpreter, img)
interpreter.invoke()
results = my_model.classify_image(interpreter)

print('highest_class_id', results['highest_class_id'])
print('highest_class_prob', results['highest_class_prob'])

```

_highest_class_id_ is selected based on labels.txt file.

More features are coming soon...
