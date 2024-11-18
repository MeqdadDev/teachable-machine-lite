## Requirements

### Python Version

```bash
Python >= 3.9
```

### Dependencies

```bash
tflite_runtime
Pillow
numpy
```

#### Important Compatibility Note:
The Google Teachable Machine platform currently exports models that are not compatible with `numpy 2.x`. This package requires `numpy 1.x` (recommended `version 1.26.4`) to function correctly. Models exported from Teachable Machine may fail or produce unexpected results with `numpy 2.x`.

**Numpy Installation recommendation:**
```bash
pip install numpy==1.26.4
```
This will ensure the compatibility with Teachable Machine models

If you experience any model loading or inference issues, please verify your numpy version using:
```bash
python -m pip list | grep numpy
```
