# Command-Line Webcam

Reads the model and labels paths from the command line, so one script works with any exported model.

## Code

```python title="examples/command_line_webcam.py"
--8<-- "examples/command_line_webcam.py"
```

## Run it

From your project folder, with the script saved as `command_line_webcam.py`:

```bash
python command_line_webcam.py --model path/to/model.tflite --labels path/to/labels.txt
```

With all the options, for example for a floating point export, an external webcam and 4 threads:

```bash
python command_line_webcam.py --model path/to/model_unquant.tflite --labels path/to/labels.txt --camera 1 --threads 4 --top-k 3
```

Put paths that contain spaces in quotes, such as `--model "C:\My Models\model.tflite"`.
Run `python command_line_webcam.py --help` to list all the options.

USB webcams work with this script on Raspberry Pi too. For the camera module, see
[Raspberry Pi Camera](picamera2.md).
