from tflite_runtime.interpreter import Interpreter 
from PIL import Image
import numpy as np
import time

class TFLiteOps:

    def transform_image(self, interpreter: Interpreter, image):
        tensor_index = interpreter.get_input_details()[0]['index']
        input_tensor = interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image
    
    def classify(self, interpreter: Interpreter, image, top_k=1):
        self.transform_image(interpreter, image)
        interpreter.invoke()
        output_details = interpreter.get_output_details()[0]
        output = np.squeeze(interpreter.get_tensor(output_details['index']))

        try:
            if output_details['dtype'] == np.uint8:
                scale, zero_point = output_details['quantization']
                output = scale * (output - zero_point)
        except:
            raise RuntimeError(
                """NotQuantizedModel: No match with uint8 data type.""")

        ordered = np.argpartition(-output, 1)
        return [(i, output[i]) for i in ordered[:top_k]][0]

class TeachableMachineLite:
    """
    A class for TeachableMachineLite package to work with TeachableMachine models in tflite environment.

    ...

    Attributes
    ----------
    `model_path` : str
        path for tflite model file. (default: `model.tflite`)
    `label_path` : str
        path for labels file. (default: `labels.txt`)
    `model_type` : str
        type of model file. (default: `tflite`)

    Methods
    -------
    `classify_frame(frameFileName)`
        Classifies the image based on the trained model file and labels file.
    """

    def __init__(self, model_path='model.tflite', labels_file_path='labels.txt', model_type='tflite') -> None:
        """
        Constructs all the necessary attributes for the TeachableMachineLite object.

        Parameters
        ----------
            `model_path` : str
                path for tflite model file. (default: `model.tflite`)
            `label_path` : str
                path for labels file. (default: `labels.txt`)
            `model_type` : str
                type of model file. (default: `tflite`)
        """
        self.model_path = model_path
        self.label_path = labels_file_path
        self.model_type = model_type

        try:
            self.teachable_machine = TFLiteOps()
            self.interpreter = Interpreter(self.model_path)
            self.interpreter.allocate_tensors()
            print('TeachableMachineLite Model loaded successfully.')
        except:
            raise RuntimeError(
                """ModelLoadingError: Error while loading model, check out the name or path.""")
    
    def _preprocess_image(self):
        _, self.height, self.width, _ = self.interpreter.get_input_details()[0]['shape']

    def _load_labels(self):
        try:
            with open(self.label_path, 'r') as f:
                return [line.strip() for i, line in enumerate(f.readlines())]
        except FileNotFoundError as fnfe:
            print("LabelsFileError: Error in labels file, check out name, path or content.")
            raise (fnfe)
    
    def classify_frame(self, frameFileName="frame.jpg"):
        """
        Classifies the image based on the trained model file and labels file.
        
        If the argument 'frameFileName' is passed, then it is assigned with `frame.jpg`.        
        
        Before using this method, you need to make sure that you have a stored image with correct file name.

        This method takes the image name (str), not the image file itself (the matrix).

        Parameters
        ----------
        `frameFileName` : str
            Pass the image file name with string type (default is `frame.jpg`)

        Returns
        -------
            `id` : int
                id for class with highest prediction result.
            `label` : str
                name of class with highest prediction result based on the content of labels.txt file.
            `time` : float
                time of frame classification in seconds.
            `confidence` : float
                classification confidence (accuracy).
        """
        self._preprocess_image()
        try:
            image = Image.open(frameFileName).convert('RGB').resize((self.width, self.height))
        except FileNotFoundError as fnfe:
            print("ImageFileNameError: Error in image file name, check out image file name or extension")
            raise(fnfe)
        
        time1 = time.time()
        label_id, prob = self.teachable_machine.classify(self.interpreter, image)
        time2 = time.time()
        classification_time = np.round(time2-time1, 3)

        labels = self._load_labels()

        classification_label = labels[label_id].split()[1]
        classification_confidence = np.round(prob*100, 2)

        return {
            "id": label_id,
            "label": classification_label,
            "time": classification_time,
            "confidence": classification_confidence,
            "highest_class_id": label_id,
            "highest_class_prob": classification_confidence
        }

    # below | old methods; for compatibility purposes with old codes.
    def get_image_dimensions(self, interpreter):
        _, height, width, _ = self.interpreter.get_input_details()[0]['shape']
        return {
            'height': height,
            'width': width
        }

    def transform_image(self, interpreter, image):
        self.teachable_machine.transform_image(self.interpreter, image)
    
    def classify_image(self, interpreter, top_k=1):
        try:
            print("Warning: This method is deprecated. Use `classify_frame` method instead of `classify_image`.")
            return self.classify_frame("frame.jpg")
        except FileNotFoundError as fnf:
            print("Hint: Try to rename your image file with \"frame.jpg\", or it's recommended to use `classify_frame` method.")
            print("ImageFileNameError: Error in image file name, check out image file name or extension")
            raise(fnf)
    