from tflite_runtime.interpreter import Interpreter
from PIL import Image
import numpy as np
import time
import logging

class TFLiteUtils:

    @staticmethod
    def build_input_tensor(interpreter: Interpreter, image):
        """
        Builds the input tensor for the TFLite interpreter using the provided image.

        Args:
            interpreter (Interpreter): The TFLite interpreter.
            image: The input image to be used for building the tensor.

        Note:
            This method modifies the interpreter's input tensor in-place.
            The image should be pre-processed and have the correct dimensions
            to match the interpreter's input tensor shape.
        """
        tensor_index = interpreter.get_input_details()[0]['index']
        input_tensor = interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    @staticmethod
    def classify(interpreter: Interpreter, image, top_k: int = 1):
        """
        Classifies the input image using the provided TFLite interpreter.

        Args:
            interpreter (Interpreter): The TFLite interpreter.
            image: The input image to be classified.
            top_k (int): The number of top predictions to return. Defaults to 1.

        Returns:
            tuple: A tuple containing (class_id, confidence_score) for the top prediction.

        Raises:
            ValueError: If the interpreter output is invalid or if there are missing keys in output details.
            RuntimeError: If an unexpected error occurs during classification.

        Note:
            This method assumes that the model output is a list of class probabilities.
        """
        try:
            TFLiteUtils.build_input_tensor(interpreter, image)
            interpreter.invoke()
            output_details = interpreter.get_output_details()[0]
            output = np.squeeze(interpreter.get_tensor(output_details['index']))

            if output_details['dtype'] == np.uint8:
                scale, zero_point = output_details['quantization']
                output = scale * (output - zero_point)

            ordered = np.argpartition(-output, top_k)
            return [(i, output[i]) for i in ordered[:top_k]][0]

        except IndexError:
            raise ValueError("Invalid interpreter output. Check if the model is correctly loaded.")
        except KeyError as e:
            raise ValueError(f"Missing key in output details: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during classification: {str(e)}")

class TeachableMachineLite:
    def __init__(self, model_path='model.tflite', labels_file_path='labels.txt', model_type='tflite') -> None:
        """
        Initialize the TeachableMachineLite object.

        Args:
            model_path (str): Path to the TFLite model file. Default is 'model.tflite'.
            labels_file_path (str): Path to the labels file. Default is 'labels.txt'.
            model_type (str): Type of the model. Default is 'tflite'.

        Raises:
            FileNotFoundError: If the model file is not found.
            RuntimeError: If there's an error loading the model.
        """
        self.model_path = model_path
        self.label_path = labels_file_path
        self.model_type = model_type
        self.interpreter = None

        self._load_labels(self.label_path)

        try:
            # Check if the model file exists
            with open(self.model_path, 'rb') as f:
                pass
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            self.interpreter = Interpreter(self.model_path)
            self.interpreter.allocate_tensors()
            logging.info(f"TeachableMachineLite Model '{self.model_path}' loaded successfully.")
            print(f"TeachableMachineLite Model '{self.model_path}' loaded successfully.")
        except Exception as e:
            error_msg = f"Error loading model '{self.model_path}': {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

    def _load_labels(self, labels_file_path):
        try:
            with open(labels_file_path, "r") as file:
                self._labels = file.readlines()
        except IOError as e:
            print("LoadingLabelsError: Error while loading labels.txt file")
            raise IOError("Error loading labels") from e
        except Exception as e:
            print("LoadingLabelsError: Error while loading labels.txt file")
            raise FileNotFoundError("Labels file not found") from e

    def _preprocess_image(self):
        _, self.height, self.width, _ = self.interpreter.get_input_details()[0]['shape']

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
        label_id, prob = TFLiteUtils.classify(self.interpreter, image)
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
        TFLiteUtils.build_input_tensor(self.interpreter, image)
    
    def classify_image(self, interpreter, top_k=1):
        try:
            print("Warning: This method is deprecated. Use `classify_frame` method instead of `classify_image`.")
            return self.classify_frame("frame.jpg")
        except FileNotFoundError as fnf:
            print("Hint: Try to rename your image file with \"frame.jpg\", or it's recommended to use `classify_frame` method.")
            print("ImageFileNameError: Error in image file name, check out image file name or extension")
            raise(fnf)
    