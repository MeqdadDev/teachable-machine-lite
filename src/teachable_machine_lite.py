from tflite_runtime.interpreter import Interpreter
from PIL import Image
import numpy as np
import time
import logging

class TeachableMachineLite:
    def __init__(self, model_path: str = 'model.tflite', labels_file_path: str = 'labels.txt', model_type: str = 'tflite') -> None:
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

        self._labels = None
        self.height = None
        self.width = None
        self.interpreter = None

        self._load_labels(self.label_path)

        try:
            with open(self.model_path, 'rb') as f:
                # To check if the model file exists
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
        input_details = self.interpreter.get_input_details()[0]
        _, self.height, self.width, _ = input_details['shape']


    def classify_image(self, image_path: str, calc_time: bool = False):
        """
        Classifies an image using the model interpreter and returns the classification results.

        This method loads an image from the specified file path, preprocesses it, and classifies it using the
        TFLite model interpreter. The classification result includes the predicted label, confidence score,
        and the time taken for classification if requested.

        Parameters:
        -----------
        image_path : str
            The file path to the image that needs to be classified.

        calc_time : bool, optional
            If True, measures and returns the time taken for the classification process.
            Defaults to False.

        Returns:
        --------
        dict
            A dictionary containing the classification results:
            - "id" : int
                The index of the predicted label.
            - "label" : str
                The predicted label corresponding to the image.
            - "time" : float or None
                The time taken to classify the image, in seconds (rounded to 3 decimal places).
                Returns None if `calc_time` is False.
            - "confidence" : float
                The confidence level of the classification as a percentage.
            - "highest_class_id" : int
                The same as "id", for consistency.
            - "highest_class_prob" : float
                The same as "confidence", for consistency.
        """
        image = self._load_image(image_path)

        classification_time = None

        if calc_time:
            start_time = time.time()

        label_id, prob = self._predict(image, 1)[0]

        if calc_time:
            classification_time = np.round(time.time() - start_time, 3)

        classification_label = self._labels[label_id].strip()
        classification_confidence = np.round(prob * 100, 2)

        return {
            "id": label_id,
            "label": classification_label,
            "time": classification_time,
            "confidence": classification_confidence,
            "highest_class_id": label_id,
            "highest_class_prob": classification_confidence
        }

    def _build_input_tensor(self, image):
        """
        Prepares the input tensor for the TFLite interpreter with the given image.

        This method sets up the input tensor for the TFLite interpreter by copying
        the provided image into the tensor's memory. The image should already be
        preprocessed and resized to match the model's expected input dimensions.

        Parameters:
        -----------
        image : np.ndarray
            The preprocessed image to be classified. The image should be a NumPy array
            with the correct shape (height, width, channels) expected by the model.

        Returns:
        --------
        None
            The method modifies the interpreter's input tensor in place and does not return a value.
        """
        tensor_index = self.interpreter.get_input_details()[0]['index']
        input_tensor = self.interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def _predict(self, image, top_k: int = 1):
        """
        Classifies the input image using the model interpreter and returns the top predictions.

        This method processes the input image through the TFLite model interpreter to generate predictions.
        It returns the top predicted class IDs along with their confidence scores.

        Args:
        -----
        image : np.ndarray
            The preprocessed input image to be classified. The image should be a NumPy array
            that has been resized and formatted according to the model's input specifications.

        top_k : int, optional
            The number of top predictions to return. Defaults to 1. If set to 1, a single prediction
            (class_id, confidence_score) is returned. If set to a higher value, a list of the top
            predictions is returned.

        Returns:
        --------
        list of tuple or tuple
            A list of tuples containing (class_id, confidence_score) for the top predictions.
            If `top_k` is 1, a single-element list is returned with the highest prediction.

        Raises:
        -------
        ValueError
            If the interpreter output is invalid, such as when the model is not correctly loaded,
            or if there are missing keys in the output details.

        RuntimeError
            If an unexpected error occurs during classification, such as issues during model inference.
        """
        try:
            self._build_input_tensor(image)

            # Run inference
            self.interpreter.invoke()

            output_details = self.interpreter.get_output_details()[0]
            output = np.squeeze(self.interpreter.get_tensor(output_details['index']))

            if output_details['dtype'] == np.uint8:
                scale, zero_point = output_details['quantization']
                output = scale * (output - zero_point)

            if top_k == 1:
                max_index = np.argmax(output)
                return [(max_index, output[max_index])]
            else:
                ordered_indices = np.argpartition(-output, top_k)[:top_k]
                top_predictions = [(i, output[i]) for i in ordered_indices]
                top_predictions.sort(key=lambda x: -x[1])  # Sort by confidence
                return top_predictions

        except IndexError:
            raise ValueError("Invalid interpreter output. Ensure the model is correctly loaded.")
        except KeyError as e:
            raise ValueError(f"Missing key in output details: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during classification: {str(e)}")

    def _load_image(self, image_path: str):
        """
        Loads and preprocesses an image for classification.

        Parameters:
        -----------
        image_path : str
            The file path to the image that needs to be loaded and preprocessed.

        Returns:
        --------
        Image.Image
            A PIL Image object that has been converted to RGB format and resized to match the model's input dimensions.

        Raises:
        -------
        IOError
            If the image file cannot be found, opened, or processed, an IOError is raised with a descriptive message.
        """
        try:
            # Load, convert, and resize the image
            image = (Image.open(image_path)
                     .convert('RGB')
                     .resize((self.width, self.height)))
            return np.array(image)
        except IOError as io_error:
            print(f"Error loading image '{image_path}': {io_error}")
            raise

    def _load_labels(self, labels_file_path: str):
        """
        Loads the classification labels from a specified file.

        This method reads labels from a given file and stores them in the instance variable `_labels`.
        The labels are expected to be in a text file where each label is on a new line.

        Parameters:
        -----------
        labels_file_path : str
            The file path to the labels file that needs to be loaded.

        Raises:
        -------
        IOError
            If the file cannot be found, opened, or read, an IOError is raised with a descriptive message.
        """
        try:
            with open(labels_file_path, "r") as file:
                self._labels = file.readlines()
        except IOError as e:
            print(f"Error loading labels from '{labels_file_path}': {e}")
            raise IOError("Error loading labels") from e
