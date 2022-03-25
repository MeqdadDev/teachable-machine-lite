import numpy as np


class TeachableMachineLite(object):
    '''
    Create your TeachableMachineLite object to run your exported AI models using TFLite.
    '''
    __supported_types = ('tflite', 'tensorflowlite', 'TensorFlowLite')

    def __init__(self, model_path='model.tflite', labels_file_path='labels.txt', model_type='tflite') -> None:
        self.model_type = model_type.lower()
        self.model_path = model_path
        self.labels_file_path = labels_file_path
        self.object_creation_status = self.model_type in self.__supported_types
        if self.object_creation_status:
            print('Teachable Machine Lite Object is created successfully.')
        else:
            raise RuntimeError(
                """*****Not supported model type: Select a supported model type like "tflite" or "tensorflowlite".*****""")

    def get_image_dimensions(self, interpreter):
        _, height, width, _ = interpreter.get_input_details()[0]['shape']
        return {
            'height': height,
            'width': width
        }

    def transform_image(self, interpreter, image):
        tensor_index = interpreter.get_input_details()[0]['index']
        input_tensor = interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def classify_image(self, interpreter, top_k=1):
        output_details = interpreter.get_output_details()[0]
        output = np.squeeze(
            interpreter.get_tensor(output_details['index']))
        try:
            if output_details['dtype'] == np.uint8:
                scale, zero_point = output_details['quantization']
                output = scale * (output - zero_point)
        except:
            raise RuntimeError(
                """Not quantized model, there is no match with uint8 data type.""")
        ordered = np.argpartition(-output, top_k)
        predictions = [(i, output[i]) for i in ordered[:top_k]]
        highest_class_id, highest_class_prob = predictions[0]
        return {
            'highest_class_id': highest_class_id,
            'highest_class_prob': highest_class_prob
        }
