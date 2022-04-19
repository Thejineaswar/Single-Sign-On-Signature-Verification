import base64
import io
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from sampleBase64 import get_sample_base_64

def decode_image(img : str):
    """
    :param img: Base 64 string of image
    :return: The image in a numpy array format(floating point values within the array)
    """
    decoded = base64.b64decode(img)
    image = Image.open(io.BytesIO(decoded))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(224,224))
    return image


def getPredictions(image_1: str, image_2: str):
    """
    :param image_1: Base 64 string of the first image
    :param image_2: Base 64 string of the second image
    :return: A dictionary with "prediction" as key for value
    """
    i_1 = decode_image(image_1)
    i_2 = decode_image(image_2)
    model = tf.keras.models.load_model('model.h5')
    predictions = model.predict([np.expand_dims(i_1, 0), np.expand_dims(i_2, 0)])

    if predictions[0][0] > 0.5:
        return {
            'prediction': "Similar"
        }
    else:
        return {
            'prediction': "Disimilar"
        }

if __name__ == '__main__':
    example = get_sample_base_64()
    getPredictions(example,example) # Passing the same image for checking