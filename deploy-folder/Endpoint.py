from definitions import ROOT_DIR
from keras.models import load_model
import pickle
from mtcnn import MTCNN
from numpy import asarray, load, expand_dims
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
from PIL import Image


class Endpoint(object):
    """
    :param x, df: A single datapoint.
    :param feature_names: Attribute names for x
    :param shap_values: shap values from the model
    :returns:
        - probability estimates: type.list([probability, 1-probability])
        - cluster estimate: type.int64(cluster_num)
        - shap values: type.list([shap values])
    """

    def __init__(self):
        self.embedding_model = load_model(ROOT_DIR + '/serialized/facenet_keras.h5')
        self.classifier_model = pickle.load(open(ROOT_DIR + '/serialized/svm.sav', 'rb'))
        self.in_encoder = Normalizer(norm='l2')

    def predict(self, x, feature_names):
        
        normalized_embedding = self.in_encoder.transform(self.get_embedding(self.embedding_model, self.extract_face(x)))
        prediction = self.classifier_model.predict(normalized_embedding)

        return [prediction]

    
    def extract_face(image, required_size=(160, 160)):
        image = image.convert('RGB')
        pixels = asarray(image)
        detector = MTCNN()
        results = detector.detect_faces(pixels)
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        return face_array

    def get_embedding(model, face_pixels):
        face_pixels = face_pixels.astype('float32')
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        samples = expand_dims(face_pixels, axis=0)
        yhat = model.predict(samples)
        return yhat[0]