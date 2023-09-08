import keras.models
import pandas as pd
import cv2
from skimage import io
import numpy as np
from PIL import Image
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input

data = pd.read_csv('data_for_db.csv')

model = keras.models.load_model('feature_extractor.keras')

feat_extractor = Model(inputs = model.input, outputs = model.get_layer("dense_1").output)

images = []
for url in data['image_link']:
    images.append(
        np.expand_dims(cv2.resize(io.imread(url), (150, 150), interpolation = cv2.INTER_AREA) / 255.0, axis = 0))
    print('success ' + url)

print(images[34].shape)

imgs_features = []
for image in images:
    imgs_features.append(feat_extractor.predict(image))
print("features successfully extracted!")
imgs_features = np.array(imgs_features)
print(imgs_features.shape)

data['image_embeddings'] = pd.Series(imgs_features.tolist())
data.to_csv('data_with_array_emb_custom.csv', index = False)
