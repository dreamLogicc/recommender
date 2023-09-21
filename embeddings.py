import pandas as pd
import cv2
from skimage import io
import numpy as np
from keras.applications import vgg19
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input

data = pd.read_csv('data_for_db.csv')

vgg_model = vgg19.VGG19(weights = 'imagenet')

feat_extractor = Model(inputs = vgg_model.input, outputs = vgg_model.get_layer("fc2").output)

images = []
for url in data['image_link']:
    images.append(np.expand_dims(cv2.resize(io.imread(url), (224, 224)), axis = 0))
    print('success ' + url)

print(images[34].shape)

images = np.vstack(images)

processed_images = preprocess_input(images.copy())

imgs_features = feat_extractor.predict(processed_images)
print("features successfully extracted!")
print(imgs_features.shape)

data['image_embeddings'] = pd.Series(imgs_features.tolist())
data.to_csv('data_vgg19.csv', index = False)
