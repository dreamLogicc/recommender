from skimage import io
import pandas as pd
import numpy as np
import cv2
import base64
from skimage import io
from keras.applications import vgg16
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from recommender import PlaceRecommender

data = pd.read_csv('data_with_array_emb.csv')

vgg_model = vgg16.VGG16(weights = 'imagenet')

feat_extractor = Model(inputs = vgg_model.input, outputs = vgg_model.get_layer("fc2").output)

image = cv2.resize(
    io.imread('https://khakassia.travel/assets/cache/images/uploads/2021/06/khakasiya-1024x576-2-1100x-062.jpg'),
    (224, 224))

ps = PlaceRecommender(data['description'])

print(ps.recommend_on_image(image, data['image_embeddings']))
