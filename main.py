from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
from recommender import PlaceRecommender
from PIL import Image
import io
import keras
import numpy as np
from keras.applications.imagenet_utils import preprocess_input

app = FastAPI(title = 'Recommender System')

routes = pd.read_csv('data_vgg19.csv')

place_recommender = PlaceRecommender()
landscape_clf = keras.models.load_model('clfv5.keras')


def read_image(file):
    try:
        image = Image.open(io.BytesIO(file))
        return image
    except Exception as ex:
        return {'error': str(ex)}


@app.post('/recommend-on-image')
async def recommend_on_image(file: UploadFile = File()):
    if file.filename.split('.')[-1].lower() not in ('jpg', 'png', 'jpeg', 'ppm', 'tiff', 'bmp', 'tif'):
        raise HTTPException(status_code = 400, detail = 'Invalid file format')

    image = read_image(await file.read())

    if np.array(image).shape[0] > 1080 or np.array(image).shape[1] > 1920 or np.array(image).shape[2] != 3:
        raise HTTPException(status_code = 400, detail = 'Invalid file shape')

    print(landscape_clf.predict(
        np.expand_dims(np.array(image.resize((150, 150))), 0) / 255.0))

    is_landscape = 1 if landscape_clf.predict(
        np.expand_dims(np.array(image.resize((150, 150))), 0) / 255.0) > 0.5 else 0

    if not is_landscape:
        raise HTTPException(status_code = 400, detail = 'Loaded image is not landscape')

    try:
        to_predict = preprocess_input(np.expand_dims(np.array(image.resize((224, 224))), 0))
        to_recommend = place_recommender.recommend_on_image(to_predict, routes['image_embeddings'])

        result = {}
        for i in range(len(to_recommend)):
            result[f'place{i}'] = {
                'name': routes.iloc[to_recommend[i]]['name'],
                'index': to_recommend[i] + 1,
            }
        return result
    except Exception as ex:
        return {'error': str(ex)}
