from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
from recommender import PlaceRecommender
from PIL import Image
import io
import keras
import numpy as np

app = FastAPI(title = 'Recommender System')

routes = pd.read_csv('data_with_array_emb_custom.csv')

place_recommender = PlaceRecommender()
landscape_clf = keras.models.load_model('clfv2.keras')


def read_image(file):
    image = Image.open(io.BytesIO(file)).resize((150, 150))
    return image

@app.post('/recommend-on-image')
async def recommend_on_image(file: UploadFile = File()):
    if file.filename.split('.')[-1].lower() not in ('jpg', 'png', 'jpeg', 'ppm', 'tiff', 'bmp'):
        raise HTTPException(status_code = 400, detail = 'Invalid file format')

    image = np.array(read_image(await file.read()))

    if np.array(image).shape[0] > 1080 or np.array(image).shape[0] > 1920 or np.array(image).shape[2] != 3:
        raise HTTPException(status_code = 400, detail = 'Invalid file shape')

    is_landscape = 1 if landscape_clf.predict(
        np.expand_dims(image, 0) / 255.0) > 0.5 else 0

    if not is_landscape:
        raise HTTPException(status_code = 400, detail = 'Loaded image is not landscape')

    try:
        to_recommend = place_recommender.recommend_on_image(
            np.expand_dims(image, 0) / 255.0,
            routes['image_embeddings'])

        result = {}
        for i in range(len(to_recommend)):
            result[f'place{i}'] = {
                'index': to_recommend[i] + 1,
                'name': routes.iloc[to_recommend[i]]['name']
            }
        return result
    except Exception as ex:
        return {'error': str(ex)}
