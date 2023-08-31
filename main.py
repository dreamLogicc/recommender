from fastapi import FastAPI, File, UploadFile
import pandas as pd
from recommender import PlaceRecommender
from PIL import Image
import io
import keras
import numpy as np

app = FastAPI(title = 'Recommender System')

routes = pd.read_csv('data_with_array_emb.csv')
route_difficulties = pd.DataFrame({'id': [1, 2, 3], 'name': ['новичок', 'знающий', 'опытный']})

place_recommender = PlaceRecommender(routes['description'] + ' ' + routes['name'])
landscape_clf = keras.models.load_model('clf.keras')


def read_image(file):
    image = Image.open(io.BytesIO(file)).resize((224, 224))
    return image


@app.post('/recommend-on-history')
def recommend_on_history(history: str):
    try:
        arr = list(map(lambda x: int(x) - 1, history.split(',')))

        to_recommend = place_recommender.recommend_on_history(arr)

        result = {}
        print(to_recommend)
        for i in range(len(to_recommend)):
            result[f'place{i}'] = {
                'index': to_recommend[i] + 1,
                'name': routes.iloc[to_recommend[i]]['name'],
            }

        return result

    except Exception as ex:
        print(ex)


@app.post('/recommend-on-image')
async def recommend_on_image(file: UploadFile = File()):
    try:
        if file.filename.split('.')[1].lower() not in ('jpg', 'png', 'jpeg', 'ppm', 'tiff', 'bmp'):
            raise Exception('Invalid file format')

        image = np.array(read_image(await file.read()))
        image = np.expand_dims(image, 0)
        is_landscape = 1 if landscape_clf.predict(image / 255.0) > 0.5 else 0

        if not is_landscape:
            raise Exception('Loaded image is not landscape')

        to_recommend = place_recommender.recommend_on_image(image, routes['image_embeddings'])

        result = {}

        for i in range(len(to_recommend)):
            result[f'place{i}'] = {
                'index': to_recommend[i] + 1,
                'name': routes.iloc[to_recommend[i]]['name']
            }
        return result
    except Exception as ex:
        return {'error': str(ex)}
