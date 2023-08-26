import cv2
from fastapi import FastAPI, File, UploadFile
import pandas as pd
from recommender import PlaceRecommender
from PIL import Image
import base64
import io
import keras
import numpy as np

app = FastAPI(title = 'Recommender System')

routes = pd.read_csv('data_with_array_emb.csv')
route_difficulties = pd.DataFrame({'id': [1, 2, 3], 'name': ['новичок', 'знающий', 'опытный']})

place_recommender = PlaceRecommender(routes['description'])

def read_image(file):
    image = Image.open(io.BytesIO(file))
    return image


@app.post('/recommend-on-servey')
def recommend_on_servey(likes: str):
    try:
        to_recommend = place_recommender.recommend_on_description(likes)

        result = {}

        for i in range(len(to_recommend)):
            result[f'place{i}'] = {
                'index': to_recommend[i],
                'name': routes.iloc[to_recommend[i]]['name'],
                'description': routes.iloc[to_recommend[i]]['description'],
                'difficulty': route_difficulties.iloc[routes.iloc[to_recommend[i]]['difficulty_id'] - 1]['name'],
                'longitude': routes.iloc[to_recommend[i]]['longitude'],
                'latitude': routes.iloc[to_recommend[i]]['latitude'],
                'rating': routes.iloc[to_recommend[i]]['rating']
            }

        return result

    except Exception as ex:
        print(ex)


@app.post('/recommend-on-image')
async def recommend_on_image(file: UploadFile = File()):
    # try:
    image = np.array(read_image(await file.read()))
    image = cv2.resize(image, (224, 224))
    to_recommend = place_recommender.recommend_on_image(image, routes['image_embeddings'])

    result = {}

    for i in range(len(to_recommend)):
        result[f'place{i}'] = {
            'index': to_recommend[i] + 1,
            'name': routes.iloc[to_recommend[i]]['name'],
            'description': routes.iloc[to_recommend[i]]['description'],
            'difficulty': route_difficulties.iloc[routes.iloc[to_recommend[i]]['difficulty_id'] - 1]['name'],
            'longitude': routes.iloc[to_recommend[i]]['longitude'],
            'latitude': routes.iloc[to_recommend[i]]['latitude'],
            'rating': routes.iloc[to_recommend[i]]['rating']
        }
    return result
# except Exception as ex:
#     print(ex)
