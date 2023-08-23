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

routes = pd.read_csv('data_for_db.csv')
route_difficulties = pd.DataFrame({'id': [1, 2, 3], 'name': ['новичок', 'знающий', 'опытный']})

place_recommender = PlaceRecommender(routes['description'])
image_classifier = keras.models.load_model('model.keras')


def read_image(file):
    image = Image.open(io.BytesIO(file))
    return image


def predict(image):
    pred = np.argmax(image_classifier.predict(np.expand_dims(image, 0))[0])
    if pred == 0:
        return 'высокие горы вершины высота'
    elif pred == 1:
        return 'старинные здания музеи архитектура'
    elif pred == 2:
        return 'ледники белый снег'
    elif pred == 3:
        return 'зеленые леса деревья'
    elif pred == 4:
        return 'реки озера водоемы берега'
    elif pred == 5:
        return 'улицы парки скверы культура кафе магазин'
    # match pred:
    #     case 0:
    #         return 'высокие горы вершины высота'
    #     case 1:
    #         return 'старинные здания музеи архитектура'
    #     case 2:
    #         return 'ледники белый снег'
    #     case 3:
    #         return 'зеленые леса деревья'
    #     case 4:
    #         return 'реки озера водоемы берега'
    #     case 5:
    #         return 'улицы парки скверы культура кафе магазин'


@app.post('/recommend-on-servey')
def recommend_on_servey(likes: str):
    try:
        to_recommend = place_recommender.recommend(likes)

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
    try:
        image = np.array(read_image(await file.read()))
        image = cv2.resize(image, (150, 150))
        caption = predict(image)
        to_recommend = place_recommender.recommend(caption)

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
    except Exception as ex:
        print(ex)