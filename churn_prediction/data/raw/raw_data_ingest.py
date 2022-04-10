# https://www.technologiesinindustry4.com/2021/07/how-to-use-kaggle-api-in-python.html

import py7zr
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
path = r'C:\Users\shreyas.dhaware2301\Documents\antern\antrix\churn_prediction\data\interim\churn_prediction'

for i in ['members_v3.csv.7z', 'transactions.csv.7z', 'user_logs.csv.7z', 'train.csv.7z']:
    api.competition_download_file(
        'kkbox-churn-prediction-challenge', i, path=path)
    with py7zr.SevenZipFile(f"{path}\{i}", mode='r') as z:
        z.extractall(path=path)
