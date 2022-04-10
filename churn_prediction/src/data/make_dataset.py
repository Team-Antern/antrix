# Downloading 4 files to the data folder
# kaggle competitions download kkbox-churn-prediction-challenge -f train.csv.7z -f transactions.csv.7z -f user_logs.csv.7z -p C:\Users\shreyas.dhaware2301\Documents\antern\antrix\churn_prediction\data\interim\churn_prediction

import numpy as np
import pandas as pd


class DataUtils:
    #  def __init__(self) -> None:

    def read_data(self, path):

        try:
            train_data = pd.read_csv(path+r"\train.csv")
            transaction_data = pd.read_csv(path+r"\transactions.csv")
            user_logs_data = pd.read_csv(path+r"\user_logs.csv")
            members_data = pd.read_csv(path+r"\members.csv")
            return train_data, transaction_data, user_logs_data, members_data
        except Exception as e:
            raise e


if __name__ == "__main__":
    data_utils = DataUtils()
    path = r"C:\Users\shreyas.dhaware2301\Documents\antern\antrix\churn_prediction\data\interim\churn_prediction"
    train_data, transaction_data, user_logs_data, members_data = data_utils.read_data(
        path)
    train_data.head(), transaction_data.head(
    ), user_logs_data.head(), members_data.head()
