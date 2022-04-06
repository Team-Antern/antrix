import numpy as np 
import pandas as pd 
import logging 
from nltk.corpus import stopwords
import nltk
from nltk import sent_tokenize, word_tokenize
import gensim
import re

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from gensim.models.fasttext import FastText
class FeatureEngineering: 
    def __init__(self, df): 
        self.df = df 
        self.df_copy = df.copy() 
    
    def remove_features(self): 
        try: 
            logging.info("Removing features from the dataframe")
            cols_to_remove = ["cid", "time", "author", "photo"]  
            self.df_copy = self.df_copy.drop(cols_to_remove, axis=1) 
            return self.df_copy 
        except Exception as e:
            logging.error("Error in removing features from the dataframe")
            logging.error(e)
            return None


    def tf_idf_vectorizer(self):
        logging.info(
            
            "In tf_idf_vectorizer method In Vectorization class: adding tf-idf features"
        )
        try:
            vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, max_features=100)
            print("column printing") 
            print(self.df_copy["text"]) 
            self.df_copy.dropna(inplace=True)
            extracted_data = list(vectorizer.fit_transform(self.df_copy["text"]).toarray())
            extracted_data = pd.DataFrame(extracted_data)
            extracted_data.columns = vectorizer.get_feature_names()

            vocab = vectorizer.vocabulary_
            mapping = vectorizer.get_feature_names()
            keys = list(vocab.keys())

            extracted_data.shape
            Modified_df = extracted_data.copy()
   
            Modified_df.reset_index(drop=True, inplace=True)
            self.df.reset_index(drop=True, inplace=True)

            Final_Training_data = pd.concat([self.df, Modified_df], axis=1)

            Final_Training_data.drop("text", axis=1, inplace=True)
            Final_Training_data.to_csv("informative-reports-creater/src/features/final_training_vectorized.csv", index=False)

            joblib.dump(vectorizer, "informative-reports-creater/models/vectorizer.pkl")
            return Final_Training_data

        except Exception as e:
            logging.info(
                f"In tf_idf_vectorizer method In Vectorization class: Error in adding tf-idf features: {e}"
            )
            raise e 
            