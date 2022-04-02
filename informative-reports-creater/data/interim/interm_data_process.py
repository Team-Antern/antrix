import numpy as np 
import pandas as pd 

import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer

import logging 

class IntermediateDataProcess:
    
    def __init__(self, raw_data_path):
        self.raw_data_path = raw_data_path 
    
    def comment_processing(self, comment):
        logging.info("Applying comment processing on the data started")
        try:
            comment = comment.lower()
            comment = comment.replace("\n", " ")
            comment = comment.replace("\r", " ")
            comment = comment.replace("\t", " ")
            comment = comment.replace("\xa0", " ")
            comment = comment.replace("\u200b", " ")
            comment = comment.replace("\u200c", " ")
            comment = comment.replace("\u200d", " ")
            comment = comment.replace("\ufeff", " ")
            comment = comment.replace("\ufeef", " ")
        except Exception as e:
            logging.error(
                "Error in applying comment processing methods on train and test data"
            )
            logging.error(e)
            return None

    def stemming(self, comment):
        logging.info("Applying stemming methods on train and test data")
        try:
            comment = comment.split()
            ps = PorterStemmer()
            comment = [ps.stem(word) for word in comment]
            comment = " ".join(comment)
        except Exception as e:
            logging.error(
                "Error in applying stemming methods on train and test data"
            )
            logging.error(e)
            return None

    def lemmatization(self, comment):
        logging.info("Applying lemmatization method on train and test data")
        try:
            comment = comment.split()
            lem = WordNetLemmatizer()
            comment = [lem.lemmatize(word) for word in comment]
            comment = " ".join(comment)
            return comment

        except Exception as e:
            logging.error(
                "Error in applying lemmatization method on train and test data"
            )
            logging.error(e)
            return None

    def remove_stopwords(self, comment):
        logging.info("Applying remove_stopwords methods on  comment")
        try:
            comment = comment.split()
            stop_words = set(stopwords.words("english"))
            comment = [word for word in comment if not word in stop_words]
            comment = " ".join(comment)
            return comment
        except Exception as e:
            logging.error(
                "Error in applying remove_stopwords method on comment"
            )
            logging.error(e)
            return None


