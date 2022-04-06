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
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
import logging 

class IntermediateDataProcess:
    
    def __init__(self, data):
        self.data = data 
        self.df =pd.read_csv(self.data)
    
    def main(self): 
        print(self.df) 
        logging.info("Applying interim main method on  comment")
        try: 
            print(self.df)
            self.df['text'] = self.df["text"].apply(
                lambda x: self.comment_processing(x)
            ) 
            self.df.to_csv('informative-reports-creater/data/interim/interm_data_process.csv', index=False) 
            return self.df
        except Exception as e:
            logging.error(
                "Error in applying main method on  comment"
            )
            logging.error(e)

    def comment_processing(self, comment):
        logging.info("Applying comment processing on the data started")
        try:
            comment = comment.lower()
            comment = comment.replace("\n", " ")
            comment = comment.replace("\r", " ")
            comment = comment.replace("\t", " ")
            return comment 
        except Exception as e:
            logging.error(
                "Error in applying comment processing methods on train and test data"
            )
            logging.error(e)

    def stemming(self, comment):
        logging.info("Applying stemming methods on train and test data")
        try:
            comment = comment.split()
            ps = PorterStemmer()
            comment = [ps.stem(word) for word in comment]
            comment = " ".join(comment) 
            return comment 
        except Exception as e:
            logging.error(
                "Error in applying stemming methods on train and test data"
            )
            logging.error(e)

    def lemmatization(self, comment):
        logging.info("Applying lemmatization method on train and test data")
        try: 
            print(comment) 
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

