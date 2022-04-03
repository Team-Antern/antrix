import numpy as np 
import pandas as pd 
import logging 

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

    def train_a_gensim_model(self):
        # train a gensim model
        logging.info("Train a gensim model")

        text_text = self.df_copy.text.apply(gensim.utils.simple_preprocess)
        model = gensim.models.Word2Vec(window=10, min_count=2, workers=4)
        model.build_vocab(text_text, progress_per=1000)
        model.train(text_text, total_examples=model.corpus_count, epochs=model.epochs)
        model.save(r"E:\Hackathon\UGAM\src\saved_model\ugam_texts.model")
        return model

    def get_word_embeddings(self, model):
        # get word embeddings
        logging.info("Get word embeddings")
        word_embeddings = model.wv
        return word_embeddings

    def get_similar(self, word, model):
        if word in model.wv:
            return model.wv.most_similar(word)[0]  # try runnign again
        else:
            return None

    def make_acolumn(self, model):
        # make a new column "most similar words" and get the most similar words for every word in text text
        logging.info(
            "Make a new column 'most similar words' and get the most similar words for every word in text text and leave the word whic is not present in the model"
        )
        #
        self.df_copy["most_similar_words"] = self.df_copy["text"].apply(
            lambda x: [
                self.get_similar(word, model) for word in word_tokenize(x)
            ]  # get the most similar words for every word in text text
        )
        self.test_data["most_similar_words"] = self.test_data["text"].apply(
            lambda x: [
                self.get_similar(word, model) for word in word_tokenize(x)
            ]  # get the most similar words for every word in text text
        )
        return self.df_copy, self.test_data

    def process_most_similar_words(self, text):

        # process most similar words
        logging.info("Process most similar words")
        # process the column most similar words row by row
        # tokenize the word
        text = word_tokenize(text)
        for j in text:
            if j.isalpha() == False:
                text.remove(j)
            if j == "None":
                text.remove(j)
            if j == "":
                text.remove(j)
                # convert str to int
            if j.isdigit():
                text.remove(j)
        text = " ".join(text)
        # remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        # remove numbers, None and empty strings
        text = re.sub(r"\d+", "", text)
        # remove None from text
        text = re.sub(r"None", "", text)
        # remove extra spaces
        text = re.sub(r"\s+", " ", text)
        # remove stop words
        from nltk.corpus import stopwords

        stop_words = set(stopwords.words("english"))
        text = [word for word in text.split() if word not in stop_words]
        # convert list to str
        text = " ".join(text)
        return text

    def fast_text_extract_features(self):
        self.logging.log(
            self.file_object,
            "In fast_text_extract_features method In Vectorization class: adding fast-text features"
        )
        try:
            def averaged_word2vec_vectorizer(corpus, model, num_features):
                vocabulary = set(model.wv.index_to_key)

                def average_word_vectors(words, model, vocabulary, num_features):
                    feature_vector = np.zeros((num_features,), dtype="float64")
                    nwords = 0.

                    for word in words:
                        if word in vocabulary:
                            nwords = nwords + 1.
                            feature_vector = np.add(feature_vector, model.wv[word])
                    if nwords:
                        feature_vector = np.divide(feature_vector, nwords)

                    return feature_vector
                features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                            for tokenized_sentence in corpus]
                return np.array(features)

            # ft_model = FastText.load("ft_model")

            tokenized_docs_train = [doc.split()
                                    for doc in list(self.df_copy['text'])]
            ft_model = FastText(tokenized_docs_train, min_count=2,
                                vector_size=100, workers=4, window=40, sg=1, epochs=100)
            doc_vecs_ft_train = averaged_word2vec_vectorizer(
                tokenized_docs_train, ft_model, 300)
            doc_vecs_ft_train = pd.DataFrame(doc_vecs_ft_train)
            ft_model.save("ft_model.model")

            self.logging.log(
                self.file_object,
                "In fast_text_extract_features method In Vectorization class: successfully added fast-text features"
            )
            return doc_vecs_ft_train

        except Exception as e:
            self.logging.log(
                self.file_object,
                f"In fast_text_extract_features method In Vectorization class: Error in adding fast-text features: {e}"
            )
            raise e

    def tf_idf_vectorizer(self, column):
        logging.info(
            
            "In tf_idf_vectorizer method In Vectorization class: adding tf-idf features"
        )
        try:
            # tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, max_features=1000, stop_words='english')
            vectorizer = TfidfVectorizer(max_features=5000)

            extracted_data = list(vectorizer.fit_transform(self.df[column]).toarray())
            extracted_data = pd.DataFrame(extracted_data)
            extracted_data.head()
            extracted_data.columns = vectorizer.get_feature_names()

            vocab = vectorizer.vocabulary_
            mapping = vectorizer.get_feature_names()
            keys = list(vocab.keys())

            extracted_data.shape
            Modified_df = extracted_data.copy()
            print(Modified_df.shape)
            Modified_df.head()
            Modified_df.reset_index(drop=True, inplace=True)
            self.df.reset_index(drop=True, inplace=True)

            Final_Training_data = pd.concat([self.df, Modified_df], axis=1)

            Final_Training_data.drop(column, axis=1, inplace=True)
            Final_Training_data.to_csv("informative-reports-creater/src/features/final_training_vectorized.csv", index=False)

            joblib.dump(vectorizer, "informative-reports-creater/models/vectorizer.pkl")
            return Final_Training_data

        except Exception as e:
            logging.info(
                f"In tf_idf_vectorizer method In Vectorization class: Error in adding tf-idf features: {e}"
            )
            raise e 
            