import pandas as pd
import gensim

from zenml.steps import step, Output
from .build_features import FeatureEngineering 

@step 
def basic_feature_processing(data: pd.DataFrame) -> Output(
       ouput_data = pd.DataFrame
):
       feature_engine = FeatureEngineering(data)
       data = feature_engine.remove_features()
       return data 

@step 
def add_features(data: pd.DataFrame) -> Output(
       ouput_data = pd.DataFrame
):
       train_gensim_model = True
       feature_engine = FeatureEngineering(data) 
       if train_gensim_model:
                model = feature_engine.train_a_gensim_model()
                data = feature_engine.make_similar_features_cols(model) 
                #   process_most_similar_words
                data['most_similar_words'] = data["most_similar_words"].apply(
                    lambda x: feature_engine.process_most_similar_words(x)
                ) 
       else: 
            # load the trained gensim model
            model = gensim.models.Word2Vec.load(r"anfiles/informative-reports-creater/models/yt_text_model.model") 
            data = feature_engine.make_similar_features_cols(model)
            #   process_most_similar_words
            data['most_similar_words'] = data["most_similar_words"].apply(
                lambda x: feature_engine.process_most_similar_words(x)
            )

       return data 

@step
def vectorization(data: pd.DataFrame) -> Output(
       ouput_data = pd.DataFrame
): 
    column = 'text'
    data_copy = data.copy() 
    feature_engine = FeatureEngineering(data_copy)
    data_copy = feature_engine.tf_idf_vectorizer(data_copy[column])
    return data_copy


